"""Money-Agent 的 LangGraph Agent 编排。

图结构（tools.py 里的 5 个工具作为 Function Calling 工具绑定给大模型）：

    parse_intent ──► agent ──(有 tool_calls)──► tools ──┐
                       ▲                                │
                       └────────────────────────────────┘
                       │
                       └──(无 tool_calls)──► ensure ──► report ──► END

各节点职责：
- parse_intent : 用大模型（不可用时用正则）解析出【标的名称】与【持仓参数】
- agent        : 绑定 5 个工具的 LLM，决定下一步调用哪些工具（Function Calling 循环）
- tools        : 执行工具；对 get_fund_history_kline 的 7/15/30 三次调用**并行**执行
- ensure       : 兜底补齐缺失的 Step1~Step5，保证报告所需数据齐全（顺序确定）
- report       : 用固定 Markdown 模板生成报告（第 1/2 章数字由工具结果确定性渲染）

约束：任何工具失败都在报告中显式标注「数据缺失」，绝不编造数字。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import (
    TOOLS_SCHEMA,
    execute_tool,
    execute_tools_parallel,
    market_status,
    warm_up,
)
import tools

logger = logging.getLogger("money_agent.agent")


def _load_dotenv() -> None:
    """自动加载项目根目录的 `.env`。

    - 真实环境变量优先（override=False），方便 CI / 主系统注入覆盖；
    - 未安装 python-dotenv 时静默跳过，不影响运行。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


# 必须在读取任何 os.getenv 之前执行
_load_dotenv()

MAX_TOOL_ROUNDS = int(os.getenv("MAX_AGENT_TURNS", "10"))

DISCLAIMER = (
    "> 免责声明：本分析仅基于历史公开数据做客观统计，**不构成任何投资建议。"
    "市场存在风险，过往收益不代表未来表现，请独立审慎决策。**"
)


# ==========================================================================
# 1. 状态定义
# ==========================================================================

class AgentState(TypedDict, total=False):
    """LangGraph 状态。"""

    query: str
    messages: Annotated[list, add_messages]

    # 解析出的标的与持仓
    fund_name: str
    fund_code: str
    holding: dict[str, Any]

    # 各工具结果
    latest_info: dict[str, Any]
    kline: dict[str, dict[str, Any]]  # {"7":..., "15":..., "30":...}
    position: dict[str, Any]
    rag: dict[str, Any]

    errors: list[str]
    tool_rounds: int
    report: str
    engine: str


# ==========================================================================
# 2. 意图解析（标的名称 + 持仓参数）
# ==========================================================================

HOLDING_MARKERS = ("我的持仓", "持仓", "成本", "份额", "买入", "建仓")
_SHARES_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:份|份额)")
_COST_RE = re.compile(r"成本(?:净值|单价|价)?\s*[:：]?\s*([\d.]+)")
_SHARES_KV_RE = re.compile(r"份额\s*[:：]?\s*([\d.]+)")
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _looks_like_code(text: str) -> bool:
    """判断一个字符串是不是 6 位基金代码。"""
    return bool(re.fullmatch(r"\d{6}", str(text or "").strip()))


def _extract_fund_code(text: str) -> str | None:
    """从文本中提取 6 位基金代码。

    必须能在基金名称表里查到才算命中，避免把「份额100000份」里的数字误当代码。
    """
    for match in _CODE_RE.finditer(str(text or "")):
        code = match.group(1)
        if tools.get_fund_name_by_code(code):
            return code
    return None


def parse_holding_rule_based(text: str) -> dict[str, Any]:
    """正则解析持仓参数（不依赖 LLM）。"""
    shares = cost = None
    m = _SHARES_RE.search(text)
    if m:
        shares = float(m.group(1).replace(",", ""))
    else:
        m = _SHARES_KV_RE.search(text)
        if m:
            shares = float(m.group(1))
    m = _COST_RE.search(text)
    if m:
        cost = float(m.group(1))
    return {"shares": shares, "cost_nav": cost}


def extract_target_text(query: str) -> str:
    """切出 query 中“只描述标的”的部分（丢掉持仓描述）。"""
    cut = len(query)
    for marker in HOLDING_MARKERS:
        idx = query.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    head = query[:cut].strip(" ，,。.：:；;、\n")
    return head or query.strip()


INTENT_PROMPT = """你是金融分析 Agent 的意图解析模块。请从用户输入中提取两个字段，只输出 JSON：

{
  "fund_name": "用户提到的基金名称（不含持仓描述、不含“今天如何”这类问句词）",
  "shares": 持仓份额（数字，未提及填 null）,
  "cost_nav": 持仓成本净值（数字，未提及填 null）
}

要求：
- fund_name 只保留标的名词本身，例如“半导体设备基金”“招商中证白酒”；
- 不要臆造用户没有提供的数字；未提及一律填 null；
- 只输出 JSON，不要任何解释。"""


def parse_intent(state: AgentState, llm: Any | None) -> dict[str, Any]:
    """LangGraph 节点：解析标的名称与持仓参数。"""
    query = state.get("query", "")
    holding = parse_holding_rule_based(query)
    fund_name = extract_target_text(query)

    if llm is not None:
        try:
            reply = llm.invoke(
                [SystemMessage(content=INTENT_PROMPT), HumanMessage(content=query)]
            )
            data = _loads_json(getattr(reply, "content", "") or "")
            if data:
                if data.get("fund_name"):
                    fund_name = str(data["fund_name"]).strip()
                for key, target in (("shares", "shares"), ("cost_nav", "cost_nav")):
                    if data.get(key) is not None:
                        holding[target] = float(data[key])
        except Exception as exc:  # noqa: BLE001 - 解析失败要能降级，不能中断
            logger.warning("LLM 意图解析失败，改用正则结果：%s", exc)

    return {"fund_name": fund_name, "holding": holding}


def _loads_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ==========================================================================
# 3. 提示词
# ==========================================================================

SYSTEM_PROMPT = """你是一个严谨的金融数据分析 Agent（Money-Agent），只能通过工具获取真实数据。

【核心原则】
1. 所有事实必须来自工具返回的数据；禁止编造行情、净值、K线、研报数字。
2. 禁止预测未来涨跌，禁止输出“一定要加仓/减仓”这类确定性建议。
3. 区分场外公募基金的「盘中估算净值（仅参考）」与「官方正式净值」。

【必须按顺序调用工具】
Step1 get_fund_code_by_name(fund_name)          —— 用基金名称换代码（若已给出 6 位代码可跳过）
Step2 get_fund_latest_info(fund_code)           —— 取最新净值（注意 tag 是「估算」还是「正式」）
Step3 get_fund_history_kline(fund_code, period_days) —— 分别用 period_days=7、15、30 各调一次
Step4 calc_user_position(fund_cost, current_net, share) —— 用户给了持仓成本与份额时才调用
Step5 rag_search_industry_report(industry_name) —— 用该基金的行业/主题名检索研报

【规则】
- 一轮可以同时发起多个工具调用；Step3 的三次调用请在同一轮内一起发起。
- 当上述步骤都已拿到结果后，不要再调用工具，直接回复“数据采集完成”。
- 工具返回 error 时不要重试超过一次，也不要编造数字，如实说明数据缺失。
- 全程使用中文。"""


def agent_node(state: AgentState, llm: Any) -> dict[str, Any]:
    """LangGraph 节点：绑定工具的 LLM，发起 Function Calling。

    模型不可用（key 无效 / 网络失败 / 服务报错）时**不抛异常中断链路**，
    而是直接跳到 ensure 节点走确定性数据采集，保证报告仍能产出。
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.get("messages", []))
    try:
        bound = llm.bind_tools(TOOLS_SCHEMA)
        reply = bound.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - 任何模型侧问题都应降级而非崩溃
        logger.warning("LLM 工具调用失败，改用确定性数据采集：%s", exc)
        errors = list(state.get("errors", []))
        errors.append("【LLM降级】模型调用失败，本次已改用规则化数据采集与文案")
        return {
            "messages": [AIMessage(content="llm-unavailable: fallback")],
            "tool_rounds": MAX_TOOL_ROUNDS,  # 强制走 ensure 分支
            "errors": errors,
        }
    return {
        "messages": [reply],
        "tool_rounds": state.get("tool_rounds", 0) + 1,
    }


def should_continue(state: AgentState) -> str:
    """条件边：还有工具调用就继续，否则收尾。"""
    if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        logger.warning("工具调用轮次达到上限 %d，强制进入收尾", MAX_TOOL_ROUNDS)
        return "ensure"
    last = (state.get("messages") or [None])[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "tools" if tool_calls else "ensure"


# ==========================================================================
# 4. 工具执行节点（并行）
# ==========================================================================

def tools_node(state: AgentState) -> dict[str, Any]:
    """执行上一轮 LLM 请求的工具调用；K 线的多次调用并行执行。"""
    last = (state.get("messages") or [None])[-1]
    raw_calls = list(getattr(last, "tool_calls", None) or [])
    if not raw_calls:
        return {}

    # 解析出 (tool_call_id, name, args)，并归一化参数名
    parsed: list[tuple[str, str, dict[str, Any]]] = []
    for call in raw_calls:
        name = call.get("name") or call.get("function", {}).get("name", "")
        args = call.get("args") or call.get("function", {}).get("arguments") or {}
        if isinstance(args, str):
            args = _loads_json(args) or {}
        call_id = call.get("id") or name
        parsed.append((str(call_id), name, _normalize_args(name, args)))

    # 关键：7/15/30 天 K 线并行拉取，其余工具也并行
    kline_calls = [(n, a) for _, n, a in parsed if n == "get_fund_history_kline"]
    other_calls = [(n, a) for _, n, a in parsed if n != "get_fund_history_kline"]
    kline_results = iter(execute_tools_parallel(kline_calls))
    other_results = iter(execute_tools_parallel(other_calls))

    messages: list[ToolMessage] = []
    errors = list(state.get("errors", []))
    # 以已有 kline 为底，避免多轮调用时覆盖掉上一轮结果
    updates: dict[str, Any] = {"kline": dict(state.get("kline") or {})}

    for call_id, name, args in parsed:
        result = next(kline_results) if name == "get_fund_history_kline" else next(other_results)
        messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call_id,
            )
        )
        if result.get("error"):
            if result["error"] not in errors:
                errors.append(result["error"])
        _absorb(updates, name, args, result)

    updates["messages"] = messages
    updates["errors"] = errors
    return updates


def _normalize_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """容忍模型把参数名写错（例如 code / days）。"""
    args = dict(args)
    if name in ("get_fund_latest_info", "get_fund_history_kline"):
        if "fund_code" not in args:
            for alt in ("code", "fundCode", "symbol"):
                if alt in args:
                    args["fund_code"] = args.pop(alt)
                    break
    if name == "get_fund_history_kline" and "period_days" not in args:
        for alt in ("days", "period", "periodDays"):
            if alt in args:
                args["period_days"] = args.pop(alt)
                break
    if name == "get_fund_code_by_name" and "fund_name" not in args:
        for alt in ("name", "fundName", "query"):
            if alt in args:
                args["fund_name"] = args.pop(alt)
                break
    if name == "rag_search_industry_report" and "industry_name" not in args:
        for alt in ("industry", "name", "theme"):
            if alt in args:
                args["industry_name"] = args.pop(alt)
                break
    return args


def _absorb(updates: dict[str, Any], name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    """把工具结果同步进结构化状态，供报告渲染使用。"""
    if result.get("error"):
        return
    if name == "get_fund_code_by_name":
        updates["fund_code"] = result.get("fund_code")
        if result.get("fund_name"):
            updates["fund_name"] = result["fund_name"]
    elif name == "get_fund_latest_info":
        updates["latest_info"] = result
        if result.get("fund_code"):
            updates["fund_code"] = result["fund_code"]
    elif name == "get_fund_history_kline":
        kline = dict(updates.get("kline") or {})
        key = str(args.get("period_days"))
        kline[key] = result
        updates["kline"] = kline
    elif name == "calc_user_position":
        updates["position"] = result
    elif name == "rag_search_industry_report":
        updates["rag"] = result


# ==========================================================================
# 5. 兜底补齐节点（保证 Step1~Step5 齐全）
# ==========================================================================

def ensure_node(state: AgentState) -> dict[str, Any]:
    """确定性补齐缺失的步骤，确保报告所需数据完整。"""
    updates: dict[str, Any] = {}
    errors = list(state.get("errors", []))
    holding = dict(state.get("holding") or {})

    raw_name = str(state.get("fund_name") or "").strip()
    fund_code = state.get("fund_code")
    fund_name = raw_name

    # 用户直接给了 6 位代码，或从原句里能提取到代码 → 跳过 Step1，并反查名称
    if not fund_code:
        candidate = _extract_fund_code(fund_code or raw_name or state.get("query", ""))
        if candidate:
            fund_code = candidate
            updates["fund_code"] = candidate

    # Step1 名称 → 代码
    if not fund_code:
        res = execute_tool("get_fund_code_by_name", {"fund_name": raw_name or state.get("query", "")})
        if res.get("error"):
            errors.append(res["error"])
            return {"errors": list(dict.fromkeys(errors)), "fund_code": None}
        fund_code = res["fund_code"]
        updates["fund_code"] = fund_code
        updates["fund_name"] = res.get("fund_name") or fund_code
        fund_name = updates["fund_name"]

    # 标题补全：名称为空或本身就是代码时，按代码反查简称
    if fund_code and (not fund_name or _looks_like_code(fund_name)):
        resolved = tools.get_fund_name_by_code(fund_code)
        if resolved:
            updates["fund_name"] = resolved
            fund_name = resolved

    # Step2 最新净值
    latest = state.get("latest_info") or {}
    if not latest.get("nav"):
        res = execute_tool("get_fund_latest_info", {"fund_code": fund_code})
        if res.get("error"):
            errors.append(res["error"])
        else:
            updates["latest_info"] = res
            latest = res

    # Step3 7/15/30 天 K 线（并行）
    kline = dict(state.get("kline") or {})
    missing = [d for d in (7, 15, 30) if str(d) not in kline]
    if missing:
        calls = [("get_fund_history_kline", {"fund_code": fund_code, "period_days": d}) for d in missing]
        for period, res in zip(missing, execute_tools_parallel(calls)):
            if res.get("error"):
                errors.append(res["error"])
            else:
                kline[str(period)] = res
        updates["kline"] = kline

    # Step4 持仓盈亏
    if holding.get("shares") is not None and holding.get("cost_nav") is not None:
        current = (latest or {}).get("nav")
        if current is None:
            errors.append("【数据缺失】缺少当前净值，无法计算持仓盈亏")
        elif not (state.get("position") or {}).get("return_pct"):
            res = execute_tool(
                "calc_user_position",
                {
                    "fund_cost": holding["cost_nav"],
                    "current_net": current,
                    "share": holding["shares"],
                },
            )
            if res.get("error"):
                errors.append(res["error"])
            else:
                updates["position"] = res

    # Step5 研报检索
    if not state.get("rag"):
        res = execute_tool("rag_search_industry_report", {"industry_name": updates.get("fund_name") or fund_name})
        updates["rag"] = res

    updates["errors"] = list(dict.fromkeys(errors))
    return updates


# ==========================================================================
# 6. 报告生成节点
# ==========================================================================

REPORT_PROMPT = """你将收到一份已通过工具实测得到的结构化数据（JSON）。
请严格只依据这份数据，撰写分析报告的 4 个定性章节，并以 JSON 输出，键固定为：

{
  "trend": "第3章 近期走势分析：结合近7/15/30日涨跌幅与近30日最大回撤，描述当前处于上涨/下跌/震荡、以及波动大小",
  "fundamentals": "第4章 板块基本面信息：依据 RAG 返回的 summary 撰写；若标记为占位(is_stub=true)或为空，必须原样引用该占位提示，不得编造研报观点",
  "options": "第5章 客观评估与可选操作方向：分别给出①思路A(考虑加仓)②思路B(考虑减仓)③思路C(继续持有)，每项写清【适用场景的数据条件】与【风险】",
  "watchlist": "第6章 板块关注推荐：客观列举可跟踪方向与理由；若缺少研报数据则如实说明无法列举"
}

【严格要求】
1. **只使用给定 JSON 中出现的数字**。JSON 里没有的指标（如单日明细、费率、持仓时长等）一律不要提及，更不要自行计算或推算。
2. 「区间涨跌幅_pct」就是近7/15/30日涨跌幅，直接引用；不要与任何其他口径的数字混用。
3. 章节正文里**不要**再写“第3章”“近期走势分析：”这类标题前缀，直接写内容即可（标题由模板统一生成）。
4. 每个章节控制在 150 字以内，简洁客观，不要罗列大段数字。
5. 不得预测未来涨跌，不得给出确定性买卖建议；options 必须含思路A/B/C 三段并强调这是可选思路。
6. 数据缺失处如实写“数据缺失”。
7. 全部使用中文，只输出 JSON。"""


_CHAPTER_PREFIX_RE = re.compile(
    r"^\s*[*_#\s]*第\s*[1-6一二三四五六]\s*[章节]\s*[^\n：:]{0,20}?[：:]\s*"
)
_CHAPTER_TITLES = (
    "近期走势分析", "板块基本面信息", "客观评估与可选操作方向",
    "客观评估 & 可选操作方向", "客观评估&可选操作方向", "板块关注推荐",
)


def _strip_chapter_prefix(text: str) -> str:
    """去掉 LLM 复述的章节编号/标题，避免与模板标题重复。"""
    text = (text or "").strip()
    text = _CHAPTER_PREFIX_RE.sub("", text, count=1)
    for title in _CHAPTER_TITLES:
        for pattern in (f"{title}：", f"{title}:"):
            if text.startswith(pattern):
                text = text[len(pattern) :].strip()
    return text.strip()


def report_node(state: AgentState, llm: Any | None) -> dict[str, Any]:
    """LangGraph 节点：渲染固定模板报告。"""
    facts = _build_facts(state)
    engine = "rule-based"
    narrative: dict[str, str] = {}

    if llm is not None:
        try:
            reply = llm.invoke(
                [
                    SystemMessage(content="你是严谨的金融分析写作助手，只依据给定数据写作，绝不编造。"),
                    HumanMessage(
                        content=REPORT_PROMPT
                        + "\n\n【实测数据 JSON】\n"
                        + json.dumps(_llm_facts(state), ensure_ascii=False, default=str)
                    ),
                ]
            )
            data = _loads_json(getattr(reply, "content", "") or "") or {}
            narrative = {
                k: _strip_chapter_prefix(str(data.get(k, "")))
                for k in ("trend", "fundamentals", "options", "watchlist")
            }
            if any(narrative.values()):
                engine = "llm"
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 报告生成失败，降级为规则化文案：%s", exc)

    if not any(narrative.values()):
        narrative = rule_based_narrative(facts)

    return {"report": render_report(state, narrative), "engine": engine}


def _build_facts(state: AgentState) -> dict[str, Any]:
    """完整事实集（供规则化文案使用，含 K 线明细）。"""
    kline = state.get("kline") or {}
    return {
        "fund_name": state.get("fund_name"),
        "fund_code": state.get("fund_code"),
        "holding": state.get("holding") or {},
        "latest_info": state.get("latest_info") or {},
        "kline": kline,
        "position": state.get("position") or {},
        "rag": state.get("rag") or {},
        "errors": state.get("errors") or [],
    }


def _llm_facts(state: AgentState) -> dict[str, Any]:
    """喂给大模型的**无歧义**事实集。

    只保留口径明确的指标，并附带口径说明：避免出现
    「近7个交易日区间涨跌幅」与「近7日收益率」两套数字被混用。
    """
    latest = state.get("latest_info") or {}
    k30 = (state.get("kline") or {}).get("30") or {}
    returns = k30.get("returns") or {}
    position = state.get("position") or {}
    holding = state.get("holding") or {}

    dd = k30.get("max_drawdown_30_pct")
    if dd is None:
        dd = k30.get("max_drawdown_pct")

    position_out: dict[str, Any] = {}
    if position and not position.get("error"):
        position_out = {
            "持仓份额": position.get("share"),
            "持仓成本净值": position.get("fund_cost"),
            "当前净值": position.get("current_net"),
            "账面收益率_pct": position.get("return_pct"),
            "账面盈亏_元": position.get("pnl_amount"),
            "市值_元": position.get("market_value"),
        }

    return {
        "标的": {
            "名称": state.get("fund_name"),
            "代码": state.get("fund_code"),
            "市场状态": latest.get("market_status"),
        },
        "最新净值": {
            "净值": latest.get("nav"),
            "口径": latest.get("nav_kind"),
            "标记": latest.get("tag"),
            "时间": latest.get("nav_time"),
            "当日涨跌幅_pct": latest.get("change_pct"),
        },
        "区间涨跌幅_pct": {
            "近7日": returns.get("7"),
            "近15日": returns.get("15"),
            "近30日": returns.get("30"),
            "口径": "自然日口径（不晚于最新净值日往前推 N 个自然日的最近一根净值）",
        },
        "最大回撤_pct": {
            "近30日": dd,
            "口径": "自然日口径：区间最高点回撤到最低点的跌幅，负数表示回撤",
        },
        "用户持仓": {
            "持仓份额": holding.get("shares"),
            "持仓成本净值": holding.get("cost_nav"),
            **(position_out or {"说明": "未提供完整持仓信息或无法计算，数据缺失"}),
        },
        "行业研报RAG": state.get("rag") or {},
        "数据缺失项": state.get("errors") or [],
    }


def rule_based_narrative(facts: dict[str, Any]) -> dict[str, str]:
    """无 LLM 时的规则化兜底文案（同样只引用已获取的数据）。"""
    k30 = (facts.get("kline") or {}).get("30") or {}
    returns = k30.get("returns") or {}
    dd = k30.get("max_drawdown_30_pct")
    if dd is None:
        dd = k30.get("max_drawdown_pct")

    def pct(v: Any) -> str:
        return "数据缺失" if v is None else f"{v:+.2f}%"

    trend = (
        f"区间表现：近7日 {pct(returns.get('7'))}、近15日 {pct(returns.get('15'))}、"
        f"近30日 {pct(returns.get('30'))}；"
        + (f"近30日最大回撤 {dd:.2f}%。" if dd is not None else "近30日最大回撤数据缺失。")
    )
    if not k30:
        trend = "【数据缺失】未获取到历史净值序列，本次无法分析走势。"

    rag = facts.get("rag") or {}
    fundamentals = rag.get("summary") or "未获取到研报数据。"

    latest = facts.get("latest_info") or {}
    nav = latest.get("nav")
    nav_txt = "数据缺失" if nav is None else f"{nav:.4f}"
    options = (
        "⚠️ 以下仅为数据对应的可选思路，不构成任何买卖建议。\n\n"
        f"① **思路A：考虑加仓**。适用场景：短期明显回调且基本面无利空、估值处于相对低位。"
        f"风险：可能继续下跌，越跌越买会放大回撤（当前近30日最大回撤 "
        f"{'数据缺失' if dd is None else f'{dd:.2f}%'}）。\n\n"
        "② **思路B：考虑减仓**。适用场景：短期快速上涨后回撤开始放大。"
        "风险：可能卖在阶段性低点、错失后续反弹，且场外基金赎回存在费用与到账时滞。\n\n"
        "③ **思路C：继续持有**。适用场景：区间震荡、基本面平稳、波动可控。"
        f"风险：若趋势转为单边下行，将被动承受回撤（参考当前净值 {nav_txt}）。"
    )

    sources = rag.get("sources") or []
    if sources:
        watch = "基于检索到的研报，可持续跟踪：\n" + "\n".join(
            f"- {s.get('title')}（{s.get('institution')}，{s.get('date')}）" for s in sources
        )
    else:
        watch = "暂无可用研报数据，无法客观列举跟踪方向。"

    return {"trend": trend, "fundamentals": fundamentals, "options": options, "watchlist": watch}


# ==========================================================================
# 7. 固定 Markdown 模板
# ==========================================================================

def _is_exchange(state: AgentState) -> bool:
    return bool((state.get("latest_info") or {}).get("is_exchange_traded"))


def _report_title(state: AgentState, name: str, code: str) -> str:
    """标题：优先「名称（代码）」；名称为空或与代码重复时退化为单一展示。"""
    name = (name or "").strip()
    code = (code or "").strip()
    if name and code and name != code:
        return f"# 标的分析报告：{name}（{code}）"
    if name or code:
        return f"# 标的分析报告：{name or code}"
    return "# 标的分析报告：未能识别标的（数据缺失）"


def render_report(state: AgentState, narrative: dict[str, str]) -> str:
    """按固定模板渲染报告；第 1、2 章数字全部来自工具结果。"""
    name = state.get("fund_name") or "未知标的"
    code = state.get("fund_code") or "未知代码"
    latest = state.get("latest_info") or {}
    holding = state.get("holding") or {}
    k30 = (state.get("kline") or {}).get("30") or {}
    returns = k30.get("returns") or {}
    position = state.get("position") or {}
    errors = state.get("errors") or []

    # ---------- 标题 ----------
    blocks: list[str] = [
        _report_title(state, name, code),
        f"> 市场状态：{latest.get('market_status', market_status())}"
        f"｜数据源：AKShare / 东方财富公开接口"
        f"｜生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
    ]

    # ---------- 1. 基础行情数据 ----------
    sec1 = ["## 1. 基础行情数据（来自AKShare）"]
    nav = latest.get("nav")
    if nav is None:
        sec1.append("- 当前价格/净值：数据缺失，数据时间：数据缺失；未能获取净值，估值不可用")
    else:
        if latest.get("tag") == "估算":
            tag = "⚠️ 盘中估算值（仅参考，非正式成交价，正式净值盘后公布）"
        elif _is_exchange(state):
            tag = "场内价格（非场外净值）"
        else:
            tag = "官方正式净值"
        change = latest.get("change_pct")
        change_txt = f"，当日涨跌幅 {change:+.2f}%" if change is not None else ""
        sec1.append(
            f"- 当前价格/净值：{nav:.4f}{change_txt}，数据时间：{latest.get('nav_time') or '未知'}；{tag}"
        )
    if latest.get("tag") == "正式" and latest.get("estimate_nav") is not None:
        sec1.append(
            f"- ⚠️ 另有盘中估算净值：{latest['estimate_nav']:.4f}"
            f"（{latest.get('estimate_time') or '时间未知'}，仅供参考）"
        )
    if latest.get("note"):
        sec1.append(f"- ℹ️ {latest['note']}")
    if _is_exchange(state) and latest.get("unit_nav") is not None:
        sec1.append(
            f"- 基金单位净值（供参考，非交易价格）：{latest['unit_nav']:.4f}"
            f"（{latest.get('unit_nav_date') or '日期未知'}）"
        )
    sec1.append(f"- 近7日涨跌幅：{_fmt_pct(returns.get('7'))}")
    sec1.append(f"- 近15日涨跌幅：{_fmt_pct(returns.get('15'))}")
    sec1.append(f"- 近30日涨跌幅：{_fmt_pct(returns.get('30'))}")
    dd30 = k30.get("max_drawdown_30_pct")
    if dd30 is None:
        dd30 = k30.get("max_drawdown_pct")
    sec1.append(
        f"- 近30日最大回撤：{'数据缺失' if dd30 is None else f'{dd30:.2f}%'}"
        "（计算：区间最低点相对区间最高点跌幅）"
    )
    blocks.append("\n".join(sec1))

    # ---------- 2. 用户持仓现状 ----------
    sec2 = ["## 2. 用户持仓现状"]
    shares, cost = holding.get("shares"), holding.get("cost_nav")
    sec2.append(
        f"持仓份额：{'未提供' if shares is None else f'{shares:,.2f}'}"
        f"；持仓成本净值：{'未提供' if cost is None else f'{cost:.4f}'}"
    )
    if position and not position.get("error"):
        sec2.append(
            f"当前账面收益率：{position['return_pct']:+.2f}%（仅账面，**不含申购赎回费、管理费**）"
        )
        sec2.append(f"持仓盈亏金额：{position['pnl_amount']:+,.2f} 元（按 {position['current_net']:.4f} 计）")
    elif shares is None or cost is None:
        sec2.append("当前账面收益率：数据缺失（未提供持仓成本或份额，无法计算）")
        sec2.append("持仓盈亏金额：数据缺失")
    else:
        sec2.append("当前账面收益率：数据缺失（缺少当前净值或计算失败，无法计算）")
        sec2.append("持仓盈亏金额：数据缺失")
    blocks.append("\n".join(sec2))

    # ---------- 3~6 ----------
    blocks.append("## 3. 近期走势分析")
    blocks.append(narrative.get("trend") or "数据不足，无法给出走势判断。")
    blocks.append("## 4. 板块基本面信息（来自行业研报RAG）")
    blocks.append(narrative.get("fundamentals") or (state.get("rag") or {}).get("summary") or "未获取到研报数据。")
    blocks.append("## 5. 客观评估 & 可选操作方向")
    blocks.append(narrative.get("options") or "数据不足，无法给出可选思路。")
    blocks.append("## 6. 板块关注推荐（客观列举，不推荐买入）")
    blocks.append(narrative.get("watchlist") or "暂无可列举的跟踪方向。")

    # ---------- 数据缺失说明 ----------
    if errors:
        gap = ["## ⚠️ 数据缺失与局限"]
        gap += [f"- {e}" for e in dict.fromkeys(errors)]
        gap.append("- 说明：接口报错或返回空数据时，相关周期分析存在局限性，请勿据此推断未列出的数字。")
        blocks.append("\n".join(gap))

    blocks.append(DISCLAIMER)
    return "\n\n".join(blocks) + "\n"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "数据缺失"
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "数据缺失"


# ==========================================================================
# 8. 组装 LangGraph
# ==========================================================================

@dataclass
class GraphDeps:
    """图运行所需的外部依赖（LLM 可为 None → 全规则化模式）。"""

    llm: Any | None = None


def build_graph(deps: GraphDeps):
    """构建并编译 LangGraph。"""

    def _parse(state: AgentState) -> dict[str, Any]:
        return parse_intent(state, deps.llm)

    def _agent(state: AgentState) -> dict[str, Any]:
        if deps.llm is None:
            # 无 LLM：跳过 Function Calling 循环，直接走确定性采集
            return {"messages": [AIMessage(content="no-llm: skip tool loop")], "tool_rounds": 1}
        return agent_node(state, deps.llm)

    def _report(state: AgentState) -> dict[str, Any]:
        return report_node(state, deps.llm)

    graph = StateGraph(AgentState)
    graph.add_node("parse_intent", _parse)
    graph.add_node("agent", _agent)
    graph.add_node("tools", tools_node)
    graph.add_node("ensure", ensure_node)
    graph.add_node("render_report", _report)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "agent")
    graph.add_conditional_edges(
        "agent",
        (lambda state: "ensure" if deps.llm is None else should_continue(state)),
        {"tools": "tools", "ensure": "ensure"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("ensure", "render_report")
    graph.add_edge("render_report", END)
    return graph.compile()


# ==========================================================================
# 9. 对外入口
# ==========================================================================

@dataclass
class FinancialAgent:
    """Money-Agent 对外封装。"""

    llm: Any | None = None
    _graph: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._graph = build_graph(GraphDeps(llm=self.llm))

    def analyze(self, query: str) -> dict[str, Any]:
        """运行完整链路，返回结构化结果（含最终报告）。"""
        # 在主线程预热公共依赖（名称表 / 交易日历），避免并行 worker 首次触发 V8
        warm_up()
        init: AgentState = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "errors": [],
            "tool_rounds": 0,
        }
        final = self._graph.invoke(init, {"recursion_limit": 50})
        return {
            "fund_name": final.get("fund_name"),
            "fund_code": final.get("fund_code"),
            "holding": final.get("holding"),
            "latest_info": final.get("latest_info"),
            "kline": final.get("kline"),
            "position": final.get("position"),
            "rag": final.get("rag"),
            "errors": final.get("errors"),
            "engine": final.get("engine"),
            "report": final.get("report", ""),
        }

    def analyze_text(self, query: str) -> str:
        return self.analyze(query)["report"]


def build_llm() -> Any | None:
    """按环境变量构造 OpenAI 兼容 LLM；未配置有效 key 时返回 None（走规则化模式）。"""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    if _is_placeholder_key(api_key):
        logger.warning("检测到 OPENAI_API_KEY 仍是占位符，已跳过 LLM，改用规则化模式")
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        logger.warning("langchain-openai 未安装，降级为规则化模式：%s", exc)
        return None
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        openai_api_key=api_key,
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        max_retries=1,
    )


def _is_placeholder_key(api_key: str) -> bool:
    """判断 key 是否还是 .env.example 里的占位符，避免拿着假 key 去请求。"""
    lowered = api_key.lower()
    return lowered.startswith("sk-xxx") or "xxxxxxxx" in lowered or lowered in {"your-api-key", "changeme"}


def create_agent(use_llm: bool = True) -> FinancialAgent:
    """便捷工厂：use_llm=False 强制走规则化模式。"""
    return FinancialAgent(llm=build_llm() if use_llm else None)
