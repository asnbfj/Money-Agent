"""编排器：把解析 → 工具采集 → 统计 → 定性文案 → 固定模板串起来。

关键保证：
- 第 1、2 章的数字全部由工具层实测数据确定性计算，LLM 无法篡改；
- 生成完整报告前，四个必备工具必须都成功返回（LLM 循环中若遗漏会自动补齐）；
- 任何工具失败/空数据都会写入报告的“数据缺失与局限”章节，禁止编造。
"""

from __future__ import annotations

import json
import re
from typing import Any

from money_agent.agent.llm import ChatLLM
from money_agent.agent.prompts import (
    GATHERING_SYSTEM_PROMPT,
    build_narrative_messages,
    rule_based_narrative,
)
from money_agent.agent.tools import MANDATORY_TOOLS, ToolContext, ToolRegistry
from money_agent.config import Settings
from money_agent.data.analytics import compute_analytics
from money_agent.exceptions import LLMError, MoneyAgentError
from money_agent.models import AnalysisReport, Holding, History
from money_agent.report import render
from money_agent.utils import now_str

# 非基金标的（A股个股/板块）的确定性采集顺序
_GENERIC_ORDER = ("get_price_history", "get_latest_quote", "search_research_reports")

_HOLDING_MARKERS = ("我的持仓", "持仓", "成本", "份额", "买入", "建仓")

_SHARES_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:份|份额)")
_COST_RE = re.compile(r"成本(?:净值|单价|价)?\s*[:：]?\s*([\d.]+)")
_BUY_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_holding(text: str) -> Holding:
    """从自然语言里尽力解析持仓（份额 / 成本 / 买入日期）。"""
    shares = None
    cost = None
    buy_date = None

    m = _SHARES_RE.search(text)
    if m:
        shares = float(m.group(1).replace(",", ""))
    else:
        m = re.search(r"份额\s*[:：]?\s*([\d.]+)", text)
        if m:
            shares = float(m.group(1))

    m = _COST_RE.search(text)
    if m:
        cost = float(m.group(1))

    if any(k in text for k in ("买入", "建仓")):
        m = _BUY_DATE_RE.search(text)
        if m:
            buy_date = m.group(1)

    if shares is None and cost is None:
        return Holding()
    return Holding(shares=shares, cost_nav=cost, buy_date=buy_date, raw=text)


def extract_target_text(query: str) -> str:
    """从整个 query 里切出“只描述标的”的部分。"""
    cut = len(query)
    for marker in _HOLDING_MARKERS:
        idx = query.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    head = query[:cut].strip(" ，,。.：:；;、\n")
    return head or query.strip()


class FinancialAgent:
    """金融数据分析 Agent。"""

    def __init__(self, settings: Settings | None = None, *, use_llm: bool = True) -> None:
        self.settings = settings or Settings()
        self.llm = ChatLLM(self.settings)
        self.use_llm = use_llm and self.llm.available

    # ---------------- 对外入口 ----------------
    def analyze(self, query: str) -> AnalysisReport:
        holding = parse_holding(query)
        target_text = extract_target_text(query)

        ctx = ToolContext(settings=self.settings)
        registry = ToolRegistry(ctx)

        if self.use_llm:
            self._run_gathering_loop(ctx, registry, target_text, query)
        self._ensure_gathered(ctx, registry, target_text)

        analytics = ctx.analytics or compute_analytics(ctx.history or History())

        report = AnalysisReport(
            target=ctx.target,
            holding=holding,
            quote=ctx.quote,
            history=ctx.history or History(),
            analytics=analytics,
            reports=ctx.reports,
            industry=ctx.industry,
        )

        # 第 2 章：走 calc_user_position 工具，保证“账面收益率/盈亏”口径唯一
        self._apply_position_tool(report, ctx)

        engine = "rule-based"
        narrative: dict[str, str] = {}
        facts = self._build_facts(ctx, analytics, report.position)
        if self.use_llm:
            try:
                narrative = self._llm_narrative(facts)
                engine = f"llm({self.settings.model})"
            except LLMError as exc:
                ctx.add_error(f"LLM 定性章节生成失败，已降级为规则化文案：{exc}")
        if not narrative:
            narrative = rule_based_narrative(facts)

        report.narrative = narrative
        report.data_gaps = self._collect_gaps(ctx, analytics)
        report.generated_at = now_str()
        report.engine = engine
        return report

    def _apply_position_tool(self, report: AnalysisReport, ctx: ToolContext) -> None:
        """若用户提供了持仓，则调用 calc_user_position 工具计算账面收益。"""
        holding = report.holding
        current = report.mark_value
        if not holding.has_position or holding.cost_nav is None or holding.shares is None:
            return
        if current is None:
            ctx.add_error("缺少当前价格/净值，无法用 calc_user_position 计算持仓收益。")
            return
        result = ctx.fn_calc_user_position(
            fund_cost=holding.cost_nav, current_net=current, share=holding.shares
        )
        report.position = result

    def analyze_text(self, query: str) -> str:
        """一步到位：分析并渲染成固定模板报告。"""
        return render(self.analyze(query))

    # ---------------- 内部实现 ----------------
    def _run_gathering_loop(
        self, ctx: ToolContext, registry: ToolRegistry, target_text: str, query: str
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": GATHERING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户请求：{query}\n"
                    f"标的相关文本为：「{target_text}」，请先调用 get_fund_code_by_name 解析基金代码。"
                ),
            },
        ]
        for _ in range(self.settings.max_agent_turns):
            try:
                message = self.llm.chat(messages, tools=registry.specs, tool_choice="auto")
            except LLMError as exc:
                ctx.add_error(f"工具采集循环中断，转为确定性采集：{exc}")
                return
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break
            for call in tool_calls:
                name = call.get("function", {}).get("name", "")
                raw_args = call.get("function", {}).get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                result = registry.dispatch(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "content": result,
                    }
                )

    def _ensure_gathered(
        self, ctx: ToolContext, registry: ToolRegistry, target_text: str
    ) -> None:
        """确保必备数据都已获取（LLM 遗漏时确定性补齐）。"""
        # 1) 先解析标的：优先按基金名称（工具1），失败再退化到个股/板块
        if ctx.target is None:
            registry.dispatch("get_fund_code_by_name", {"fund_name": target_text})
        if ctx.target is None:
            registry.dispatch_internal("resolve_target", {"query": target_text})
        if ctx.target is None:
            raise MoneyAgentError("无法解析标的，分析终止")

        if ctx.target.kind == "fund":
            code = ctx.target.code
            if "get_fund_latest_info" not in ctx.called:
                registry.dispatch("get_fund_latest_info", {"fund_code": code})
            if "get_fund_history_kline" not in ctx.called:
                registry.dispatch(
                    "get_fund_history_kline", {"fund_code": code, "period_days": 30}
                )
            if "rag_search_industry_report" not in ctx.called:
                registry.dispatch(
                    "rag_search_industry_report", {"industry_name": ctx.target.name}
                )
            # 若先取了 latest 再取 history，latest 可能没享受到历史末值降级，补一次
            if (
                ctx.quote is None or ctx.quote.value is None
            ) and ctx.history and ctx.history.ok:
                registry.dispatch("get_fund_latest_info", {"fund_code": code})
            return

        # 非基金标的：个股 / 板块
        for name in _GENERIC_ORDER:
            if name not in ctx.called:
                registry.dispatch_internal(name, {})

    def _build_facts(
        self, ctx: ToolContext, analytics, position: dict[str, Any]
    ) -> dict[str, Any]:
        history = ctx.history
        recent = [p.to_dict() for p in (history.points[-20:] if history and history.points else [])]
        return {
            "target": ctx.target.to_dict() if ctx.target else {},
            "quote": ctx.quote.to_dict() if ctx.quote else {},
            "analytics": analytics.to_dict(),
            "history_recent": recent,
            "reports": [r.to_dict() for r in ctx.reports],
            "industry": ctx.industry,
            "position": position,
            "news": ctx.news,
            "data_gaps": ctx.errors,
        }

    def _llm_narrative(self, facts: dict[str, Any]) -> dict[str, str]:
        data = self.llm.chat_json(build_narrative_messages(facts))
        keys = ("trend", "fundamentals", "options", "watchlist")
        narrative = {k: str(data.get(k, "")).strip() for k in keys}
        if not narrative["options"]:
            narrative["options"] = rule_based_narrative(facts)["options"]
        return narrative

    def _collect_gaps(self, ctx: ToolContext, analytics) -> list[str]:
        gaps = list(ctx.errors)

        def covered(keyword: str) -> bool:
            return any(keyword in gap for gap in gaps)

        history = ctx.history
        if history is None or not history.ok:
            if not covered("历史行情"):
                gaps.append(
                    "历史行情接口获取失败，缺少近7/15/30日周期数据，本次分析存在局限性。"
                )
        else:
            for label, value in (
                ("近7日", analytics.ret_7),
                ("近15日", analytics.ret_15),
                ("近30日", analytics.ret_30),
            ):
                if value is None:
                    gaps.append(f"缺少足够的历史样本，无法计算{label}涨跌幅。")

        if ctx.quote is None or ctx.quote.value is None:
            if not (covered("估值") or covered("盘口") or covered("行情")):
                gaps.append("最新行情/净值缺失，无法按最新价计算持仓收益。")

        if not ctx.reports and not covered("研报"):
            gaps.append("未检索到公开券商研报，第4、6章基本面信息缺失。")

        return list(dict.fromkeys(gaps))
