"""提示词：数据采集循环的系统指令 + 定性章节生成的 JSON 指令。"""

from __future__ import annotations

import json
from typing import Any

GATHERING_SYSTEM_PROMPT = """你是一个严谨的金融数据分析 Agent（Money-Agent）。

【核心原则】
1. 所有事实必须来自工具返回的数据，禁止编造行情、净值、K线、研报数字。
2. 禁止预测未来涨跌，禁止输出确定性买卖建议（如“必须加仓/减仓”）。
3. 区分场外公募基金的「盘中估算净值（仅参考）」与「官方正式净值」。

【可用工具】
- get_fund_code_by_name(fund_name)：按基金名称查基金代码与类型
- get_fund_latest_info(fund_code)：最新净值（交易时段为盘中估算，盘后为官方正式）
- get_fund_history_kline(fund_code, period_days)：历史净值序列 + 区间涨跌幅 / 最大回撤
- rag_search_industry_report(industry_name)：行业公开研报检索
- calc_user_position(fund_cost, current_net, share)：持仓账面收益率与盈亏

【工作流程】
在给出任何分析之前，你必须按顺序调用以下工具获取真实数据：
1. get_fund_code_by_name —— 用用户提到的基金名称解析出代码（必须先做）
2. get_fund_latest_info —— 取得最新净值（注意返回里的 tag 是「估算」还是「正式」）
3. get_fund_history_kline —— 取近 30 天历史净值与区间统计
4. rag_search_industry_report —— 用该基金的行业/主题名检索公开研报

只有当以上 4 个工具都成功返回后，才可以结束工具调用。
如某工具返回 error 或空数据，请在最终说明中如实指出“数据缺失”，不得用推测数字填补。
请用中文与调用方沟通。"""


def build_narrative_messages(facts: dict[str, Any]) -> list[dict[str, str]]:
    """构造生成定性章节的对话（要求严格 JSON 输出）。"""
    schema_hint = {
        "trend": "第3章 近期走势分析：描述近7/15/30天K线特征（震荡/持续上涨/持续下跌）、波动与回撤情况。仅基于给定数据。",
        "fundamentals": "第4章 板块基本面信息：提炼行业现状、催化因素、风险点；若研报为空则说明无法给出基本面解读。",
        "options": "第5章 客观评估与可选操作方向：列出①思路A(考虑加仓)②思路B(考虑减仓)③思路C(继续持有)，每项给出【适用场景的数据条件】与【风险】。严禁给出结论性买卖建议。",
        "watchlist": "第6章 板块关注推荐：基于研报与行情客观列举值得跟踪的方向与简要理由，不推荐买入。",
    }
    instructions = f"""你将收到一份已经通过工具实测得到的结构化数据（JSON）。
请严格只依据这份数据，撰写固定报告的 4 个定性章节，并以 JSON 输出，键为 trend / fundamentals / options / watchlist。

要求：
- 不得编造数据里没有的数字、事件或研报；数据缺失时如实说明。
- 不得预测未来涨跌，不得给出确定性买卖建议。
- options 章节必须包含“思路A / 思路B / 思路C”三种可选方向及其适用场景与风险，并强调这是可选思路而非建议。
- 使用 Markdown 排版，可含小标题与列表；不要重复第1、2章的原始数字表格。
- 全部使用中文。

各章节写作要求：
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}

只输出 JSON，不要输出其它内容。"""
    return [
        {"role": "system", "content": "你是严谨的金融分析写作助手，只依据给定数据写作，绝不编造。"},
        {
            "role": "user",
            "content": instructions
            + "\n\n【实测数据 JSON】\n"
            + json.dumps(facts, ensure_ascii=False, default=str),
        },
    ]


def rule_based_narrative(facts: dict[str, Any]) -> dict[str, str]:
    """无 LLM 时的规则化兜底文案（同样只引用已获取的数据）。"""
    target = facts.get("target", {})
    analytics = facts.get("analytics", {}) or {}
    quote = facts.get("quote", {}) or {}
    reports = facts.get("reports", []) or []

    def pct(v: Any) -> str:
        return "数据缺失" if v is None else f"{v:+.2f}%"

    trend_label = analytics.get("trend_label") or "数据不足，无法判断"
    ret7, ret15, ret30 = analytics.get("ret_7"), analytics.get("ret_15"), analytics.get("ret_30")
    dd = analytics.get("max_drawdown_30")
    vol = analytics.get("volatility_30_annualized")

    trend_bits = [
        f"区间表现：近7日 {pct(ret7)}、近15日 {pct(ret15)}、近30日 {pct(ret30)}。",
    ]
    if dd is not None:
        trend_bits.append(f"近30日最大回撤 {dd:.2f}%。")
    if vol is not None:
        trend_bits.append(f"近30日年化波动率约 {vol:.2f}%。")
    trend_bits.append(f"综合判断：{trend_label}。")
    trend_text = " ".join(trend_bits)

    if reports:
        names = "、".join(
            sorted({r.get("institution", "") for r in reports if r.get("institution")})
        )
        industry = next((r.get("industry") for r in reports if r.get("industry")), "")
        fundamentals = (
            f"本次共检索到 {len(reports)} 篇相关公开研报"
            + (f"（机构：{names}）" if names else "")
            + (f"，所属行业：{industry}。" if industry else "。")
            + "具体观点请点击报告链接查阅原文（详见本节研报来源列表）。"
        )
    else:
        fundamentals = "未检索到可用公开研报，本次无法给出板块基本面解读（数据缺失）。"

    cur = quote.get("official_nav") or quote.get("value")
    cur_txt = "数据缺失" if cur is None else f"{cur:.4f}"
    kind = (facts.get("target") or {}).get("kind")
    exit_note = (
        "场外基金赎回存在费用与到账时滞"
        if kind == "fund"
        else "卖出可能承担交易成本与冲击成本"
    )
    options = (
        "⚠️ 以下仅为数据对应的可选思路，不构成任何买卖建议。\n\n"
        f"① **思路A：考虑加仓**。适用场景：短期明显回调且基本面无利空、估值处于相对低位。"
        f"风险：可能继续下跌，越跌越买会放大回撤（当前近30日最大回撤 "
        f"{'数据缺失' if dd is None else f'{dd:.2f}%'}）。\n\n"
        "② **思路B：考虑减仓**。适用场景：短期快速上涨后回撤开始放大，或板块出现明确利空。"
        f"风险：可能卖在阶段性低点、错失后续反弹，且{exit_note}。\n\n"
        "③ **思路C：继续持有**。适用场景：区间震荡、基本面平稳、波动处于可承受范围。"
        f"风险：若趋势转为单边下行，将被动承受回撤（参考当前净值 {cur_txt}）。"
    )

    if reports:
        watch = "基于本次检索到的研报，可持续跟踪以下方向：\n"
        for item in reports[:6]:
            reason = "，".join(
                x for x in [item.get("industry"), item.get("institution"), item.get("rating")] if x
            )
            watch += f"- {item.get('title')}" + (f"（{reason}）" if reason else "") + "\n"
        watch += "\n以上仅为客观列举，不代表推荐买入。"
    else:
        watch = "暂无可列举的跟踪方向（未检索到研报）。"

    return {
        "trend": trend_text,
        "fundamentals": fundamentals,
        "options": options,
        "watchlist": watch,
    }
