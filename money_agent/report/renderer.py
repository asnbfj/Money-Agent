"""固定模板渲染器。

输出结构严格对齐主系统约定：6 个章节 + 强制免责声明。
第 1、2 章的数字全部来自工具层实测数据，由本模块确定性计算，LLM 不参与；
第 3~6 章为 LLM（或规则）生成的定性文案。任何数据缺失都会显式写入报告。
"""

from __future__ import annotations

from money_agent.models import AnalysisReport
from money_agent.utils import fmt_num, fmt_pct

DISCLAIMER = (
    "> 免责声明：本分析仅基于历史公开数据做客观统计，**不构成任何投资建议。"
    "市场存在风险，过往收益不代表未来表现，请独立审慎决策。**"
)


def _is_exchange_fund(report: AnalysisReport) -> bool:
    return report.target.kind == "fund" and bool(report.target.meta.get("场内"))


def _target_label(report: AnalysisReport) -> str:
    if _is_exchange_fund(report):
        return "场内基金（ETF/LOF）"
    return report.target.kind_label


def _mark_to_market(report: AnalysisReport) -> float | None:
    """当前价值口径统一由 `AnalysisReport.mark_value` 提供。"""
    return report.mark_value


def _holding_section(report: AnalysisReport) -> str:
    holding = report.holding
    lines: list[str] = ["## 2. 用户持仓现状"]

    shares = holding.shares
    cost = holding.cost_nav
    lines.append(f"持仓份额：{fmt_num(shares, 2) if shares is not None else '未提供'}"
                 f"；持仓成本净值：{fmt_num(cost, 4) if cost is not None else '未提供'}")

    # 优先使用 calc_user_position 工具的计算结果，保证与工具口径一致
    position = report.position or {}
    if position and "error" not in position:
        lines.append(
            f"当前账面收益率：{position['return_pct']:+.2f}%"
            "（仅账面，**不含申购赎回费、管理费**）"
        )
        lines.append(
            f"持仓盈亏金额：{position['pnl_amount']:+,.2f} 元"
            f"（按 {fmt_num(position['current_net'], 4)} 计）"
        )
        return "\n".join(lines)

    current = _mark_to_market(report)
    if shares is not None and cost not in (None, 0) and current is not None:
        pnl_pct = (current / cost - 1) * 100
        pnl_amt = (current - cost) * shares
        lines.append(f"当前账面收益率：{pnl_pct:+.2f}%（仅账面，**不含申购赎回费、管理费**）")
        lines.append(f"持仓盈亏金额：{pnl_amt:+,.2f} 元（按 {fmt_num(current, 4)} 计）")
    else:
        reason = "缺少持仓成本或份额" if (shares is None or cost is None) else "缺少当前价格/净值"
        lines.append(f"当前账面收益率：数据缺失（{reason}，无法计算）")
        lines.append("持仓盈亏金额：数据缺失")
    return "\n".join(lines)


def _quote_section(report: AnalysisReport) -> str:
    quote = report.quote
    lines = ["## 1. 基础行情数据（来自AKShare）"]

    value = quote.value if quote.value is not None else _mark_to_market(report)
    if value is None:
        lines.append("- 当前价格/净值：数据缺失，数据时间：数据缺失；未能获取行情，估值不可用")
    else:
        if report.target.kind == "fund" and _is_exchange_fund(report):
            tag = "场内价格（收盘/实时市价，非场外净值）"
        elif report.target.kind == "fund" and quote.is_intraday_estimate:
            tag = "⚠️ 盘中估算值（仅参考，非正式成交价，正式净值盘后公布）"
        elif report.target.kind == "fund":
            tag = "官方正式净值"
        else:
            tag = "行情最新价"
        change = f"，当日涨跌幅 {fmt_pct(quote.change_pct)}" if quote.change_pct is not None else ""
        lines.append(
            f"- 当前价格/净值：{fmt_num(value, 4)}{change}，数据时间：{quote.data_time or '未知'}；{tag}"
        )

    # 场外基金额外补充：官方净值 vs 盘中估算
    if report.target.kind == "fund":
        if _is_exchange_fund(report):
            if quote.official_nav is not None:
                lines.append(
                    f"- 基金单位净值（供参考，非交易价格）：{fmt_num(quote.official_nav, 4)}"
                    f"（{quote.official_nav_date or '日期未知'}）"
                )
        else:
            if quote.official_nav is not None:
                lines.append(
                    f"- 官方正式单位净值：{fmt_num(quote.official_nav, 4)}"
                    f"（{quote.official_nav_date or '日期未知'}）"
                )
            if quote.estimate_value is not None:
                lines.append(
                    f"- ⚠️ 盘中估算净值：{fmt_num(quote.estimate_value, 4)}"
                    f"（{fmt_pct(quote.estimate_change_pct)}，{quote.estimate_time or '时间未知'}，"
                    "仅供参考，不作为正式交易价格）"
                )

    analytics = report.analytics
    lines.append(f"- 近7日涨跌幅：{fmt_pct(analytics.ret_7)}")
    lines.append(f"- 近15日涨跌幅：{fmt_pct(analytics.ret_15)}")
    lines.append(f"- 近30日涨跌幅：{fmt_pct(analytics.ret_30)}")
    if analytics.max_drawdown_30 is None:
        lines.append("- 近30日最大回撤：数据缺失")
    else:
        lines.append(
            f"- 近30日最大回撤：{analytics.max_drawdown_30:.2f}%"
            "（计算：区间最低点相对区间最高点跌幅）"
        )
    if analytics.volatility_30_annualized is not None:
        lines.append(f"- 近30日年化波动率：{analytics.volatility_30_annualized:.2f}%（附加统计）")
    return "\n".join(lines)


def _reports_section(report: AnalysisReport) -> str:
    lines = ["## 4. 板块基本面信息（来自行业研报RAG）"]
    narrative = report.narrative.get("fundamentals", "").strip()
    lines.append(narrative if narrative else "未获取到可用研报，本节无法给出基本面解读。")

    # 基金（尤其宽基/多行业）研报来自重仓股，可能横跨多个行业，需如实提示
    industries = {r.industry for r in report.reports if r.industry}
    if report.target.kind == "fund" and len(industries) > 1:
        lines.append(
            f"\n> 注：该基金为多行业配置（研报覆盖 {'、'.join(sorted(industries))} 等 "
            f"{len(industries)} 个行业），本节研报来自其重仓股，基本面解读仅供参考。"
        )

    if report.reports:
        lines.append("")
        lines.append("**研报来源（公开券商/机构）：**")
        for item in report.reports:
            meta = "，".join(
                x for x in [item.institution, item.rating, item.date] if x
            )
            name = f"{item.stock_name}" if item.stock_name else ""
            tag = f"[{name}] " if name else ""
            lines.append(f"- {tag}《{item.title}》（{meta}）")
    return "\n".join(lines)


def _title(report: AnalysisReport) -> str:
    """板块无独立代码时避免出现“半导体（半导体）”这种重复标题。"""
    target = report.target
    if target.code and target.code != target.name:
        return f"# 标的分析报告：{target.name}（{target.code}）"
    return f"# 标的分析报告：{target.name}"


def render(report: AnalysisReport) -> str:
    """渲染最终报告字符串。"""
    narrative = report.narrative

    blocks: list[str] = [
        _title(report),
        f"> 标的类型：{_target_label(report)}｜数据源：AKShare / 东方财富公开接口"
        f"｜生成时间：{report.generated_at}｜分析引擎：{report.engine}",
        _quote_section(report),
        _holding_section(report),
        "## 3. 近期走势分析",
        narrative.get("trend", "").strip() or "数据不足，无法给出走势判断。",
        _reports_section(report),
        "## 5. 客观评估 & 可选操作方向",
        narrative.get("options", "").strip() or "数据不足，无法给出可选思路。",
        "## 6. 板块关注推荐（客观列举，不推荐买入）",
        narrative.get("watchlist", "").strip() or "暂无可列举的跟踪方向。",
    ]

    if report.data_gaps:
        gap_lines = ["## ⚠️ 数据缺失与局限"]
        gap_lines += [f"- {gap}" for gap in report.data_gaps]
        gap_lines.append(
            "- 说明：行情接口获取失败或数据为空时，相关周期分析存在局限性，请勿据此推断未列出的数字。"
        )
        blocks.append("\n".join(gap_lines))

    blocks.append(DISCLAIMER)
    return "\n\n".join(blocks) + "\n"
