"""Agent 可用工具定义（Function Calling）。

严格对应主系统约定的 5 个函数，签名与返回结构固定：

1. `get_fund_code_by_name(fund_name)`        → 基金代码、基金类型
2. `get_fund_latest_info(fund_code)`         → 最新净值、时间、标记【估算/正式】
3. `get_fund_history_kline(code, period_days)` → data 列表、区间涨跌幅、最大回撤
4. `rag_search_industry_report(industry_name)` → 摘要、催化因素、风险点、来源
5. `calc_user_position(fund_cost, current_net, share)` → 账面收益率、盈亏金额

约束：
- 所有函数只返回真实抓取到的数据；失败时返回 `{"error": ...}`，绝不返回编造数字。
- `rag_search_industry_report` 的“催化因素/风险点”是对**检索到的研报标题**做关键词提取，
  并非研报原文结论，返回里会显式标注 `analysis_basis`。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from money_agent.config import Settings
from money_agent.data import analytics as ana
from money_agent.data.history import fetch_history
from money_agent.data.quotes import fetch_quote, is_a_share_trading_time
from money_agent.data.resolver import (
    clean_query,
    find_fund_by_name,
    target_from_fund_code,
)
from money_agent.data.retry import sanitize_error
from money_agent.models import History, ReportItem, Target
from money_agent.rag.reports import search_industry_reports

# 研报标题里的“利好/利空”关键词表（仅用于客观归类，不代表研报结论）
_CATALYST_KEYWORDS = (
    "增长", "放量", "加速", "超预期", "提升", "复苏", "涨价", "扩产", "突破", "订单",
    "景气", "上调", "创新高", "受益", "拐点", "国产替代", "需求旺盛", "高增", "盈利改善",
)
_RISK_KEYWORDS = (
    "风险", "下滑", "承压", "下调", "亏损", "降价", "竞争加剧", "不及预期", "减产",
    "库存", "减值", "放缓", "退坡", "降价压力", "成本上升", "需求疲软", "不确定性",
)


# --------------------------------------------------------------------------
# 1) 基金名称 → 代码
# --------------------------------------------------------------------------
def get_fund_code_by_name(fund_name: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """根据基金中文名称，查询对应的基金代码与类型。"""
    settings = settings or Settings()
    if not fund_name or not str(fund_name).strip():
        return {"error": "入参 fund_name 为空"}
    hit = find_fund_by_name(str(fund_name), settings)
    if not hit:
        return {
            "error": f"未在基金名称表中找到「{fund_name}」，请确认名称是否正确",
            "query": fund_name,
        }
    return {
        "fund_code": hit["基金代码"],
        "fund_name": hit["基金简称"],
        "fund_type": hit["基金类型"],
        "_query": fund_name,
    }


# --------------------------------------------------------------------------
# 2) 基金最新净值（盘中估算 / 盘后正式）
# --------------------------------------------------------------------------
def get_fund_latest_info(
    fund_code: str, *, settings: Settings | None = None
) -> dict[str, Any]:
    """获取基金最新净值：交易时段返回盘中估算净值，盘后返回官方正式净值。"""
    settings = settings or Settings()
    if not fund_code or not str(fund_code).strip():
        return {"error": "入参 fund_code 为空"}

    target = target_from_fund_code(str(fund_code), settings)
    quote = fetch_quote(target, settings, None)

    if quote.value is None and quote.official_nav is None and quote.estimate_value is None:
        return {
            "error": "基金净值接口未返回数据"
            + (f"：{quote.note}" if quote.note else ""),
            "fund_code": target.code,
        }

    if target.meta.get("场内"):
        # 场内 ETF/LOF：交易价格优先；净值作为参考一并给出
        return {
            "fund_code": target.code,
            "fund_name": target.name,
            "is_exchange_traded": True,
            "nav": quote.market_price if quote.market_price is not None else quote.value,
            "nav_time": quote.market_time or quote.data_time,
            "tag": "正式",
            "nav_kind": "场内交易价格",
            "unit_nav": quote.official_nav,
            "unit_nav_date": quote.official_nav_date,
            "change_pct": quote.change_pct,
            "source": quote.source,
            "note": quote.note,
        }

    if quote.is_intraday_estimate:
        return {
            "fund_code": target.code,
            "fund_name": target.name,
            "is_exchange_traded": False,
            "nav": quote.estimate_value,
            "nav_time": quote.estimate_time,
            "tag": "估算",
            "nav_kind": "盘中估算净值（仅参考，非正式成交价）",
            "official_nav": quote.official_nav,
            "official_nav_date": quote.official_nav_date,
            "change_pct": quote.estimate_change_pct,
            "source": quote.source,
            "note": quote.note,
        }

    return {
        "fund_code": target.code,
        "fund_name": target.name,
        "is_exchange_traded": False,
        "nav": quote.official_nav if quote.official_nav is not None else quote.value,
        "nav_time": quote.official_nav_date or quote.data_time,
        "tag": "正式",
        "nav_kind": "官方正式单位净值",
        "change_pct": quote.change_pct,
        "estimate_nav": quote.estimate_value,
        "estimate_time": quote.estimate_time,
        "source": quote.source,
        "note": quote.note
        or (
            "当前非交易时段（或无盘中估算），返回官方正式净值；"
            "盘中估算需在交易日 09:30-15:00 获取"
            if not is_a_share_trading_time()
            else ""
        ),
    }


# --------------------------------------------------------------------------
# 3) 基金历史净值序列 + 区间统计
# --------------------------------------------------------------------------
def get_fund_history_kline(
    fund_code: str, period_days: int, *, settings: Settings | None = None
) -> dict[str, Any]:
    """获取基金历史净值序列（7/15/30 天），并计算区间涨跌幅与最大回撤。

    口径说明：
    - `data` 为最近 `period_days` 个**交易日**的净值点（含日涨跌幅），保证序列有足够可读点数；
    - `interval_change_pct` / `max_drawdown_pct` 基于返回的 `data` 首尾区间计算；
    - `returns` 额外给出按**自然日**口径的 7/15/30 日涨跌幅，与报告模板一致。
    """
    settings = settings or Settings()
    if not fund_code or not str(fund_code).strip():
        return {"error": "入参 fund_code 为空"}
    try:
        period_days = int(period_days)
    except (TypeError, ValueError):
        return {"error": f"入参 period_days 非法：{period_days!r}"}
    if period_days <= 0:
        return {"error": "入参 period_days 必须为正整数"}

    target = target_from_fund_code(str(fund_code), settings)
    # 抓取窗口至少覆盖请求周期 + 30 天冗余，保证区间统计有足够样本
    fetch_days = max(settings.history_calendar_days, period_days * 2 + 45)
    history = fetch_history(target, replace(settings, history_calendar_days=fetch_days))

    if not history.ok:
        return {
            "error": "历史净值接口获取失败，缺少该周期数据"
            + (f"：{history.error}" if history.error else ""),
            "fund_code": target.code,
            "period_days": period_days,
        }

    points = sorted(history.points, key=lambda p: p.date)
    window = ana.last_n_points(points, period_days)

    return {
        "fund_code": target.code,
        "fund_name": target.name,
        "period_days": period_days,
        "period_basis": "交易日",
        "value_name": history.value_name,
        "source": history.source,
        "data": ana.points_with_change(window),
        "interval_change_pct": ana.change_over(window),
        "interval_baseline_date": window[0].date if window else None,
        "max_drawdown_pct": ana.max_drawdown_of(window),
        "annualized_volatility_pct": ana.annualized_volatility(points, period_days),
        # 报告固定需要 7/15/30 三档（自然日口径），一并返回供上层直接使用
        "returns": {
            "7": ana.period_return(points, 7)[0],
            "15": ana.period_return(points, 15)[0],
            "30": ana.period_return(points, 30)[0],
        },
        "max_drawdown_30_pct": ana.period_max_drawdown(points, 30),
        "latest_date": points[-1].date,
        "latest_nav": points[-1].close,
    }


# --------------------------------------------------------------------------
# 4) 行业研报 RAG 检索
# --------------------------------------------------------------------------
def build_industry_report_payload(
    reports: list[ReportItem], used: str, err: str | None
) -> dict[str, Any]:
    """把检索到的研报整理成 `rag_search_industry_report` 的标准返回结构。"""
    if not reports:
        return {
            "industry": used,
            "summary": "未检索到可用的公开券商研报，无法给出基本面摘要。",
            "catalysts": [],
            "risks": [],
            "sources": [],
            "analysis_basis": "无研报数据",
            "error": err or "未检索到研报",
        }

    catalysts = _pick_by_keywords(reports, _CATALYST_KEYWORDS)
    risks = _pick_by_keywords(reports, _RISK_KEYWORDS)
    institutions = sorted({r.institution for r in reports if r.institution})
    ratings = [r.rating for r in reports if r.rating]
    rating_text = "、".join(sorted(set(ratings))) if ratings else "未标注"

    summary = (
        f"共检索到 {len(reports)} 篇与「{used}」相关的公开券商研报，"
        f"覆盖机构 {len(institutions)} 家（{'、'.join(institutions)}），"
        f"评级分布：{rating_text}。"
        f"最新一篇为 {reports[0].date or '日期未知'}《{reports[0].title}》。"
    )

    return {
        "industry": used,
        "summary": summary,
        "catalysts": catalysts,
        "risks": risks,
        "sources": [
            {
                "title": r.title,
                "institution": r.institution,
                "rating": r.rating,
                "industry": r.industry,
                "date": r.date,
                "url": r.url,
                "stock": r.stock_name,
            }
            for r in reports
        ],
        "analysis_basis": (
            "catalysts / risks 由检索到的研报标题关键词客观归类得到，"
            "并非研报原文结论；详细观点请查阅 sources 中的原文链接。"
        ),
        "error": None if not err else err,
    }


def rag_search_industry_report(
    industry_name: str,
    *,
    settings: Settings | None = None,
    fund_target: Target | None = None,
) -> dict[str, Any]:
    """检索行业公开研报：返回摘要、催化因素、风险点与来源信息。"""
    settings = settings or Settings()
    query = clean_query(str(industry_name or "")) or str(industry_name or "")
    reports, used, err = search_industry_reports(
        query, settings, fund_target=fund_target, top_k=settings.report_top_k
    )
    return build_industry_report_payload(reports, used or query, err)


def _pick_by_keywords(reports: list[ReportItem], keywords: tuple[str, ...]) -> list[dict[str, str]]:
    picked: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in reports:
        title = item.title
        if title in seen:
            continue
        if any(k in title for k in keywords):
            seen.add(title)
            picked.append(
                {
                    "title": title,
                    "date": item.date,
                    "institution": item.institution,
                    "url": item.url,
                }
            )
    return picked


# --------------------------------------------------------------------------
# 5) 持仓账面收益计算
# --------------------------------------------------------------------------
def calc_user_position(
    fund_cost: float, current_net: float, share: float
) -> dict[str, Any]:
    """计算用户持仓账面收益率与账面盈亏（不含任何费用）。"""
    try:
        cost = float(fund_cost)
        current = float(current_net)
        shares = float(share)
    except (TypeError, ValueError):
        return {"error": "入参必须为数字：fund_cost / current_net / share"}

    if cost <= 0:
        return {"error": "持仓成本净值 fund_cost 必须大于 0"}
    if shares < 0:
        return {"error": "持仓份额 share 不能为负数"}

    return_pct = (current / cost - 1) * 100
    pnl_amount = (current - cost) * shares
    return {
        "fund_cost": round(cost, 4),
        "current_net": round(current, 4),
        "share": round(shares, 2),
        "return_pct": round(return_pct, 2),
        "pnl_amount": round(pnl_amount, 2),
        "market_value": round(current * shares, 2),
        "cost_value": round(cost * shares, 2),
        "note": "仅账面计算，不含申购赎回费、管理费与税费",
    }


# --------------------------------------------------------------------------
# Function Calling 定义（OpenAI tools 格式）
# --------------------------------------------------------------------------
FUNCTION_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_fund_code_by_name",
            "description": "根据基金中文名称，调用akshare查询对应的基金代码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_name": {
                        "type": "string",
                        "description": "基金名称字符串，例如“天弘半导体设备指数A”“招商中证白酒”",
                    }
                },
                "required": ["fund_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_latest_info",
            "description": "获取基金最新净值；交易时段返回天天基金盘中估算净值；盘后获取akshare官方正式净值。返回净值数值、时间与标记（估算/正式）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_code": {"type": "string", "description": "6 位基金代码，例如 000001"}
                },
                "required": ["fund_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_history_kline",
            "description": "调用akshare获取基金历史净值序列，支持 7/15/30 天，返回每日日期、单位净值、日涨跌幅，并自动计算区间涨跌幅与最大回撤。data 为最近 period_days 个交易日的净值点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_code": {"type": "string", "description": "6 位基金代码"},
                    "period_days": {
                        "type": "integer",
                        "description": "统计周期（自然日），常用 7 / 15 / 30",
                        "enum": [7, 15, 30],
                    },
                },
                "required": ["fund_code", "period_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search_industry_report",
            "description": "RAG检索行业公开研报，返回研报摘要、核心利好/利空、来源信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry_name": {
                        "type": "string",
                        "description": "行业或主题名称，例如“半导体”“白酒”“新能源”",
                    }
                },
                "required": ["industry_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_user_position",
            "description": "计算用户持仓账面收益率与账面盈亏。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_cost": {"type": "number", "description": "持仓成本净值"},
                    "current_net": {"type": "number", "description": "当前净值"},
                    "share": {"type": "number", "description": "持仓份额"},
                },
                "required": ["fund_cost", "current_net", "share"],
            },
        },
    },
]

FUNCTION_NAMES = tuple(spec["function"]["name"] for spec in FUNCTION_SPECS)
