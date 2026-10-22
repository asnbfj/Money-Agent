"""工具注册表：把 Function Calling 工具接到共享上下文上。

暴露给 LLM 的工具 = `functions.FUNCTION_SPECS`（严格对应主系统约定的 5 个函数）：
    get_fund_code_by_name / get_fund_latest_info / get_fund_history_kline
    / rag_search_industry_report / calc_user_position

`ToolContext` 缓存已抓取结果，避免重复请求；`dispatch` 会在调用工具的同时把结果
同步进上下文的 `target / quote / history / analytics / reports / position`，
这样第 1、2 章的数字仍然由确定性逻辑计算，LLM 无法篡改。

对于 A股个股 / 板块，额外保留一组**内部**（不暴露给 LLM）的通用抓取方法，
由编排器在确定性采集阶段调用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import akshare as ak

import money_agent.agent.functions as fns
from money_agent.agent.functions import FUNCTION_NAMES, FUNCTION_SPECS
from money_agent.config import Settings
from money_agent.data.analytics import compute_analytics
from money_agent.data.history import fetch_history
from money_agent.data.quotes import fetch_quote
from money_agent.data.resolver import resolve_target, target_from_fund_code
from money_agent.data.retry import sanitize_error
from money_agent.exceptions import MoneyAgentError
from money_agent.models import Analytics, History, Quote, ReportItem, Target

# 生成完整基金报告前“必须”成功调用的工具（calc_user_position 由编排器确定性调用）
MANDATORY_TOOLS = (
    "get_fund_code_by_name",
    "get_fund_latest_info",
    "get_fund_history_kline",
    "rag_search_industry_report",
)

__all__ = [
    "ToolContext",
    "ToolRegistry",
    "MANDATORY_TOOLS",
    "FUNCTION_NAMES",
    "FUNCTION_SPECS",
]


@dataclass
class ToolContext:
    """一次分析过程中的共享状态。"""

    settings: Settings
    target: Target | None = None
    quote: Quote | None = None
    history: History | None = None
    analytics: Analytics | None = None
    reports: list[ReportItem] = field(default_factory=list)
    industry: str = ""
    position: dict[str, Any] = field(default_factory=dict)
    news: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    called: set[str] = field(default_factory=set)

    def add_error(self, message: str) -> None:
        if message and message not in self.errors:
            self.errors.append(message)

    def ensure_target_from_code(self, code: str) -> Target:
        if self.target is not None and self.target.code == str(code).zfill(6):
            return self.target
        self.target = target_from_fund_code(str(code), self.settings)
        return self.target

    # ---------------- 内部通用抓取（A股个股 / 板块用，不暴露给 LLM） ----------------
    def do_resolve(self, query: str) -> dict[str, Any]:
        target = resolve_target(query, self.settings)
        self.target = target
        self.called.add("resolve_target")
        return target.to_dict()

    def do_quote(self) -> dict[str, Any]:
        if self.target is None:
            raise MoneyAgentError("尚未解析标的，无法获取行情")
        quote = fetch_quote(self.target, self.settings, self.history)
        self.quote = quote
        self.called.add("get_latest_quote")
        if quote.note:
            self.add_error(quote.note)
        return quote.to_dict()

    def do_history(self) -> dict[str, Any]:
        if self.target is None:
            raise MoneyAgentError("尚未解析标的，无法获取历史行情")
        history = fetch_history(self.target, self.settings)
        self.history = history
        self.analytics = compute_analytics(history)
        self.called.add("get_price_history")
        if not history.ok:
            self.add_error(
                f"历史行情接口获取失败，缺少近7/15/30日K线数据（{history.error or '空数据'}）"
            )
        return _history_summary(history, self.analytics)

    def do_reports(self) -> dict[str, Any]:
        if self.target is None:
            raise MoneyAgentError("尚未解析标的，无法检索研报")
        from money_agent.rag.reports import fetch_reports

        reports, err = fetch_reports(self.target, self.settings)
        self.reports = reports
        self.called.add("search_research_reports")
        if err:
            self.add_error(f"研报检索不完整：{err}")
        return {"count": len(reports), "reports": [r.to_dict() for r in reports]}

    def do_news(self) -> dict[str, Any]:
        if self.target is None or self.target.kind != "stock":
            return {"count": 0, "news": [], "note": "仅个股提供新闻检索"}
        try:
            df = ak.stock_news_em(symbol=self.target.code)
        except Exception as exc:  # noqa: BLE001
            self.add_error(f"个股新闻获取失败：{sanitize_error(exc)}")
            return {"count": 0, "news": []}
        items: list[dict[str, Any]] = []
        if df is not None and not df.empty:
            for _, row in df.head(8).iterrows():
                items.append(
                    {
                        "title": str(row.get("新闻标题", "")),
                        "time": str(row.get("发布时间", "")),
                        "source": str(row.get("文章来源", "")),
                    }
                )
        self.news = items
        return {"count": len(items), "news": items}

    # ---------------- 官方工具对应的上下文动作 ----------------
    def fn_get_fund_code_by_name(self, fund_name: str) -> dict[str, Any]:
        result = fns.get_fund_code_by_name(fund_name, settings=self.settings)
        if "error" not in result:
            self.target = target_from_fund_code(result["fund_code"], self.settings)
            self.called.add("get_fund_code_by_name")
        else:
            self.add_error(result["error"])
        return result

    def fn_get_fund_latest_info(self, fund_code: str) -> dict[str, Any]:
        code = fund_code or (self.target.code if self.target else "")
        if not code:
            return {"error": "缺少 fund_code，且尚未解析出标的"}
        target = self.ensure_target_from_code(code)
        result = fns.get_fund_latest_info(target.code, settings=self.settings)
        if "error" in result:
            self.add_error(result["error"])
            return result
        self.quote = fetch_quote(target, self.settings, self.history)
        self.called.add("get_fund_latest_info")
        return result

    def fn_get_fund_history_kline(self, fund_code: str, period_days: int) -> dict[str, Any]:
        code = fund_code or (self.target.code if self.target else "")
        if not code:
            return {"error": "缺少 fund_code，且尚未解析出标的"}
        target = self.ensure_target_from_code(code)
        result = fns.get_fund_history_kline(target.code, period_days, settings=self.settings)
        if "error" in result:
            self.add_error(result["error"])
            return result
        # 重新抓取完整序列存入上下文，供报告第 1 章统一口径使用
        history = fetch_history(target, self.settings)
        self.history = history
        self.analytics = compute_analytics(history)
        self.called.add("get_fund_history_kline")
        if not history.ok:
            self.add_error(f"历史净值接口获取失败，缺少近7/15/30日数据（{history.error or '空数据'}）")
        return result

    def fn_rag_search_industry_report(self, industry_name: str) -> dict[str, Any]:
        from money_agent.rag.reports import search_industry_reports

        query = industry_name
        # 未给出行业名时，用当前标的（基金名称）作为主题
        if not str(query or "").strip() and self.target is not None:
            query = self.target.name
        reports, used, err = search_industry_reports(
            str(query or ""), self.settings, fund_target=self.target, top_k=self.settings.report_top_k
        )
        self.reports = reports
        self.industry = used
        self.called.add("rag_search_industry_report")
        if err:
            self.add_error(f"研报检索不完整：{err}")
        return fns.build_industry_report_payload(reports, used or str(query), err)

    def fn_calc_user_position(self, fund_cost: float, current_net: float, share: float) -> dict[str, Any]:
        result = fns.calc_user_position(fund_cost, current_net, share)
        if "error" in result:
            self.add_error(result["error"])
        else:
            self.position = result
            self.called.add("calc_user_position")
        return result


def _history_summary(history: History, analytics: Analytics | None) -> dict[str, Any]:
    points = history.points
    tail = points[-35:] if len(points) > 35 else points
    return {
        "ok": history.ok,
        "source": history.source,
        "value_name": history.value_name,
        "error": history.error,
        "count": len(points),
        "first_date": points[0].date if points else None,
        "last_date": points[-1].date if points else None,
        "recent_points": [p.to_dict() for p in tail],
        "analytics": analytics.to_dict() if analytics else None,
    }


class ToolRegistry:
    """把工具名映射到上下文动作，并统一返回 JSON 字符串。"""

    # 内部通用抓取（不暴露给 LLM，供非基金标的的确定性采集使用）
    _INTERNAL = {
        "resolve_target": "do_resolve",
        "get_price_history": "do_history",
        "get_latest_quote": "do_quote",
        "search_research_reports": "do_reports",
        "get_related_news": "do_news",
    }

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    @property
    def specs(self) -> list[dict[str, Any]]:
        return FUNCTION_SPECS

    def dispatch(self, name: str, arguments: dict[str, Any] | None) -> str:
        """暴露给 LLM 的 5 个官方工具。"""
        handler: Callable[..., dict[str, Any]] | None = getattr(
            self.ctx, f"fn_{name}", None
        )
        if handler is None or name not in FUNCTION_NAMES:
            return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)
        return self._invoke(name, handler, arguments or {})

    def dispatch_internal(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """内部工具（A股个股/板块），不对外暴露。"""
        method_name = self._INTERNAL.get(name)
        handler: Callable[..., dict[str, Any]] | None = (
            getattr(self.ctx, method_name, None) if method_name else None
        )
        if handler is None:
            return json.dumps({"error": f"未知内部工具：{name}"}, ensure_ascii=False)
        return self._invoke(name, handler, arguments or {})

    def _invoke(
        self, name: str, handler: Callable[..., dict[str, Any]], arguments: dict[str, Any]
    ) -> str:
        try:
            result = handler(**arguments)
            return json.dumps(result, ensure_ascii=False, default=str)
        except MoneyAgentError as exc:
            self.ctx.add_error(str(exc))
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        except TypeError as exc:
            message = f"{name} 参数错误：{sanitize_error(exc)}"
            self.ctx.add_error(message)
            return json.dumps({"error": message}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            message = f"{name} 执行失败：{sanitize_error(exc)}"
            self.ctx.add_error(message)
            return json.dumps({"error": message}, ensure_ascii=False)
