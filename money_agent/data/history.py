"""历史行情 / 净值序列抓取，并统一归一化为「日期 + 收盘价/单位净值」。

不同标的走不同 akshare 接口：
- 场外公募基金：`fund_open_fund_info_em`（单位净值走势）
- 场内 ETF/LOF：`fund_etf_hist_em`（价格 K 线）
- A股个股：`stock_zh_a_hist`（前复权日线）
- 行业板块：`stock_board_industry_hist_em`（日 K）
- 概念板块：`stock_board_concept_hist_em`（日 K）

全部失败时返回带 `error` 的 History，由上层写入“数据缺失”说明。
"""

from __future__ import annotations

import akshare as ak
import pandas as pd

from money_agent.config import Settings
from money_agent.data.resolver import is_exchange_traded_fund
from money_agent.data.retry import safe_call
from money_agent.models import History, HistoryPoint, Target
from money_agent.utils import days_ago, to_compact, to_date


def _normalize(
    df: pd.DataFrame, date_col: str, value_col: str, *, limit_days: int
) -> list[HistoryPoint]:
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return []
    frame = df[[date_col, value_col]].copy()
    frame.columns = ["date", "close"]
    frame = frame.dropna()
    frame["date"] = frame["date"].map(lambda v: to_date(v))
    frame = frame[frame["date"].notna()]
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame[frame["close"].notna()]
    cutoff = days_ago(limit_days)
    frame = frame[frame["date"] >= cutoff]
    frame = frame.drop_duplicates(subset=["date"]).sort_values("date")
    return [
        HistoryPoint(date=row["date"].isoformat(), close=float(row["close"]))
        for _, row in frame.iterrows()
    ]


def get_fund_nav_history(code: str, start: str, end: str, limit_days: int) -> list[HistoryPoint]:
    df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
    return _normalize(df, "净值日期", "单位净值", limit_days=limit_days)


def get_fund_etf_history(code: str, start: str, end: str, limit_days: int) -> list[HistoryPoint]:
    df = ak.fund_etf_hist_em(
        symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
    )
    return _normalize(df, "日期", "收盘", limit_days=limit_days)


def get_stock_history(code: str, start: str, end: str, limit_days: int) -> list[HistoryPoint]:
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
    )
    return _normalize(df, "日期", "收盘", limit_days=limit_days)


def get_industry_history(name: str, start: str, end: str, limit_days: int) -> list[HistoryPoint]:
    df = ak.stock_board_industry_hist_em(
        symbol=name, start_date=start, end_date=end, period="日k"
    )
    return _normalize(df, "日期", "收盘", limit_days=limit_days)


def get_concept_history(name: str, start: str, end: str, limit_days: int) -> list[HistoryPoint]:
    df = ak.stock_board_concept_hist_em(
        symbol=name, period="daily", start_date=start, end_date=end, adjust=""
    )
    return _normalize(df, "日期", "收盘", limit_days=limit_days)


def _is_exchange_traded_fund(target: Target) -> bool:
    """判断是否为场内交易的 ETF/LOF（用价格 K 线），而非场外基金（用净值）。"""
    return is_exchange_traded_fund(target)


def fetch_history(target: Target, settings: Settings) -> History:
    """按标的类型抓取历史序列，自动重试并在失败时降级。"""
    limit_days = settings.history_calendar_days
    start = to_compact(days_ago(limit_days + 35))  # 多取一段，防止节假日截断
    end = to_compact(days_ago(-1))
    attempts = settings.retry_attempts
    delay = settings.retry_delay

    if target.kind == "fund":
        is_etf = _is_exchange_traded_fund(target)
        # 场外优先用净值走势；场内 ETF/LOF 用价格 K 线
        if is_etf:
            points, err = safe_call(
                lambda: get_fund_etf_history(target.code, start, end, limit_days),
                label=f"获取 {target.code} 场内价格历史",
                attempts=attempts,
                delay=delay,
            )
            if points:
                return History(points=points, source="akshare.fund_etf_hist_em", value_name="收盘价")
            points, err2 = safe_call(
                lambda: get_fund_nav_history(target.code, start, end, limit_days),
                label=f"获取 {target.code} 历史净值",
                attempts=attempts,
                delay=delay,
            )
            if points:
                return History(points=points, source="akshare.fund_open_fund_info_em", value_name="单位净值")
            return History(error=err or err2, value_name="单位净值")
        points, err = safe_call(
            lambda: get_fund_nav_history(target.code, start, end, limit_days),
            label=f"获取 {target.code} 历史净值",
            attempts=attempts,
            delay=delay,
        )
        if points:
            return History(points=points, source="akshare.fund_open_fund_info_em", value_name="单位净值")
        return History(error=err, value_name="单位净值")

    if target.kind == "stock":
        points, err = safe_call(
            lambda: get_stock_history(target.code, start, end, limit_days),
            label=f"获取 {target.code} 历史K线",
            attempts=attempts,
            delay=delay,
        )
        return History(
            points=points or [],
            source="akshare.stock_zh_a_hist",
            value_name="收盘价（前复权）",
            error=None if points else err,
        )

    if target.kind == "industry":
        points, err = safe_call(
            lambda: get_industry_history(target.name, start, end, limit_days),
            label=f"获取板块 {target.name} 历史K线",
            attempts=attempts,
            delay=delay,
        )
        return History(
            points=points or [],
            source="akshare.stock_board_industry_hist_em",
            value_name="收盘价",
            error=None if points else err,
        )

    # concept
    points, err = safe_call(
        lambda: get_concept_history(target.name, start, end, limit_days),
        label=f"获取概念板块 {target.name} 历史K线",
        attempts=attempts,
        delay=delay,
    )
    return History(
        points=points or [],
        source="akshare.stock_board_concept_hist_em",
        value_name="收盘价",
        error=None if points else err,
    )
