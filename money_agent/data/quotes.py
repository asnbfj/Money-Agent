"""最新行情 / 净值快照抓取。

关键区分（贯穿全项目）：
- 场外公募基金：盘中只有「估值」（⚠️ 仅参考，非正式成交价），
  官方正式净值盘后才公布 → 分别放进 `estimate_*` 与 `official_*` 字段。
- 场内 ETF/LOF：市价实时可查（`market_price`）。
- A股个股 / 板块：直接取最新价与涨跌幅（`value` / `change_pct`）。

所有接口都做重试与降级：拿不到就留空，交由上层写入“数据缺失”。
"""

from __future__ import annotations

from datetime import datetime, time

import akshare as ak
import pandas as pd
import requests

from money_agent.config import Settings
from money_agent.data.resolver import is_exchange_traded_fund
from money_agent.data.retry import safe_call
from money_agent.models import History, Quote, Target
from money_agent.utils import now_str

# 东方财富手机版接口：单只基金的实时估值 + 官方净值
_FUND_MOBILE_API = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNFInfo"
# 必须使用手机端 UA，桌面 Chrome UA 会被拒绝（返回 Datas=null）
_MOBILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    )
}


def is_a_share_trading_time(now: datetime | None = None) -> bool:
    """粗略判断是否处于 A 股交易时段（未考虑法定节假日）。"""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    return time(9, 25) <= now.time() <= time(15, 5)


def _float(value) -> float | None:
    if value is None or value == "" or value == "--":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mobile_fund_snapshot(code: str, timeout: int) -> dict:
    params = {
        "Fcodes": code,
        "pageIndex": 1,
        "pageSize": 200,
        "plat": "Android",
        "appType": "ttjj",
        "product": "EFund",
        "Version": "1",
        "deviceid": "money-agent",
    }
    resp = requests.get(_FUND_MOBILE_API, params=params, headers=_MOBILE_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _fund_quote(target: Target, settings: Settings, history: History | None) -> Quote:
    exchange = is_exchange_traded_fund(target)
    quote = Quote(name=target.name, code=target.code)
    payload, err = safe_call(
        lambda: _mobile_fund_snapshot(target.code, settings.http_timeout),
        label=f"获取 {target.code} 基金估值",
        attempts=settings.retry_attempts,
        delay=settings.retry_delay,
    )
    datas = (payload or {}).get("Datas") or []
    if datas:
        d = datas[0]
        expansion = payload.get("Expansion") or {}
        quote.estimate_value = _float(d.get("GSZ"))
        quote.estimate_change_pct = _float(d.get("GSZZL"))
        quote.estimate_time = d.get("GZTIME") or expansion.get("GZTIME")
        quote.official_nav = _float(d.get("NAV"))
        quote.official_nav_date = d.get("PDATE")
        quote.market_price = _float(d.get("NEWPRICE"))
        quote.market_time = d.get("HQDATE")
        quote.source = "东方财富基金估值接口(fundmobapi)"

    if exchange:
        # 场内 ETF/LOF：当前价值用市价，其次是日线收盘；不算“盘中估算”
        if quote.market_price is None and history and history.points:
            last = history.points[-1]
            quote.market_price = last.close
            quote.market_time = last.date
        quote.value = quote.market_price
        quote.data_time = quote.market_time
        if datas:
            quote.change_pct = _float(datas[0].get("CHANGERATIO")) or _float(
                datas[0].get("NAVCHGRT")
            )
        return quote

    # 场外基金：盘中估算（仅参考）与官方正式净值分开
    if datas:
        quote.is_intraday_estimate = (
            quote.estimate_value is not None and is_a_share_trading_time()
        )
        if quote.is_intraday_estimate:
            quote.value = quote.estimate_value
            quote.change_pct = quote.estimate_change_pct
            quote.data_time = quote.estimate_time
        else:
            quote.value = quote.official_nav
            quote.change_pct = _float(datas[0].get("NAVCHGRT"))
            quote.data_time = quote.official_nav_date
        return quote

    # 降级：用历史净值的最后一点作为官方净值
    if quote.official_nav is None and history and history.points:
        last = history.points[-1]
        quote.official_nav = last.close
        quote.official_nav_date = last.date
        quote.value = last.close
        quote.data_time = last.date
    quote.source = "东方财富基金估值接口(fundmobapi)"
    quote.note = "基金估值接口未返回数据" + (f"：{err}" if err else "")
    return quote


def _stock_quote(target: Target, settings: Settings, history: History | None) -> Quote:
    quote = Quote(name=target.name, code=target.code)

    def _load() -> pd.DataFrame:
        return ak.stock_bid_ask_em(symbol=target.code)

    df, err = safe_call(
        _load,
        label=f"获取 {target.code} 实时盘口",
        attempts=settings.retry_attempts,
        delay=settings.retry_delay,
    )
    if df is not None and not df.empty and {"item", "value"} <= set(df.columns):
        mapping = dict(zip(df["item"].astype(str), df["value"]))
        quote.value = _float(mapping.get("最新"))
        quote.change_pct = _float(mapping.get("涨幅"))
        quote.source = "akshare.stock_bid_ask_em"
        quote.data_time = now_str()
        return quote

    # 降级：用日线最后一根
    if history and history.points:
        last = history.points[-1]
        quote.value = last.close
        quote.data_time = last.date
    quote.source = "akshare.stock_bid_ask_em"
    quote.note = "实时盘口获取失败，改用日线收盘" + (f"：{err}" if err else "")
    return quote


def _board_quote(target: Target, settings: Settings, history: History | None) -> Quote:
    quote = Quote(name=target.name, code=target.code)
    loader = (
        (lambda: ak.stock_board_industry_spot_em(symbol=target.name))
        if target.kind == "industry"
        else (lambda: ak.stock_board_concept_spot_em(symbol=target.name))
    )
    df, err = safe_call(
        loader,
        label=f"获取板块 {target.name} 实时行情",
        attempts=settings.retry_attempts,
        delay=settings.retry_delay,
    )
    if df is not None and not df.empty:
        record = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]))
        quote.value = _float(record.get("最新价") or record.get("最新"))
        quote.change_pct = _float(record.get("涨跌幅"))
        quote.source = "akshare.板块实时行情"
        quote.data_time = now_str()
        return quote

    if history and history.points:
        last = history.points[-1]
        quote.value = last.close
        quote.data_time = last.date
    quote.source = "akshare.板块实时行情"
    quote.note = "板块实时行情获取失败，改用日线收盘" + (f"：{err}" if err else "")
    return quote


def fetch_quote(target: Target, settings: Settings, history: History | None = None) -> Quote:
    if target.kind == "fund":
        return _fund_quote(target, settings, history)
    if target.kind == "stock":
        return _stock_quote(target, settings, history)
    return _board_quote(target, settings, history)
