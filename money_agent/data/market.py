"""行情辅助数据：腾讯净值（备用源）、交易日判断、当日分时估值。

参考 fund-baby 的做法：
- 东方财富估值取不到，或当晚已公布新净值时，用腾讯 `qt.gtimg.cn/q=jj{代码}` 补正式净值；
- 用上证指数最近的行情日期判断今天是否开市（能识别法定节假日）；
- 用腾讯 `getSsgz` 取当日每分钟的估值，画分时曲线。

解析函数不联网，联网函数都接受超时参数，调用方负责兜底。
"""

from __future__ import annotations

import re
import threading
import time as _time
from datetime import date, datetime, time
from typing import Callable

import requests

TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q={symbol}"
TENCENT_INTRADAY_URL = "https://web.ifzq.gtimg.cn/fund/newfund/fundSsgz/getSsgz"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_VAR_RE = re.compile(r'v_(\w+)="([^"]*)"')

SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(15, 0)


def _float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


# --------------------------------------------------------------------------
# 腾讯基金净值（备用源）
# --------------------------------------------------------------------------
def parse_tencent_fund(text: str) -> dict | None:
    """解析 `v_jj110020="110020~名称~…~单位净值~…~日涨跌幅~净值日期…"`。

    字段位置与 fund-baby 一致：[1] 名称、[5] 单位净值、[7] 日涨跌幅(%)、[8] 净值日期。
    """
    match = _VAR_RE.search(text or "")
    if not match:
        return None
    parts = match.group(2).split("~")
    if len(parts) < 9:
        return None
    nav = _float(parts[5])
    if nav is None or nav <= 0:
        return None
    nav_date = parts[8][:10] if parts[8] else None
    return {
        "name": parts[1] or None,
        "nav": nav,
        "change_pct": _float(parts[7]),
        "nav_date": nav_date,
    }


def fetch_tencent_fund(code: str, timeout: int = 8) -> dict | None:
    resp = requests.get(TENCENT_QUOTE_URL.format(symbol=f"jj{code}"), headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return parse_tencent_fund(resp.content.decode("gbk", errors="replace"))


# --------------------------------------------------------------------------
# 交易日判断
# --------------------------------------------------------------------------
def parse_index_date(text: str) -> str | None:
    """从上证指数行情里取最近一次行情的日期（YYYYMMDD），在第 30 个字段。"""
    match = _VAR_RE.search(text or "")
    if not match:
        return None
    parts = match.group(2).split("~")
    if len(parts) <= 30 or not re.match(r"\d{8}", parts[30]):
        return None
    return parts[30][:8]


def fetch_index_date(timeout: int = 6) -> str | None:
    resp = requests.get(TENCENT_QUOTE_URL.format(symbol="sh000001"), headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return parse_index_date(resp.content.decode("gbk", errors="replace"))


def decide_trading_day(now: datetime, index_date: str | None) -> bool:
    """周末休市；工作日看上证指数今天有没有行情。9:30 前还没有行情，先按开市处理。"""
    if now.weekday() >= 5:
        return False
    if index_date is None:
        return True  # 接口不可用时退回到“工作日即开市”
    if index_date == now.strftime("%Y%m%d"):
        return True
    return now.time() < SESSION_OPEN


_trading_cache: dict = {}
_trading_lock = threading.Lock()
TRADING_CHECK_SECONDS = 120


def is_trading_day(
    now: datetime | None = None,
    fetch: Callable[[], str | None] = fetch_index_date,
) -> bool:
    """今天是否为 A 股交易日（含法定节假日判断），结果缓存 2 分钟。"""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    key = now.strftime("%Y%m%d")
    with _trading_lock:
        cached = _trading_cache.get(key)
        if cached and _time.monotonic() - cached[0] < TRADING_CHECK_SECONDS:
            return cached[1]
    try:
        index_date = fetch()
    except Exception:  # noqa: BLE001 - 判断失败不影响主流程
        index_date = None
    result = decide_trading_day(now, index_date)
    with _trading_lock:
        _trading_cache.clear()
        _trading_cache[key] = (_time.monotonic(), result)
    return result


def market_status(now: datetime | None = None, trading_day: bool | None = None) -> dict:
    """当前市场阶段：closed（休市日）/ pre / open / lunch / after。"""
    now = now or datetime.now()
    if trading_day is None:
        trading_day = is_trading_day(now)
    t = now.time()
    if not trading_day:
        phase, label = "closed", "今日休市"
    elif t < SESSION_OPEN:
        phase, label = "pre", "未开盘"
    elif t < time(11, 30):
        phase, label = "open", "交易中"
    elif t < time(13, 0):
        phase, label = "lunch", "午间休市"
    elif t < SESSION_CLOSE:
        phase, label = "open", "交易中"
    else:
        phase, label = "after", "已收盘"
    return {"trading_day": trading_day, "phase": phase, "label": label}


# --------------------------------------------------------------------------
# 当日分时估值
# --------------------------------------------------------------------------
def parse_intraday(payload: dict) -> dict | None:
    """解析腾讯 getSsgz：data.data = [["0930", 1.1846, -0.0036], …]，data.yesterdayDwjz = 昨日净值。"""
    if not isinstance(payload, dict) or payload.get("code") != 0:
        return None
    data = payload.get("data") or {}
    rows = data.get("data")
    base = _float(data.get("yesterdayDwjz"))
    if not isinstance(rows, list) or not base:
        return None
    points = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        stamp = str(row[0]).zfill(4)
        value = _float(row[1])
        if value is None or not re.fullmatch(r"\d{4}", stamp):
            continue
        points.append({"time": f"{stamp[:2]}:{stamp[2:]}", "value": value})
    return {"yesterday_nav": base, "points": points} if points else None


def fetch_intraday(code: str, timeout: int = 8) -> dict | None:
    params = {"app": "web", "symbol": f"jj{code}", "_": int(_time.time() * 1000)}
    resp = requests.get(TENCENT_INTRADAY_URL, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return parse_intraday(resp.json())


def session_minute(stamp: str) -> int | None:
    """把 "HH:MM" 换成交易时段内的分钟序号（09:30=0，11:30=120，13:00=120，15:00=240）。"""
    try:
        hour, minute = (int(x) for x in stamp.split(":"))
    except ValueError:
        return None
    total = hour * 60 + minute
    if total < 9 * 60 + 30:
        return 0
    if total <= 11 * 60 + 30:
        return total - (9 * 60 + 30)
    if total < 13 * 60:
        return 120
    return min(120 + total - 13 * 60, 240)


def portfolio_intraday(series: list[dict]) -> dict | None:
    """按份额把每只基金的分时估值加总成组合的分时涨跌。

    series: [{"shares": 1000, "yesterday_nav": 1.23, "points": [{"time": "09:30", "value": 1.24}, …]}, …]
    没有分时数据的基金按昨日净值计入（不涨不跌），并在 coverage 里体现占比。
    """
    usable = [s for s in series if s.get("shares") and s.get("yesterday_nav")]
    if not usable:
        return None
    base_total = sum(s["shares"] * s["yesterday_nav"] for s in usable)
    if base_total <= 0:
        return None
    covered = [s for s in usable if s.get("points")]
    coverage = sum(s["shares"] * s["yesterday_nav"] for s in covered) / base_total
    stamps = sorted({p["time"] for s in covered for p in s["points"]}, key=lambda t: session_minute(t) or 0)
    if not stamps:
        return None
    lookups = [{p["time"]: p["value"] for p in s["points"]} for s in covered]
    last_values = [s["yesterday_nav"] for s in covered]
    uncovered_value = base_total - sum(s["shares"] * s["yesterday_nav"] for s in covered)
    points = []
    for stamp in stamps:
        total = uncovered_value
        for i, s in enumerate(covered):
            if stamp in lookups[i]:
                last_values[i] = lookups[i][stamp]
            total += s["shares"] * last_values[i]
        points.append({
            "time": stamp,
            "minute": session_minute(stamp),
            "value": round(total, 2),
            "pct": round((total / base_total - 1) * 100, 4),
        })
    return {"base_value": round(base_total, 2), "coverage": round(coverage, 4), "points": points}


def is_today(stamp: str | None, today: date | None = None) -> bool:
    today = today or date.today()
    return bool(stamp) and str(stamp)[:10] == today.isoformat()
