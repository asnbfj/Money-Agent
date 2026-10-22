"""通用工具：日期归一化与数值格式化。"""

from __future__ import annotations

from datetime import date, datetime, timedelta


def today_iso() -> str:
    return date.today().isoformat()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def to_date(value: str | date | datetime) -> date | None:
    """把 '2026-10-09' / '20261009' / '2026-10-09 12:00' 等统一成 date。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text[: len(fmt) + 2].strip(), fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def days_ago(n: int) -> date:
    return date.today() - timedelta(days=n)


def to_compact(value: date | str) -> str:
    """转成 akshare 常用的 YYYYMMDD。"""
    d = to_date(value)
    return d.strftime("%Y%m%d") if d else str(value)


def fmt_pct(value: float | None, digits: int = 2) -> str:
    """百分比格式化；缺失时返回“数据缺失”。"""
    if value is None:
        return "数据缺失"
    return f"{value:+.{digits}f}%" if value >= 0 else f"{value:.{digits}f}%"


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "数据缺失"
    return f"{value:.{digits}f}"
