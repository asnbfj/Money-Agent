"""客观统计量计算：区间涨跌幅、最大回撤、年化波动率、逐日涨跌幅。

口径说明（必须与报告、工具返回一致）：
- 近 N 日涨跌幅：以最新一个数据日为终点，取「不晚于 (最新日 - N 自然日)」的最近一根，
  即 (最新值 / 基准值 - 1) × 100%。
- 近 N 日最大回撤：窗口内最低点相对窗口内最高点的跌幅（负数）。
- 波动率：窗口内每日涨跌幅的样本标准差 × √252（年化），单位 %。
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from money_agent.models import Analytics, History, HistoryPoint
from money_agent.utils import to_date


def latest_date(points: list[HistoryPoint]) -> date | None:
    if not points:
        return None
    return to_date(points[-1].date)


def window_points(
    points: list[HistoryPoint], days: int, latest: date | None = None
) -> list[HistoryPoint]:
    """取最近 `days` 自然日内的数据点。"""
    if not points:
        return []
    latest = latest or latest_date(points)
    if latest is None:
        return list(points)
    cutoff = latest - timedelta(days=days)
    return [p for p in points if (to_date(p.date) or latest) >= cutoff]


def period_baseline(
    points: list[HistoryPoint], days: int, latest: date | None = None
) -> HistoryPoint | None:
    """区间涨跌幅的基准点：不晚于 (最新日 - days) 的最近一根。"""
    latest = latest or latest_date(points)
    if latest is None:
        return None
    cutoff = latest - timedelta(days=days)
    candidates = [p for p in points if (to_date(p.date) or latest) <= cutoff]
    return candidates[-1] if candidates else None


def period_return(
    points: list[HistoryPoint], days: int
) -> tuple[float | None, str | None]:
    """区间涨跌幅（%）与基准日期。"""
    if len(points) < 2:
        return None, None
    latest_point = points[-1]
    base = period_baseline(points, days)
    if base is None or base.close == 0:
        return None, None
    pct = (latest_point.close / base.close - 1) * 100
    return round(pct, 2), base.date


def last_n_points(points: list[HistoryPoint], n: int) -> list[HistoryPoint]:
    """取最近 n 个数据点（交易日口径），保证 K 线序列有足够的可读点数。"""
    if n <= 0 or not points:
        return []
    return points[-n:]


def max_drawdown_of(points: list[HistoryPoint]) -> float | None:
    """给定序列的最大回撤（%，负数）。"""
    if len(points) < 2:
        return None
    peak = points[0].close
    max_dd = 0.0
    for point in points:
        peak = max(peak, point.close)
        if peak > 0:
            max_dd = max(max_dd, (peak - point.close) / peak)
    return round(-max_dd * 100, 2)


def change_over(points: list[HistoryPoint]) -> float | None:
    """给定序列首尾的区间涨跌幅（%）。"""
    if len(points) < 2 or points[0].close == 0:
        return None
    return round((points[-1].close / points[0].close - 1) * 100, 2)


def period_max_drawdown(points: list[HistoryPoint], days: int) -> float | None:
    """区间最大回撤（%，负数）。"""
    return max_drawdown_of(window_points(points, days))


def annualized_volatility(points: list[HistoryPoint], days: int) -> float | None:
    """区间年化波动率（%）。"""
    closes = [p.close for p in window_points(points, days)]
    if len(closes) < 3:
        return None
    returns = [
        closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] != 0
    ]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return round(math.sqrt(variance) * math.sqrt(252) * 100, 2)


def points_with_change(points: list[HistoryPoint]) -> list[dict]:
    """逐点带上日涨跌幅，供工具返回 data 列表。"""
    result: list[dict] = []
    for i, point in enumerate(points):
        change = None
        if i > 0 and points[i - 1].close:
            change = round((point.close / points[i - 1].close - 1) * 100, 4)
        result.append({"date": point.date, "nav": point.close, "change_pct": change})
    return result


def trend_label(
    ret_7: float | None, ret_15: float | None, ret_30: float | None, vol: float | None
) -> str:
    available = [r for r in (ret_7, ret_15, ret_30) if r is not None]
    if not available:
        return "数据不足，无法判断"
    short = ret_7 if ret_7 is not None else available[-1]
    mid = ret_30 if ret_30 is not None else available[0]
    if short > 1.5 and mid > 3:
        base = "近期持续上涨"
    elif short < -1.5 and mid < -3:
        base = "近期持续下跌"
    elif abs(short) <= 1.5 and abs(mid) <= 3:
        base = "近期区间震荡"
    elif short > 0 >= mid:
        base = "短期反弹、中期仍偏弱"
    elif short < 0 <= mid:
        base = "中期偏强、短期回调"
    else:
        base = "近期涨跌互现"
    if vol is not None:
        if vol >= 30:
            base += "，波动显著偏高"
        elif vol >= 15:
            base += "，波动中等"
        else:
            base += "，波动相对温和"
    return base


def compute_analytics(history: History) -> Analytics:
    """基于归一化历史序列计算报告所需的统计量。"""
    if not history.ok:
        return Analytics(error=history.error or "无历史数据", trend_label="数据缺失，无法判断")

    points = sorted(history.points, key=lambda p: p.date)
    latest = latest_date(points)
    if latest is None:
        return Analytics(error="历史数据日期解析失败", trend_label="数据缺失，无法判断")

    ret_7, base_7 = period_return(points, 7)
    ret_15, base_15 = period_return(points, 15)
    ret_30, base_30 = period_return(points, 30)
    vol = annualized_volatility(points, 30)

    return Analytics(
        ret_7=ret_7,
        ret_15=ret_15,
        ret_30=ret_30,
        max_drawdown_30=period_max_drawdown(points, 30),
        volatility_30_annualized=vol,
        trend_label=trend_label(ret_7, ret_15, ret_30, vol),
        baseline={
            "7": base_7 or "",
            "15": base_15 or "",
            "30": base_30 or "",
            "latest": points[-1].date,
        },
    )
