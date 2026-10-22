"""自建盘中估值：官方盘中估值接口下线后的替代方案。

背景
----
东方财富 / 腾讯的基金「盘中估值」（GSZ、getSsgz）已于 2026 年下线
（见 fund_estimate.py 的实测记录），`fetch_quote` 因此只能回落到 T-1 的
官方正式净值 —— 交易时段里数字一整天不动。

本模块用两条路径自己算一个「估算涨跌」：

1. **持仓加权**（默认）：取基金最近一期季报的重仓股与权重，
   逐个取实时股价，按权重加权得到估算涨跌。
   未被季报覆盖的部分（现金、打新、其他持仓）按「与已覆盖部分同涨同跌」
   处理，即对权重做归一化，避免引入无关基准的噪音。

2. **代理标的**（季报股票仓位极低的 ETF 联接基金）：这类基金主要持有目标
   ETF，股票投资明细几乎为空。改用一只高相关 ETF 的实时涨跌 × 历史 beta。
   PROXIES 里的 bp/beta 是在 120 个交易日上回归得到的，不是拍脑袋的系数。

准确性
------
- 只覆盖 A 股 / 港股（港股通）持仓；季报权重有滞后，个股停牌会缺失。
- QDII（美股、韩股等）不支持，直接返回原因，不做猜测。
- 估算值仅用于盘中观察涨跌方向，不等于基金公司公布的正式净值。
"""

from __future__ import annotations

import re
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

from money_agent.cache import cached_dataframe
from money_agent.data.market import is_trading_day

# 实时行情内存缓存（与服务端报价缓存同量级，避免一轮组合请求打太多次接口）
QUOTE_TTL_SECONDS = 20.0
# 季报持仓变化慢，磁盘缓存 12 小时
HOLDINGS_TTL_HOURS = 12.0
# 季报股票权重覆盖低于该比例时认为「不够代表整只基金」，改走代理标的
MIN_COVERAGE = 0.5
# 净值基准过期阈值：超过这么多天没有新净值就不估算（QDII / 长期停牌）
MAX_BASE_AGE_DAYS = 5

QUOTE_URL = "https://qt.gtimg.cn/q="
MAX_SYMBOLS_PER_REQUEST = 60
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# 季报股票仓位不足时的代理标的：(腾讯代码, beta, 说明)
# beta 由 120 个交易日的「基金日收益 ~ ETF 日收益」回归得到
PROXIES: dict[str, tuple[str, float, str]] = {
    "020465": ("sh512760", 0.965, "芯片ETF(512760)"),
    "021894": ("sh512760", 0.976, "芯片ETF(512760)"),
}

_quote_cache: dict[str, tuple[float, dict]] = {}
_quote_lock = threading.Lock()
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _names_compatible(expected: str, actual: str) -> bool:
    """防止「代码撞号」污染估值。

    季报里的境外标的是纯数字代码时（如韩股 SK海力士 = 000660），会被行情接口
    当成深市 A 股代码（sz000660 = *ST南华）。用名称里共同的中文字符做校验：
    完全无关就丢弃这条行情，宁可不估也不要用错价格。
    """
    left = "".join(_CJK_RE.findall(str(expected or "")))
    right = "".join(_CJK_RE.findall(str(actual or "")))
    if not left or not right:
        return True  # 名称缺失时不做判断，保持原有行为
    if left in right or right in left:
        return True
    return len(set(left) & set(right)) >= 2


# --------------------------------------------------------------------------
# 实时行情
# --------------------------------------------------------------------------
def tencent_symbol(code: str) -> str | None:
    """把 6 位 A 股 / 5 位港股代码换成腾讯行情代码；无法识别的返回 None。"""
    code = str(code or "").strip()
    if len(code) == 5 and code.isdigit():
        return f"hk{code}"
    if len(code) != 6 or not code.isdigit():
        return None
    if code[0] == "6" or code.startswith(("5", "9")):
        return f"sh{code}"
    if code[0] in "03" or code.startswith(("1", "2")):
        return f"sz{code}"
    if code[0] in "48":
        return f"bj{code}"
    return None


def _parse_quotes(text: str) -> dict[str, dict]:
    """解析 `v_sh600519="1~贵州茅台~600519~现价~昨收~…"`，涨跌幅用现价/昨收自行计算。

    不依赖腾讯的字段位置（A 股/港股/ETF 的列序并不完全一致），只取前 5 段。
    """
    result: dict[str, dict] = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        payload = line.split('"', 2)
        if len(payload) < 2:
            continue
        parts = payload[1].split("~")
        if len(parts) < 5:
            continue
        code = parts[2].strip()
        try:
            price = float(parts[3])
            prev = float(parts[4])
        except (TypeError, ValueError):
            continue
        if not code or price <= 0 or prev <= 0:
            continue
        result[code] = {
            "name": parts[1].strip(),
            "price": price,
            "prev_close": prev,
            "change_pct": round((price / prev - 1) * 100, 2),
        }
    return result


def quotes(codes: list[str]) -> dict[str, dict]:
    """批量取实时行情（带 20 秒内存缓存）。返回 {原始代码: {...}}。"""
    wanted: dict[str, str] = {}
    for code in codes:
        symbol = tencent_symbol(code)
        if symbol:
            wanted[code] = symbol
    if not wanted:
        return {}

    now = time.monotonic()
    out: dict[str, dict] = {}
    missing: dict[str, str] = {}
    with _quote_lock:
        for code, symbol in wanted.items():
            cached = _quote_cache.get(symbol)
            if cached and now - cached[0] < QUOTE_TTL_SECONDS:
                out[code] = cached[1]
            else:
                missing[code] = symbol
    if not missing:
        return out

    symbols = list(dict.fromkeys(missing.values()))
    fetched: dict[str, dict] = {}
    for start in range(0, len(symbols), MAX_SYMBOLS_PER_REQUEST):
        chunk = symbols[start : start + MAX_SYMBOLS_PER_REQUEST]
        try:
            resp = requests.get(QUOTE_URL + ",".join(chunk), headers=_HEADERS, timeout=8)
            resp.encoding = "gbk"
            fetched.update(_parse_quotes(resp.text))
        except Exception:  # noqa: BLE001 - 取不到就少几只，不影响其余持仓
            continue

    with _quote_lock:
        for code, symbol in missing.items():
            by_code = fetched.get(symbol[2:])
            if by_code:
                _quote_cache[symbol] = (now, by_code)
                out[code] = by_code
    return out


# --------------------------------------------------------------------------
# 季报持仓
# --------------------------------------------------------------------------
def _load_holdings(code: str):
    """抓季报股票持仓（akshare）。优先用「覆盖度够高」的最近一期，避免季报披露不全。

    返回 (DataFrame, 季度标签)；失败返回 (None, "")。
    """
    import akshare as ak
    import pandas as pd

    best = None
    for attempt in range(2):
        for year in (date.today().year, date.today().year - 1):
            try:
                df = ak.fund_portfolio_hold_em(symbol=code, date=str(year))
            except Exception:  # noqa: BLE001 - 514 限流等，退避後重试
                time.sleep(3 + attempt * 3)
                continue
            if df is None or df.empty:
                continue
            chosen = None
            for quarter in df["季度"].unique():
                sub = df[df["季度"] == quarter].copy()
                sub["占净值比例"] = pd.to_numeric(sub["占净值比例"], errors="coerce")
                sub = sub[sub["占净值比例"] > 0]
                if sub.empty:
                    continue
                # 越新的季度越可能披露不全（季报只列前十大），
                # 这里以「权重覆盖 50% 以上」为可用门槛，取最近一期达标的
                total = float(sub["占净值比例"].sum())
                if total >= MIN_COVERAGE * 100:
                    chosen = (sub, str(quarter))
            if chosen is not None:
                best = chosen
                break
            # 没有达标的季度，退而求其次用最近一期
            last = df["季度"].unique()[-1]
            sub = df[df["季度"] == last].copy()
            sub["占净值比例"] = pd.to_numeric(sub["占净值比例"], errors="coerce")
            sub = sub[sub["占净值比例"] > 0]
            if not sub.empty:
                best = (sub, str(last))
        if best is not None:
            break
    if best is None:
        return None
    sub, quarter = best
    frame = sub[["股票代码", "股票名称", "占净值比例"]].reset_index(drop=True)
    frame.attrs["quarter"] = quarter
    return frame


def holdings(code: str):
    """带磁盘缓存的季报持仓（12 小时）。返回 DataFrame 或 None。"""
    frame = cached_dataframe(
        key=f"fund_holdings_v2_{code}",
        loader=lambda: _load_holdings(code),
        ttl_hours=HOLDINGS_TTL_HOURS,
        empty_ttl_hours=0.5,
        columns=["股票代码", "股票名称", "占净值比例"],
    )
    if frame is not None and frame.empty:
        return None
    return frame


# --------------------------------------------------------------------------
# 估算
# --------------------------------------------------------------------------
def _base_age_days(base_date: str | None) -> float:
    if not base_date:
        return 9999.0
    try:
        stamp = datetime.strptime(str(base_date)[:10], "%Y-%m-%d").date()
    except ValueError:
        return 9999.0
    return float((date.today() - stamp).days)


def estimate_change(code: str) -> dict[str, Any]:
    """算一只基金的估算涨跌幅（%）。失败时返回 {"error": "原因"}。"""
    frame = holdings(code)
    quarter = ""
    reported = 0.0
    covered = 0.0
    weighted = 0.0
    detail: list[dict] = []

    if frame is not None and not frame.empty:
        quarter = frame.attrs.get("quarter") or "最近一期季报"
        try:
            reported = float(frame["占净值比例"].sum()) / 100.0
        except (TypeError, ValueError):
            reported = 0.0
        market = quotes([str(c) for c in frame["股票代码"].tolist()])
        for _, row in frame.iterrows():
            stock = str(row["股票代码"])
            quote = market.get(stock)
            if not quote or not _names_compatible(row["股票名称"], quote.get("name")):
                continue
            weight = float(row["占净值比例"])
            weighted += weight * quote["change_pct"]
            covered += weight
            detail.append({
                "code": stock,
                "name": str(row["股票名称"]),
                "weight": round(weight, 2),
                "change_pct": quote["change_pct"],
            })

    # 关键守卫：季报覆盖率高不代表能定价（QDII 的美股/韩股拿不到行情），
    # 必须按「真正取到实时价格」的权重来判断能不能算。
    if covered and covered / 100.0 >= MIN_COVERAGE:
        detail.sort(key=lambda x: -x["weight"])
        return {
            "change_pct": round(weighted / covered, 2),
            "method": "holdings",
            "method_label": "季报重仓股加权",
            "coverage": round(covered / 100.0, 4),
            "reported_coverage": round(reported, 4),
            "quarter": quarter,
            "constituents": detail[:10],
            "constituents_total": len(detail),
            "updated_at": time.strftime("%H:%M:%S"),
        }

    proxy = PROXIES.get(str(code))
    if proxy:
        symbol, beta, label = proxy
        try:
            resp = requests.get(QUOTE_URL + symbol, headers=_HEADERS, timeout=8)
            resp.encoding = "gbk"
            parsed = _parse_quotes(resp.text)
        except Exception:  # noqa: BLE001
            parsed = {}
        item = next(iter(parsed.values()), None)
        if not item:
            return {"error": f"代理标的 {label} 行情不可用"}
        return {
            "change_pct": round(item["change_pct"] * beta, 2),
            "method": "proxy",
            "method_label": f"{label} × {beta:g}",
            "coverage": round(beta, 4),
            "reported_coverage": round(reported, 4),
            "quarter": quarter,
            "constituents": [{
                "code": symbol[2:],
                "name": label,
                "weight": round(beta * 100, 1),
                "change_pct": item["change_pct"],
            }],
            "constituents_total": 1,
            "updated_at": time.strftime("%H:%M:%S"),
        }

    if frame is not None and not frame.empty:
        priced = covered / 100.0
        if reported >= MIN_COVERAGE:
            return {"error": f"季报持仓有 {reported * 100:.0f}%，但只有 {priced * 100:.1f}% 能取到实时行情（多为境外标的），暂不估算"}
        return {"error": f"季报股票仓位仅 {reported * 100:.1f}%，且没有配置代理标的"}
    return {"error": "没有可用的季报持仓数据"}


def estimate_for(
    code: str,
    base_nav: float | None,
    base_date: str | None,
    *,
    trading_day: bool,
) -> dict[str, Any] | None:
    """给一只基金生成估算结果（含估算净值）。不适用时返回 None。

    只有在「今天是交易日」且「今天的正式净值还没公布」时才有意义；
    其余情况沿用官方净值，避免用昨天的股价去估算今天的净值。
    """
    if not trading_day:
        return None
    if base_nav is None or float(base_nav) <= 0:
        return None
    if _base_age_days(base_date) > MAX_BASE_AGE_DAYS:
        return {"error": "最近净值太久没更新，估算基准不可用"}
    if str(base_date or "")[:10] == date.today().isoformat():
        return None  # 今天的正式净值已经公布，直接用官方值

    result = estimate_change(code)
    if result.get("error"):
        return result
    change = float(result["change_pct"])
    result["base_nav"] = round(float(base_nav), 4)
    result["base_date"] = str(base_date)[:10] if base_date else None
    result["estimate_nav"] = round(float(base_nav) * (1 + change / 100), 4)
    return result
