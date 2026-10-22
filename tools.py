"""Money-Agent 工具层（Function Calling tools）。

本文件只做一件事：把真实行情数据用 5 个工具暴露出去。
所有工具都只返回**真实抓取到的数据**；一旦异常/空数据，返回带 `error` 的字典，
由上层在报告中明确标注“数据缺失”，绝不返回臆造数字。

工具清单：
    1. get_fund_code_by_name(fund_name)              名称 → 基金代码
    2. get_fund_latest_info(fund_code)               最新净值（估算 / 正式）
    3. get_fund_history_kline(fund_code, period_days) 历史净值序列 + 涨跌幅 + 最大回撤
    4. rag_search_industry_report(industry_name)     研报检索（当前为桩函数）
    5. calc_user_position(fund_cost, current_net, share) 持仓账面收益

统一返回约定：
    成功 -> {...业务字段...}
    失败 -> {"error": "【错误类型】友好提示（原始原因）", "error_type": "...", "data_missing": True}
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from datetime import time as dtime
from typing import Any, Callable

# akshare 内部使用 tqdm 打进度条，会污染 Agent 输出；必须在导入前关闭
os.environ.setdefault("TQDM_DISABLE", "1")

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

logger = logging.getLogger("money_agent.tools")

# ==========================================================================
# 常量
# ==========================================================================

# 东方财富手机版接口：单只基金的实时估值 + 官方净值
# 注意：必须用手机端 UA，桌面 Chrome UA 会被拒绝并返回 Datas=null
_FUND_MOBILE_API = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNFInfo"
_MOBILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    )
}

# 场内 ETF/LOF 常见代码前缀（沪 50/51/52/56/58，深 15/16）
_EXCHANGE_CODE_PREFIXES = ("15", "16", "50", "51", "52", "56", "58")

# 东方财富基金全量数据 JS（净值序列的真实来源，akshare 内部也是取它）
# 这里直接解析，避免使用 py_mini_racer(V8) —— 它不是线程安全的，并行调用会崩溃。
_NAV_JS_API = "https://fund.eastmoney.com/pingzhongdata/{code}.js"
_DESKTOP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# 少数 akshare 接口内部使用 py_mini_racer(V8)，非线程安全，需要串行保护
_AKSHARE_LOCK = threading.Lock()

# 研报 RAG 桩提示
RAG_NOT_READY = "【当前RAG模块待接入，此处为板块占位，后续接入研报知识库】"

# 网络与重试参数（可用环境变量覆盖）
_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "15"))
_RETRY = int(os.getenv("FETCH_RETRY", "3"))
_RETRY_DELAY = float(os.getenv("FETCH_RETRY_DELAY", "1.0"))

_URL_RE = re.compile(r"https?://[^\s'\")]+")
_STOPWORDS = (
    "今天", "今日", "现在", "目前", "最近", "近期", "如何", "怎么样", "怎样", "分析",
    "一下", "帮我", "看看", "走势", "行情", "情况", "该", "我的", "持仓", "成本",
    "份额", "表现", "值得", "吗", "呢", "？", "?", "、", "和", "与", "的", "板块", "行业",
)


# ==========================================================================
# 异常捕获与友好提示
# ==========================================================================

def sanitize_error(exc: object, limit: int = 120) -> str:
    """把底层异常压成可读短句：去长 URL、折叠空白、截断。"""
    cleaned = _URL_RE.sub("<接口地址>", str(exc))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "…"


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """把异常归类为 (错误类型, 友好提示)。覆盖网络超时 / 限流 / 连接失败 / 解析失败。"""
    text = str(exc)
    lower = text.lower()
    name = type(exc).__name__.lower()

    if isinstance(exc, requests.Timeout) or "timeout" in name or "timed out" in lower:
        return "网络超时", "请求行情数据超时，请稍后重试"
    if "429" in text or "限流" in text or "too many requests" in lower or "rate limit" in lower:
        return "接口限流", "数据源返回限流，请降低调用频率后重试"
    if "proxyerror" in lower or isinstance(
        exc, (requests.ConnectionError, requests.RequestException)
    ) or "connection" in name:
        return "网络连接异常", "无法连接数据源，请检查网络或代理设置"
    if isinstance(exc, ValueError) or "json" in lower or "decode" in lower:
        return "数据解析异常", "数据源返回内容异常，无法解析"
    return "接口异常", "接口调用失败"


def friendly_error(exc: BaseException) -> dict[str, Any]:
    """构造统一的失败返回结构。"""
    kind, hint = classify_exception(exc)
    return {
        "error": f"【{kind}】{hint}（{sanitize_error(exc)}）",
        "error_type": kind,
        "data_missing": True,
    }


def empty_error(what: str) -> dict[str, Any]:
    """接口未抛错但返回空数据的情况，同样算“数据缺失”。"""
    return {
        "error": f"【数据为空】AKShare 未返回{what}",
        "error_type": "数据为空",
        "data_missing": True,
    }


def _retry(fn: Callable[[], Any], *, label: str, attempts: int | None = None) -> Any:
    """带指数退避的重试；全部失败后抛出最后一次异常，由调用方转成友好提示。

    只对“可重试”的异常重试：网络类问题重试有意义；参数/解析类错误重试无意义。
    """
    attempts = attempts or _RETRY
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 需兜住数据源五花八门的异常
            last = exc
            kind, _ = classify_exception(exc)
            if kind in ("数据解析异常", "接口异常"):
                break  # 重试大概率无用，直接失败
            if i < attempts:
                logger.warning("%s 第 %d 次失败：%s", label, i, sanitize_error(exc))
                time.sleep(_RETRY_DELAY * (1.6 ** (i - 1)))
    assert last is not None
    raise last


# ==========================================================================
# 时间逻辑：交易日 / 盘中
# ==========================================================================

_trade_calendar: dict[str, Any] = {"dates": None, "ts": 0.0, "failed": False}


def _load_trade_calendar() -> set[str] | None:
    """加载 A 股交易日历。

    注意：akshare 的交易日历接口内部使用 py_mini_racer(V8)，**不是线程安全的**，
    因此这里加锁串行，并且失败时做负缓存（1 小时内不再重试），避免在并行线程里
    反复触发 V8 导致进程崩溃。
    """
    now = time.time()
    # 成功缓存 1 天；失败缓存 1 小时
    ttl = 3600 if _trade_calendar["failed"] else 86400
    if _trade_calendar["dates"] is not None or _trade_calendar["failed"]:
        if now - _trade_calendar["ts"] < ttl:
            return _trade_calendar["dates"]  # type: ignore[return-value]
    try:
        with _AKSHARE_LOCK:
            df = ak.tool_trade_date_hist_sina()
        dates = {str(d)[:10] for d in df["trade_date"].astype(str)}
        _trade_calendar.update({"dates": dates, "ts": now, "failed": False})
        return dates
    except Exception as exc:  # noqa: BLE001
        logger.warning("交易日历获取失败，退化为仅判断周末：%s", sanitize_error(exc))
        _trade_calendar.update({"dates": None, "ts": now, "failed": True})
        return None


def is_trading_day(day: date | None = None) -> bool:
    """是否 A 股交易日。优先查交易日历；日历不可用时只排除周末。"""
    day = day or date.today()
    if day.weekday() >= 5:
        return False
    dates = _load_trade_calendar()
    if dates is None:
        return True  # 日历拿不到，退化为“工作日即交易日”
    return day.isoformat() in dates


def is_trading_time(now: datetime | None = None) -> bool:
    """是否处于盘中交易时段（交易日 09:30~15:00）。"""
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return False
    return dtime(9, 30) <= now.time() <= dtime(15, 0)


def market_status(now: datetime | None = None) -> str:
    """返回人类可读的市场状态，用于报告标注。"""
    now = now or datetime.now()
    if not is_trading_day(now.date()):
        return "非交易日"
    if now.time() < dtime(9, 30):
        return "交易日盘前"
    if now.time() > dtime(15, 0):
        return "交易日盘后"
    return "交易日盘中"


# ==========================================================================
# 基金名称表（带缓存）
# ==========================================================================

_fund_table: dict[str, Any] = {"df": None, "ts": 0.0}
_FUND_TABLE_TTL = 12 * 3600


def _get_fund_table() -> pd.DataFrame:
    """基金名称/代码/类型表，磁盘无关的内存缓存（12 小时）。"""
    now = time.time()
    if _fund_table["df"] is not None and now - _fund_table["ts"] < _FUND_TABLE_TTL:
        return _fund_table["df"]  # type: ignore[return-value]
    df = _retry(lambda: _call_akshare(ak.fund_name_em), label="获取基金名称表")
    if df is None or df.empty:
        raise RuntimeError("基金名称表为空")
    _fund_table.update({"df": df, "ts": now})
    return df


def _call_akshare(fn: Callable[[], Any]) -> Any:
    """串行执行 akshare 调用（部分接口内部用非线程安全的 py_mini_racer）。"""
    with _AKSHARE_LOCK:
        return fn()


def _clean_query(text: str) -> str:
    cleaned = text
    for word in _STOPWORDS:
        cleaned = cleaned.replace(word, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _match_name(query: str, names: list[str]) -> str | None:
    """精确 → 子串（双向）→ 模糊，逐级降级。"""
    if not names or not query:
        return None
    if query in names:
        return query
    # 双向子串，命中越完整得分越高
    best, best_score = None, 0.0
    for name in names:
        if query in name:
            score = 0.6 + 0.4 * (len(query) / max(len(name), 1))
        elif name in query and len(name) >= 2:
            score = 0.55 + 0.3 * (len(name) / max(len(query), 1))
        else:
            continue
        if score > best_score:
            best, best_score = name, score
    if best:
        return best
    matches = difflib.get_close_matches(query, names, n=1, cutoff=0.5)
    return matches[0] if matches else None


def _is_exchange_traded(code: str, name: str = "", fund_type: str = "") -> bool:
    """是否场内交易（ETF/LOF）。场外联接基金不算场内。"""
    if "联接" in name:
        return False
    if "场内" in fund_type:
        return True
    upper = name.upper()
    if "ETF" not in upper and "LOF" not in upper:
        return False
    # 还需代码前缀匹配，避免把“XX ETF联接C（如 012414）”误判成场内
    return code[:2] in _EXCHANGE_CODE_PREFIXES


# ==========================================================================
# 工具 1：名称 → 代码
# ==========================================================================

def get_fund_code_by_name(fund_name: str) -> dict[str, Any]:
    """根据基金中文名称，调用 akshare 查询对应的基金代码。

    :param fund_name: 基金名称，例如「半导体设备基金」「招商中证白酒」
    :return: {"fund_code", "fund_name", "fund_type"} 或 {"error", ...}
    """
    if not fund_name or not str(fund_name).strip():
        return {"error": "【入参错误】fund_name 不能为空", "error_type": "入参错误", "data_missing": True}

    try:
        table = _get_fund_table()
    except Exception as exc:  # noqa: BLE001
        return friendly_error(exc)

    df = table
    names = df["基金简称"].astype(str).tolist()
    query = str(fund_name).strip()
    hit = _match_name(query, names) or _match_name(_clean_query(query), names)

    if hit is None:
        return {
            "error": f"【未找到】未在基金名称表中找到「{fund_name}」，请确认名称是否正确",
            "error_type": "未找到",
            "data_missing": True,
        }

    row = df[df["基金简称"].astype(str) == hit].iloc[0]
    return {
        "fund_code": str(row["基金代码"]),
        "fund_name": str(row["基金简称"]),
        "fund_type": str(row.get("基金类型", "")),
        "_query": fund_name,
    }


def _lookup_fund_meta(fund_code: str) -> dict[str, str]:
    """补充基金名称与类型（用于判断场内/场外）；查不到返回空。"""
    try:
        table = _get_fund_table()
    except Exception:  # noqa: BLE001
        return {}
    hit = table[table["基金代码"].astype(str) == str(fund_code).zfill(6)]
    if hit.empty:
        return {}
    row = hit.iloc[0]
    return {"fund_name": str(row["基金简称"]), "fund_type": str(row.get("基金类型", ""))}


def get_fund_name_by_code(fund_code: str) -> str:
    """按代码反查基金简称（用户直接给代码时用于补全标题）；查不到返回空串。"""
    if not fund_code:
        return ""
    return _lookup_fund_meta(str(fund_code).zfill(6)).get("fund_name", "")


# ==========================================================================
# 工具 2：最新净值（盘中估算 / 盘后正式）
# ==========================================================================

def _to_float(value: Any) -> float | None:
    if value is None or value in ("", "--"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_fund_mobile(fund_code: str) -> dict[str, Any]:
    params = {
        "Fcodes": fund_code,
        "pageIndex": 1,
        "pageSize": 200,
        "plat": "Android",
        "appType": "ttjj",
        "product": "EFund",
        "Version": "1",
        "deviceid": "money-agent",
    }
    resp = requests.get(_FUND_MOBILE_API, params=params, headers=_MOBILE_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_fund_latest_info(fund_code: str) -> dict[str, Any]:
    """获取基金最新净值。

    - 交易日盘中（09:30~15:00）：返回天天基金盘中**估算净值**（tag="估算"，仅供参考）
    - 其他时间：返回 AKShare / 东财**官方正式净值**（tag="正式"）
    - 场内 ETF/LOF：返回**场内交易价格**（tag="正式"），并附带单位净值

    :return: {"nav", "nav_time", "tag", "nav_kind", "change_pct", ...} 或 {"error", ...}
    """
    if not fund_code or not str(fund_code).strip():
        return {"error": "【入参错误】fund_code 不能为空", "error_type": "入参错误", "data_missing": True}

    code = str(fund_code).strip().zfill(6)
    meta = _lookup_fund_meta(code)
    name = meta.get("fund_name", code)
    exchange = _is_exchange_traded(code, meta.get("fund_name", ""), meta.get("fund_type", ""))

    try:
        payload = _retry(lambda: _fetch_fund_mobile(code), label=f"获取 {code} 基金估值")
    except Exception as exc:  # noqa: BLE001
        return friendly_error(exc)

    datas = (payload or {}).get("Datas") or []
    if not datas:
        return empty_error(f"{code} 的最新净值数据")

    d = datas[0]
    expansion = payload.get("Expansion") or {}
    estimate_nav = _to_float(d.get("GSZ"))
    estimate_pct = _to_float(d.get("GSZZL"))
    estimate_time = d.get("GZTIME") or expansion.get("GZTIME")
    official_nav = _to_float(d.get("NAV"))
    official_date = d.get("PDATE")
    market_price = _to_float(d.get("NEWPRICE"))
    market_time = d.get("HQDATE")
    day_pct = _to_float(d.get("NAVCHGRT"))
    source = "东方财富基金估值接口(fundmobapi)"

    if exchange:
        nav = market_price if market_price is not None else official_nav
        if nav is None:
            return empty_error(f"{code} 的场内价格与单位净值")
        return {
            "fund_code": code,
            "fund_name": name,
            "is_exchange_traded": True,
            "nav": nav,
            "nav_time": market_time or official_date,
            "tag": "正式",
            "nav_kind": "场内交易价格（非场外净值）",
            "change_pct": _to_float(d.get("CHANGERATIO")) or day_pct,
            "unit_nav": official_nav,
            "unit_nav_date": official_date,
            "market_status": market_status(),
            "source": source,
            "note": "",
        }

    intraday = estimate_nav is not None and is_trading_time()
    if intraday:
        return {
            "fund_code": code,
            "fund_name": name,
            "is_exchange_traded": False,
            "nav": estimate_nav,
            "nav_time": estimate_time,
            "tag": "估算",
            "nav_kind": "盘中估算净值（仅参考，非正式成交价）",
            "change_pct": estimate_pct,
            "official_nav": official_nav,
            "official_nav_date": official_date,
            "market_status": market_status(),
            "source": source,
            "note": "盘中估算值由第三方平台测算，不能作为正式交易价格；正式净值盘后公布。",
        }

    if official_nav is None:
        return empty_error(f"{code} 的官方正式净值")

    note = ""
    if estimate_nav is None:
        if is_trading_time():
            note = (
                "当前已处于盘中，但数据源当前未提供该基金的盘中估算值（东方财富已下线该字段），"
                "此处返回官方正式净值。"
            )
        elif is_trading_day():
            note = "当前不在交易时段（09:30~15:00 之外），返回官方正式净值。"

    return {
        "fund_code": code,
        "fund_name": name,
        "is_exchange_traded": False,
        "nav": official_nav,
        "nav_time": official_date,
        "tag": "正式",
        "nav_kind": "官方正式单位净值",
        "change_pct": day_pct,
        "estimate_nav": estimate_nav,
        "estimate_time": estimate_time,
        "market_status": market_status(),
        "source": source,
        "note": note,
    }


# ==========================================================================
# 工具 3：历史净值序列 + 区间统计
# ==========================================================================

def _fetch_nav_direct(fund_code: str) -> list[dict[str, Any]]:
    """直连东财 pingzhongdata 解析单位净值序列（纯 requests，线程安全）。

    akshare 的 `fund_open_fund_info_em` 也是取这个数据，但内部用 py_mini_racer(V8)
    解析 JS，V8 非线程安全 → 并行调用会崩溃。这里用正则取出 JSON 数组直接解析。
    """
    resp = requests.get(
        _NAV_JS_API.format(code=fund_code), headers=_DESKTOP_HEADERS, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    match = re.search(r"var\s+Data_netWorthTrend\s*=\s*(\[.*?\])\s*;", resp.text, re.S)
    if not match:
        return []
    raw = match.group(1)
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        return []

    series: list[dict[str, Any]] = []
    for item in arr:
        ts, nav = item.get("x"), item.get("y")
        if ts is None or nav is None:
            continue
        # x 是北京时间 00:00 的毫秒时间戳 → 加 8 小时后按 UTC 取日期
        day = datetime.utcfromtimestamp(ts / 1000 + 8 * 3600).strftime("%Y-%m-%d")
        series.append({"date": day, "nav": float(nav)})

    series.sort(key=lambda p: p["date"])
    dedup: dict[str, float] = {}
    for point in series:
        dedup[point["date"]] = point["nav"]
    return [{"date": d, "nav": n} for d, n in sorted(dedup.items())]


def _fetch_nav_via_akshare(fund_code: str) -> pd.DataFrame:
    """akshare 兜底取净值（内部用 V8，需串行）。"""
    with _AKSHARE_LOCK:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        if df is None or df.empty:
            # 场内 ETF/LOF 用净值接口兜底
            end = date.today().strftime("%Y%m%d")
            start = (date.today() - timedelta(days=900)).strftime("%Y%m%d")
            df = ak.fund_etf_fund_info_em(fund=fund_code, start_date=start, end_date=end)
    return df


def _fetch_nav_series(fund_code: str) -> list[dict[str, Any]]:
    """获取净值序列（升序去重）。优先直连东财，失败再退回 akshare。"""
    try:
        series = _fetch_nav_direct(fund_code)
        if series:
            return series
        logger.warning("直连净值接口未返回数据，改用 akshare：%s", fund_code)
    except Exception as exc:  # noqa: BLE001
        logger.warning("直连净值接口失败，改用 akshare：%s", sanitize_error(exc))
    return _normalize_series(_fetch_nav_via_akshare(fund_code))


def _normalize_series(df: pd.DataFrame) -> list[dict[str, Any]]:
    """把不同接口的列名统一成 [{'date','nav'}]，按日期升序且去重。"""
    if df is None or df.empty:
        return []
    rename = {
        "净值日期": "date", "日期": "date",
        "单位净值": "nav", "净值": "nav", "累计净值": "nav",
    }
    frame = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "date" not in frame.columns or "nav" not in frame.columns:
        return []
    frame = frame[["date", "nav"]].dropna()
    frame["date"] = frame["date"].astype(str).str.slice(0, 10)
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = frame[frame["nav"].notna()]
    frame = frame.drop_duplicates(subset=["date"]).sort_values("date")
    return [{"date": r["date"], "nav": float(r["nav"])} for _, r in frame.iterrows()]


def _max_drawdown(series: list[dict[str, Any]]) -> float | None:
    """最大回撤（%）：区间最低点相对区间最高点的跌幅。"""
    if len(series) < 2:
        return None
    peak = series[0]["nav"]
    worst = 0.0
    for point in series:
        peak = max(peak, point["nav"])
        if peak > 0:
            worst = max(worst, (peak - point["nav"]) / peak)
    return round(-worst * 100, 2)


def _change_pct(series: list[dict[str, Any]]) -> float | None:
    """序列首尾涨跌幅（%）。"""
    if len(series) < 2 or series[0]["nav"] == 0:
        return None
    return round((series[-1]["nav"] / series[0]["nav"] - 1) * 100, 2)


def _with_daily_change(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, point in enumerate(series):
        prev = series[i - 1]["nav"] if i > 0 else None
        change = round((point["nav"] / prev - 1) * 100, 4) if prev else None
        out.append({"date": point["date"], "nav": point["nav"], "change_pct": change})
    return out


def get_fund_history_kline(fund_code: str, period_days: int) -> dict[str, Any]:
    """获取基金历史净值序列，并计算区间涨跌幅与最大回撤。

    :param fund_code: 6 位基金代码
    :param period_days: 统计周期，常用 7 / 15 / 30
    :return:
        {
          "data": [{"date","nav","change_pct"}, ...],   # 最近 period_days 个交易日
          "interval_change_pct": 区间涨跌幅(%),
          "max_drawdown_pct": 最大回撤(%),
          "latest_date", "latest_nav",
          "returns": {"7":..,"15":..,"30":..},          # 自然日口径，供报告模板使用
          "max_drawdown_30_pct": 30 自然日最大回撤(%),
        }
        或 {"error", ...}
    """
    if not fund_code or not str(fund_code).strip():
        return {"error": "【入参错误】fund_code 不能为空", "error_type": "入参错误", "data_missing": True}
    try:
        period_days = int(period_days)
    except (TypeError, ValueError):
        return {
            "error": f"【入参错误】period_days 必须为整数，收到 {period_days!r}",
            "error_type": "入参错误",
            "data_missing": True,
        }
    if period_days <= 0:
        return {"error": "【入参错误】period_days 必须为正整数", "error_type": "入参错误", "data_missing": True}

    code = str(fund_code).strip().zfill(6)
    try:
        full = _retry(lambda: _fetch_nav_series(code), label=f"获取 {code} 历史净值")
    except Exception as exc:  # noqa: BLE001
        return friendly_error(exc)

    if not full:
        return empty_error(f"{code} 的历史净值序列")

    # data 取最近 period_days 个交易日，保证长假后序列仍有足够可读点数
    window = full[-period_days:] if period_days <= len(full) else full

    def natural_return(days: int) -> float | None:
        """自然日口径：不晚于 (最新日 - days) 的最近一根作为基准。"""
        if len(full) < 2:
            return None
        latest = date.fromisoformat(full[-1]["date"])
        cutoff = latest - timedelta(days=days)
        candidates = [p for p in full if date.fromisoformat(p["date"]) <= cutoff]
        if not candidates or candidates[-1]["nav"] == 0:
            return None
        return round((full[-1]["nav"] / candidates[-1]["nav"] - 1) * 100, 2)

    cutoff30 = date.fromisoformat(full[-1]["date"]) - timedelta(days=30)

    return {
        "fund_code": code,
        "period_days": period_days,
        "period_basis": "交易日",
        "data": _with_daily_change(window),
        "interval_change_pct": _change_pct(window),
        "interval_baseline_date": window[0]["date"] if window else None,
        "max_drawdown_pct": _max_drawdown(window),
        "returns": {"7": natural_return(7), "15": natural_return(15), "30": natural_return(30)},
        "max_drawdown_30_pct": _max_drawdown(
            [p for p in full if date.fromisoformat(p["date"]) >= cutoff30]
        ),
        "latest_date": full[-1]["date"],
        "latest_nav": full[-1]["nav"],
        "points_total": len(full),
    }


# ==========================================================================
# 工具 4：研报 RAG（桩函数）
# ==========================================================================

def rag_search_industry_report(industry_name: str) -> dict[str, Any]:
    """检索行业公开研报。

    ⚠️ 当前为**桩函数**，尚未接入研报知识库，固定返回占位提示。
    后续接入向量库时，只需替换本函数实现，返回结构保持不变。

    :param industry_name: 行业/主题名称，例如「半导体设备」
    :return: {"industry", "summary", "catalysts", "risks", "sources"}
    """
    return {
        "industry": str(industry_name or "").strip(),
        "summary": RAG_NOT_READY,
        "catalysts": [],
        "risks": [],
        "sources": [],
        "is_stub": True,
    }


# ==========================================================================
# 工具 5：持仓账面收益
# ==========================================================================

def calc_user_position(fund_cost: float, current_net: float, share: float) -> dict[str, Any]:
    """计算用户持仓的账面收益率与账面盈亏（不含任何费用）。

    :param fund_cost: 持仓成本净值
    :param current_net: 当前净值
    :param share: 持仓份额
    """
    try:
        cost = float(fund_cost)
        current = float(current_net)
        shares = float(share)
    except (TypeError, ValueError):
        return {
            "error": "【入参错误】fund_cost / current_net / share 必须为数字",
            "error_type": "入参错误",
            "data_missing": True,
        }
    if cost <= 0:
        return {"error": "【入参错误】持仓成本净值必须大于 0", "error_type": "入参错误", "data_missing": True}
    if shares < 0:
        return {"error": "【入参错误】持仓份额不能为负数", "error_type": "入参错误", "data_missing": True}

    return {
        "fund_cost": round(cost, 4),
        "current_net": round(current, 4),
        "share": round(shares, 2),
        "return_pct": round((current / cost - 1) * 100, 2),
        "pnl_amount": round((current - cost) * shares, 2),
        "market_value": round(current * shares, 2),
        "cost_value": round(cost * shares, 2),
        "note": "仅账面计算，不含申购赎回费、管理费与税费",
    }


# ==========================================================================
# Function Calling 定义与执行入口
# ==========================================================================

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_fund_code_by_name",
            "description": "根据基金中文名称，调用akshare查询对应的基金代码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_name": {"type": "string", "description": "基金名称字符串，例如“半导体设备基金”"}
                },
                "required": ["fund_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_latest_info",
            "description": "获取基金最新净值。交易时段返回盘中估算净值；盘后获取官方正式净值。返回净值数值、时间、标记（估算/正式）。",
            "parameters": {
                "type": "object",
                "properties": {"fund_code": {"type": "string", "description": "6 位基金代码"}},
                "required": ["fund_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_history_kline",
            "description": "获取基金历史净值序列，支持 7/15/30 天，返回每日日期、单位净值、日涨跌幅，并自动计算区间涨跌幅与最大回撤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_code": {"type": "string", "description": "6 位基金代码"},
                    "period_days": {
                        "type": "integer",
                        "description": "统计周期（交易日），常用 7 / 15 / 30",
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
            "description": "检索行业公开研报，返回研报摘要、核心利好/利空与来源信息。（当前为占位实现）",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry_name": {"type": "string", "description": "行业或主题名称，例如“半导体设备”"}
                },
                "required": ["industry_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_user_position",
            "description": "计算用户持仓的账面收益率与账面盈亏。",
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

TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_fund_code_by_name": get_fund_code_by_name,
    "get_fund_latest_info": get_fund_latest_info,
    "get_fund_history_kline": get_fund_history_kline,
    "rag_search_industry_report": rag_search_industry_report,
    "calc_user_position": calc_user_position,
}

TOOL_NAMES = tuple(TOOL_FUNCTIONS)


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行单个工具，永不抛异常；失败统一返回友好错误字典。

    这是 Agent 与工具之间唯一的执行入口。
    """
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {
            "error": f"【未知工具】不存在名为 {name} 的工具",
            "error_type": "未知工具",
            "data_missing": True,
        }
    try:
        return fn(**(arguments or {}))
    except TypeError as exc:
        return {
            "error": f"【参数错误】{name} 调用参数不匹配（{sanitize_error(exc)}）",
            "error_type": "参数错误",
            "data_missing": True,
        }
    except Exception as exc:  # noqa: BLE001
        return friendly_error(exc)


def execute_tools_parallel(calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """并行执行多个工具调用（用于同时拉取 7/15/30 天 K 线）。"""
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as executor:
        futures = [executor.submit(execute_tool, name, args) for name, args in calls]
        return [f.result() for f in futures]


def warm_up() -> dict[str, Any]:
    """在主线程预热“非线程安全”的公共依赖，避免在并行 worker 中首次触发。

    - akshare 基金名称表（内部可能用到 py_mini_racer）
    - A 股交易日历（内部使用 py_mini_racer，V8 非线程安全）

    返回预热状态，便于日志与测试观察。可在 Agent 启动时调用一次。
    """
    status: dict[str, Any] = {}
    try:
        status["fund_table"] = len(_get_fund_table())
    except Exception as exc:  # noqa: BLE001
        status["fund_table"] = f"失败：{sanitize_error(exc)}"
    try:
        dates = _load_trade_calendar()
        status["trade_calendar"] = "未获取（退化为仅判断周末）" if dates is None else len(dates)
    except Exception as exc:  # noqa: BLE001
        status["trade_calendar"] = f"失败：{sanitize_error(exc)}"
    status["market_status"] = market_status()
    return status
