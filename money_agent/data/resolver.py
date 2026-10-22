"""标的解析：把用户输入的“名称/代码”解析成明确的 Target。

解析优先级：
1. 显式 6 位代码（先基金后股票，再看板块代码）
2. 名称子串匹配（双向）
3. difflib 模糊匹配
再结合“基金 / 板块 / 行业 / 概念 / 指数”等提示词做类型加权。

所有名称表都走磁盘缓存（TTL 见 Settings.cache_ttl_hours），避免每次重复拉取。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import akshare as ak
import pandas as pd

from money_agent.cache import cached_dataframe
from money_agent.config import Settings
from money_agent.exceptions import TargetNotFound
from money_agent.models import Target, TargetKind

# 解析时需要从用户输入里剔除的填充词（不影响标的识别）
_STOPWORDS = [
    "今天", "今日", "现在", "目前", "最近", "近期", "如何", "怎么样", "怎样", "分析",
    "一下", "帮我", "看看", "走势", "行情", "情况", "怎么样", "该", "我的", "持仓",
    "成本", "份额", "成本价", "成本净值", "表现", "值得", "吗", "呢", "？", "?", "、",
    "和", "与", "的",
]

_CODE_RE = re.compile(r"\b(\d{6})\b")

# 类型提示词：用于加权，同时在匹配前从查询里剥离，避免“半导体板块”匹配不到“半导体”
_KIND_WORDS = ("板块", "行业", "概念", "指数", "基金", "ETF", "etf", "LOF", "lof")


@dataclass
class _Candidate:
    kind: TargetKind
    name: str
    code: str
    score: float
    meta: dict


def _load_funds(settings: Settings) -> pd.DataFrame:
    df = cached_dataframe(
        "fund_name_em",
        ak.fund_name_em,
        ttl_hours=settings.cache_ttl_hours,
        columns=["基金代码", "基金简称", "基金类型"],
    )
    if df is None or "基金简称" not in df.columns:
        return pd.DataFrame(columns=["基金代码", "基金简称", "基金类型"])
    return df


def _load_stocks(settings: Settings) -> pd.DataFrame:
    df = cached_dataframe(
        "stock_info_a_code_name",
        ak.stock_info_a_code_name,
        ttl_hours=settings.cache_ttl_hours,
        columns=["code", "name"],
    )
    if df is None or len(df.columns) < 2:
        return pd.DataFrame(columns=["code", "name"])
    return df.rename(columns={df.columns[0]: "code", df.columns[1]: "name"})


def _load_industries(settings: Settings) -> pd.DataFrame:
    df = cached_dataframe(
        "stock_board_industry_name_em",
        ak.stock_board_industry_name_em,
        ttl_hours=settings.cache_ttl_hours,
        columns=["板块名称", "板块代码"],
    )
    if df is None or "板块名称" not in getattr(df, "columns", []):
        return pd.DataFrame(columns=["板块名称", "板块代码"])
    return df


def _load_concepts(settings: Settings) -> pd.DataFrame:
    df = cached_dataframe(
        "stock_board_concept_name_em",
        ak.stock_board_concept_name_em,
        ttl_hours=settings.cache_ttl_hours,
        columns=["板块名称", "板块代码"],
    )
    if df is None or "板块名称" not in getattr(df, "columns", []):
        return pd.DataFrame(columns=["板块名称", "板块代码"])
    return df


def clean_query(text: str) -> str:
    """去掉会导致误匹配的持仓描述与填充词，仅保留标的相关文本。"""
    cleaned = text
    # 去掉“持仓：xxx”这类片段
    cleaned = re.split(r"(持仓|买入|成本|份额)", cleaned)[0]
    for word in _STOPWORDS:
        cleaned = cleaned.replace(word, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _fuzzy(query: str, names: list[str], cutoff: float = 0.5) -> tuple[str | None, float]:
    if not names:
        return None, 0.0
    matches = difflib.get_close_matches(query, names, n=1, cutoff=cutoff)
    if not matches:
        return None, 0.0
    best = matches[0]
    ratio = difflib.SequenceMatcher(None, query, best).ratio()
    return best, ratio


def _substring(query: str, names: list[str]) -> tuple[str | None, float]:
    """双向子串匹配，命中越“完整”得分越高。"""
    if not query or not names:
        return None, 0.0
    best, best_score = None, 0.0
    for name in names:
        if query == name:
            return name, 1.0
        if query in name:
            score = 0.6 + 0.4 * (len(query) / max(len(name), 1))
            if score > best_score:
                best, best_score = name, score
        elif name in query and len(name) >= 2:
            score = 0.55 + 0.3 * (len(name) / max(len(query), 1))
            if score > best_score:
                best, best_score = name, score
    return best, best_score


def _match_table(
    query: str, df: pd.DataFrame, name_col: str, kind: TargetKind, code_col: str | None
) -> _Candidate | None:
    names = df[name_col].astype(str).tolist()
    name, score = _substring(query, names)
    if name is None:
        name, score = _fuzzy(query, names)
    if name is None:
        return None
    row = df[df[name_col].astype(str) == name].iloc[0]
    code = str(row[code_col]) if code_col and code_col in df.columns else name
    return _Candidate(kind=kind, name=name, code=code, score=score, meta={})


def resolve_target(text: str, settings: Settings) -> Target:
    """把用户输入解析为唯一 Target；解析不出则抛 `TargetNotFound`。"""
    raw = text.strip()
    query = clean_query(raw) or raw
    core = _strip_kind_words(query) or query
    hint_kind = _hint_kind(raw)

    candidates: list[_Candidate] = []

    # ---------- 显式代码 ----------
    code_match = _CODE_RE.search(raw)
    if code_match:
        code = code_match.group(1)
        funds = _load_funds(settings)
        hit = funds[funds["基金代码"].astype(str) == code]
        if not hit.empty:
            row = hit.iloc[0]
            name = str(row["基金简称"])
            fund_type = str(row.get("基金类型", ""))
            return Target(
                raw=raw,
                name=name,
                code=code,
                kind="fund",
                meta={"基金类型": fund_type, "场内": _detect_exchange_traded(name, fund_type, code)},
            )
        stocks = _load_stocks(settings)
        hit = stocks[stocks["code"].astype(str) == code]
        if not hit.empty:
            return Target(raw=raw, name=str(hit.iloc[0]["name"]), code=code, kind="stock")

    # ---------- 名称匹配 ----------
    funds = _load_funds(settings)
    cand = _match_table(core, funds, "基金简称", "fund", "基金代码")
    if cand:
        row = funds[funds["基金简称"].astype(str) == cand.name].iloc[0]
        fund_type = str(row.get("基金类型", ""))
        cand.meta = {
            "基金类型": fund_type,
            "场内": _detect_exchange_traded(cand.name, fund_type, cand.code),
        }
        # “基金”字样出现时强烈加权
        if "基金" in raw or hint_kind == "fund":
            cand.score += 0.35
        candidates.append(cand)

    stocks = _load_stocks(settings)
    cand = _match_table(core, stocks, "name", "stock", "code")
    if cand:
        if hint_kind is None:
            cand.score += 0.1  # 裸名称默认偏向个股
        candidates.append(cand)

    # 板块名称表较慢，仅在查询可能指向板块、或基金/个股都没匹配上时才加载
    if hint_kind in ("industry", "concept") or not candidates:
        industries = _load_industries(settings)
        cand = _match_table(core, industries, "板块名称", "industry", "板块代码")
        if cand:
            if hint_kind in ("industry", "concept"):
                cand.score += 0.35
            candidates.append(cand)

        concepts = _load_concepts(settings)
        cand = _match_table(core, concepts, "板块名称", "concept", "板块代码")
        if cand:
            if hint_kind in ("concept", "industry"):
                cand.score += 0.3
            candidates.append(cand)

    if hint_kind in ("industry", "concept"):
        # 用户明确说了“板块/行业/概念”，就应给出板块标的
        board = [c for c in candidates if c.kind in ("industry", "concept")]
        if board:
            best = max(board, key=lambda c: c.score)
            return Target(raw=raw, name=best.name, code=best.code, kind=best.kind, meta=best.meta)
        # 板块名称表可能因网络问题为空 → 尽力而为地用核心词构造板块标的
        return Target(raw=raw, name=core, code=core, kind=hint_kind)

    if not candidates:
        raise TargetNotFound(f"无法从输入中解析出标的：{raw!r}")

    if hint_kind == "fund":
        funds_only = [c for c in candidates if c.kind == "fund"]
        if funds_only:
            candidates = funds_only

    best = max(candidates, key=lambda c: c.score)
    return Target(raw=raw, name=best.name, code=best.code, kind=best.kind, meta=best.meta)


def _strip_kind_words(query: str) -> str:
    """剥离“板块/行业/基金/ETF”等类型词，得到用于匹配的核心名词。"""
    core = query
    for word in _KIND_WORDS:
        core = core.replace(word, " ")
    return re.sub(r"\s+", " ", core).strip()


# 场内 ETF/LOF 常见代码前缀（沪 50/51/52/56/58，深 15/16）
_EXCHANGE_CODE_PREFIXES = ("15", "16", "50", "51", "52", "56", "58")


def _detect_exchange_traded(name: str, fund_type: str, code: str = "") -> bool:
    """ETF/LOF 场外联接不算场内；还需代码前缀匹配，避免把场外 C 类误判为场内。"""
    if "联接" in name:
        return False
    if "场内" in fund_type:
        return True
    upper = name.upper()
    if "ETF" not in upper and "LOF" not in upper:
        return False
    if code and code[:2] not in _EXCHANGE_CODE_PREFIXES:
        return False
    return True


def is_exchange_traded_fund(target: Target) -> bool:
    """判断基金是否为场内交易（ETF/LOF），应使用价格而非场外净值。"""
    if target.kind != "fund":
        return False
    if "场内" in target.meta:
        return bool(target.meta["场内"])
    return _detect_exchange_traded(
        target.name, str(target.meta.get("基金类型", "")), target.code
    )


def _hint_kind(raw: str) -> TargetKind | None:
    if "概念" in raw:
        return "concept"
    if "板块" in raw or "行业" in raw:
        return "industry"
    if "基金" in raw or "ETF" in raw or "etf" in raw or "LOF" in raw:
        return "fund"
    return None


def target_from_fund_code(code: str, settings: Settings, *, name: str = "") -> Target:
    """由基金代码构造 Target；名称表不可用时也能退化工作。"""
    funds = _load_funds(settings)
    if not funds.empty:
        hit = funds[funds["基金代码"].astype(str) == str(code).zfill(6)]
        if not hit.empty:
            row = hit.iloc[0]
            fund_name = str(row["基金简称"])
            fund_type = str(row.get("基金类型", ""))
            return Target(
                raw=code,
                name=fund_name,
                code=str(code).zfill(6),
                kind="fund",
                meta={
                    "基金类型": fund_type,
                    "场内": _detect_exchange_traded(fund_name, fund_type, str(code).zfill(6)),
                },
            )
    fund_code = str(code).zfill(6)
    return Target(
        raw=code,
        name=name or fund_code,
        code=fund_code,
        kind="fund",
        meta={"场内": fund_code[:2] in _EXCHANGE_CODE_PREFIXES},
    )


def find_fund_by_name(name: str, settings: Settings) -> dict | None:
    """按中文名称查基金，返回 {基金代码, 基金简称, 基金类型} 或 None。"""
    funds = _load_funds(settings)
    if funds.empty:
        return None
    cleaned = clean_query(name)
    for query in (cleaned, _strip_kind_words(cleaned) or cleaned, name):
        cand = _match_table(query, funds, "基金简称", "fund", "基金代码")
        if cand is not None:
            row = funds[funds["基金简称"].astype(str) == cand.name].iloc[0]
            return {
                "基金代码": str(row["基金代码"]),
                "基金简称": str(row["基金简称"]),
                "基金类型": str(row.get("基金类型", "")),
            }
    return None


def match_board(name: str, settings: Settings) -> tuple[str, TargetKind] | None:
    """把行业/概念名称匹配到东财板块名，返回 (板块名称, kind) 或 None。"""
    core = _strip_kind_words(name) or name
    for loader, kind in ((_load_industries, "industry"), (_load_concepts, "concept")):
        df = loader(settings)
        if df.empty or "板块名称" not in df.columns:
            continue
        options = df["板块名称"].astype(str).tolist()
        matched, _score = _substring(core, options)
        if matched is None:
            matched, _score = _fuzzy(core, options, cutoff=0.45)
        if matched:
            return matched, kind
    return None
