"""券商研报抓取与轻量检索排序。"""

from __future__ import annotations

from collections import Counter
from datetime import date

import akshare as ak
import pandas as pd

from money_agent.config import Settings
from money_agent.data.resolver import match_board
from money_agent.data.retry import safe_call
from money_agent.models import ReportItem, Target
from money_agent.utils import to_date

# 中文停用词/通用词，参与关键词匹配时剔除，避免噪声
_STOP_BIGRAMS = {
    "公司", "行业", "报告", "点评", "研究", "证券", "股份", "集团", "科技", "维持",
    "首次", "覆盖", "更新", "事件", "深度", "系列", "跟踪", "中国", "市场", "季度",
    "年度", "半年", "业绩", "增长", "预期", "估值", "评级", "买入", "增持",
}


def _bigrams(text: str) -> set[str]:
    cleaned = "".join(ch for ch in text if ch.isalnum())
    grams = {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}
    return {g for g in grams if g not in _STOP_BIGRAMS}


def _normalize_reports(df: pd.DataFrame) -> list[ReportItem]:
    if df is None or df.empty:
        return []
    cols = {str(c): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    title_col = pick("报告名称", "标题", "研报名称")
    if title_col is None:
        return []

    items: list[ReportItem] = []
    for _, row in df.iterrows():
        def get(*names: str) -> str:
            col = pick(*names)
            if col is None:
                return ""
            value = row.get(col)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            return str(value).strip()

        items.append(
            ReportItem(
                title=get("报告名称", "标题"),
                institution=get("机构", "机构名称"),
                rating=get("东财评级", "评级"),
                industry=get("行业", "行业名称"),
                date=get("日期"),
                url=get("报告PDF链接", "链接"),
                stock_name=get("股票简称", "股票名称"),
                stock_code=get("股票代码", "代码"),
            )
        )

    seen: set[str] = set()
    deduped: list[ReportItem] = []
    for item in items:
        if item.title and item.title not in seen:
            seen.add(item.title)
            deduped.append(item)
    return deduped


def _reports_for_stock(code: str, settings: Settings) -> list[ReportItem]:
    df, _ = safe_call(
        lambda: ak.stock_research_report_em(symbol=code),
        label=f"获取 {code} 研报",
        attempts=settings.retry_attempts,
        delay=settings.retry_delay,
    )
    return _normalize_reports(df) if df is not None else []


def _fund_holdings(code: str, settings: Settings, top: int) -> list[tuple[str, str]]:
    """取基金最新一期重仓股 (代码, 名称)。"""
    for year in (date.today().year, date.today().year - 1):
        df, _ = safe_call(
            lambda y=year: ak.fund_portfolio_hold_em(symbol=code, date=str(y)),
            label=f"获取 {code} 持仓",
            attempts=max(1, settings.retry_attempts - 1),
            delay=settings.retry_delay,
        )
        if df is None or df.empty:
            continue
        code_col = next((c for c in ("股票代码", "证券代码") if c in df.columns), None)
        name_col = next((c for c in ("股票名称", "证券名称") if c in df.columns), None)
        quarter_col = next((c for c in ("季度", "报告期") if c in df.columns), None)
        if code_col is None:
            continue
        if quarter_col is not None:
            latest_quarter = df[quarter_col].astype(str).max()
            df = df[df[quarter_col].astype(str) == latest_quarter]
        pairs: list[tuple[str, str]] = []
        for _, row in df.head(top).iterrows():
            stock_code = str(row[code_col]).zfill(6)
            stock_name = str(row[name_col]) if name_col else ""
            pairs.append((stock_code, stock_name))
        if pairs:
            return pairs
    return []


def _top_constituents(board_name: str, kind: str, settings: Settings, top: int) -> list[tuple[str, str]]:
    loader = (
        (lambda: ak.stock_board_industry_cons_em(symbol=board_name))
        if kind == "industry"
        else (lambda: ak.stock_board_concept_cons_em(symbol=board_name))
    )
    df, _ = safe_call(
        loader, label=f"获取板块 {board_name} 成分股", attempts=settings.retry_attempts,
        delay=settings.retry_delay,
    )
    if df is None or df.empty:
        return []
    code_col = next((c for c in ("代码", "股票代码") if c in df.columns), None)
    name_col = next((c for c in ("名称", "股票名称") if c in df.columns), None)
    if code_col is None:
        return []
    pairs = []
    for _, row in df.head(top).iterrows():
        pairs.append((str(row[code_col]).zfill(6), str(row[name_col]) if name_col else ""))
    return pairs


def rank_reports(
    items: list[ReportItem],
    query: str,
    *,
    top_k: int = 6,
    today: date | None = None,
) -> list[ReportItem]:
    """在研报元数据上做「关键词重叠 + 时效」加权排序。"""
    today = today or date.today()
    query_terms = _bigrams(query)

    for item in items:
        text = f"{item.title} {item.industry} {item.stock_name}"
        terms = _bigrams(text)
        overlap = len(query_terms & terms) / max(len(query_terms), 1) if query_terms else 0.0
        published = to_date(item.date)
        days = (today - published).days if published else 365
        recency = 1.0 / (1.0 + max(days, 0) / 30.0)
        item.relevance = round(0.6 * overlap + 0.4 * recency, 4)

    items.sort(key=lambda i: (i.relevance, i.date), reverse=True)
    return items[:top_k]


def search_industry_reports(
    industry_name: str,
    settings: Settings,
    *,
    fund_target: Target | None = None,
    top_k: int | None = None,
) -> tuple[list[ReportItem], str, str | None]:
    """按行业/主题名检索研报（供 `rag_search_industry_report` 工具调用）。

    策略：板块名 → 成分股研报；取不到时若当前标的是基金 → 用其重仓股研报。
    :return: (研报列表, 实际使用的行业/主题名, 错误信息)
    """
    collected: list[ReportItem] = []
    used = industry_name

    board = match_board(industry_name, settings) if industry_name else None
    if board is not None:
        candidates = [(board[0], board[1])]
    elif industry_name:
        # 名称表拿不到时，仍可直接把入参当板块名试一次成分股
        candidates = [(industry_name, "industry"), (industry_name, "concept")]
    else:
        candidates = []

    for board_name, kind in candidates:
        stocks = _top_constituents(board_name, kind, settings, settings.report_scan_stocks)
        if stocks:
            used = board_name
            for code, _name in stocks:
                collected += _reports_for_stock(code, settings)
            if collected:
                break

    if not collected and fund_target is not None and fund_target.kind == "fund":
        holdings = _fund_holdings(fund_target.code, settings, settings.report_scan_stocks)
        for code, _name in holdings:
            collected += _reports_for_stock(code, settings)
        if holdings and board is None:
            # 板块名没匹配上、用的是重仓股研报 → 用研报里出现最多的行业作为实际主题
            industries = [r.industry for r in collected if r.industry]
            if industries:
                used = Counter(industries).most_common(1)[0][0]

    if not collected:
        return [], used, f"未检索到与「{industry_name or '该主题'}」相关的公开券商研报"

    ranked = rank_reports(
        collected, f"{industry_name} {used}", top_k=top_k or settings.report_top_k
    )
    return ranked, used, None


def fetch_reports(target: Target, settings: Settings) -> tuple[list[ReportItem], str | None]:
    """抓取并排序研报。返回 (研报列表, 错误信息)。"""
    collected: list[ReportItem] = []

    if target.kind == "stock":
        collected += _reports_for_stock(target.code, settings)

    elif target.kind in ("industry", "concept"):
        stocks = _top_constituents(target.name, target.kind, settings, settings.report_scan_stocks)
        for code, _name in stocks:
            collected += _reports_for_stock(code, settings)

    else:  # fund
        holdings = _fund_holdings(target.code, settings, settings.report_scan_stocks)
        if holdings:
            for code, _name in holdings:
                collected += _reports_for_stock(code, settings)
        else:
            # 退化：用基金名称关键词匹配行业板块
            from money_agent.data.resolver import _load_industries  # 延迟导入避免循环依赖

            industries = _load_industries(settings)
            if not industries.empty:
                from money_agent.data.resolver import clean_query

                q = clean_query(target.name)
                subset = industries[industries["板块名称"].astype(str).apply(
                    lambda n: n in target.name or target.name in n or n[:2] in q
                )]
                for board in subset["板块名称"].astype(str).head(3):
                    stocks = _top_constituents(board, "industry", settings, settings.report_scan_stocks)
                    for code, _name in stocks:
                        collected += _reports_for_stock(code, settings)

    if not collected:
        return [], f"未检索到与「{target.name}」相关的公开券商研报"

    ranked = rank_reports(collected, target.name + " " + target.raw, top_k=settings.report_top_k)
    return ranked, None
