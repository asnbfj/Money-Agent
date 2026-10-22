"""研报检索（轻量 RAG）。

数据源：`akshare.stock_research_report_em` —— 东方财富公开的券商研报元数据
（标题 / 机构 / 评级 / 行业 / 日期 / PDF 链接）。不解 PDF 正文，仅在元数据上做
「关键词 + 时效」检索排序，零额外依赖。

映射策略：
- 个股 → 直接取该股研报
- 行业/概念板块 → 取板块成分股，聚合其中若干只代表股的研报
- 场外公募基金 → 取其最新季报重仓股，聚合重仓股研报；
  拿不到持仓时退化为「基金名称关键词 → 行业板块」再聚合
"""

from money_agent.rag.reports import fetch_reports, rank_reports

__all__ = ["fetch_reports", "rank_reports"]
