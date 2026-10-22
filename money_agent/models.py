"""数据模型：整个 Agent 在各层之间传递的结构化对象。

统一用 dataclass + `to_dict()`，方便序列化给 LLM，也方便 Money-Agent 主系统消费。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TargetKind = Literal["fund", "stock", "industry", "concept"]


@dataclass
class Target:
    """解析后的标的。"""

    raw: str  # 用户原始输入片段
    name: str  # 标准名称
    code: str  # 基金代码 / 股票代码；板块无代码时用名称
    kind: TargetKind
    meta: dict[str, Any] = field(default_factory=dict)  # 例如基金类型、板块涨跌幅等

    @property
    def kind_label(self) -> str:
        return {
            "fund": "场外公募基金",
            "stock": "A股个股",
            "industry": "A股行业板块",
            "concept": "A股概念板块",
        }[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Holding:
    """用户持仓信息（全部可为空）。"""

    shares: float | None = None
    cost_nav: float | None = None
    buy_date: str | None = None
    raw: str | None = None

    @property
    def has_position(self) -> bool:
        return self.shares is not None or self.cost_nav is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Quote:
    """最新行情 / 净值快照。"""

    name: str = ""
    code: str = ""
    # 展示用“当前价格/净值”（基金优先用官方净值，盘中另有估算值）
    value: float | None = None
    change_pct: float | None = None
    data_time: str | None = None

    # 场外基金：盘中估算（⚠️ 仅参考，非正式成交价）
    estimate_value: float | None = None
    estimate_change_pct: float | None = None
    estimate_time: str | None = None

    # 场外基金：官方公布净值
    official_nav: float | None = None
    official_nav_date: str | None = None

    # 场内（ETF/LOF）市价
    market_price: float | None = None
    market_time: str | None = None

    is_intraday_estimate: bool = False
    source: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HistoryPoint:
    date: str  # ISO 格式 YYYY-MM-DD
    close: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class History:
    """标准化后的历史序列（统一为收盘价/单位净值）。"""

    points: list[HistoryPoint] = field(default_factory=list)
    source: str = ""
    value_name: str = "收盘价"  # 或 "单位净值"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "value_name": self.value_name,
            "error": self.error,
            "count": len(self.points),
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class Analytics:
    """基于历史序列计算的客观统计量。"""

    ret_7: float | None = None
    ret_15: float | None = None
    ret_30: float | None = None
    max_drawdown_30: float | None = None
    volatility_30_annualized: float | None = None
    trend_label: str = ""
    # 每个区间实际使用的起止日期，便于在报告里注明口径
    baseline: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportItem:
    """券商研报条目（轻量 RAG 的检索单元）。"""

    title: str
    institution: str = ""
    rating: str = ""
    industry: str = ""
    date: str = ""
    url: str = ""
    stock_name: str = ""
    stock_code: str = ""
    relevance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    """一次分析的完整结果。"""

    target: Target
    holding: Holding = field(default_factory=Holding)
    quote: Quote = field(default_factory=Quote)
    history: History = field(default_factory=History)
    analytics: Analytics = field(default_factory=Analytics)
    reports: list[ReportItem] = field(default_factory=list)
    # calc_user_position 工具的计算结果（账面收益率/盈亏），第 2 章直接引用
    position: dict[str, Any] = field(default_factory=dict)
    # 研报实际对应的行业/主题名
    industry: str = ""
    # LLM 生成的定性章节：走势分析 / 基本面 / 可选思路 / 板块关注
    narrative: dict[str, str] = field(default_factory=dict)
    # 数据缺失与局限说明（必须如实写入报告）
    data_gaps: list[str] = field(default_factory=list)
    generated_at: str = ""
    engine: str = ""  # "llm" 或 "rule-based"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "holding": self.holding.to_dict(),
            "quote": self.quote.to_dict(),
            "history": self.history.to_dict(),
            "analytics": self.analytics.to_dict(),
            "reports": [r.to_dict() for r in self.reports],
            "position": self.position,
            "industry": self.industry,
            "narrative": self.narrative,
            "data_gaps": self.data_gaps,
            "generated_at": self.generated_at,
            "engine": self.engine,
        }

    @property
    def mark_value(self) -> float | None:
        """账面收益使用的“当前价值”口径。

        场内基金用市价，场外基金用官方净值，个股/板块用行情最新价。
        统一在这里定义，保证报告与 calc_user_position 口径一致。
        """
        quote = self.quote
        if self.target.kind == "fund":
            if self.target.meta.get("场内"):
                return quote.market_price if quote.market_price is not None else quote.value
            return quote.official_nav if quote.official_nav is not None else quote.value
        return quote.value
