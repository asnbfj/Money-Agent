"""Money-Agent：基于真实行情数据的客观金融分析 Agent。

核心原则（贯穿整个包）：
1. 所有结论必须来自工具拉取的真实数据，禁止编造行情/净值/K线；
   没有拿到数据就如实标注数据缺失。
2. 不预测未来涨跌、不输出确定性投资建议，只做客观解读 + 可选操作思路 + 免责声明。
3. 区分场外公募基金 / A股板块与股票；区分盘中估算净值与官方正式净值。
"""

from __future__ import annotations

import os as _os

# akshare 内部大量使用 tqdm 打印进度条，会污染 CLI 输出。
# 在导入 akshare（及其子模块）之前关掉，可用 MONEY_AGENT_TQDM=1 重新打开。
if _os.getenv("MONEY_AGENT_TQDM") != "1":
    _os.environ.setdefault("TQDM_DISABLE", "1")

from money_agent.config import Settings
from money_agent.models import (
    AnalysisReport,
    Analytics,
    Holding,
    History,
    HistoryPoint,
    Quote,
    ReportItem,
    Target,
)

__all__ = [
    "Settings",
    "Target",
    "Holding",
    "Quote",
    "History",
    "HistoryPoint",
    "Analytics",
    "ReportItem",
    "AnalysisReport",
]

__version__ = "0.1.0"
