"""集中配置：LLM 接入、抓取参数、输出约束。

所有配置均可通过环境变量覆盖，方便在 Money-Agent 主系统里注入。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


@dataclass
class Settings:
    """运行期配置。"""

    # ---------------- LLM（OpenAI 兼容接口） ----------------
    # 适用于 DeepSeek / Qwen / vLLM / Ollama / OpenAI 等任何兼容 /chat/completions 的服务
    base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    )
    api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "deepseek-chat"))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.3")))
    llm_timeout: int = field(default_factory=lambda: int(_env("LLM_TIMEOUT", "120")))

    # ---------------- 抓取与重试 ----------------
    http_timeout: int = field(default_factory=lambda: int(_env("FETCH_TIMEOUT", "15")))
    retry_attempts: int = field(default_factory=lambda: int(_env("FETCH_RETRY", "3")))
    retry_delay: float = field(default_factory=lambda: float(_env("FETCH_RETRY_DELAY", "1.0")))

    # 抓取多少自然日的历史，用于覆盖近 30 个交易日（含节假日冗余）
    history_calendar_days: int = field(default_factory=lambda: int(_env("HISTORY_DAYS", "75")))

    # ---------------- 研报 RAG ----------------
    report_top_k: int = field(default_factory=lambda: int(_env("REPORT_TOP_K", "6")))
    # 板块/主题场景下，扫描多少只成分股的研报再聚合
    report_scan_stocks: int = field(default_factory=lambda: int(_env("REPORT_SCAN_STOCKS", "5")))

    # ---------------- Agent ----------------
    max_agent_turns: int = field(default_factory=lambda: int(_env("MAX_AGENT_TURNS", "10")))
    cache_ttl_hours: int = field(default_factory=lambda: int(_env("CACHE_TTL_HOURS", "12")))

    @property
    def llm_enabled(self) -> bool:
        """是否配置了 LLM（未配置时自动降级为确定性编排 + 规则化文案）。"""
        return bool(self.api_key)
