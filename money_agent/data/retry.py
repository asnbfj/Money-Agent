"""统一的带重试调用。

背景：东方财富的若干行情节点（push2his / 17.push2 等）在代理或高并发下会偶发
`ProxyError` / 连接中断。重试是数据层稳定性的关键。

约定：重试耗尽后抛 `DataUnavailable`，调用方负责降级并把“数据缺失”写进报告。
"""

from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

from money_agent.exceptions import DataUnavailable

T = TypeVar("T")

_URL_RE = re.compile(r"https?://[^\s'\")]+")


def sanitize_error(text: object, limit: int = 160) -> str:
    """把底层异常压成可读的一句话：去掉长 URL、折叠空白、截断。

    异常里常带几百字符的接口 URL 和代理堆栈，直接写进报告非常难看。
    """
    cleaned = _URL_RE.sub("<接口地址>", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "…"


def call_with_retry(
    fn: Callable[[], T],
    *,
    label: str,
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 1.6,
) -> T:
    """执行 `fn`，失败时按指数退避重试。

    :param label: 用于错误信息的可读标签，例如 "获取 000001 历史净值"
    :raises DataUnavailable: 全部尝试失败
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 数据源异常类型繁杂，需全部兜住
            last_error = exc
            if attempt < attempts:
                time.sleep(delay * (backoff ** (attempt - 1)))
    raise DataUnavailable(
        f"{label}失败（重试 {attempts} 次）：{sanitize_error(last_error)}"
    ) from last_error


def safe_call(
    fn: Callable[[], T],
    *,
    label: str,
    attempts: int = 3,
    delay: float = 1.0,
) -> tuple[T | None, str | None]:
    """`call_with_retry` 的“不抛异常”版本，返回 (结果, 错误信息)。

    适合数据缺失时仍需继续生成报告的链路。
    """
    try:
        return call_with_retry(fn, label=label, attempts=attempts, delay=delay), None
    except DataUnavailable as exc:
        return None, str(exc)
