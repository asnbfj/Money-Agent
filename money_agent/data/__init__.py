"""数据抓取工具层。

对外只暴露 `money_agent.data` 里的高层函数，内部统一处理重试与降级。
"""

from money_agent.data.retry import call_with_retry

__all__ = ["call_with_retry"]
