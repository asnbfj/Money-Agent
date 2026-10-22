"""统一的异常类型。"""

from __future__ import annotations


class MoneyAgentError(Exception):
    """本包所有异常的基类。"""


class TargetNotFound(MoneyAgentError):
    """无法从用户输入解析出明确标的。"""


class DataUnavailable(MoneyAgentError):
    """行情/净值/研报等数据拉取失败或为空。

    这个异常代表“数据缺失”，报告里必须如实标注，禁止用推测数字替代。
    """


class LLMError(MoneyAgentError):
    """LLM 调用失败。"""
