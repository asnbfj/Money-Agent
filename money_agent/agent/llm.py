"""OpenAI 兼容的 LLM 客户端（裸 HTTP，零额外依赖）。

只依赖 `requests`，兼容 DeepSeek / Qwen / vLLM / Ollama / OpenAI 等任何实现了
`POST {base_url}/chat/completions` 且支持 `tools`（function calling）的服务。
"""

from __future__ import annotations

import json
from typing import Any

import requests

from money_agent.config import Settings
from money_agent.exceptions import LLMError


class ChatLLM:
    """极简 Chat Completions 封装，支持 function calling。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return self.settings.llm_enabled

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """调用一次 chat completions，返回 assistant message 字典。

        message 结构：{"role": "assistant", "content": str | None, "tool_calls": [...]}
        """
        if not self.available:
            raise LLMError("未配置 OPENAI_API_KEY，LLM 不可用")

        url = f"{self.settings.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self.settings.llm_timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"LLM 请求失败：{exc}") from exc

        if resp.status_code >= 400:
            raise LLMError(f"LLM 返回错误 {resp.status_code}：{resp.text[:400]}")

        try:
            data = resp.json()
            return data["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM 响应解析失败：{resp.text[:400]}") from exc

    def chat_json(self, messages: list[dict[str, Any]], *, temperature: float | None = None) -> dict[str, Any]:
        """要求模型输出 JSON，并尽力解析（容忍 ```json 代码块包裹）。

        部分服务（如部分 Ollama/自建网关）不支持 `response_format`，遇到 400 会自动去掉重试。
        """
        try:
            message = self.chat(
                messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except LLMError:
            message = self.chat(messages, temperature=temperature)
        content = message.get("content") or ""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"LLM 未返回可解析的 JSON：{content[:300]}")
