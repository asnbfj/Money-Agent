"""命令行入口。

用法示例：
    python -m money_agent "半导体设备基金今天如何，我的持仓：份额1000份，成本净值1.42"
    python -m money_agent "比亚迪" --json
    python -m money_agent --no-llm "510300"     # 不依赖 LLM，规则化文案
    python -m money_agent                        # 交互模式
"""

from __future__ import annotations

import argparse
import json
import sys

from money_agent.agent import FinancialAgent
from money_agent.config import Settings
from money_agent.exceptions import MoneyAgentError
from money_agent.report import render


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="money-agent",
        description="Money-Agent：基于真实行情数据的客观金融分析 Agent（不构成投资建议）",
    )
    parser.add_argument("query", nargs="?", help="分析请求，例如“半导体设备基金今天如何，持仓1000份成本1.42”")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM，改用规则化文案")
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON（含原始数据）")
    parser.add_argument("--model", help="覆盖 OPENAI_MODEL")
    parser.add_argument("--base-url", help="覆盖 OPENAI_BASE_URL")
    return parser


def _run_one(agent: FinancialAgent, query: str, as_json: bool) -> int:
    try:
        report = agent.analyze(query)
    except MoneyAgentError as exc:
        print(f"[分析失败] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if as_json else render(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    settings = Settings()
    if args.model:
        settings.model = args.model
    if args.base_url:
        settings.base_url = args.base_url.rstrip("/")

    agent = FinancialAgent(settings, use_llm=not args.no_llm)
    if not agent.use_llm and not args.no_llm:
        print("[提示] 未检测到 OPENAI_API_KEY，已降级为规则化文案模式。\n", file=sys.stderr)

    if args.query:
        return _run_one(agent, args.query, args.json)

    # 交互模式
    print("Money-Agent 交互模式（输入 q 退出）")
    while True:
        try:
            query = input("\n请输入分析请求> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"q", "quit", "exit"}:
            return 0
        if not query:
            continue
        print()
        _run_one(agent, query, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
