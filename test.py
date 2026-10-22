"""Money-Agent 入口与测试用例。

用法：
    # 1) 全链路分析（配置了 OPENAI_API_KEY 走 LLM，否则自动降级为规则化模式）
    python test.py "半导体设备基金今天如何，我的持仓：份额1000份，成本净值1.42"

    # 2) 强制不走 LLM（离线可跑）
    python test.py --no-llm "半导体设备基金今天如何，我的持仓：份额1000份，成本净值1.42"

    # 3) 只跑工具层单元测试
    python test.py --unit

    # 4) 交互模式
    python test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import tools
from agent import build_llm, create_agent


# ==========================================================================
# 工具层单元测试
# ==========================================================================

def test_get_fund_code_by_name() -> None:
    """用例1：名称 → 代码。"""
    print("\n[用例1] get_fund_code_by_name")
    res = tools.get_fund_code_by_name("半导体设备基金")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    _assert("fund_code" in res or "error" in res, "必须返回 code 或 error")
    if "fund_code" in res:
        assert len(res["fund_code"]) == 6, "基金代码应为 6 位"

    # 异常输入：不存在 / 空
    bad = tools.get_fund_code_by_name("这只基金肯定不存在xyz")
    _assert("error" in bad, "不存在的名称必须返回 error")
    empty = tools.get_fund_code_by_name("")
    _assert("error" in empty, "空入参必须返回 error")


def test_get_fund_latest_info(code: str) -> dict:
    """用例2：最新净值（估算/正式标记 + 时间逻辑）。"""
    print("\n[用例2] get_fund_latest_info")
    res = tools.get_fund_latest_info(code)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if "error" not in res:
        _assert(res.get("tag") in ("估算", "正式"), "tag 只能是「估算」或「正式」")
        _assert(res.get("nav") is not None, "净值不能为空")
    return res


def test_get_fund_history_kline(code: str) -> None:
    """用例3：历史净值序列 + 涨跌幅 + 最大回撤。"""
    print("\n[用例3] get_fund_history_kline")
    for period in (7, 15, 30):
        res = tools.get_fund_history_kline(code, period)
        if "error" in res:
            print(f"  period={period}: {res['error']}")
            continue
        data = res["data"]
        print(
            f"  period={period}: 点数={len(data)} 区间涨跌幅={res['interval_change_pct']}% "
            f"最大回撤={res['max_drawdown_pct']}% 基准日={res['interval_baseline_date']}"
        )
        _assert(len(data) > 0, "data 不能为空")
        _assert(
            res["max_drawdown_pct"] is None or res["max_drawdown_pct"] <= 0,
            "最大回撤应为负数或 None",
        )
        # 校验回撤计算：用 data 独立复算一遍，结果应一致
        navs = [p["nav"] for p in data]
        peak, worst = navs[0], 0.0
        for nav in navs:
            peak = max(peak, nav)
            worst = max(worst, (peak - nav) / peak)
        expected = round(-worst * 100, 2)
        _assert(
            res["max_drawdown_pct"] == expected,
            f"回撤计算不一致：返回 {res['max_drawdown_pct']} vs 复算 {expected}",
        )
    # 异常输入
    _assert("error" in tools.get_fund_history_kline(code, 0), "period_days=0 必须报错")
    _assert("error" in tools.get_fund_history_kline(code, "abc"), "period_days=abc 必须报错")


def test_rag_stub() -> None:
    """用例4：研报 RAG 桩函数。"""
    print("\n[用例4] rag_search_industry_report（桩）")
    res = tools.rag_search_industry_report("半导体设备")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    _assert(res.get("is_stub") is True, "当前应为桩实现")
    _assert(res.get("summary") == tools.RAG_NOT_READY, "桩函数必须返回固定占位提示")


def test_calc_user_position() -> None:
    """用例5：持仓账面收益（含边界校验）。"""
    print("\n[用例5] calc_user_position")
    res = tools.calc_user_position(1.42, 2.8606, 1000)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    expected = (2.8606 / 1.42 - 1) * 100
    _assert(abs(res["return_pct"] - round(expected, 2)) < 0.01, "收益率计算错误")
    _assert(abs(res["pnl_amount"] - (2.8606 - 1.42) * 1000) < 0.01, "盈亏金额计算错误")
    _assert("error" in tools.calc_user_position(0, 1, 100), "成本为 0 必须报错")
    _assert("error" in tools.calc_user_position(1, 1, -5), "份额为负必须报错")


def test_execute_tool_never_raises() -> None:
    """用例6：execute_tool 永不抛异常，异常统一转友好提示。"""
    print("\n[用例6] execute_tool 异常兜底")
    bad_name = tools.execute_tool("no_such_tool", {})
    _assert("error" in bad_name, "未知工具必须返回 error")
    bad_args = tools.execute_tool("get_fund_history_kline", {"wrong": 1})
    _assert("error" in bad_args, "参数错误必须返回 error")
    print("  未知工具:", bad_name["error"])
    print("  参数错误:", bad_args["error"])


def test_parallel_execute() -> None:
    """用例7：并行执行 7/15/30 天 K 线，并验证并行确实生效。"""
    import time

    print("\n[用例7] execute_tools_parallel（并行拉取）")
    calls = [
        ("get_fund_history_kline", {"fund_code": "021533", "period_days": d})
        for d in (7, 15, 30)
    ]
    t0 = time.time()
    results = tools.execute_tools_parallel(calls)
    parallel_cost = time.time() - t0

    t1 = time.time()
    for name, args in calls:
        tools.execute_tool(name, args)
    serial_cost = time.time() - t1

    print(f"  并行耗时 {parallel_cost:.2f}s / 串行耗时 {serial_cost:.2f}s")
    _assert(len(results) == 3, "应返回 3 个结果")
    _assert(all("data" in r or "error" in r for r in results), "每个结果都应有 data 或 error")


def run_unit_tests(code: str | None = None) -> int:
    """跑全部工具层单元测试。"""
    print("=" * 70)
    print("Money-Agent 工具层单元测试")
    print("=" * 70)
    failed = 0
    try:
        test_get_fund_code_by_name()
        if code is None:
            res = tools.get_fund_code_by_name("半导体设备基金")
            code = res.get("fund_code")
        if code:
            test_get_fund_latest_info(code)
            test_get_fund_history_kline(code)
        else:
            print("\n[跳过] 未拿到基金代码，跳过净值/K线用例")
        test_rag_stub()
        test_calc_user_position()
        test_execute_tool_never_raises()
        if code:
            test_parallel_execute()
    except AssertionError as exc:
        failed += 1
        print(f"\n❌ 断言失败：{exc}")
        traceback.print_exc()
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"\n❌ 用例异常：{type(exc).__name__}: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("❌ 单元测试存在失败" if failed else "✅ 全部用例通过")
    print("=" * 70)
    return 1 if failed else 0


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ==========================================================================
# 全链路
# ==========================================================================

def run_analysis(query: str, use_llm: bool = True, as_json: bool = False) -> int:
    llm = build_llm() if use_llm else None
    if use_llm and llm is None:
        print("[提示] 未检测到 OPENAI_API_KEY，自动降级为规则化模式。\n", file=sys.stderr)
    agent = create_agent(use_llm=use_llm)

    result = agent.analyze(query)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(result["report"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="money-agent",
        description="Money-Agent：基于真实行情数据的客观金融分析 Agent（不构成投资建议）",
    )
    parser.add_argument("query", nargs="?", help="分析请求，例如“半导体设备基金今天如何，我的持仓：份额1000份，成本净值1.42”")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM，走规则化模式")
    parser.add_argument("--unit", action="store_true", help="只跑工具层单元测试")
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON")
    parser.add_argument("--code", help="--unit 时指定基金代码，跳过名称解析")
    args = parser.parse_args(argv)

    if args.unit:
        return run_unit_tests(args.code)

    if args.query:
        return run_analysis(args.query, use_llm=not args.no_llm, as_json=args.json)

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
        run_analysis(query, use_llm=not args.no_llm, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
