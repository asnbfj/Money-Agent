"""自建盘中估值的单元测试（离线可跑，不依赖网络）。

运行：python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from money_agent.data import estimate


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["股票代码", "股票名称", "占净值比例"])


class TencentSymbolTest(unittest.TestCase):
    def test_a_share_and_hk(self):
        self.assertEqual(estimate.tencent_symbol("600519"), "sh600519")
        self.assertEqual(estimate.tencent_symbol("688012"), "sh688012")
        self.assertEqual(estimate.tencent_symbol("000001"), "sz000001")
        self.assertEqual(estimate.tencent_symbol("300502"), "sz300502")
        self.assertEqual(estimate.tencent_symbol("512760"), "sh512760")
        self.assertEqual(estimate.tencent_symbol("01347"), "hk01347")

    def test_unsupported(self):
        for code in ("SNDK", "MU", "", "abc", "1234567"):
            self.assertIsNone(estimate.tencent_symbol(code), code)

    def test_numeric_foreign_code_collides_on_format(self):
        # 韩股 SK海力士 000660 与深市代码同形，只能靠名称校验兜住（见 _names_compatible）
        self.assertEqual(estimate.tencent_symbol("000660"), "sz000660")


class ParseQuotesTest(unittest.TestCase):
    def test_computes_change_from_price_and_prev_close(self):
        text = (
            'v_sh600519="1~贵州茅台~600519~1320.00~1300.00~1301.00~1~2~3";\n'
            'v_hk01347="100~华虹宏力~01347~66.55~73.50~73.50~0~0~0";\n'
            'v_sz000001="51~平安银行~000001~0.00~11.41~11.41~0~0~0";\n'
        )
        parsed = estimate._parse_quotes(text)
        self.assertAlmostEqual(parsed["600519"]["change_pct"], 1.54, places=2)
        self.assertAlmostEqual(parsed["01347"]["change_pct"], -9.46, places=2)
        self.assertNotIn("000001", parsed, "价格为 0（停牌）应被跳过")

    def test_empty_payload(self):
        self.assertEqual(estimate._parse_quotes(""), {})
        self.assertEqual(estimate._parse_quotes('v_pv_none_match="1";'), {})


class EstimateChangeTest(unittest.TestCase):
    def test_holdings_weighted_and_renormalised(self):
        frame = _frame([("600519", "贵州茅台", 60.0), ("000001", "平安银行", 20.0), ("SNDK", "闪迪", 20.0)])
        quotes = {"600519": {"change_pct": 1.0}, "000001": {"change_pct": -3.0}}
        with mock.patch.object(estimate, "holdings", return_value=frame), mock.patch.object(
            estimate, "quotes", return_value=quotes
        ):
            result = estimate.estimate_change("000001")
        self.assertNotIn("error", result)
        # 只有 80% 权重能定价：(60*1.0 + 20*(-3.0)) / 80 = 0.0
        self.assertAlmostEqual(result["change_pct"], 0.0, places=2)
        self.assertAlmostEqual(result["coverage"], 0.8, places=4)
        self.assertEqual(result["method"], "holdings")

    def test_foreign_holdings_refused(self):
        frame = _frame([("SNDK", "闪迪", 50.0), ("MU", "美光科技", 31.0)])
        with mock.patch.object(estimate, "holdings", return_value=frame), mock.patch.object(
            estimate, "quotes", return_value={}
        ):
            result = estimate.estimate_change("008254")
        self.assertIn("error", result)
        self.assertIn("境外", result["error"])

    def test_proxy_branch(self):
        frame = _frame([("688012", "中微公司", 0.3)])
        text = 'v_sh512760="1~芯片ETF~512760~0.90~1.00~1.00~0~0~0";'
        with mock.patch.object(estimate, "holdings", return_value=frame), mock.patch.object(
            estimate, "quotes", return_value={}
        ), mock.patch.object(estimate.requests, "get", return_value=_Resp(text)):
            result = estimate.estimate_change("020465")
        self.assertNotIn("error", result)
        self.assertEqual(result["method"], "proxy")
        # 代理跌 10%，beta=0.965 → -9.65%
        self.assertAlmostEqual(result["change_pct"], -9.65, places=1)


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.encoding = None


class NameGuardTest(unittest.TestCase):
    def test_rejects_code_collision(self):
        self.assertFalse(estimate._names_compatible("SK海力士", "*ST南华"))

    def test_accepts_same_company(self):
        self.assertTrue(estimate._names_compatible("贵州茅台", "贵州茅台"))
        self.assertTrue(estimate._names_compatible("ST南华", "*ST南华"))
        self.assertTrue(estimate._names_compatible("中芯国际", "中芯国际"))

    def test_letter_tickers_never_reach_guard(self):
        # 美股字母代码在 tencent_symbol 阶段就被丢弃，不会走到名称校验
        self.assertIsNone(estimate.tencent_symbol("TSM"))
        self.assertTrue(estimate._names_compatible("TSM", "某某科技"))

    def test_missing_name_not_blocked(self):
        self.assertTrue(estimate._names_compatible("", "某某"))
        self.assertTrue(estimate._names_compatible("某某", ""))


class EstimateForTest(unittest.TestCase):
    def test_not_trading_day(self):
        self.assertIsNone(estimate.estimate_for("017811", 2.0, "2026-10-21", trading_day=False))

    def test_official_nav_already_today(self):
        import datetime

        today = datetime.date.today().isoformat()
        self.assertIsNone(estimate.estimate_for("017811", 2.0, today, trading_day=True))

    def test_stale_base_refused(self):
        result = estimate.estimate_for("017811", 2.0, "2020-01-01", trading_day=True)
        self.assertIn("error", result)

    def test_nav_scaled_by_change(self):
        with mock.patch.object(estimate, "estimate_change", return_value={"change_pct": 2.0, "method": "holdings"}):
            result = estimate.estimate_for("017811", 2.0, "2026-10-21", trading_day=True)
        self.assertAlmostEqual(result["estimate_nav"], 2.04, places=4)
        self.assertEqual(result["base_nav"], 2.0)


if __name__ == "__main__":
    unittest.main()
