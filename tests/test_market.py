"""腾讯备用净值、交易日判断、分时汇总、估值口径切换的单元测试（离线可跑）。

运行：python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest import mock

from money_agent.data import market, quotes
from money_agent.config import Settings
from money_agent.models import Target

TENCENT_FUND = 'v_jj110020="110020~易方达沪深300ETF联接A~110020~1.8426~1.8521~1.8426~0.00~-0.51~2026-10-21 00:00:00~";'
SH_INDEX = 'v_sh000001="1~上证指数~000001~3301.25~3290.12~3295.00~' + "~".join(["0"] * 24) + '~20261022150003~12.3~0.37~";'


class TencentFundTest(unittest.TestCase):
    def test_parse_fields(self):
        self.assertEqual(
            market.parse_tencent_fund(TENCENT_FUND),
            {"name": "易方达沪深300ETF联接A", "nav": 1.8426, "change_pct": -0.51, "nav_date": "2026-10-21"},
        )

    def test_rejects_empty_or_short(self):
        self.assertIsNone(market.parse_tencent_fund('v_jj000000="";'))
        self.assertIsNone(market.parse_tencent_fund("pv_none_match=1;"))


class TradingDayTest(unittest.TestCase):
    def test_index_date_field(self):
        self.assertEqual(market.parse_index_date(SH_INDEX), "20261022")

    def test_weekend(self):
        self.assertFalse(market.decide_trading_day(datetime(2026, 10, 24, 10, 0), "20261023"))

    def test_holiday_detected_after_open(self):
        # 国庆假期周四：上证指数最近行情停在 9 月 30 日
        self.assertFalse(market.decide_trading_day(datetime(2026, 10, 1, 10, 0), "20260930"))

    def test_before_open_assumes_trading(self):
        self.assertTrue(market.decide_trading_day(datetime(2026, 10, 22, 9, 0), "20261021"))

    def test_index_updated_today(self):
        self.assertTrue(market.decide_trading_day(datetime(2026, 10, 22, 14, 0), "20261022"))

    def test_api_failure_falls_back_to_weekday(self):
        self.assertTrue(market.decide_trading_day(datetime(2026, 10, 22, 14, 0), None))

    def test_is_trading_day_caches_and_survives_errors(self):
        market._trading_cache.clear()
        boom = mock.Mock(side_effect=OSError("blocked"))
        self.assertTrue(market.is_trading_day(datetime(2026, 10, 22, 14, 0), fetch=boom))
        self.assertTrue(market.is_trading_day(datetime(2026, 10, 22, 14, 0), fetch=boom))
        self.assertEqual(boom.call_count, 1)
        market._trading_cache.clear()

    def test_market_status_phases(self):
        status = lambda h, m: market.market_status(datetime(2026, 10, 22, h, m), trading_day=True)["phase"]
        self.assertEqual([status(9, 0), status(10, 0), status(12, 0), status(14, 0), status(16, 0)],
                         ["pre", "open", "lunch", "open", "after"])
        self.assertEqual(market.market_status(datetime(2026, 10, 1, 10, 0), trading_day=False)["label"], "今日休市")


class IntradayTest(unittest.TestCase):
    def test_parse_getssgz(self):
        payload = {"code": 0, "data": {"yesterdayDwjz": "1.0000", "data": [["0930", 1.001, 0.001], ["931", 1.002, 0.002], ["bad", "x"]]}}
        parsed = market.parse_intraday(payload)
        self.assertEqual(parsed["yesterday_nav"], 1.0)
        self.assertEqual([p["time"] for p in parsed["points"]], ["09:30", "09:31"])

    def test_parse_error_payload(self):
        self.assertIsNone(market.parse_intraday({"code": -1}))

    def test_session_minutes_skip_lunch(self):
        self.assertEqual([market.session_minute(t) for t in ("09:30", "11:30", "13:00", "13:01", "15:00")], [0, 120, 120, 121, 240])

    def test_portfolio_weights_by_shares_and_forward_fills(self):
        series = [
            {"shares": 1000, "yesterday_nav": 1.0, "points": [{"time": "09:30", "value": 1.01}, {"time": "09:31", "value": 1.02}]},
            {"shares": 500, "yesterday_nav": 2.0, "points": [{"time": "09:30", "value": 1.98}]},
            {"shares": 100, "yesterday_nav": 10.0, "points": []},  # 没有分时，按昨日净值计入
        ]
        result = market.portfolio_intraday(series)
        self.assertEqual(result["base_value"], 3000.0)
        self.assertAlmostEqual(result["coverage"], 2000 / 3000, places=4)
        first, second = result["points"]
        self.assertEqual(first["value"], 1000 * 1.01 + 500 * 1.98 + 1000)
        # 09:31 第二只基金沿用 09:30 的估值
        self.assertEqual(second["value"], 1000 * 1.02 + 500 * 1.98 + 1000)
        self.assertAlmostEqual(second["pct"], (3010 / 3000 - 1) * 100, places=4)

    def test_portfolio_without_data(self):
        self.assertIsNone(market.portfolio_intraday([{"shares": 10, "yesterday_nav": 1.0, "points": []}]))


class FundQuoteWindowTest(unittest.TestCase):
    """估值口径：当天净值没公布前一直用当天估值；公布后换成正式净值和实际涨跌。"""

    def _quote(self, now, eastmoney, tencent=None):
        target = Target(raw="110020", name="易方达沪深300ETF联接A", code="110020", kind="fund", meta={"场内": False})
        settings = Settings()
        settings.retry_attempts = 1
        with mock.patch.object(quotes, "datetime") as dt, \
             mock.patch.object(quotes, "is_today", side_effect=lambda s, today=None: bool(s) and str(s)[:10] == now.date().isoformat()), \
             mock.patch.object(quotes, "is_trading_day", return_value=True), \
             mock.patch.object(quotes, "is_exchange_traded_fund", return_value=False), \
             mock.patch.object(quotes, "_mobile_fund_snapshot", return_value=eastmoney), \
             mock.patch.object(quotes, "fetch_tencent_fund", return_value=tencent) as tq:
            dt.now.return_value = now
            quote = quotes._fund_quote(target, settings, None)
        return quote, tq

    @staticmethod
    def _em(nav_date, gztime, navchg=-0.51):
        return {"Datas": [{"GSZ": "1.8600", "GSZZL": "0.95", "GZTIME": gztime, "NAV": "1.8426", "PDATE": nav_date, "NAVCHGRT": str(navchg)}]}

    def test_evening_before_nav_published_keeps_estimate(self):
        now = datetime(2026, 10, 22, 19, 0)
        quote, _ = self._quote(now, self._em("2026-10-21", "2026-10-22 15:00"))
        self.assertTrue(quote.is_intraday_estimate)
        self.assertEqual((quote.value, quote.change_pct), (1.86, 0.95))

    def test_tencent_newer_nav_switches_to_actual(self):
        now = datetime(2026, 10, 22, 21, 0)
        tencent = {"name": "x", "nav": 1.8601, "change_pct": 0.95, "nav_date": "2026-10-22"}
        quote, tq = self._quote(now, self._em("2026-10-21", "2026-10-22 15:00"), tencent)
        tq.assert_called_once()
        self.assertFalse(quote.is_intraday_estimate)
        self.assertEqual((quote.value, quote.official_nav_date, quote.change_pct), (1.8601, "2026-10-22", 0.95))

    def test_morning_uses_yesterday_nav(self):
        now = datetime(2026, 10, 22, 9, 10)
        quote, tq = self._quote(now, self._em("2026-10-21", "2026-10-21 15:00"))
        tq.assert_not_called()
        self.assertFalse(quote.is_intraday_estimate)
        self.assertEqual((quote.value, quote.change_pct), (1.8426, -0.51))

    def test_eastmoney_down_uses_tencent(self):
        now = datetime(2026, 10, 22, 10, 0)
        tencent = {"name": "x", "nav": 1.8426, "change_pct": -0.51, "nav_date": "2026-10-21"}
        quote, tq = self._quote(now, {"Datas": None}, tencent)
        tq.assert_called_once()
        self.assertEqual((quote.value, quote.official_nav_date), (1.8426, "2026-10-21"))
        self.assertIn("腾讯", quote.source)


if __name__ == "__main__":
    unittest.main()
