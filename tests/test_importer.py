"""持仓截图解析与名称匹配的单元测试（离线可跑）。

运行：python3 -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from money_agent.importer import (
    FundIndex,
    Observation,
    codes_in_text,
    derive_position,
    observations_from_payload,
    parse_holdings,
)


def _obs(text, x, y, w=None, h=0.018):
    return Observation(text=text, x=x, y=y, w=w if w is not None else 0.012 * len(text), h=h)


def alipay_screen() -> list[Observation]:
    """模拟支付宝「全部持有」页的 Vision OCR 输出（坐标原点在左上角）。"""
    return [
        _obs("13:36", 0.05, 0.010), _obs("5G", 0.80, 0.010), _obs("50", 0.88, 0.010),
        _obs("全部持有", 0.05, 0.080), _obs("收益明细", 0.35, 0.080), _obs("交易记录", 0.65, 0.080),
        _obs("全部", 0.05, 0.130), _obs("持有收益排序~", 0.70, 0.130),
        _obs("名称/金额", 0.05, 0.180), _obs("日收益", 0.47, 0.180), _obs("持有收益/率", 0.74, 0.180),
        # 卡片 1：普通情况 + 金选标签
        _obs("易方达沪深300ETF联接A", 0.05, 0.230), _obs("金选", 0.62, 0.231),
        _obs("82,460.20", 0.05, 0.265), _obs("-346.20", 0.47, 0.265), _obs("+6,320.40", 0.76, 0.265),
        _obs("+8.30%", 0.80, 0.292),
        # 卡片 2：名称折成两行
        _obs("富国中证人工智能主题ETF", 0.05, 0.340), _obs("联接C", 0.05, 0.362),
        _obs("32,518.00", 0.05, 0.395), _obs("+405.12", 0.47, 0.395), _obs("+5,202.00", 0.76, 0.395),
        _obs("+19.04%", 0.79, 0.422),
        # 卡片 3：名称被截断；金额和日收益挤在同一个文字块
        _obs("华夏国证半导体芯片ETF联…", 0.05, 0.470),
        _obs("56,820.10   +711.20", 0.05, 0.505, w=0.48), _obs("+8,716.80", 0.76, 0.505),
        _obs("+18.13%", 0.79, 0.532),
        # 卡片 4：名称和数字之间有交易状态
        _obs("广发中证消费ETF联接C", 0.05, 0.580), _obs("1笔交易进行中", 0.05, 0.605),
        _obs("48,260.22", 0.05, 0.635), _obs("+173.40", 0.47, 0.635), _obs("−1,140.32", 0.76, 0.635),
        _obs("−2.31%", 0.80, 0.662),
        # 卡片 5：只有金额和持有收益（日收益未更新）
        _obs("南方中证全债指数A", 0.05, 0.710),
        _obs("40,372.00", 0.05, 0.745), _obs("+1,182.00", 0.76, 0.745),
        _obs("+3.02%", 0.80, 0.772),
        # 底部导航
        _obs("首页", 0.08, 0.960), _obs("理财", 0.30, 0.960), _obs("我的", 0.85, 0.960),
    ]


FUNDS = [
    ("110020", "易方达沪深300ETF联接A", "指数型-股票"),
    ("007339", "易方达沪深300ETF联接C", "指数型-股票"),
    ("012900", "富国中证人工智能主题ETF联接C", "指数型-股票"),
    ("012899", "富国中证人工智能主题ETF联接A", "指数型-股票"),
    ("008887", "华夏国证半导体芯片ETF联接A", "指数型-股票"),
    ("008888", "华夏国证半导体芯片ETF联接C", "指数型-股票"),
    ("009329", "广发中证消费ETF联接C", "指数型-股票"),
    ("202101", "南方中证全债指数A", "指数型-固收"),
    ("000001", "华夏成长混合", "混合型-偏股"),
]


class ParseHoldingsTest(unittest.TestCase):
    def setUp(self):
        self.rows = {h.ocr_name: h for h in parse_holdings(alipay_screen())}

    def test_finds_every_card_and_skips_ui_text(self):
        self.assertEqual(
            list(self.rows),
            [
                "易方达沪深300ETF联接A",
                "富国中证人工智能主题ETF联接C",
                "华夏国证半导体芯片ETF联...",
                "广发中证消费ETF联接C",
                "南方中证全债指数A",
            ],
        )

    def test_reads_amount_daily_and_holding_columns(self):
        row = self.rows["易方达沪深300ETF联接A"]
        self.assertEqual((row.amount, row.daily_pnl, row.holding_pnl, row.holding_rate), (82460.20, -346.20, 6320.40, 8.30))

    def test_splits_numbers_merged_into_one_block(self):
        row = self.rows["华夏国证半导体芯片ETF联..."]
        self.assertEqual((row.amount, row.daily_pnl, row.holding_pnl), (56820.10, 711.20, 8716.80))

    def test_unicode_minus_and_status_line(self):
        row = self.rows["广发中证消费ETF联接C"]
        self.assertEqual((row.holding_pnl, row.holding_rate), (-1140.32, -2.31))

    def test_two_numbers_use_header_columns(self):
        row = self.rows["南方中证全债指数A"]
        self.assertEqual((row.amount, row.daily_pnl, row.holding_pnl), (40372.00, None, 1182.00))

    def test_plain_text_ocr_without_coordinates(self):
        text = "\n".join([
            "13:361", ": ! ! ! 5G", "50", "全部持有", "收益明细", "交易记录", "全部", "持有收益排序~",
            "名称/金额", "日收益", "持有收益/率",
            "易方达沪深300ETF联接A", "82,460.20", "-346.20", "+6,320.40", "+8.30%",
            "南方中证全债指数A", "40,372.00", "0.00", "+1,182.00", "+3.02%",
        ])
        rows = parse_holdings(observations_from_payload(text))
        self.assertEqual([r.ocr_name for r in rows], ["易方达沪深300ETF联接A", "南方中证全债指数A"])
        self.assertEqual((rows[0].amount, rows[0].daily_pnl, rows[0].holding_pnl, rows[0].holding_rate), (82460.20, -346.20, 6320.40, 8.30))

    def test_json_payload_round_trip(self):
        payload = {"observations": [{"text": o.text, "x": o.x, "y": o.y, "w": o.w, "h": o.h} for o in alipay_screen()]}
        self.assertEqual(len(parse_holdings(observations_from_payload(payload))), 5)

    def test_codes_in_text(self):
        self.assertEqual(codes_in_text("110020 易方达\n代码:008887 110020"), ["110020", "008887"])


class FundIndexTest(unittest.TestCase):
    def setUp(self):
        self.index = FundIndex(FUNDS)

    def test_exact_match_prefers_same_share_class(self):
        best = self.index.match("易方达沪深300ETF联接A")[0]
        self.assertEqual((best["fund_code"], best["score"]), ("110020", 1.0))

    def test_full_width_and_spaces(self):
        self.assertEqual(self.index.match("易方达沪深 300ＥＴＦ联接Ａ")[0]["fund_code"], "110020")

    def test_truncated_name_lists_share_classes(self):
        codes = [m["fund_code"] for m in self.index.match("华夏国证半导体芯片ETF联…")]
        self.assertEqual(set(codes[:2]), {"008887", "008888"})

    def test_ocr_typo_still_matches(self):
        self.assertEqual(self.index.match("广发中证消费ETE联接C")[0]["fund_code"], "009329")

    def test_unrelated_name_returns_nothing(self):
        self.assertEqual(self.index.match("持有收益排序"), [])

    def test_lookup_code(self):
        self.assertEqual(self.index.lookup_code("202101")["fund_name"], "南方中证全债指数A")


class DerivePositionTest(unittest.TestCase):
    def test_shares_and_cost_from_amount_and_profit(self):
        result = derive_position(82460.20, 6320.40, 1.8426)
        self.assertAlmostEqual(result["shares"], 44752.09, places=2)
        self.assertAlmostEqual(result["cost_nav"], 1.7014, places=4)

    def test_cost_from_rate_when_profit_missing(self):
        result = derive_position(10800.0, None, 1.08, holding_rate=8.0)
        self.assertEqual(result, {"shares": 10000.0, "cost_nav": 1.0})

    def test_missing_nav(self):
        self.assertEqual(derive_position(1000.0, 10.0, None), {"shares": None, "cost_nav": None})


if __name__ == "__main__":
    unittest.main()
