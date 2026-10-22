"""
获取单只基金的实时估值（东方财富手机版 API）——【已废弃，仅作接口探测参考】。

⚠️ 现状（2026-10-12 盘中实测确认）：东方财富已整体下线基金盘中估值。
   本脚本依赖的 GSZ / GSZZL / GZTIME 三个字段在所有基金上均返回 None，
   因此它拿不到当日盘中估值，只能拿到官方正式净值（NAV / PDATE）。
   行情推送本身仍实时正常（HQDATE / NEWPRICE 有值），只是估值字段被移除。

   已下线/失效的来源（实测）：
     - fundgz.1234567.com.cn/js/{code}.js             → 404 页面
     - fund.eastmoney.com/fundguzhi.html              → 404 页面
     - api.fund.eastmoney.com/FundGuZhi/GetFundGZList → ErrCode 4 / 404
     - 移动端 FundMNFInfo 的 GSZ / GSZZL / GZTIME      → 返回 None
     - 基金详情页 id="gz_gsz"                          → 恒为 "--"

   ❗ 请改用 tools.py 的 get_fund_latest_info()：它在估值缺失时会自动回落
      到官方正式净值，并在报告中明确说明原因。本脚本保留仅用于排查接口变化。

------------------------------------------------------------
以下为历史说明（部分前提已不成立，见上）：

为什么不用 ak.fund_value_estimation_em：
    1. akshare 的 symbol="全部"/"指数型" 走的是 lof_fundguzhiN.html 静态页，
       返回的其实是 LOF/指数型 的一批基金（约 788 只），并不包含像 000001
       这种普通混合型基金，所以过滤后得到空表。
    2. 该接口的数据源在非交易时段（周末、节假日、收盘后）本身就只保留
       上一个交易日的估值，拿不到“今天”的数据，这是数据源的限制，不是 bug。

东方财富手机版接口 FundMNFInfo 可以直接按基金代码取到：
    GSZ      实时估值          ← 已下线，恒为 None
    GSZZL    实时估算涨跌幅     ← 已下线，恒为 None
    GZTIME   估值的日期/时间    ← 已下线，恒为 None
    PDATE    最新公布净值的日期  ← 仍有效
    NAV      最新单位净值        ← 仍有效
"""

from __future__ import annotations

from datetime import datetime, time

import requests

MOBILE_API = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNFInfo"
# 注意：必须使用手机端 UA。桌面 Chrome 的 UA 会被该接口拒绝，返回 Datas=null。
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    ),
}


def is_a_share_trading_time(now: datetime | None = None) -> bool:
    """粗略判断当前是否处于 A 股（及基金估值）交易时段。

    未考虑法定节假日，仅用于解释“为什么今天没数据”。
    """
    now = now or datetime.now()
    if now.weekday() >= 5:  # 周六/周日
        return False
    return time(9, 25) <= now.time() <= time(15, 5)


def get_fund_realtime_estimate(code: str) -> dict:
    """获取单只基金的实时估值信息。

    :param code: 6 位基金代码，例如 "000001"
    :return: 估值信息字典；取不到数据时返回空字典
    """
    params = {
        "Fcodes": code,
        "pageIndex": 1,
        "pageSize": 200,
        "plat": "Android",
        "appType": "ttjj",
        "product": "EFund",
        "Version": "1",
        "deviceid": "money-agent",
    }
    resp = requests.get(MOBILE_API, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    datas = payload.get("Datas") or []
    if not datas:
        return {}

    d = datas[0]
    expansion = payload.get("Expansion") or {}
    return {
        "基金代码": d.get("FCODE"),
        "基金名称": d.get("SHORTNAME"),
        "估值": d.get("GSZ"),  # 实时估值（非交易时段为 None）
        "估算增长率": d.get("GSZZL"),  # 单位：%
        "估值日期": d.get("GZTIME") or expansion.get("GZTIME"),
        "净值日期": d.get("PDATE"),
        "单位净值": d.get("NAV"),
        "场内价格": d.get("NEWPRICE"),  # 仅 ETF/LOF 场内基金有
        "行情时间": d.get("HQDATE"),
    }


if __name__ == "__main__":
    for fund_code in ("000001", "161725", "510300"):
        info = get_fund_realtime_estimate(fund_code)
        print(f"===== {fund_code} =====")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()

    print("当前是否交易时段(粗略判断):", is_a_share_trading_time())
