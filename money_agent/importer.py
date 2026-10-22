"""持仓截图解析：把 OCR 结果还原成「基金名称 + 金额 + 收益」，再匹配基金代码。

支付宝「全部持有」页不显示基金代码，每只基金是一张卡片：

    易方达沪深300ETF联接A                      ← 名称（可能折成两行、被省略号截断）
    82,460.20        -346.20       +6,320.40   ← 持有金额 / 日收益 / 持有收益
                                     +8.30%    ← 持有收益率

做法与养基宝等记账工具一致：按坐标把文字还原成行，识别出卡片，
用基金名称表匹配代码，再用「金额 ÷ 最新正式净值」推算份额、
「(金额 − 持有收益) ÷ 份额」推算成本净值，最后交给用户确认。
本模块只做纯计算，不访问网络，方便单测。
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

_NUMBER_RE = re.compile(r"^[+\-−]?¥?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^[+\-−]?¥?\d+(?:\.\d+)?$")
_PERCENT_RE = re.compile(r"^[+\-−]?\d+(?:\.\d+)?%$")
_TOKEN_RE = re.compile(r"[+\-−]?¥?[\d,]+(?:\.\d+)?%?")
_CJK_RE = re.compile(r"[一-鿿]")
_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")

# 支付宝页面上的界面文字与状态标签，不是基金名称
_UI_LABELS = {
    "全部持有", "收益明细", "交易记录", "全部", "持有收益排序", "名称/金额", "名称", "金额",
    "日收益", "持有收益/率", "持有收益", "持有收益率", "排序", "基金", "我的持有", "我的基金",
    "总金额", "持有资产", "昨日收益", "累计收益", "资产", "金选", "定投", "自选", "理财", "首页",
    "市场", "机会", "我的", "更多", "偏股", "偏债", "指数", "黄金", "货币", "筛选", "搜索",
    "资产分析", "收益分析", "基金经理", "持仓分析", "查看全部", "定投计划", "全部基金",
}
_UI_PATTERNS = (
    re.compile(r"(买入|卖出|转换|定投|赎回|申购|交易)(中|确认中|进行中)"),
    re.compile(r"\d+\s*笔"),
    re.compile(r"(待确认|已更新|更新中|未更新|收益更新|净值更新|今日收益)"),
    re.compile(r"排序"),
    re.compile(r"^(持有|名称|金额|日收益|收益)[/／]"),
    re.compile(r"^[\(（]?元[\)）]?$"),
)


@dataclass
class Observation:
    """一段 OCR 文字及其位置（坐标为 0–1，原点在左上角）。"""

    text: str
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.02

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class ParsedHolding:
    """从截图里读出的一只基金。"""

    ocr_name: str
    amount: float | None = None
    daily_pnl: float | None = None
    holding_pnl: float | None = None
    holding_rate: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ocr_name": self.ocr_name,
            "amount": self.amount,
            "daily_pnl": self.daily_pnl,
            "holding_pnl": self.holding_pnl,
            "holding_rate": self.holding_rate,
        }


# --------------------------------------------------------------------------
# OCR 结果 → 行
# --------------------------------------------------------------------------
def observations_from_payload(payload: object) -> list[Observation]:
    """兼容新版（带坐标的 JSON）与旧版（纯文本行）OCR 输出。"""
    if isinstance(payload, dict):
        payload = payload.get("observations", [])
    result: list[Observation] = []
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                result.append(
                    Observation(
                        text=str(item["text"]).strip(),
                        x=float(item.get("x") or 0),
                        y=float(item.get("y") if item.get("y") is not None else index * 0.03),
                        w=float(item.get("w") or 0),
                        h=float(item.get("h") or 0.02),
                    )
                )
    elif isinstance(payload, str):
        # 旧版纯文本：每行当作独立的一行，按出现顺序排列
        for index, line in enumerate(payload.splitlines()):
            if line.strip():
                result.append(Observation(text=line.strip(), y=index * 0.03, h=0.02))
    return result


def _split_numeric(obs: Observation) -> list[Observation]:
    """一个文字块里挤了多个数字时（如「82,460.20 -346.20」），按字符位置拆开。"""
    tokens = list(_TOKEN_RE.finditer(obs.text))
    rest = _TOKEN_RE.sub("", obs.text).replace(" ", "").replace("¥", "")
    if len(tokens) < 2 or rest:
        return [obs]
    length = max(len(obs.text), 1)
    pieces = []
    for match in tokens:
        start, end = match.span()
        pieces.append(
            Observation(
                text=match.group(0),
                x=obs.x + obs.w * start / length,
                y=obs.y,
                w=obs.w * (end - start) / length,
                h=obs.h,
            )
        )
    return pieces


def group_rows(observations: list[Observation]) -> list[list[Observation]]:
    """按纵坐标把文字块聚成视觉上的一行，行内按横坐标排序。"""
    items: list[Observation] = []
    for obs in observations:
        items.extend(_split_numeric(obs))
    items.sort(key=lambda o: (o.cy, o.x))
    rows: list[list[Observation]] = []
    for obs in items:
        if rows:
            last = rows[-1]
            ref = sum(o.cy for o in last) / len(last)
            tolerance = max(min(o.h for o in last + [obs]) * 0.6, 0.004)
            if abs(obs.cy - ref) <= tolerance:
                last.append(obs)
                continue
        rows.append([obs])
    for row in rows:
        row.sort(key=lambda o: o.x)
    return rows


# --------------------------------------------------------------------------
# 行 → 持仓
# --------------------------------------------------------------------------
def parse_number(text: str) -> float | None:
    cleaned = text.strip().replace("−", "-").replace("¥", "").replace(",", "").replace("，", "")
    cleaned = cleaned.rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_number(text: str) -> bool:
    return bool(_NUMBER_RE.match(text.strip().replace("，", ",")))


def _is_percent(text: str) -> bool:
    return bool(_PERCENT_RE.match(text.strip().replace("−", "-")))


def is_ui_label(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if compact in _UI_LABELS:
        return True
    return any(p.search(compact) for p in _UI_PATTERNS)


def _is_name_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(_CJK_RE.findall(compact)) < 3 or is_ui_label(compact):
        return False
    # 纯时间、电量等状态栏文字
    return not re.fullmatch(r"[\d:：.%\sA-Za-z!！]+", compact)


def _is_name_continuation(text: str) -> bool:
    """长名称折行后的第二行可能很短（如「联接C」），只要不是界面文字就接上。"""
    compact = re.sub(r"\s+", "", text)
    return bool(_CJK_RE.search(compact)) and not is_ui_label(compact) and not _is_number(compact)


def _column_centers(rows: list[list[Observation]]) -> dict[str, float]:
    """读取表头「名称/金额 · 日收益 · 持有收益/率」的横坐标，用于对齐数字列。"""
    centers: dict[str, float] = {}
    for row in rows:
        for obs in row:
            compact = re.sub(r"\s+", "", obs.text)
            if "日收益" in compact and "daily" not in centers:
                centers["daily"] = obs.cx
            elif compact.startswith("持有收益") and "holding" not in centers:
                centers["holding"] = obs.cx
            elif "金额" in compact and "amount" not in centers:
                centers["amount"] = obs.cx
    return centers


def _assign_columns(numbers: list[Observation], centers: dict[str, float]) -> dict[str, float]:
    values = [parse_number(o.text) for o in numbers]
    if len(numbers) >= 3:
        return {"amount": values[0], "daily": values[1], "holding": values[2]}
    if len(numbers) == 1:
        return {"amount": values[0]}
    # 两个数字：金额一定在最左；另一个靠表头判断是日收益还是持有收益
    second = numbers[1]
    if "daily" in centers and "holding" in centers:
        key = "daily" if abs(second.cx - centers["daily"]) < abs(second.cx - centers["holding"]) else "holding"
    else:
        key = "holding"
    return {"amount": values[0], key: values[1]}


def parse_holdings(observations: list[Observation]) -> list[ParsedHolding]:
    """从支付宝等平台的持仓列表截图中读出每只基金的名称和金额。"""
    rows = group_rows(observations)
    centers = _column_centers(rows)
    # 旧版 OCR 只有文字没有坐标：同一张卡片的数字会分散在连续几行，按顺序补齐
    positional = any(o.w > 0 for o in observations)
    holdings: list[ParsedHolding] = []
    pending_name: list[str] = []
    current: ParsedHolding | None = None

    for row in rows:
        names = [o for o in row if _is_name_text(o.text) or (pending_name and _is_name_continuation(o.text))]
        numbers = [o for o in row if _is_number(o.text)]
        percents = [o for o in row if _is_percent(o.text)]

        if names and not numbers:
            name_text = "".join(o.text for o in names)
            if current is not None and current.amount is not None:
                current = None  # 上一张卡片已完整，开始新卡片
            pending_name.append(name_text)
            continue

        if numbers and pending_name:
            cols = _assign_columns(numbers, centers)
            name = _clean_name("".join(pending_name))
            pending_name = []
            current = ParsedHolding(
                ocr_name=name,
                amount=cols.get("amount"),
                daily_pnl=cols.get("daily"),
                holding_pnl=cols.get("holding"),
            )
            if percents:
                current.holding_rate = parse_number(percents[-1].text)
            if current.amount is not None and current.amount > 0:
                holdings.append(current)
            else:
                current = None
            continue

        if numbers and not positional and current is not None and not pending_name:
            for obs in numbers:
                slot = "daily_pnl" if current.daily_pnl is None else "holding_pnl" if current.holding_pnl is None else None
                if slot:
                    setattr(current, slot, parse_number(obs.text))
            if percents and current.holding_rate is None:
                current.holding_rate = parse_number(percents[-1].text)
            continue

        if percents and current is not None and current.holding_rate is None and not numbers:
            current.holding_rate = parse_number(percents[-1].text)
        # 其余是状态标签、表头等：跳过，不打断正在拼接的名称
    return holdings


def _clean_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"\s+", "", name)
    return name.strip("·|-_ ")


def codes_in_text(text: str) -> list[str]:
    """截图里直接出现的 6 位基金代码（天天基金等平台会显示）。"""
    seen: list[str] = []
    for code in _CODE_RE.findall(text):
        if code not in seen:
            seen.append(code)
    return seen


# --------------------------------------------------------------------------
# 名称 → 基金代码
# --------------------------------------------------------------------------
def normalize_fund_name(name: str) -> str:
    """统一全半角、大小写，去掉空格、省略号与括号写法差异。"""
    text = unicodedata.normalize("NFKC", str(name)).upper()
    text = text.replace("…", "").replace("...", "").replace("..", "")
    text = re.sub(r"[\s·・\-_]", "", text)
    text = text.replace("(", "").replace(")", "")
    return text


class FundIndex:
    """基金名称表的匹配索引。"""

    def __init__(self, records: list[tuple[str, str, str]]):
        # records: (代码, 简称, 类型)
        self.records = [(str(c).zfill(6), str(n), str(t)) for c, n, t in records]
        self.normalized = [normalize_fund_name(n) for _, n, _ in self.records]
        self.by_norm: dict[str, list[int]] = {}
        for index, norm in enumerate(self.normalized):
            self.by_norm.setdefault(norm, []).append(index)
        self.by_code = {code: index for index, (code, _, _) in enumerate(self.records)}

    def lookup_code(self, code: str) -> dict | None:
        index = self.by_code.get(str(code).zfill(6))
        return self._as_match(index, 1.0) if index is not None else None

    def _as_match(self, index: int, score: float) -> dict:
        code, name, kind = self.records[index]
        return {"fund_code": code, "fund_name": name, "fund_type": kind, "score": round(score, 3)}

    def match(self, ocr_name: str, limit: int = 3) -> list[dict]:
        """返回按可信度排序的候选基金；第一项为推荐匹配。"""
        query = normalize_fund_name(ocr_name)
        if len(query) < 2:
            return []
        truncated = bool(re.search(r"(…|\.{2,})$", str(ocr_name).strip()))
        scored: dict[int, float] = {}

        for index in self.by_norm.get(query, []):
            scored[index] = 1.0

        # 名称被截断：以识别出的文字开头的基金都是候选
        if len(scored) < limit and len(query) >= 4:
            for index, norm in enumerate(self.normalized):
                if index not in scored and norm.startswith(query):
                    bonus = 0.05 if truncated else 0.0
                    scored[index] = 0.86 + bonus + 0.08 * len(query) / max(len(norm), 1)

        # OCR 错字或平台简称不同：模糊匹配
        if len(scored) < limit:
            matcher = difflib.SequenceMatcher(autojunk=False)
            matcher.set_seq2(query)
            fuzzy: list[tuple[float, int]] = []
            for index, norm in enumerate(self.normalized):
                if index in scored:
                    continue
                matcher.set_seq1(norm)
                if matcher.real_quick_ratio() < 0.6 or matcher.quick_ratio() < 0.6:
                    continue
                ratio = matcher.ratio()
                if ratio >= 0.6:
                    fuzzy.append((ratio * 0.95, index))
            fuzzy.sort(reverse=True)
            for ratio, index in fuzzy[: limit * 2]:
                scored.setdefault(index, ratio)

        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], len(self.normalized[kv[0]])))
        return [self._as_match(index, min(score, 1.0)) for index, score in ranked[:limit]]


# --------------------------------------------------------------------------
# 金额 → 份额与成本
# --------------------------------------------------------------------------
def derive_position(
    amount: float | None,
    holding_pnl: float | None,
    nav: float | None,
    holding_rate: float | None = None,
) -> dict:
    """按「金额 ÷ 净值」推算份额，按「(金额 − 持有收益) ÷ 份额」推算成本净值。"""
    if amount is None or amount <= 0 or nav is None or nav <= 0:
        return {"shares": None, "cost_nav": None}
    shares = amount / nav
    cost_value = None
    if holding_pnl is not None:
        cost_value = amount - holding_pnl
    elif holding_rate is not None and holding_rate > -100:
        cost_value = amount / (1 + holding_rate / 100)
    cost_nav = cost_value / shares if cost_value and cost_value > 0 else None
    return {
        "shares": round(shares, 2),
        "cost_nav": round(cost_nav, 4) if cost_nav is not None else None,
    }
