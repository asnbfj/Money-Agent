"""轻量本地 Web 服务：托管前端、截图 OCR 与基金估值 API。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from money_agent.data.estimate import estimate_for
from money_agent.importer import (
    FundIndex,
    codes_in_text,
    derive_position,
    observations_from_payload,
    parse_holdings,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DATA_FILE = ROOT / ".money-agent-data" / "holdings.json"
OCR_SCRIPT = Path(__file__).with_name("ocr.swift")
MAX_BODY_BYTES = 12 * 1024 * 1024
QUOTE_CACHE_SECONDS = float(os.getenv("MONEY_AGENT_QUOTE_CACHE", "20"))
INTRADAY_CACHE_SECONDS = 50
_intraday_cache: dict[str, tuple[float, dict | None]] = {}
_quote_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()
FUND_INDEX_SECONDS = 6 * 3600
_fund_index: tuple[float, FundIndex] | None = None
_fund_index_lock = threading.Lock()


def _holdings() -> list[dict]:
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_holdings(items: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = DATA_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(DATA_FILE)


def _validate_code(code: object) -> str:
    """校验 6 位基金代码，不合法直接抛 ValueError。"""
    code = str(code or "").strip()
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("基金代码应为 6 位数字")
    return code


def _normalize_holding(item: object) -> dict:
    """校验并规范化单条持仓（导入 / 新增 / 交易共用）。"""
    if not isinstance(item, dict):
        raise ValueError("持仓条目应为对象")
    code = str(item.get("fund_code", "")).strip()
    name = str(item.get("fund_name", "")).strip()[:80]
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"基金代码 {code or '（空）'} 应为 6 位数字")
    try:
        shares = float(item.get("shares") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name or code} 的持有份额应为数字") from exc
    cost_raw = item.get("cost_nav")
    try:
        cost = float(cost_raw) if cost_raw not in (None, "") else None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name or code} 的成本净值应为数字") from exc
    if shares <= 0:
        raise ValueError(f"{name or code} 的持有份额必须大于 0")
    if cost is not None and cost <= 0:
        raise ValueError(f"{name or code} 的成本净值须大于 0")
    return {
        "fund_code": code,
        "fund_name": name or code,
        "shares": round(shares, 2),
        "cost_nav": round(cost, 4) if cost is not None else None,
    }


def _extract_candidates(text: str) -> list[dict]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    candidates: list[dict] = []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        for match in re.finditer(r"(?<!\d)(\d{6})(?!\d)", line):
            code = match.group(1)
            if code in seen:
                continue
            seen.add(code)
            start, end = max(0, i - 3), min(len(lines), i + 5)
            context = "\n".join(lines[start:end])
            name = ""
            for possible in reversed(lines[max(0, i - 3):i + 1]):
                cleaned = re.sub(r"\d{6}|基金代码|基金\s*代码|代码|[:：]", "", possible).strip(" -·|")
                if re.search(r"[\u4e00-\u9fffA-Za-z]{2,}", cleaned) and not re.search(
                    r"(持有收益|累计收益|昨日收益|持有金额|持仓金额|持有份额|份额|净值|估值)", cleaned
                ):
                    name = cleaned[:48]
                    break
            shares = _labeled_number(context, r"(?:持有|持仓)?份额")
            cost = _labeled_number(context, r"(?:成本净值|持仓成本|成本价|买入净值)")
            market_value = _labeled_number(context, r"(?:持有金额|持仓金额|市值)")
            profit = _labeled_number(context, r"(?:持有收益|累计收益|持仓收益)")
            if cost is None and shares and market_value is not None and profit is not None:
                cost = (market_value - profit) / shares
            candidates.append({"fund_code": code, "fund_name": name, "shares": shares, "cost_nav": cost})
    return candidates


def _labeled_number(text: str, label_pattern: str) -> float | None:
    match = re.search(rf"{label_pattern}[^\d+\-−]{{0,36}}([+\-−]?[\d,]+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def _ocr_image(image: bytes, suffix: str) -> str:
    """用 macOS Vision 在本机识别截图，不向第三方上传图像。"""
    if not OCR_SCRIPT.exists():
        raise RuntimeError("本地 OCR 程序缺失")
    module_cache = Path(tempfile.gettempdir()) / "money-agent-swift-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix) as image_file:
        image_file.write(image)
        image_file.flush()
        try:
            result = subprocess.run(
                ["swift", "-module-cache-path", str(module_cache), str(OCR_SCRIPT), image_file.name],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 Swift 运行环境，无法启动本机 OCR") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("截图识别超时，请尝试裁剪后重新上传") from exc
    if result.returncode:
        message = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"截图识别失败：{message or 'Vision OCR 错误'}")
    output = result.stdout.strip()
    try:
        payload: object = json.loads(output)
    except json.JSONDecodeError:
        payload = output  # 旧版 OCR 脚本只输出文字
    return observations_from_payload(payload)


def _load_fund_index() -> FundIndex | None:
    """基金名称表（akshare，磁盘缓存 12 小时）做成内存索引；拿不到时返回 None。"""
    global _fund_index
    with _fund_index_lock:
        if _fund_index and time.monotonic() - _fund_index[0] < FUND_INDEX_SECONDS:
            return _fund_index[1]
        try:
            from money_agent.config import Settings
            from money_agent.data.resolver import fund_table

            df = fund_table(Settings())
        except Exception:  # noqa: BLE001 - 依赖缺失或网络失败时退化为手动填写代码
            return None
        if df is None or df.empty:
            return None
        kinds = df["基金类型"].astype(str) if "基金类型" in df.columns else [""] * len(df)
        records = list(zip(df["基金代码"].astype(str), df["基金简称"].astype(str), kinds))
        index = FundIndex(records)
        _fund_index = (time.monotonic(), index)
        return index


def _confirmed_nav(code: str) -> tuple[float | None, str | None]:
    """最近一个已公布的正式净值。支付宝的持有金额按它计算，不用盘中估值。"""
    try:
        from money_agent.agent.functions import get_fund_latest_info

        info = get_fund_latest_info(code)
    except Exception:  # noqa: BLE001
        return None, None
    if info.get("error"):
        return None, None
    if info.get("tag") == "估算":
        nav, date = info.get("official_nav"), info.get("official_nav_date")
    elif info.get("is_exchange_traded"):
        nav, date = info.get("unit_nav") or info.get("nav"), info.get("unit_nav_date") or info.get("nav_time")
    else:
        nav, date = info.get("nav"), info.get("nav_time")
    try:
        return (float(nav), str(date) if date else None) if nav is not None else (None, None)
    except (TypeError, ValueError):
        return None, None


def _recognize_holdings(observations: list) -> dict:
    """截图 → 待确认的持仓列表：名称匹配代码，再按正式净值推算份额和成本。"""
    text = "\n".join(o.text for o in observations)
    parsed = parse_holdings(observations)
    index = _load_fund_index()
    rows: list[dict] = []
    if parsed:
        source = "holdings"
        for item in parsed:
            matches = index.match(item.ocr_name) if index else []
            best = matches[0] if matches else None
            confidence = "none"
            if best:
                close = len(matches) > 1 and matches[1]["score"] >= best["score"] - 0.02
                confidence = "high" if best["score"] >= 0.97 and not close else "check"
            rows.append({
                **item.to_dict(),
                "matches": matches,
                "fund_code": best["fund_code"] if best else "",
                "fund_name": best["fund_name"] if best else item.ocr_name,
                "confidence": confidence,
            })
    else:
        source = "codes" if codes_in_text(text) else "none"
        for cand in _extract_candidates(text):
            hit = index.lookup_code(cand["fund_code"]) if index else None
            rows.append({
                **cand,
                "ocr_name": cand.get("fund_name") or cand["fund_code"],
                "fund_name": cand.get("fund_name") or (hit["fund_name"] if hit else ""),
                "matches": [hit] if hit else [],
                "confidence": "high" if hit else "check",
            })

    def fill(row: dict) -> None:
        if not row.get("fund_code") or row.get("amount") is None:
            return
        nav, nav_date = _confirmed_nav(row["fund_code"])
        row["nav"], row["nav_date"] = nav, nav_date
        row.update(derive_position(row.get("amount"), row.get("holding_pnl"), nav, row.get("holding_rate")))

    if source == "holdings" and rows:
        with ThreadPoolExecutor(max_workers=min(6, len(rows))) as pool:
            list(pool.map(fill, rows))
    return {
        "text": text,
        "source": source,
        "candidates": rows,
        "fund_list_available": index is not None,
    }


def _quote_for(holding: dict) -> dict:
    code = holding["fund_code"]
    now = time.monotonic()
    with _cache_lock:
        cached = _quote_cache.get(code)
        if cached and now - cached[0] < QUOTE_CACHE_SECONDS:
            quote = dict(cached[1])
            quote["cached"] = True
            return quote
    try:
        from money_agent.agent.functions import get_fund_latest_info

        quote = get_fund_latest_info(code)
    except Exception as exc:  # noqa: BLE001 - 单只基金失败不影响组合其余条目
        quote = {"fund_code": code, "error": f"估值获取失败：{exc}"}
    with _cache_lock:
        _quote_cache[code] = (now, dict(quote))
    quote["cached"] = False
    return quote


def _estimate_quote(quote: dict, trading_day: bool) -> dict | None:
    """给一只基金的报价附上自建盘中估算。

    只在「交易日 + 今日正式净值尚未公布」时生效（判定逻辑在 estimate.estimate_for）。
    """
    if not isinstance(quote, dict) or quote.get("error"):
        return None
    if quote.get("tag") != "正式":
        return None
    return estimate_for(
        str(quote.get("fund_code") or ""),
        quote.get("nav"),
        quote.get("nav_time"),
        trading_day=trading_day,
    )


def _apply_estimate(quote: dict, estimate: dict | None) -> dict:
    """把估算结果套用到报价上：净值、涨跌幅、标签都换成估算值，并保留官方基准。"""
    if not estimate:
        return quote
    if estimate.get("error"):
        return {**quote, "estimate": estimate}
    official_nav = quote.get("nav")
    official_date = quote.get("nav_time")
    return {
        **quote,
        "nav": estimate.get("estimate_nav"),
        "nav_time": estimate.get("base_date"),
        "change_pct": estimate.get("change_pct"),
        "tag": "估算",
        "nav_kind": "自建盘中估算（仅供参考）",
        "official_nav": official_nav,
        "official_nav_date": official_date,
        "estimate": estimate,
    }


def _portfolio_payload() -> dict:
    holdings = _holdings()
    market = _market_status()
    trading_day = bool(market.get("trading_day"))
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(holdings)))) as pool:
        quotes = list(pool.map(_quote_for, holdings)) if holdings else []
        estimates = list(pool.map(lambda q: _estimate_quote(q, trading_day), quotes)) if quotes else []
    rows = []
    total_value = 0.0
    total_pnl = 0.0
    pnl_complete = bool(holdings)
    for holding, quote, estimate in zip(holdings, quotes, estimates):
        quote = _apply_estimate(quote, estimate)
        code = holding["fund_code"]
        nav = quote.get("nav")
        shares = holding.get("shares")
        cost = holding.get("cost_nav")
        value = round(float(nav) * float(shares), 2) if nav is not None and shares is not None else None
        pnl = round((float(nav) - float(cost)) * float(shares), 2) if value is not None and cost is not None else None
        if value is not None:
            total_value += value
        if pnl is not None:
            total_pnl += pnl
        else:
            pnl_complete = False
        resolved_name = quote.get("fund_name") if holding.get("fund_name") == code else None
        rows.append({**holding, "fund_name": resolved_name or holding.get("fund_name") or code, "quote": quote, "market_value": value, "pnl_amount": pnl})
    return {
        "holdings": rows,
        "total_value": round(total_value, 2) if rows and any(r["market_value"] is not None for r in rows) else None,
        "total_pnl": round(total_pnl, 2) if pnl_complete and holdings else None,
        "refreshed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market": market,
        "quote_cache_seconds": QUOTE_CACHE_SECONDS,
        "estimate": {
            "enabled": trading_day,
            "note": "自建盘中估算：季报重仓股 × 实时股价；未覆盖部分按同比例归一化",
        },
    }


def _market_status() -> dict:
    try:
        from money_agent.data.market import market_status

        return market_status()
    except Exception:  # noqa: BLE001 - 依赖缺失时前端按普通刷新处理
        return {"trading_day": None, "phase": "unknown", "label": ""}


def _intraday_for(code: str) -> dict | None:
    """单只基金的当日分时估值（腾讯 getSsgz），缓存 50 秒。"""
    now = time.monotonic()
    with _cache_lock:
        cached = _intraday_cache.get(code)
        if cached and now - cached[0] < INTRADAY_CACHE_SECONDS:
            return cached[1]
    try:
        from money_agent.data.market import fetch_intraday

        data = fetch_intraday(code)
    except Exception:  # noqa: BLE001 - 单只失败不影响组合
        data = None
    with _cache_lock:
        _intraday_cache[code] = (now, data)
    return data


def _portfolio_intraday_payload() -> dict:
    """组合当日分时：按份额加总各基金的分时估值。"""
    from money_agent.data.market import portfolio_intraday

    holdings = _holdings()
    if not holdings:
        return {"available": False, "reason": "还没有持仓", "market": _market_status()}
    with ThreadPoolExecutor(max_workers=min(5, len(holdings))) as pool:
        datas = list(pool.map(lambda h: _intraday_for(h["fund_code"]), holdings))
    series = []
    funds = []
    for holding, data in zip(holdings, datas):
        shares = float(holding.get("shares") or 0)
        base = data.get("yesterday_nav") if data else None
        if base is None:
            # 没有分时的基金按最近正式净值计入组合，不影响涨跌
            quote = _quote_for(holding)
            base = quote.get("official_nav") or quote.get("nav")
        series.append({"shares": shares, "yesterday_nav": base, "points": (data or {}).get("points", [])})
        funds.append({"fund_code": holding["fund_code"], "has_intraday": bool(data and data.get("points"))})
    result = portfolio_intraday(series)
    if not result:
        return {"available": False, "reason": "暂时没有当日分时估值", "funds": funds, "market": _market_status()}
    return {"available": True, **result, "funds": funds, "market": _market_status()}


def _clear_quote_cache() -> None:
    """持仓变动后必须清缓存，否则新份额/新成本会配上旧估值。"""
    with _cache_lock:
        _quote_cache.clear()
        _intraday_cache.clear()


class MoneyAgentHandler(SimpleHTTPRequestHandler):
    """本地 UI 静态文件与 JSON API。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/intraday":
            try:
                self._send_json(200, _portfolio_intraday_payload())
            except Exception as exc:  # noqa: BLE001
                self.log_error("intraday failed: %s", exc)
                self._send_json(500, {"error": "分时估值暂时不可用"})
            return
        if path.startswith("/api/intraday/"):
            code = unquote(path[len("/api/intraday/"):]).strip("/")
            try:
                code = _validate_code(code)
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            data = _intraday_for(code)
            self._send_json(200 if data else 404, data or {"error": "暂时没有该基金的分时估值"})
            return
        if path.startswith("/api/fund/"):
            self._fund_info(unquote(path[len("/api/fund/"):]).strip("/"))
            return
        if path.startswith("/api/estimate/"):
            code = unquote(path[len("/api/estimate/"):]).strip("/")
            try:
                code = _validate_code(code)
            except ValueError as exc:
                self._send_json(422, {"error": str(exc)})
                return
            quote = _quote_for({"fund_code": code})
            estimate = _estimate_quote(quote, bool(_market_status().get("trading_day")))
            if estimate is None:
                self._send_json(404, {"error": "当前不适用盘中估算（非交易日，或今日正式净值已公布）"})
                return
            self._send_json(200, {"fund_code": code, "quote": quote, "estimate": estimate})
            return
        if path == "/api/portfolio":
            try:
                self._send_json(200, _portfolio_payload())
            except Exception as exc:  # noqa: BLE001
                self.log_error("portfolio refresh failed: %s", exc)
                self._send_json(500, {"error": "组合估值暂时不可用"})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/import-screenshot":
            self._import_screenshot()
        elif path == "/api/holdings":
            self._save_imported_holdings()
        elif path.startswith("/api/holdings/"):
            self._upsert_holding(unquote(path[len("/api/holdings/"):]).strip("/"))
        elif path == "/api/analyze":
            self._analyze()
        else:
            self.send_error(404, "Not found")

    def do_PATCH(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/holdings/"):
            self._trade_holding(unquote(path[len("/api/holdings/"):]).strip("/"))
        else:
            self.send_error(404, "Not found")

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/holdings/"):
            self._delete_holding(unquote(path[len("/api/holdings/"):]).strip("/"))
        else:
            self.send_error(404, "Not found")

    def _fund_info(self, code: str) -> None:
        """单只基金的名称与最近正式净值，导入确认时改了代码用来重算份额和成本。"""
        try:
            code = _validate_code(code)
        except ValueError as exc:
            self._send_json(422, {"error": str(exc)})
            return
        index = _load_fund_index()
        hit = index.lookup_code(code) if index else None
        nav, nav_date = _confirmed_nav(code)
        if hit is None and nav is None:
            self._send_json(404, {"error": f"没有找到代码为 {code} 的基金，或行情暂时不可用"})
            return
        self._send_json(200, {
            "fund_code": code,
            "fund_name": hit["fund_name"] if hit else "",
            "fund_type": hit["fund_type"] if hit else "",
            "nav": nav,
            "nav_date": nav_date,
        })

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("上传内容为空或超过 12 MB")
        return self.rfile.read(length)

    def _import_screenshot(self) -> None:
        try:
            body = self._read_body()
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("请上传图片文件")
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
            )
            image = None
            image_type = ""
            for part in message.iter_parts():
                if part.get_content_disposition() == "form-data" and part.get_param(
                    "name", header="content-disposition"
                ) == "screenshot":
                    image = part.get_payload(decode=True)
                    image_type = part.get_content_type()
                    break
            suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(image_type)
            if not image or not suffix:
                raise ValueError("支持 PNG、JPG 或 WebP 图片")
            self._send_json(200, _recognize_holdings(_ocr_image(image, suffix)))
        except (ValueError, RuntimeError) as exc:
            self._send_json(422, {"error": str(exc)})

    def _save_imported_holdings(self) -> None:
        try:
            payload = json.loads(self._read_body())
            items = payload.get("holdings") if isinstance(payload, dict) else None
            if not isinstance(items, list) or not items or len(items) > 100:
                raise ValueError("请至少确认一只基金，最多支持 100 只")
            normalized: list[dict] = []
            indexes: dict[str, int] = {}
            for item in items:
                record = _normalize_holding(item)
                code = record["fund_code"]
                if code in indexes:
                    existing = normalized[indexes[code]]
                    combined = existing["shares"] + record["shares"]
                    if existing["cost_nav"] is not None and record["cost_nav"] is not None:
                        existing["cost_nav"] = round(
                            (
                                existing["cost_nav"] * existing["shares"]
                                + record["cost_nav"] * record["shares"]
                            )
                            / combined,
                            4,
                        )
                    else:
                        existing["cost_nav"] = None
                    existing["shares"] = round(combined, 2)
                else:
                    indexes[code] = len(normalized)
                    normalized.append(record)
            _save_holdings(normalized)
            _clear_quote_cache()
            self._send_json(200, {"holdings": normalized, "count": len(normalized)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(422, {"error": str(exc)})

    def _upsert_holding(self, code: str) -> None:
        """新增或覆盖单只持仓（新增持仓 / 编辑份额与成本）。"""
        try:
            code = _validate_code(code)
            payload = json.loads(self._read_body())
            if not isinstance(payload, dict):
                raise ValueError("请求体应为 JSON 对象")
            payload["fund_code"] = code
            record = _normalize_holding(payload)
            holdings = _holdings()
            for index, holding in enumerate(holdings):
                if holding["fund_code"] == code:
                    holdings[index] = record
                    break
            else:
                if len(holdings) >= 100:
                    raise ValueError("最多支持 100 只基金")
                holdings.append(record)
            _save_holdings(holdings)
            _clear_quote_cache()
            self._send_json(200, {"holding": record, "count": len(holdings), "message": "持仓已保存"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(422, {"error": str(exc)})

    def _trade_holding(self, code: str) -> None:
        """加仓 / 减仓。加仓按加权平均重算成本净值；减仓不改变剩余成本，减至 0 移除。"""
        try:
            code = _validate_code(code)
            payload = json.loads(self._read_body())
            if not isinstance(payload, dict):
                raise ValueError("请求体应为 JSON 对象")
            action = str(payload.get("action", "")).strip()
            if action not in ("add", "reduce"):
                raise ValueError("操作类型只支持 add（加仓）或 reduce（减仓）")
            try:
                delta = float(payload.get("shares") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError("变动份额应为数字") from exc
            if delta <= 0:
                raise ValueError("变动份额必须大于 0")
            price_raw = payload.get("price")
            try:
                price = float(price_raw) if price_raw not in (None, "") else None
            except (TypeError, ValueError) as exc:
                raise ValueError("成交净值应为数字") from exc
            if price is not None and price <= 0:
                raise ValueError("成交净值必须大于 0")

            holdings = _holdings()
            target = next((h for h in holdings if h["fund_code"] == code), None)
            if target is None:
                raise ValueError(f"当前持仓中没有 {code} 这只基金，请先新增持仓")

            if action == "add":
                old_shares = float(target["shares"])
                old_cost = target.get("cost_nav")
                new_shares = old_shares + delta
                if old_cost is not None and price is not None:
                    target["cost_nav"] = round((old_cost * old_shares + price * delta) / new_shares, 4)
                elif old_cost is None and price is not None:
                    target["cost_nav"] = round(price, 4)
                target["shares"] = round(new_shares, 2)
                message = f"已加仓 {delta:g} 份，现持有 {target['shares']:g} 份"
            else:
                if delta >= float(target["shares"]) - 1e-9:
                    holdings = [h for h in holdings if h["fund_code"] != code]
                    message = "已减仓至 0，该持仓已移除"
                else:
                    target["shares"] = round(float(target["shares"]) - delta, 2)
                    message = f"已减仓 {delta:g} 份，现持有 {target['shares']:g} 份"

            _save_holdings(holdings)
            _clear_quote_cache()
            self._send_json(200, {"holdings": holdings, "count": len(holdings), "message": message})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(422, {"error": str(exc)})

    def _delete_holding(self, code: str) -> None:
        """删除一只持仓。"""
        try:
            code = _validate_code(code)
            holdings = _holdings()
            remaining = [h for h in holdings if h["fund_code"] != code]
            if len(remaining) == len(holdings):
                raise ValueError(f"当前持仓中没有 {code} 这只基金")
            _save_holdings(remaining)
            _clear_quote_cache()
            self._send_json(
                200,
                {"holdings": remaining, "count": len(remaining), "message": f"已删除 {code}"},
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(422, {"error": str(exc)})

    def _analyze(self) -> None:
        try:
            payload = json.loads(self._read_body())
            query = str(payload.get("query", "")).strip() if isinstance(payload, dict) else ""
            if not query:
                raise ValueError("请输入基金名称或代码")
            if len(query) > 2_000:
                raise ValueError("分析请求最多支持 2000 个字符")
            try:
                from money_agent.agent import FinancialAgent
                from money_agent.report import render
            except ImportError as exc:
                raise RuntimeError("项目依赖尚未安装，请先运行 pip install -r requirements.txt") from exc
            report = FinancialAgent().analyze(query)
            self._send_json(200, {"report": report.to_dict(), "text": render(report)})
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(422, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self.log_error("analysis failed: %s", exc)
            self._send_json(500, {"error": "分析服务暂时不可用，请稍后重试"})

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """启动仅监听本机的开发服务。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    host = os.getenv("MONEY_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("MONEY_AGENT_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), MoneyAgentHandler)
    print(f"Money Agent 前端已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
