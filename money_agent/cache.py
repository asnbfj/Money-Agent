"""磁盘缓存：用于基金/股票/板块名称列表等“体量大、变化慢”的数据。

akshare 的 `stock_info_a_code_name` / `stock_board_industry_name_em` 等接口较慢，
缓存可以显著降低分析延迟，也降低被数据源限流的概率。

另外对“失败/空结果”做短时负缓存：某些行情节点偶发不可用，如果每次都重新发起请求，
解析一个标的可能要多等十几秒。失败结果按 `empty_ttl_hours` 缓存，稍后自动重试。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

CACHE_DIR = Path(
    os.getenv("MONEY_AGENT_CACHE", str(Path.home() / ".cache" / "money_agent"))
)


def _path_for(key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:40]
    return CACHE_DIR / f"{safe}-{digest}.json"


def _write(key: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {**payload, "saved_at": time.time()}
        _path_for(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # 缓存不是关键路径，写失败不影响分析


def load_dataframe(
    key: str, ttl_hours: float, empty_ttl_hours: float = 0.25
) -> pd.DataFrame | None:
    """读取缓存 DataFrame；过期、不存在或失败缓存已过期时返回 None。

    注意：返回空 DataFrame 表示“缓存命中但内容为空”，与 None（未命中）语义不同。
    """
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    age_hours = (time.time() - payload.get("saved_at", 0)) / 3600

    if payload.get("failed"):
        if empty_ttl_hours >= 0 and age_hours > empty_ttl_hours:
            return None
        columns = payload.get("columns") or None
        return pd.DataFrame(columns=columns)

    limit = empty_ttl_hours if payload.get("empty") else ttl_hours
    if limit >= 0 and age_hours > limit:
        return None
    try:
        return pd.DataFrame(payload["records"], columns=payload.get("columns"))
    except Exception:
        return None


def save_dataframe(key: str, df: pd.DataFrame) -> None:
    _write(
        key,
        {
            "columns": list(map(str, df.columns)),
            "records": json.loads(df.to_json(orient="records", force_ascii=False)),
            "empty": bool(df.empty),
        },
    )


def mark_failed(key: str, columns: Sequence[str] | None = None) -> None:
    """记录一次失败，做短时负缓存，避免短时间内反复重试拖慢响应。"""
    _write(key, {"failed": True, "columns": list(columns or [])})


def cached_dataframe(
    key: str,
    loader: Callable[[], pd.DataFrame],
    ttl_hours: float = 12.0,
    empty_ttl_hours: float = 0.25,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """带缓存的 DataFrame 加载：优先缓存，其次调用 loader 并写回。

    loader 抛错时返回 None 并写入短时负缓存，保证不会中断整条链路。
    """
    cached = load_dataframe(key, ttl_hours, empty_ttl_hours)
    if cached is not None:
        return cached
    try:
        df = loader()
    except Exception:
        mark_failed(key, columns)
        return None
    if df is None:
        mark_failed(key, columns)
        return None
    save_dataframe(key, df)
    return df
