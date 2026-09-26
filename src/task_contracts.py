"""处理任务的数据契约（任务 A：请求与缓存身份）。

`PipelineRequest` 冻结一次处理请求所需的全部输入，并给出两个互不相关的标识：

* `cache_key` —— 由全部字段 + `SCHEMA_VERSION` 规范化后取 SHA256，用于判断
  "这次请求能否复用上次的结果"。任一字段（含日期、坐标、窗口、地形、阈值）
  变化都会得到不同的键，因此不会把别的请求的缓存当成自己的。
* `new_run_id()` —— 每次执行独立的 UUID，用于日志/恢复；与缓存键无关。

坐标不做任何四舍五入：身份就是用户给的原值，避免"附近但不是同一地点"误命中。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Dict

SCHEMA_VERSION = 2
TERRAIN_PROFILES = ("unspecified", "plain", "hilly", "mountain", "urban", "coastal", "wetland", "arid")
MIN_SIZE, MAX_SIZE = 64, 4096
_REQUIRED = ("lon", "lat", "pre_start", "pre_end", "post_start", "post_end")


def _iso_date(value: Any, name: str) -> str:
    """校验真实的、规范的 ISO 日期（YYYY-MM-DD）。"""
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{name} 需为 YYYY-MM-DD 字符串：{value!r}")
    try:
        parsed = _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} 不是合法日期：{value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} 需为规范 ISO 日期：{value!r}")
    return value


def _number(value: Any, name: str) -> float:
    """校验有限数字（拒绝 bool / 字符串 / NaN / inf）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 需为数字：{value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} 需为有限数字：{value!r}")
    return out


@dataclass(frozen=True)
class PipelineRequest:
    """一次遥感处理请求的完整身份。"""

    lon: float
    lat: float
    pre_start: str
    pre_end: str
    post_start: str
    post_end: str
    size: int = 768
    terrain_profile: str = "unspecified"
    max_cloud_pct: float = 35.0
    min_valid_pct: float = 50.0
    detection_strategy: str = "baseline"
    water_index: str = "auto"
    index_threshold: float | None = None
    slope_threshold_deg: float = 15.0
    band_order: str = "auto"

    def __post_init__(self) -> None:
        lon = _number(self.lon, "lon")
        lat = _number(self.lat, "lat")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"lon 超出 [-180, 180]：{lon}")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"lat 超出 [-90, 90]：{lat}")
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise ValueError(f"size 需为整数：{self.size!r}")
        if not MIN_SIZE <= self.size <= MAX_SIZE:
            raise ValueError(f"size 需在 {MIN_SIZE}..{MAX_SIZE}：{self.size}")
        if self.terrain_profile not in TERRAIN_PROFILES:
            raise ValueError(f"terrain_profile 非法：{self.terrain_profile!r}")
        if self.detection_strategy not in ("baseline", "adaptive"):
            raise ValueError("detection_strategy 需为 baseline/adaptive")
        if self.water_index not in ("auto", "ndwi", "mndwi"):
            raise ValueError("water_index 需为 auto/ndwi/mndwi")
        from src.preprocess import BAND_ORDERS
        if self.band_order not in ("auto", *BAND_ORDERS):
            raise ValueError("未知 band_order")
        slope = _number(self.slope_threshold_deg, "slope_threshold_deg")
        if not 0 <= slope <= 90:
            raise ValueError("slope_threshold_deg 需在 0..90")
        object.__setattr__(self, "slope_threshold_deg", slope)
        if self.index_threshold is not None:
            threshold = _number(self.index_threshold, "index_threshold")
            if not -1 <= threshold <= 1:
                raise ValueError("index_threshold 需在 -1..1")
            object.__setattr__(self, "index_threshold", threshold)
        if self.detection_strategy == "baseline" and (self.water_index != "auto" or self.index_threshold is not None):
            raise ValueError("water_index/index_threshold 自定义需要 detection_strategy=adaptive")
        cloud = _number(self.max_cloud_pct, "max_cloud_pct")
        valid = _number(self.min_valid_pct, "min_valid_pct")
        for name, pct in (("max_cloud_pct", cloud), ("min_valid_pct", valid)):
            if not 0.0 <= pct <= 100.0:
                raise ValueError(f"{name} 需在 0..100：{pct}")
        pre_s = _iso_date(self.pre_start, "pre_start")
        pre_e = _iso_date(self.pre_end, "pre_end")
        post_s = _iso_date(self.post_start, "post_start")
        post_e = _iso_date(self.post_end, "post_end")
        if not pre_s <= pre_e < post_s <= post_e:
            raise ValueError("需满足 pre_start<=pre_end<post_start<=post_end")
        for name, value in (
            ("lon", lon), ("lat", lat), ("size", self.size),
            ("max_cloud_pct", cloud), ("min_valid_pct", valid),
            ("pre_start", pre_s), ("pre_end", pre_e),
            ("post_start", post_s), ("post_end", post_e),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineRequest":
        """从字典还原；未知字段与缺失必填字段都直接拒绝。"""
        if not isinstance(payload, Mapping):
            raise ValueError("from_dict 需要映射对象")
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"未知字段：{unknown}")
        missing = [name for name in _REQUIRED if name not in payload]
        if missing:
            raise ValueError(f"缺少必填字段：{missing}")
        return cls(**dict(payload))

    @property
    def cache_key(self) -> str:
        """全部字段 + schema 版本的规范化 SHA256。"""
        body = self.to_dict()
        body["schema_version"] = SCHEMA_VERSION
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def new_run_id() -> str:
        """本次执行独立的运行标识（UUID），与 cache_key 无关。"""
        return uuid.uuid4().hex


def new_run_id() -> str:
    """Module-level entry point for orchestration callers."""
    return PipelineRequest.new_run_id()
