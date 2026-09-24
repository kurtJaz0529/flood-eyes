"""
慧眼识灾 · 本地 GeoTIFF 多时相光谱监测
=========================================

在本地 GeoTIFF 上计算植被/水体指数，并按时间相邻两景统计指数变化。
本模块复用既有 ``src.preprocess.load_scene`` 读取影像；配准时逐波段排除
NoData 对邻近像元的影响。不改动洪水识别流水线，不联网。

指数与所需真实波段（禁止用无 NIR 的代理波段）::

    ndvi = (nir - red) / (nir + red)                需要 nir, red
    savi = 1.5 * (nir - red) / (nir + red + 0.5)    需要 nir, red
    ndwi = (green - nir) / (green + nir)            需要 green, nir

关键约定
--------
* 全部场景相对**首景**对齐到同一网格；任一景缺 CRS/transform 直接报错。
* 单景有效区 = 非 nodata ∧ 所需波段为有限值 ∧ 分母非零 ∧
  SCL sidecar 不属于 ``{0, 1, 3, 8, 9, 10, 11}``
  （0=NoData 1=饱和/缺陷 3=云影 8/9=云 10=卷云 11=雪/冰）。
* 指数差值只在相邻两景的**共同有效区**计算。
* 每景及相邻两景共同区均使用 ``min_valid_pct`` 门禁；未过门禁时，
  均值/共同有效面积报告为 ``null``（status=missing/insufficient），绝不写成 0。
* 栅格输出：每景指数 ``float32``（无效 -9999）、相邻差值 ``float32``
  （无效 -9999）、共同有效区 ``uint8`` 0/1（0 是真实类别，不声明 nodata），
  沿用首景真实 CRS/transform。
* 共同有效面积仅在 CRS 为**米制投影**且 transform 行列式可给出投影网格像元面积时才
  输出 km²，否则为 ``null`` 并附说明。
* 日期只能由用户在 CLI 显式提供，否则标记 ``unverified``；绝不从文件名臆测。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date as calendar_date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .preprocess import Scene, load_scene

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

#: 指数 -> 公式文本（写入 summary.json，便于复核）
INDEX_FORMULAS: Dict[str, str] = {
    "ndvi": "(nir - red) / (nir + red)",
    "savi": "1.5 * (nir - red) / (nir + red + 0.5)",
    "ndwi": "(green - nir) / (green + nir)",
}

#: 指数 -> 严格需要的真实波段（不允许替代波段）
REQUIRED_BANDS: Dict[str, Tuple[str, ...]] = {
    "ndvi": ("nir", "red"),
    "savi": ("nir", "red"),
    "ndwi": ("green", "nir"),
}

#: SCL 中视为"不可用"的类别。有效类别为其补集（2/4/5/6/7）。
SCL_INVALID_CLASSES: Tuple[int, ...] = (0, 1, 3, 8, 9, 10, 11)

#: 指数栅格与差值栅格的无效值。绝不使用 0，因为 0 是合法指数值。
INDEX_NODATA: float = -9999.0

#: 分母绝对值不超过该值即视为无信号（严格意义上"分母非零"）。
DENOM_EPS: float = 1e-12

_GEOTIFF_EXTS = frozenset({".tif", ".tiff", ".vrt", ".img"})
_METER_UNITS = frozenset({"metre", "meter", "m", "metres", "meters"})
_UNVERIFIED_NOTE = (
    "未提供日期：所有场景 date=null 且 date_source='unverified'；"
    "未从文件名或影像元数据推断日期。"
)
_USER_DATE_NOTE = (
    "日期由用户在 CLI 显式提供（date_source='user_provided'），"
    "未与影像元数据二次核验。"
)


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def supported_indices() -> Tuple[str, ...]:
    """返回支持的指数名。"""
    return tuple(INDEX_FORMULAS)


def _normalize_index(index: Any) -> str:
    key = str(index).strip().lower()
    if key not in INDEX_FORMULAS:
        raise ValueError(f"未知指数 '{index}'，可选：{sorted(INDEX_FORMULAS)}")
    return key


def required_bands(index: str) -> Tuple[str, ...]:
    """返回该指数严格要求存在的真实波段名。"""
    return REQUIRED_BANDS[_normalize_index(index)]


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """分块计算文件 SHA256（用于 summary.json 记录输入指纹）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _transform_list(transform: Any) -> Optional[List[float]]:
    if transform is None:
        return None
    try:
        return [float(transform.a), float(transform.b), float(transform.c),
                float(transform.d), float(transform.e), float(transform.f)]
    except Exception:  # pragma: no cover - 防御性分支
        return None


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in stem)
    return cleaned.strip("_") or "scene"


def _require_arrays(bands: Mapping[str, Any], index: str) -> Dict[str, np.ndarray]:
    """取出该指数需要的真实波段，缺任何一个都报错。"""
    arrays: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for name in REQUIRED_BANDS[index]:
        arr = bands.get(name)
        if arr is None:
            missing.append(name)
        else:
            arrays[name] = np.asarray(arr, dtype=np.float32)
    if missing:
        raise ValueError(
            f"指数 {index} 严格要求真实波段 {list(REQUIRED_BANDS[index])}，"
            f"当前缺少 {missing}（可用波段：{sorted(bands)}）。"
            "禁止使用无 NIR 的代理波段或其他替代波段。"
        )
    shapes = {arr.shape for arr in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"指数 {index} 所需波段尺寸不一致：{sorted(shapes)}")
    return arrays


def _terms_from_arrays(arrays: Mapping[str, np.ndarray], index: str) -> Tuple[np.ndarray, np.ndarray]:
    if index in ("ndvi", "savi"):
        nir = arrays["nir"]
        red = arrays["red"]
        num = nir - red
        den = nir + red
        if index == "savi":
            num = np.float32(1.5) * num
            den = den + np.float32(0.5)
    else:  # ndwi
        green = arrays["green"]
        nir = arrays["nir"]
        num = green - nir
        den = green + nir
    return (
        np.asarray(num, dtype=np.float32),
        np.asarray(den, dtype=np.float32),
    )


def index_terms(bands: Mapping[str, Any], index: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 ``(分子, 分母)``，两者均为 float32，便于逐项核验公式。"""
    key = _normalize_index(index)
    return _terms_from_arrays(_require_arrays(bands, key), key)


def compute_index(bands: Mapping[str, Any], index: str) -> np.ndarray:
    """计算原始指数 float32；分母为零或非有限处为 ``nan``（不做无效值替换）。"""
    key = _normalize_index(index)
    num, den = _terms_from_arrays(_require_arrays(bands, key), key)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > DENOM_EPS)
    np.divide(num, den, out=out, where=ok)
    return out


def compute_masked_index(bands: Mapping[str, Any], index: str, valid: np.ndarray) -> np.ndarray:
    """在 ``valid`` 内计算指数，其余位置写 ``INDEX_NODATA``（-9999）。"""
    key = _normalize_index(index)
    num, den = _terms_from_arrays(_require_arrays(bands, key), key)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != num.shape:
        raise ValueError(
            f"有效掩膜尺寸 {valid.shape} 与指数尺寸 {num.shape} 不一致"
        )
    out = np.full(num.shape, INDEX_NODATA, dtype=np.float32)
    ok = valid & np.isfinite(num) & np.isfinite(den) & (np.abs(den) > DENOM_EPS)
    np.divide(num, den, out=out, where=ok)
    out[~ok] = INDEX_NODATA
    return out


def scene_valid_mask(scene: Scene, index: str) -> Tuple[np.ndarray, bool]:
    """单景有效区。

    组成：非 nodata ∧ 所需波段有限 ∧ 分子分母有限且分母非零 ∧
    SCL sidecar 不属于 :data:`SCL_INVALID_CLASSES`。

    返回 ``(valid_bool, scl_available)``。没有 SCL sidecar 时不施加 SCL 约束，
    由调用方在 summary 中显式告警。
    """
    key = _normalize_index(index)
    arrays = _require_arrays(scene.bands, key)
    shape = scene.shape
    for name, arr in arrays.items():
        if arr.shape[:2] != shape:
            raise ValueError(f"波段 {name} 尺寸 {arr.shape[:2]} 与影像 {shape} 不一致")

    valid = np.ones(shape, dtype=bool)

    nodata = scene.nodata_mask
    if nodata is not None:
        nodata = np.asarray(nodata)
        if nodata.shape[:2] != shape:
            raise ValueError(f"nodata_mask 尺寸 {nodata.shape[:2]} 与影像 {shape} 不一致")
        valid &= ~nodata.astype(bool)

    for arr in arrays.values():
        valid &= np.isfinite(arr)
    band_valid = scene.meta.get("spectral_band_valid") or {}
    for name in arrays:
        per_band = band_valid.get(name)
        if per_band is not None:
            per_band = np.asarray(per_band, dtype=bool)
            if per_band.shape != shape:
                raise ValueError(f"波段 {name} 的有效区尺寸与影像不一致")
            valid &= per_band

    num, den = _terms_from_arrays(arrays, key)
    valid &= np.isfinite(num) & np.isfinite(den) & (np.abs(den) > DENOM_EPS)

    scl_available = False
    scl = scene.meta.get("scl") if isinstance(scene.meta, dict) else None
    if scl is not None:
        scl_arr = np.asarray(scl)
        if scl_arr.shape[:2] == shape:
            scl_available = True
            valid &= ~np.isin(scl_arr, SCL_INVALID_CLASSES)
    return valid, scl_available


def metric_pixel_area_m2(transform: Any, crs: Any) -> Tuple[Optional[float], str]:
    """仅在米制投影且行列式可计算时返回投影网格像元面积（m²），否则 ``(None, 原因)``。"""
    if transform is None:
        return None, "缺少 transform，无法计算投影网格像元面积"
    if crs is None:
        return None, "缺少 CRS，无法判定是否为米制投影，面积不输出 km²"
    try:
        from rasterio.crs import CRS

        parsed = CRS.from_user_input(crs)
    except Exception:
        return None, "CRS 无法解析，面积不输出 km²"
    if not bool(parsed.is_projected):
        return None, f"CRS {parsed.to_string()} 是地理坐标系而非投影坐标系，面积不输出 km²"
    units = (parsed.linear_units or "").strip().lower()
    if units not in _METER_UNITS:
        return None, f"投影单位为 {parsed.linear_units!r}（非米制），面积不输出 km²"
    try:
        a = float(transform.a)
        b = float(transform.b)
        d = float(transform.d)
        e = float(transform.e)
    except Exception:
        return None, "transform 无法读取行列式，面积不输出 km²"
    det = a * e - b * d
    if not math.isfinite(det) or abs(det) <= 0.0:
        return None, "transform 行列式退化（像元面积为零），面积不输出 km²"
    area = abs(det)
    return area, f"米制投影 {parsed.to_string()}，像元面积 {area:.6g} m²"


# --------------------------------------------------------------------------
# 加载与对齐
# --------------------------------------------------------------------------


def _quality_sidecar(path: str, scene: Scene) -> Optional[Dict[str, str]]:
    """核对同名 SCL。GeoTIFF 必须与主影像同网格；PNG 只能确认尺寸。"""
    import rasterio
    from PIL import Image

    stem, _ = os.path.splitext(path)
    for suffix in ("_scl.tif", "_scl.tiff", "_scl.png"):
        candidate = stem + suffix
        if not os.path.isfile(candidate):
            continue
        if suffix.endswith(".png"):
            with Image.open(candidate) as im:
                if im.size != (scene.width, scene.height):
                    raise ValueError(f"SCL PNG 尺寸与影像不一致：{candidate}")
            alignment = "pixel_order_only"
        else:
            with rasterio.open(candidate) as ds:
                if (ds.width, ds.height) != (scene.width, scene.height):
                    raise ValueError(f"SCL GeoTIFF 尺寸与影像不一致：{candidate}")
                if ds.crs != scene.crs or ds.transform != scene.transform:
                    raise ValueError(f"SCL GeoTIFF 网格/CRS 与影像不一致：{candidate}")
            alignment = "verified_grid"
        if scene.meta.get("scl") is None:
            raise ValueError(f"SCL 文件存在但未能读取质量类别：{candidate}")
        return {"path": os.path.abspath(candidate), "sha256": sha256_file(candidate),
                "alignment": alignment}
    return None


def _same_grid_exact(source: Scene, reference: Scene) -> bool:
    """地理原点不能使用相对容差：投影坐标很大时 1m 偏移也要重采样。"""
    from rasterio.crs import CRS

    return (source.shape == reference.shape
            and CRS.from_user_input(source.crs) == CRS.from_user_input(reference.crs)
            and source.transform == reference.transform)


def _align_scene_safe(source: Scene, reference: Scene) -> Scene:
    """按有效权重重采样反射率，避免单波段 NoData 被双线性混入邻域。"""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    shape = reference.shape
    source_quality = source.meta.get("spectral_band_valid") or {}

    def warp(arr: np.ndarray, resampling: Resampling) -> np.ndarray:
        dst = np.zeros(shape, dtype=np.float32)
        reproject(np.asarray(arr, dtype=np.float32), dst,
                  src_transform=source.transform, src_crs=source.crs,
                  dst_transform=reference.transform, dst_crs=reference.crs,
                  src_nodata=None, dst_nodata=0.0, resampling=resampling)
        return dst

    coverage = warp(np.ones(source.shape, dtype=np.float32), Resampling.nearest) >= 0.5
    if source.nodata_mask is not None:
        coverage &= warp((~np.asarray(source.nodata_mask, dtype=bool)).astype(np.float32),
                         Resampling.nearest) >= 0.5
    bands: Dict[str, np.ndarray] = {}
    warped_quality: Dict[str, np.ndarray] = {}
    for name, band in source.bands.items():
        valid = np.asarray(source_quality.get(name, np.ones(source.shape, dtype=bool)), dtype=bool).copy()
        valid &= np.isfinite(band)
        if source.nodata_mask is not None:
            valid &= ~np.asarray(source.nodata_mask, dtype=bool)
        nearest = warp(valid.astype(np.float32), Resampling.nearest) >= 0.5
        weight = warp(valid.astype(np.float32), Resampling.bilinear)
        numerator = warp(np.where(valid, band, 0.0), Resampling.bilinear)
        good = coverage & nearest & (weight > 1e-6)
        out = np.zeros(shape, dtype=np.float32)
        np.divide(numerator, weight, out=out, where=good)
        bands[name] = out
        warped_quality[name] = good

    meta = dict(source.meta)
    meta["spectral_band_valid"] = warped_quality
    if meta.get("scl") is not None:
        meta["scl"] = warp(np.asarray(meta["scl"], dtype=np.float32), Resampling.nearest).astype(np.uint8)
    meta["reprojected_to"] = reference.path or "reference"
    return Scene(bands=bands, transform=reference.transform, crs=reference.crs,
                 path=source.path, pixel_size_m=reference.pixel_size_m,
                 nodata_mask=~coverage, meta=meta)


def load_scenes(
    paths: Sequence[str],
    index: str,
    band_order: str = "auto",
) -> List[Scene]:
    """读取本地 GeoTIFF 并校验指数所需真实波段（缺波段直接报错）。"""
    key = _normalize_index(index)
    import rasterio

    scenes: List[Scene] = []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"影像不存在：{path}")
        if os.path.splitext(path)[1].lower() not in _GEOTIFF_EXTS:
            raise ValueError(f"仅支持本地 GeoTIFF（.tif/.tiff/.vrt/.img），收到：{path}")
        scene = load_scene(path, band_order=band_order)
        missing = [b for b in REQUIRED_BANDS[key] if b not in scene.bands]
        if missing:
            raise ValueError(
                f"场景 {os.path.basename(path)} 缺少指数 {key} 所需真实波段 {missing}"
                f"（解析出：{sorted(scene.bands)}）。"
                "禁止使用无 NIR 的代理波段或其他替代波段。"
            )
        band_map = scene.meta.get("band_map") or {}
        with rasterio.open(path) as ds:
            scene.meta["spectral_band_valid"] = {
                name: ds.read_masks(int(band_map[name]) + 1) > 0
                for name in scene.bands if name in band_map
            }
        scene.meta["quality_sidecar"] = _quality_sidecar(path, scene)
        scenes.append(scene)
    return scenes


def align_scenes(scenes: Sequence[Scene]) -> List[Scene]:
    """全部场景相对首景对齐到同一网格。缺 CRS/transform 明确报错。"""
    if not scenes:
        raise ValueError("至少需要一景影像才能建立参考网格")
    ref = scenes[0]
    if ref.transform is None or ref.crs is None:
        raise ValueError(
            f"首景缺少 {'transform' if ref.transform is None else ''}"
            f"{'/' if ref.transform is None and ref.crs is None else ''}"
            f"{'CRS' if ref.crs is None else ''}，无法建立参考网格：{ref.path or ref.name}"
        )
    aligned = [ref]
    for scene in scenes[1:]:
        if scene.transform is None or scene.crs is None:
            raise ValueError(
                f"场景 {scene.path or scene.name} 缺少 "
                f"{'transform' if scene.transform is None else 'CRS'}，无法对齐到首景网格"
            )
        if _same_grid_exact(scene, ref):
            aligned.append(scene)
        else:
            aligned.append(_align_scene_safe(scene, ref))
    return aligned


# --------------------------------------------------------------------------
# 结果数据结构
# --------------------------------------------------------------------------


@dataclass
class SceneIndex:
    """单景指数结果与质量信息。"""

    order: int
    path: str
    name: str
    index: str
    data: np.ndarray            # float32，无效 = INDEX_NODATA
    valid: np.ndarray           # bool
    valid_pixels: int
    total_pixels: int
    valid_pct: float
    mean: Optional[float]       # 低于门禁或全无效时为 None，绝不写 0
    status: str                 # "ok" | "missing"
    reason: str
    date: Optional[str]
    date_source: str
    sha256: str
    crs: str
    transform: Optional[List[float]]
    width: int
    height: int
    reprojected: bool
    scl_available: bool
    quality_sidecar: Optional[Dict[str, str]] = None
    warnings: List[str] = field(default_factory=list)
    index_tif: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "order": int(self.order),
            "path": self.path,
            "name": self.name,
            "date": self.date,
            "date_source": self.date_source,
            "sha256": self.sha256,
            "index": self.index,
            "crs": self.crs,
            "transform": self.transform,
            "width": int(self.width),
            "height": int(self.height),
            "reprojected_to_reference": bool(self.reprojected),
            "scl_available": bool(self.scl_available),
            "quality_sidecar": self.quality_sidecar,
            "valid_pixels": int(self.valid_pixels),
            "total_pixels": int(self.total_pixels),
            "valid_pct": _round(self.valid_pct, 6),
            "mean": _round(self.mean, 6),
            "status": self.status,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "index_tif": self.index_tif,
            "index_tif_name": os.path.basename(self.index_tif) if self.index_tif else None,
        }


@dataclass
class ChangeResult:
    """相邻两景的变化结果与质量信息。"""

    left: SceneIndex
    right: SceneIndex
    common_valid: np.ndarray    # uint8 0/1，0 是真实类别（无 nodata）
    diff: np.ndarray            # float32，无效 = INDEX_NODATA
    common_valid_pixels: int
    total_pixels: int
    common_valid_pct: float
    mean_change: Optional[float]   # 门禁未过或共同有效区为空时为 None
    status: str                    # "ok" | "insufficient"
    reason: str
    observable_area_km2: Optional[float]
    area_note: str
    change_tif: Optional[str] = None
    common_valid_tif: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "from_order": int(self.left.order),
            "to_order": int(self.right.order),
            "from_path": self.left.path,
            "to_path": self.right.path,
            "from_date": self.left.date,
            "to_date": self.right.date,
            "common_valid_pixels": int(self.common_valid_pixels),
            "total_pixels": int(self.total_pixels),
            "common_valid_pct": _round(self.common_valid_pct, 6),
            "mean_change": _round(self.mean_change, 6),
            "status": self.status,
            "reason": self.reason,
            "observable_area_km2": _round(self.observable_area_km2, 9),
            "observable_area_basis": (
                "共同有效区像元数 × 投影网格像元面积"
                if self.observable_area_km2 is not None else None
            ),
            "area_note": self.area_note,
            "change_tif": self.change_tif,
            "change_tif_name": os.path.basename(self.change_tif) if self.change_tif else None,
            "common_valid_tif": self.common_valid_tif,
            "common_valid_tif_name": (
                os.path.basename(self.common_valid_tif) if self.common_valid_tif else None
            ),
        }


# --------------------------------------------------------------------------
# 单景 / 相邻变化
# --------------------------------------------------------------------------


def build_scene_index(
    scene: Scene,
    index: str,
    min_valid_pct: float,
    *,
    order: int = 0,
    date: Optional[str] = None,
    date_source: str = "unverified",
    sha256: Optional[str] = None,
    reprojected: bool = False,
) -> SceneIndex:
    """计算单景指数并按 ``min_valid_pct`` 门禁决定均值是否报告。"""
    key = _normalize_index(index)
    threshold = float(min_valid_pct)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
        raise ValueError(f"min_valid_pct 必须在 0~100 之间，收到 {min_valid_pct}")

    valid, scl_available = scene_valid_mask(scene, key)
    data = compute_masked_index(scene.bands, key, valid)

    total = int(valid.size)
    valid_pixels = int(np.count_nonzero(valid))
    valid_pct = (100.0 * valid_pixels / total) if total else 0.0

    warnings: List[str] = []
    if not scl_available:
        warnings.append("缺少 SCL sidecar，云/云影/雪/饱和像元未纳入有效区筛选")
    sidecar = scene.meta.get("quality_sidecar")
    if sidecar and sidecar.get("alignment") == "pixel_order_only":
        warnings.append("SCL PNG 只有像素顺序，无独立地理参考；请核验其与影像逐像元对齐")
    if (str(scene.meta.get("band_order", "")).startswith("auto")
            and not any(scene.meta.get("band_descriptions") or [])):
        warnings.append("影像未提供波段描述；自动波段顺序按通道数推断，请核验实际波段")

    if valid_pixels == 0:
        mean: Optional[float] = None
        status = "missing"
        reason = "无有效像元，均值报告为 null（不写成 0）"
    elif valid_pct < threshold:
        mean = None
        status = "missing"
        reason = f"有效比例 {valid_pct:.4g}% 低于 min_valid_pct={threshold:g}%，均值报告为 null"
    else:
        mean = float(data[valid].mean())
        status = "ok"
        reason = ""

    digest = sha256 if sha256 is not None else sha256_file(scene.path)

    return SceneIndex(
        order=int(order),
        path=scene.path,
        name=scene.name,
        index=key,
        data=data,
        valid=valid,
        valid_pixels=valid_pixels,
        total_pixels=total,
        valid_pct=valid_pct,
        mean=mean,
        status=status,
        reason=reason,
        date=date,
        date_source=date_source,
        sha256=digest,
        crs=str(scene.crs),
        transform=_transform_list(scene.transform),
        width=int(scene.width),
        height=int(scene.height),
        reprojected=bool(reprojected),
        scl_available=scl_available,
        quality_sidecar=sidecar,
        warnings=warnings,
    )


def compare_scene_indices(
    left: SceneIndex,
    right: SceneIndex,
    pixel_area_m2: Optional[float] = None,
    area_note: str = "",
    min_common_valid_pct: float = 0.0,
) -> ChangeResult:
    """只在相邻两景共同有效区计算差值；门禁未过或共同区为空时报告 null。"""
    if left.index != right.index:
        raise ValueError(f"指数不一致：{left.index} vs {right.index}")
    if left.valid.shape != right.valid.shape:
        raise ValueError(
            f"两景网格不一致（{left.valid.shape} vs {right.valid.shape}），无法比较"
        )

    common = left.valid & right.valid
    shape = common.shape
    diff = np.full(shape, INDEX_NODATA, dtype=np.float32)
    np.subtract(right.data, left.data, out=diff, where=common)
    diff[~common] = INDEX_NODATA

    total = int(common.size)
    common_pixels = int(np.count_nonzero(common))
    common_pct = (100.0 * common_pixels / total) if total else 0.0
    common_u8 = common.astype(np.uint8)

    reasons: List[str] = []
    if common_pixels == 0:
        reasons.append("相邻两景共同有效区为空，变化未计算")
    elif common_pct < min_common_valid_pct:
        reasons.append(
            f"共同有效比例 {common_pct:.4g}% 低于 min_valid_pct={min_common_valid_pct:g}%"
        )
    if left.status != "ok":
        reasons.append(
            f"前一时相（序 {left.order}）有效比例 {left.valid_pct:.4g}% 低于门禁或缺测"
        )
    if right.status != "ok":
        reasons.append(
            f"后一时相（序 {right.order}）有效比例 {right.valid_pct:.4g}% 低于门禁或缺测"
        )

    computed = (common_pixels > 0 and common_pct >= min_common_valid_pct
                and left.status == "ok" and right.status == "ok")
    mean_change = float(diff[common].mean()) if computed else None
    if not computed:
        diff.fill(INDEX_NODATA)
    status = "ok" if computed else "insufficient"
    reason = "；".join(reasons) if reasons else ""

    if common_pixels == 0:
        area: Optional[float] = None
        note = "共同有效区为空，共同有效面积报告为 null（不写成 0）"
    elif not computed:
        area = None
        note = "有效观测未通过门禁，共同有效面积报告为 null"
    elif pixel_area_m2 is None:
        area = None
        note = area_note or "无法计算投影网格像元面积，面积不输出 km²"
    else:
        area = common_pixels * float(pixel_area_m2) / 1_000_000.0
        note = f"{area_note}；共同有效面积 = {common_pixels} 像元 × {float(pixel_area_m2):.6g} m²"

    return ChangeResult(
        left=left,
        right=right,
        common_valid=common_u8,
        diff=diff,
        common_valid_pixels=common_pixels,
        total_pixels=total,
        common_valid_pct=common_pct,
        mean_change=mean_change,
        status=status,
        reason=reason,
        observable_area_km2=area,
        area_note=note,
    )


# --------------------------------------------------------------------------
# 栅格写出
# --------------------------------------------------------------------------


def _write_float32_tif(
    path: str,
    arr: np.ndarray,
    transform: Any,
    crs: Any,
    nodata: float = INDEX_NODATA,
) -> str:
    import rasterio

    data = np.asarray(arr, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"只写出二维栅格，收到 shape={data.shape}")
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float(nodata),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)
    return path


def _write_uint8_tif(path: str, arr: np.ndarray, transform: Any, crs: Any) -> str:
    """共同有效区 0/1 栅格。0 是真实类别，因此**不声明 nodata**。"""
    import rasterio

    data = np.asarray(arr, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError(f"只写出二维栅格，收到 shape={data.shape}")
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)
    return path


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def run_monitor(
    images: Sequence[str],
    index: str,
    out_dir: str,
    *,
    min_valid_pct: float = 10.0,
    dates: Optional[Sequence[str]] = None,
    band_order: str = "auto",
) -> Dict[str, Any]:
    """执行多时相光谱监测，写出栅格与 ``summary.json``，返回摘要字典。

    参数
    ----
    images : 本地 GeoTIFF 路径序列（按时间先后给出）
    index : ``ndvi`` / ``savi`` / ``ndwi``
    out_dir : 输出目录
    min_valid_pct : 单景及相邻共同有效比例门禁（0~100）
    dates : 可选，与 ``images`` 等长的日期字符串；缺省则标记 unverified
    band_order : 传给 ``load_scene`` 的波段顺序，默认 ``auto``
    """
    key = _normalize_index(index)
    if isinstance(images, (str, os.PathLike)):
        images = [images]
    paths = [os.path.abspath(os.fspath(p)) for p in images]
    if not paths:
        raise ValueError("至少需要一景本地 GeoTIFF 影像（--images）")

    threshold = float(min_valid_pct)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
        raise ValueError(f"min_valid_pct 必须在 0~100 之间，收到 {min_valid_pct}")

    if dates is not None:
        dates = [str(d) for d in dates]
        if len(dates) != len(paths):
            raise ValueError(
                f"日期数量（{len(dates)}）与影像数量（{len(paths)}）不一致"
            )
        try:
            parsed_dates = [calendar_date.fromisoformat(d) for d in dates]
        except ValueError as exc:
            raise ValueError("日期必须是合法的 YYYY-MM-DD") from exc
        if any(d.isoformat() != raw for d, raw in zip(parsed_dates, dates)):
            raise ValueError("日期必须是规范的 YYYY-MM-DD")
        if any(a >= b for a, b in zip(parsed_dates, parsed_dates[1:])):
            raise ValueError("--dates 必须按影像时间严格递增")

    out_dir = os.path.abspath(os.fspath(out_dir))
    scenes = load_scenes(paths, key, band_order=band_order)
    aligned = align_scenes(scenes)
    ref = aligned[0]

    pixel_area_m2, area_note = metric_pixel_area_m2(ref.transform, ref.crs)

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    try:
        os.mkdir(out_dir)
    except FileExistsError as exc:
        raise FileExistsError(f"输出目录必须是新目录，避免覆盖或混入旧成果：{out_dir}") from exc

    global_warnings: List[str] = []
    scene_results: List[SceneIndex] = []
    for i, scene in enumerate(aligned):
        scene_date = dates[i] if dates is not None else None
        date_source = "user_provided" if dates is not None else "unverified"
        result = build_scene_index(
            scene,
            key,
            threshold,
            order=i,
            date=scene_date,
            date_source=date_source,
            reprojected=("reprojected_to" in (scene.meta or {})),
        )
        filename = f"scene_{i:02d}_{_safe_stem(scene.path)}_{key}.tif"
        result.index_tif = _write_float32_tif(
            os.path.join(out_dir, filename), result.data, scene.transform, scene.crs
        )
        scene_results.append(result)
        global_warnings.extend(f"场景 {os.path.basename(scene.path)}：{w}" for w in result.warnings)

    change_results: List[ChangeResult] = []
    for i in range(len(aligned) - 1):
        change = compare_scene_indices(
            scene_results[i], scene_results[i + 1], pixel_area_m2, area_note,
            min_common_valid_pct=threshold,
        )
        change.change_tif = _write_float32_tif(
            os.path.join(out_dir, f"change_{i:02d}_{i + 1:02d}_{key}.tif"),
            change.diff,
            ref.transform,
            ref.crs,
        )
        change.common_valid_tif = _write_uint8_tif(
            os.path.join(out_dir, f"common_valid_{i:02d}_{i + 1:02d}.tif"),
            change.common_valid,
            ref.transform,
            ref.crs,
        )
        change_results.append(change)

    if len(aligned) < 2:
        global_warnings.append("仅一景输入，没有相邻变化结果")

    summary: Dict[str, Any] = {
        "tool": "spectral_monitor",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index": key,
        "formula": INDEX_FORMULAS[key],
        "required_bands": list(REQUIRED_BANDS[key]),
        "min_valid_pct": threshold,
        "scl_invalid_classes": list(SCL_INVALID_CLASSES),
        "invalid_nodata": INDEX_NODATA,
        "denominator_epsilon": DENOM_EPS,
        "band_order": band_order,
        "dates_user_provided": dates is not None,
        "dates_unverified": True,
        "dates_verified_against_source": False,
        "dates_guessed_from_filename": False,
        "date_source": "user_provided" if dates is not None else "unverified",
        "date_note": _USER_DATE_NOTE if dates is not None else _UNVERIFIED_NOTE,
        "out_dir": out_dir,
        "reference_grid": {
            "scene_order": 0,
            "path": ref.path,
            "crs": str(ref.crs),
            "transform": _transform_list(ref.transform),
            "width": int(ref.width),
            "height": int(ref.height),
        },
        "area": {
            "available": pixel_area_m2 is not None,
            "pixel_area_m2": _round(pixel_area_m2, 9),
            "pixel_area_km2": _round(
                pixel_area_m2 / 1_000_000.0 if pixel_area_m2 is not None else None, 12
            ),
            "note": area_note,
        },
        "scenes": [s.to_summary() for s in scene_results],
        "changes": [c.to_summary() for c in change_results],
        "warnings": global_warnings,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    summary["summary_json"] = summary_path
    temp_path = summary_path + f".tmp.{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(temp_path, summary_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return summary


def format_summary_text(summary: Mapping[str, Any]) -> str:
    """把摘要渲染成便于终端阅读的短文本（不替代 summary.json）。"""
    lines: List[str] = []
    lines.append(f"指数：{summary.get('index')}  公式：{summary.get('formula')}")
    lines.append(
        f"门禁：min_valid_pct = {summary.get('min_valid_pct')}%  "
        f"无效值：{summary.get('invalid_nodata')}"
    )
    grid = summary.get("reference_grid", {})
    lines.append(
        f"参考网格：{os.path.basename(str(grid.get('path', '')))} "
        f"({grid.get('crs')}, {grid.get('width')}x{grid.get('height')})"
    )
    lines.append(
        "日期：" + ("用户显式提供（未与元数据核验）" if summary.get("dates_user_provided")
                    else "未核验（未从文件名推断）")
    )

    lines.append("[每期]")
    for scene in summary.get("scenes", []):
        mean = scene.get("mean")
        mean_text = "null" if mean is None else f"{mean:.6g}"
        lines.append(
            f"  {scene.get('order'):02d}  {scene.get('name')}  "
            f"有效 {scene.get('valid_pct'):.4g}%  均值 {mean_text}  "
            f"状态 {scene.get('status')}"
            + (f"  原因：{scene.get('reason')}" if scene.get("reason") else "")
        )

    lines.append("[相邻变化]")
    if not summary.get("changes"):
        lines.append("  （无）")
    for change in summary.get("changes", []):
        mean = change.get("mean_change")
        mean_text = "null" if mean is None else f"{mean:.6g}"
        area = change.get("observable_area_km2")
        area_text = "null" if area is None else f"{area:.6g} km²"
        lines.append(
            f"  {change.get('from_order'):02d}->{change.get('to_order'):02d}  "
            f"共同有效 {change.get('common_valid_pct'):.4g}%  "
            f"平均变化 {mean_text}  共同有效面积 {area_text}  "
            f"状态 {change.get('status')}"
        )
        if area is None and change.get("area_note"):
            lines.append(f"      面积说明：{change.get('area_note')}")

    for warning in summary.get("warnings", []):
        lines.append(f"[警告] {warning}")
    return "\n".join(lines)


__all__ = [
    "INDEX_FORMULAS",
    "REQUIRED_BANDS",
    "SCL_INVALID_CLASSES",
    "INDEX_NODATA",
    "DENOM_EPS",
    "SceneIndex",
    "ChangeResult",
    "supported_indices",
    "required_bands",
    "sha256_file",
    "index_terms",
    "compute_index",
    "compute_masked_index",
    "scene_valid_mask",
    "metric_pixel_area_m2",
    "load_scenes",
    "align_scenes",
    "build_scene_index",
    "compare_scene_indices",
    "run_monitor",
    "format_summary_text",
]
