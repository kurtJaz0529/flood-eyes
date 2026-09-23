"""
慧眼识灾 · 地形背景与 DEM 坡度筛查
==================================

本模块只做两件事，且都保持"用户声明 / 人工复核"的边界：

1. ``terrain_context(profile)``：把用户声明的地形背景翻译成提示语。它**不会**
   根据背景去修改水体阈值或掩膜——地形背景只用于提示复核重点。
2. ``assess_dem(dem_path, scene, slope_threshold_deg)``：把本地 DEM 重投影到
   影像网格，计算坡度并输出"需复核"标记。坡度阈值只是筛查阈值，不是通用水文
   淹没判据；坡度标记仅供应急人员复核。

风险等级：0=低（坡度 < 阈值），1=需复核（坡度 ≥ 阈值），255=未评估
（DEM 无数据/覆盖范围外、影像无效、或缺少有效邻域像元）。未评估**不会**
被当成 0° 缓坡处理。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np

UNASSESSED = 255

# 允许的地形背景；键 -> (中文标签, 提示语)
_PROFILES: Dict[str, Tuple[str, List[str]]] = {
    "unspecified": (
        "未声明地形背景",
        ["未声明地形背景：地形阴影、城市建筑阴影和潮汐水面都可能造成误判，请结合地形与影像时相人工复核。"],
    ),
    "plain": (
        "平原/河湖周边",
        ["平原区永久水体、季节性水面和养殖塘在汛期前即已存在，需与灾前影像逐期对比，避免把季节性水面当成新增淹没。"],
    ),
    "hilly": (
        "丘陵",
        ["丘陵区地形阴影与坡面几何会改变反射率/后向散射，阴坡可能被误判为水体；请结合灾前影像与坡向复核边界。"],
    ),
    "mountain": (
        "山地",
        ["山地区地形阴影、陡坡几何与雷达叠掩/透视收缩会扭曲水体判读；坡面投影还会放大面积误差，山地结果需人工复核。"],
    ),
    "urban": (
        "城市",
        ["城市建筑阴影与屋顶低反射区可能被误判为积水；内涝范围需结合建筑高度与街谷阴影复核，不能只看光谱掩膜。"],
    ),
    "coastal": (
        "沿海/感潮河段",
        ["沿海潮汐与风暴潮会让同一位置在不同时相呈现水陆变化；比较时应记录潮位，优先使用同一潮位附近的影像。"],
    ),
}

_SCOPE_NOTE = "地形背景为用户声明，仅用于提示复核重点；系统不会据此修改水体阈值或掩膜。"


def terrain_context(profile: str = "unspecified") -> Dict[str, Any]:
    """把用户声明的地形背景转成 JSON 安全的结果元数据。

    允许的取值：unspecified / plain / hilly / mountain / urban / coastal。
    返回 dict 含 profile、label、warnings、dem_assessed=False（声明阶段不做 DEM 评估）。
    """
    key = str(profile if profile is not None else "unspecified").strip().lower()
    if key not in _PROFILES:
        raise ValueError(f"未知地形背景 '{profile}'，可选：{sorted(_PROFILES)}")
    label, warnings = _PROFILES[key]
    return {
        "profile": key,
        "label": label,
        "warnings": list(warnings),
        "dem_assessed": False,
        "scope": "user_declared_only",
        "note": _SCOPE_NOTE,
    }


def _require_projected_metre(crs: Any) -> str:
    """坡度需要米制投影坐标；返回单位名用于摘要。"""
    try:
        from rasterio.crs import CRS

        c = crs if hasattr(crs, "is_projected") else CRS.from_user_input(crs)
    except Exception as exc:  # pragma: no cover - 防御性分支
        raise ValueError(f"无法解析影像坐标系：{crs!r}（{exc}）") from exc
    if not bool(c.is_projected):
        raise ValueError("DEM 坡度评估需要投影坐标系（米），当前影像为地理坐标系；请先投影到米制 CRS")
    factor = None
    try:
        factor = float(c.linear_units_factor[1])
    except Exception:
        factor = None
    if factor is not None:
        if abs(factor - 1.0) > 1e-6:
            raise ValueError("DEM 坡度评估需要米制投影坐标系，当前线性单位为非米")
    elif "met" not in str(getattr(c, "linear_units", "") or "").lower():
        raise ValueError("DEM 坡度评估需要米制投影坐标系")
    return str(getattr(c, "linear_units", "") or "metre")


def _require_north_up(transform: Any) -> None:
    """旋转/错切网格上按行列差分算梯度会得到错误方向，直接拒绝。"""
    b = float(getattr(transform, "b", 0.0) or 0.0)
    d = float(getattr(transform, "d", 0.0) or 0.0)
    if abs(b) > 1e-9 or abs(d) > 1e-9:
        raise ValueError("影像网格存在旋转/错切，坡度计算需要北向正置（north-up）网格")


def _slope_degrees(z: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """中心差分坡度（度）。任一参与差分的邻域像元无效（NaN）则该像元返回 NaN。

    只在有限邻域上做差分：无数据/覆盖范围外不会被当作 0° 缓坡。
    """
    h, w = z.shape
    finite = np.isfinite(z)
    gx = np.full((h, w), np.nan, dtype=np.float64)
    gy = np.full((h, w), np.nan, dtype=np.float64)
    if w >= 3:
        # 中心像元本身也必须有限：否则无数据点会被相邻有效点"插值成缓坡"
        ok = finite[:, 2:] & finite[:, :-2] & finite[:, 1:-1]
        gx[:, 1:-1] = np.where(ok, (z[:, 2:] - z[:, :-2]) / (2.0 * dx), np.nan)
    if w >= 2:
        gx[:, 0] = np.where(finite[:, 1] & finite[:, 0], (z[:, 1] - z[:, 0]) / dx, np.nan)
        gx[:, -1] = np.where(finite[:, -1] & finite[:, -2], (z[:, -1] - z[:, -2]) / dx, np.nan)
    if h >= 3:
        ok = finite[2:, :] & finite[:-2, :] & finite[1:-1, :]
        gy[1:-1, :] = np.where(ok, (z[2:, :] - z[:-2, :]) / (2.0 * dy), np.nan)
    if h >= 2:
        gy[0, :] = np.where(finite[1, :] & finite[0, :], (z[1, :] - z[0, :]) / dy, np.nan)
        gy[-1, :] = np.where(finite[-1, :] & finite[-2, :], (z[-1, :] - z[-2, :]) / dy, np.nan)
    mag = np.hypot(gx, gy)  # 任一维 NaN 会传播，得到未评估
    return np.degrees(np.arctan(mag))


def assess_dem(
    dem_path: str,
    scene: Any,
    slope_threshold_deg: float = 15.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """把本地 DEM 重投影到影像网格并做坡度筛查。

    返回 ``(risk_mask, summary)``：
        risk_mask: uint8，(H, W)，0=低（坡度 < 阈值），1=需复核，255=未评估
        summary:   JSON 安全的统计与免责说明

    DEM 缺失/读取失败会显式抛错，绝不用 0 值伪造地形。
    """
    try:
        threshold = float(slope_threshold_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"坡度阈值必须是数值，收到 {slope_threshold_deg!r}") from exc
    if not np.isfinite(threshold) or not (0.0 <= threshold <= 90.0):
        raise ValueError(f"坡度阈值必须是 0~90 之间的有限数值，收到 {slope_threshold_deg!r}")

    if scene is None:
        raise ValueError("缺少影像对象，无法确定 DEM 的目标网格")
    transform = getattr(scene, "transform", None)
    crs = getattr(scene, "crs", None)
    if transform is None or crs is None:
        raise ValueError("影像缺少 CRS/仿射变换，无法把 DEM 重投影到影像网格")
    linear_unit = _require_projected_metre(crs)
    _require_north_up(transform)

    if not dem_path or not os.path.isfile(dem_path):
        raise FileNotFoundError(f"DEM 文件不存在：{dem_path}")

    try:
        import rasterio
        from rasterio.warp import Resampling, reproject
    except Exception as exc:  # pragma: no cover - 环境缺 rasterio
        raise RuntimeError(f"缺少 rasterio，无法读取 DEM：{exc}") from exc

    try:
        with rasterio.open(dem_path) as ds:
            dem = ds.read(1).astype(np.float64)
            dem_transform, dem_crs, dem_nodata = ds.transform, ds.crs, ds.nodata
            if dem_transform is None or dem_crs is None:
                raise ValueError("DEM 缺少 CRS/仿射变换，无法重投影到影像网格")
            valid = np.isfinite(dem)
            try:
                valid &= ds.read_masks(1) > 0
            except Exception:
                pass
            if dem_nodata is not None:
                nod = float(dem_nodata)
                if np.isnan(nod):
                    valid &= ~np.isnan(dem)
                else:
                    # 固定容差：默认 rtol 会把接近 nodata 的合法高程也误判为无数据
                    valid &= ~np.isclose(dem, nod, rtol=0.0, atol=1e-6)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"无法读取 DEM（{type(exc).__name__}: {exc}）：{dem_path}") from exc

    # 先把 DEM 无效像元置 NaN：否则 nodata 值（如 -9999）会被双线性插值当成真实高程，
    # 在无效区边缘制造假陡坡。
    dem = np.where(valid, dem, np.nan)

    shape = tuple(scene.shape)
    elevation = np.full(shape, np.nan, dtype=np.float32)
    coverage = np.zeros(shape, dtype=np.float32)
    reproject(
        source=dem.astype(np.float32),
        destination=elevation,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    # 覆盖范围用最近邻单独重投影一次：范围外留在 0，不会被插值成"缓坡"
    reproject(
        source=valid.astype(np.float32),
        destination=coverage,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.nearest,
        src_nodata=None,
        dst_nodata=0.0,
    )
    assessed = np.isfinite(elevation) & (coverage >= 0.5)
    scene_nodata = getattr(scene, "nodata_mask", None)
    if scene_nodata is not None:
        mask = np.asarray(scene_nodata, dtype=bool)
        if mask.shape == shape:
            assessed &= ~mask
    z = np.where(assessed, elevation.astype(np.float64), np.nan)

    dx = abs(float(transform.a))
    dy = abs(float(transform.e))
    if not (dx > 0 and dy > 0):
        raise ValueError("影像像元尺寸无效，无法计算坡度")
    slope = _slope_degrees(z, dy, dx)

    risk = np.full(shape, UNASSESSED, dtype=np.uint8)
    ok = np.isfinite(slope)
    risk[ok] = np.where(slope[ok] >= threshold, 1, 0).astype(np.uint8)

    values = slope[ok]
    n_total = int(risk.size)
    n_assessed = int(values.size)
    risk_px = int(np.count_nonzero(risk == 1))
    summary: Dict[str, Any] = {
        "dem_assessed": True,
        "dem_file": os.path.basename(str(dem_path)),
        "crs": str(crs),
        "linear_unit": linear_unit,
        "pixel_size_m": float(0.5 * (dx + dy)),
        "screening_threshold_deg": threshold,
        "coverage_pct": round(100.0 * n_assessed / max(n_total, 1), 2),
        "assessed_pixels": n_assessed,
        "unassessed_pixels": n_total - n_assessed,
        "unassessed_pct": round(100.0 * (n_total - n_assessed) / max(n_total, 1), 2),
        "slope_min_deg": float(values.min()) if n_assessed else None,
        "slope_max_deg": float(values.max()) if n_assessed else None,
        "slope_mean_deg": float(values.mean()) if n_assessed else None,
        "risk_pixels": risk_px,
        "risk_pct_of_assessed": round(100.0 * risk_px / max(n_assessed, 1), 2),
        "risk_area_km2": risk_px * dx * dy / 1_000_000.0,
        "warnings": [
            "坡度标记仅供人工复核：坡度阈值是筛查阈值，不是通用水文淹没判据。",
            "坡度由重投影后的 DEM 与影像像元尺寸计算，DEM 垂直精度与重采样会影响结果。",
        ],
        "note": "risk=1 表示坡度≥阈值需复核，risk=0 表示低坡度，255 表示 DEM/影像未覆盖、无法评估。",
    }
    return risk, summary
