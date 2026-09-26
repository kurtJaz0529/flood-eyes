"""Opt-in, evidence-based optical water recipes; defaults require local validation.

Scores are spectral decision scores, not calibrated probabilities. Review pixels
are excluded from comparable observations, never converted into dry land.
"""
from __future__ import annotations

import numpy as np

from . import baseline
from .terrain import assess_dem, terrain_context

RECIPE_VERSION = 1
SWIR_PROFILES = frozenset({"urban", "coastal", "arid"})
RELIEF_PROFILES = frozenset({"hilly", "mountain"})
SHADOW_PROFILES = RELIEF_PROFILES | {"urban"}


def select_index(profile, scenes, requested="auto"):
    terrain_context(profile)  # validate declared context
    if requested not in ("auto", "ndwi", "mndwi"):
        raise ValueError("water_index 需为 auto/ndwi/mndwi")
    for scene in scenes:
        if not {"green", "nir"}.issubset(scene.bands):
            raise ValueError("场景适配洪水识别需要真实 green 和 nir 波段，禁止代理波段")
    preferred = "mndwi" if profile in SWIR_PROFILES else "ndwi"
    choice = preferred if requested == "auto" else requested
    missing = any("swir1" not in scene.bands for scene in scenes)
    if choice == "mndwi" and missing:
        if requested != "auto":
            raise ValueError("MNDWI 需要每个时相都具有真实 swir1 波段")
        return "ndwi", ["所选场景优先使用 MNDWI，但至少一个时相缺少 SWIR1；两期统一降级为 NDWI。"]
    return choice, []


def morphology(profile, *, min_area_px, open_radius, close_radius, fill_holes):
    """Preserve narrow channels/street water, including smaller user area limits."""
    if profile in RELIEF_PROFILES | {"wetland", "coastal", "urban"}:
        return dict(min_area_px=min(min_area_px, 9), open_radius=0,
                    close_radius=0, fill_holes=False)
    return dict(min_area_px=min_area_px, open_radius=open_radius,
                close_radius=close_radius, fill_holes=fill_holes)


def predict(scene, profile, valid, *, index, fixed_threshold=None,
            softness=0.05, dem_path=None, slope_threshold_deg=15.0):
    required = ("green", "nir", "swir1") if index == "mndwi" else ("green", "nir")
    usable = np.asarray(valid, dtype=bool).copy()
    quality = scene.meta.get("spectral_band_valid") or {}
    for name in required:
        band = scene.bands[name]
        usable &= np.isfinite(band)
        if name in quality:
            usable &= np.asarray(quality[name], dtype=bool)
    other = scene.bands["swir1" if index == "mndwi" else "nir"]
    den = scene.bands["green"] + other
    usable &= np.isfinite(den) & (np.abs(den) > 1e-8)
    values = np.full(scene.shape, np.nan, dtype=np.float32)
    np.divide(scene.bands["green"] - other, den, out=values, where=usable)
    method = "local" if profile in RELIEF_PROFILES else "hybrid"
    _, prob, meta = baseline.predict(
        values, method=method, fixed_threshold=fixed_threshold,
        nodata_mask=~usable, nir=scene.bands["nir"], softness=softness,
    )
    candidate = (prob >= 0.5) & usable
    review = np.zeros(scene.shape, dtype=np.uint8)
    warnings = list(terrain_context(profile)["warnings"])
    evidence = []
    if profile in SHADOW_PROFILES:
        names = ("blue", "green", "nir", "swir1", "swir2")
        if all(name in scene.bands for name in names):
            good = usable.copy()
            for name in names:
                good &= np.isfinite(scene.bands[name])
                if name in quality:
                    good &= np.asarray(quality[name], dtype=bool)
            # Feyisa et al. AWEI_sh: an additional shadow-sensitive index.
            b = scene.bands
            awei = b["blue"] + 2.5 * b["green"] - 1.5 * (b["nir"] + b["swir1"]) - 0.25 * b["swir2"]
            review[candidate & good & (awei <= 0)] |= 1
            review[candidate & ~good] |= 4
            evidence.append("awei_sh")
        else:
            warnings.append("缺少 AWEI 阴影交叉核查所需波段；建筑/山体阴影仍可能误判。")
    if dem_path is not None:
        risk, summary = assess_dem(dem_path, scene, slope_threshold_deg)
        meta["terrain_risk_mask"] = risk
        meta["terrain_risk_summary"] = summary
        if profile in RELIEF_PROFILES:
            review[candidate & (risk == 1)] |= 2
            review[candidate & (risk == 255)] |= 4
            evidence.append("dem_slope_screening")
    elif profile in RELIEF_PROFILES:
        warnings.append("未提供 DEM，未执行坡度核查；当前山地/丘陵结果为降级光谱识别。")
    review[~usable] = 255
    review_pixels = usable & (review != 0)
    comparable = usable & ~review_pixels
    prob[~comparable] = 0.0
    meta["review_mask"] = review
    meta["adaptive"] = {
        "recipe_version": RECIPE_VERSION, "terrain_profile": profile,
        "water_index": index, "threshold_method": method,
        "evidence": evidence, "review_pixels": int(review_pixels.sum()),
        "observed_pixels": int(usable.sum()),
        "review_codes": {"0": "无需本规则复核", "1": "AWEI阴影证据冲突",
                         "2": "DEM陡坡候选水体", "4": "辅助证据无数据",
                         "255": "光学无效；1/2/4可按位组合"},
        "calibration": "未作区域真值标定；默认参数为可复现的实验配方",
        "score_semantics": "光谱判别分数，非标定概率",
    }
    meta["adaptive_warnings"] = warnings
    meta["threshold_desc"] = baseline.describe_threshold(meta).replace("NDWI", index.upper())
    return prob, meta, comparable
