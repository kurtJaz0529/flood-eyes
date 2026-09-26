"""
慧眼识灾 · 基线模型：NDWI + Otsu 自适应阈值
=========================================

为什么要有基线？
    1. 第一天就能演示（无需 GPU、无需训练数据、无需权重）
    2. 作为深度模型的对照实验（论文/答辩里"可解释、有 baseline"是加分项）
    3. 训练数据不足的灾种/传感器上，它是兜底方案

原理：
    NDWI = (Green - NIR) / (Green + NIR)
    水体在 NDWI 上呈明显高值，阈值分割即可得到水体掩膜。
    全局 Otsu 解决"每景影像亮度不同"的问题；
    局部 Otsu 解决"大范围阴影/山地"导致的阈值漂移。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

EPS = 1e-8

# Otsu 结果落在该区间之外时视为失效（例如整景都是陆地/都是水）
THRESH_VALID_RANGE = (-0.35, 0.55)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """一维 Otsu 阈值。返回 nan 表示无法计算（样本过少/全为无效值）。"""
    v = values[np.isfinite(values)]
    if v.size < 64:
        return float("nan")
    try:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(v, nbins=nbins))
    except Exception:
        # 纯 numpy 兜底实现，保证无 skimage 也能跑
        hist, edges = np.histogram(v, bins=nbins)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return float("nan")
        centers = (edges[:-1] + edges[1:]) / 2.0
        w0 = np.cumsum(hist)
        w1 = total - w0
        valid = (w0 > 0) & (w1 > 0)
        if not valid.any():
            return float("nan")
        mu = np.cumsum(hist * centers)
        mu_total = mu[-1]
        mu0 = np.divide(mu, w0, out=np.zeros_like(mu), where=w0 > 0)
        mu1 = np.divide(mu_total - mu, w1, out=np.zeros_like(mu), where=w1 > 0)
        between = w0 * w1 * (mu0 - mu1) ** 2
        between[~valid] = -1.0
        return float(centers[int(np.argmax(between))])


def _box_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """均值滤波（替代 scipy.ndimage.uniform_filter，避免打包时拖入整个 scipy）。"""
    size = int(size)
    if size <= 1:
        return arr
    if size % 2 == 0:
        size += 1
    try:
        import cv2

        return cv2.blur(arr.astype(np.float32), (size, size))
    except Exception:
        pass
    # 纯 numpy 可分离卷积兜底
    k = size // 2
    padded = np.pad(arr.astype(np.float32), k, mode="edge")
    kernel = np.ones(size, dtype=np.float32) / size
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, padded)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, tmp)
    return out.astype(np.float32)


def local_otsu_threshold(ndwi: np.ndarray, block: int = 256, smooth: int = 3) -> np.ndarray:
    """逐块 Otsu -> 平滑成阈值图（每个像元一个阈值）。"""
    h, w = ndwi.shape
    thr_map = np.full((h, w), np.nan, dtype=np.float32)
    stride = max(32, block // 2)
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1, x1 = min(y0 + block, h), min(x0 + block, w)
            patch = ndwi[y0:y1, x0:x1]
            t = otsu_threshold(patch)
            if not np.isfinite(t):
                continue
            if not (THRESH_VALID_RANGE[0] <= t <= THRESH_VALID_RANGE[1]):
                continue
            thr_map[y0:y1, x0:x1] = np.where(
                np.isfinite(thr_map[y0:y1, x0:x1]), thr_map[y0:y1, x0:x1], t
            )
    # 未覆盖区域用全局阈值填充
    global_t = otsu_threshold(ndwi)
    if not np.isfinite(global_t):
        global_t = 0.0
    thr_map = np.where(np.isfinite(thr_map), thr_map, global_t).astype(np.float32)
    if smooth and smooth > 1:
        thr_map = _box_filter(thr_map, smooth).astype(np.float32)
    return thr_map


def predict(
    ndwi: np.ndarray,
    method: str = "hybrid",
    fixed_threshold: Optional[float] = None,
    block: int = 256,
    softness: float = 0.05,
    nodata_mask: Optional[np.ndarray] = None,
    nir: Optional[np.ndarray] = None,
    nir_max: float = 0.12,
    nir_softness: float = 0.03,
    local_delta: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """NDWI 阈值分割。

    参数
    ----
    ndwi : (H, W) 水体指数
    method : "otsu" 全局阈值 / "local" 局部阈值 / "hybrid" 两者软融合（默认）
    fixed_threshold : 指定阈值（覆盖 Otsu，用于复现论文或人工调参）
    softness : 软概率的过渡宽度，越小边界越"硬"
    nodata_mask : True 表示无效像元，输出会被置 0
    nir : 可选近红外反射率。水体近红外反射率极低（<0.05），而城市/裸土偏高，
         加入该"物理闸门"可显著压掉城市误判——这是纯 NDWI 最大的假阳性来源。
    nir_max : 近红外闸门阈值（反射率），水体通常远低于此值
    local_delta : 局部阈值相对全局阈值的最大偏移，防止"整块陆地"被局部阈值翻成水体

    返回
    ----
    mask : (H, W) bool 水体掩膜
    prob : (H, W) float32 0~1 置信度（越大越确信是水）
    meta : 阈值等诊断信息
    """
    ndwi = np.asarray(ndwi, dtype=np.float32)
    if nodata_mask is not None:
        if np.asarray(nodata_mask).shape != ndwi.shape:
            raise ValueError("无效掩膜与指数尺寸不一致")
        ndwi = np.where(nodata_mask, np.nan, ndwi)
    h, w = ndwi.shape
    meta: Dict[str, Any] = {"method": method, "softness": float(softness)}

    # ---- 退化场景：整景指数几乎无变化（全陆/全水/云覆盖）----
    finite = ndwi[np.isfinite(ndwi)]
    std = float(finite.std()) if finite.size else 0.0
    if std < 1e-4 and fixed_threshold is None:
        const = float(finite.mean()) if finite.size else 0.0
        is_water = const > 0.1  # NDWI > 0.1 才认为整景为水
        prob = np.full(ndwi.shape, 1.0 if is_water else 0.0, dtype=np.float32)
        if nir is not None:
            if np.asarray(nir).shape != ndwi.shape:
                raise ValueError("近红外影像尺寸与指数不一致")
            prob *= _sigmoid((nir_max - np.asarray(nir)) / max(nir_softness, 1e-3))
            meta["nir_gate"] = float(nir_max)
        prob[~np.isfinite(ndwi)] = 0.0
        if nodata_mask is not None:
            prob[nodata_mask] = 0.0
        mask = prob > 0.5
        meta.update(
            {
                "threshold_source": "no_signal",
                "threshold_global": 0.1,
                "constant_index": const,
                "note": f"水体指数几乎无空间变化（std={std:.2e}，均值={const:.3f}）",
                "water_fraction_raw": float(np.count_nonzero(mask)) / max(mask.size, 1),
            }
        )
        return mask, prob, meta

    # ---- 全局阈值 ----
    if fixed_threshold is not None:
        t_global = float(fixed_threshold)
        meta["threshold_source"] = "fixed"
    else:
        t_global = otsu_threshold(ndwi)
        meta["threshold_source"] = "otsu"
        if not np.isfinite(t_global):
            t_global = 0.0
            meta["threshold_source"] = "fallback"
        elif not (THRESH_VALID_RANGE[0] <= t_global <= THRESH_VALID_RANGE[1]):
            # Otsu 失效（全水或全陆）：退回经验阈值
            meta["threshold_source"] = "fallback_clamped"
            meta["otsu_raw"] = float(t_global)
            t_global = float(np.clip(t_global, *THRESH_VALID_RANGE))
    meta["threshold_global"] = float(t_global)

    # ---- 软概率 ----
    prob_global = _sigmoid((ndwi - t_global) / max(softness, 1e-3))

    if method in ("local", "hybrid") and fixed_threshold is None:
        thr_map = local_otsu_threshold(ndwi, block=block)
        # 关键：局部阈值只能在全局阈值附近微调，否则"全是陆地"的块会把城市翻成水
        thr_map = np.clip(thr_map, t_global - local_delta, t_global + local_delta)
        meta["threshold_local_range"] = [float(thr_map.min()), float(thr_map.max())]
        prob_local = _sigmoid((ndwi - thr_map) / max(softness, 1e-3))
        prob = 0.5 * (prob_global + prob_local) if method == "hybrid" else prob_local
    else:
        prob = prob_global

    # ---- 近红外物理闸门（模糊与运算）----
    if nir is not None:
        nir = np.asarray(nir, dtype=np.float32)
        if nir.shape != ndwi.shape:
            raise ValueError(f"近红外影像尺寸 {nir.shape} 与 NDWI {ndwi.shape} 不一致")
        gate = _sigmoid((nir_max - nir) / max(nir_softness, 1e-3))
        prob = prob * gate
        meta["nir_gate"] = float(nir_max)
        meta["nir_gate_rejected_pct"] = 100.0 * float(np.count_nonzero(gate < 0.5)) / max(gate.size, 1)

    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    prob[~np.isfinite(prob) | ~np.isfinite(ndwi)] = 0.0
    if nodata_mask is not None:
        prob[nodata_mask] = 0.0

    mask = prob > 0.5
    meta["water_fraction_raw"] = float(np.count_nonzero(mask)) / max(mask.size, 1)
    return mask, prob, meta


def describe_threshold(meta: Dict[str, Any]) -> str:
    """把 meta 变成一句人话，直接显示在界面上。"""
    t = meta.get("threshold_global", 0.0)
    src = {
        "otsu": "Otsu 自适应",
        "fixed": "人工指定",
        "fallback": "经验阈值(样本不足)",
        "fallback_clamped": "经验阈值(Otsu 失效)",
    }.get(meta.get("threshold_source", ""), meta.get("threshold_source", "?"))
    extra = ""
    if meta.get("threshold_source") == "fallback_clamped":
        extra = f"（原始 Otsu={meta.get('otsu_raw'):.3f}）"
    return f"NDWI 阈值 {t:+.3f}，来源：{src}{extra}"
