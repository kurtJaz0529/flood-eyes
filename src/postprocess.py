"""
慧眼识灾 · 后处理与统计模块
===========================

模型输出的是"像素级概率"，应急部门要的是"哪里淹了、淹了多少平方公里"。
这一层负责：
    形态学去噪 -> 连通域过滤 -> 空洞填充 -> 轮廓叠加 -> 面积统计
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-8


# --------------------------------------------------------------------------
# 形态学与连通域
# --------------------------------------------------------------------------


def _get_cv2():
    try:
        import cv2  # noqa: WPS433

        return cv2
    except Exception:  # pragma: no cover
        return None


def _fill_small_holes(water: np.ndarray, max_hole_px: int) -> np.ndarray:
    """只填小于 max_hole_px 的孔，保留被水包围的岛屿/建筑。max_hole_px<=0 则全填。"""
    cv2 = _get_cv2()
    m = np.asarray(water).astype(np.uint8)
    h, w = m.shape
    if cv2 is not None:
        pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        mask2 = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(pad, mask2, (0, 0), 1)
        holes = (pad[1:-1, 1:-1] == 0).astype(np.uint8)
        if max_hole_px <= 0:
            return np.maximum(m, holes)
        if not holes.any():
            return m
        n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        fill = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] <= max_hole_px:
                fill[labels == i] = 1
        return np.maximum(m, fill)
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover
        return m.astype(bool)
    filled = ndimage.binary_fill_holes(m.astype(bool))
    holes = filled & ~m.astype(bool)
    if max_hole_px <= 0:
        return filled.astype(np.uint8)
    if not holes.any():
        return m
    lab, n = ndimage.label(holes)
    keep = np.zeros_like(holes)
    if n > 0:
        sizes = ndimage.sum(holes, lab, range(1, n + 1))
        for i, s in enumerate(sizes, start=1):
            if s <= max_hole_px:
                keep[lab == i] = True
    return (m.astype(bool) | keep).astype(np.uint8)


def clean_mask(
    mask: np.ndarray,
    open_radius: int = 2,
    close_radius: int = 3,
    min_area_px: int = 120,
    fill_holes: bool = True,
    max_hole_px: int = 500,
) -> np.ndarray:
    """形态学清洗 + 小连通域剔除 + 小孔填充。

    - 开运算：去掉零星椒盐噪点（如建筑屋顶、云影误判）
    - 闭运算：连接被道路/堤坝切断的水体
    - 连通域过滤：面积小于 min_area_px 的碎块直接删除
    - 孔洞：只填 ≤ max_hole_px 的小孔，避免把岛屿/城区填成水
    """
    mask = np.asarray(mask).astype(bool)
    if mask.size == 0 or not mask.any():
        return mask

    cv2 = _get_cv2()
    if cv2 is not None:
        m = mask.astype(np.uint8)
        if open_radius > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1,) * 2)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if close_radius > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if fill_holes:
            m = _fill_small_holes(m, max_hole_px)
        if min_area_px > 0:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.zeros_like(m)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                    keep[labels == i] = 1
            m = keep
        return m.astype(bool)

    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover
        return mask
    m = mask
    if open_radius > 0:
        m = ndimage.binary_opening(m, iterations=open_radius)
    if close_radius > 0:
        m = ndimage.binary_closing(m, iterations=close_radius)
    if fill_holes:
        m = _fill_small_holes(np.asarray(m).astype(np.uint8), max_hole_px).astype(bool)
    if min_area_px > 0:
        lab, n = ndimage.label(m)
        if n > 0:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            keep = np.zeros_like(m)
            for i, s in enumerate(sizes, start=1):
                if s >= min_area_px:
                    keep[lab == i] = True
            m = keep
    return m.astype(bool)


def components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """连通域标记，返回 (标签图, 个数)。"""
    mask = np.asarray(mask).astype(np.uint8)
    cv2 = _get_cv2()
    if cv2 is not None:
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        return labels, max(0, n - 1)
    try:  # 打包版不带 scipy，这里仅作源码环境的兜底
        from scipy import ndimage
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 OpenCV，且没有 scipy 兜底实现；请安装 opencv-python") from exc

    labels, n = ndimage.label(mask)
    return labels, int(n)


def extract_contours(mask: np.ndarray, simplify: float = 1.5) -> List[np.ndarray]:
    """提取外轮廓，用于叠加显示。返回 (N, 1, 2) 的 int32 点集列表。"""
    cv2 = _get_cv2()
    if cv2 is None:
        return []
    m = np.asarray(mask).astype(np.uint8)
    if not m.any():
        return []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if simplify > 0:
        cnts = [cv2.approxPolyDP(c, simplify, True) for c in cnts]
    return cnts


# --------------------------------------------------------------------------
# 面积统计
# --------------------------------------------------------------------------


def area_stats(
    mask: np.ndarray,
    pixel_size_m: float = 10.0,
    nodata_mask: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """把像素掩膜换算成应急部门看得懂的指标。

    nodata_mask：True 表示无效像元，不计入分母。
    valid_mask 是旧参数名，语义与 nodata_mask 相同，勿把「有效像元」传进来。
    """
    if nodata_mask is None:
        nodata_mask = valid_mask
    mask = np.asarray(mask).astype(bool)
    total_px = int(mask.size) if nodata_mask is None else int(np.count_nonzero(~np.asarray(nodata_mask)))
    water_px = int(np.count_nonzero(mask))
    px_km2 = (float(pixel_size_m) ** 2) / 1_000_000.0

    labels, n_comp = components(mask)
    comps: List[Dict[str, Any]] = []
    if n_comp > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        order = np.argsort(counts)[::-1][:10]
        for idx in order:
            if counts[idx] <= 0:
                continue
            ys, xs = np.where(labels == idx)
            comps.append(
                {
                    "area_km2": float(counts[idx]) * px_km2,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "centroid": [float(xs.mean()), float(ys.mean())],
                }
            )
        comps.sort(key=lambda c: c["area_km2"], reverse=True)

    perimeter_km = 0.0
    cv2 = _get_cv2()
    if cv2 is not None and water_px > 0:
        cnts = extract_contours(mask, simplify=0)
        perimeter_px = sum(float(cv2.arcLength(c, True)) for c in cnts)
        perimeter_km = perimeter_px * float(pixel_size_m) / 1000.0

    return {
        "water_pixels": water_px,
        "total_pixels": total_px,
        "water_area_km2": water_px * px_km2,
        "total_area_km2": total_px * px_km2,
        "water_fraction_pct": 100.0 * water_px / max(total_px, 1),
        "n_components": int(n_comp),
        "largest_area_km2": comps[0]["area_km2"] if comps else 0.0,
        "perimeter_km": perimeter_km,
        "components": comps,
        "pixel_size_m": float(pixel_size_m),
    }


def confidence_stats(prob: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """置信度统计：水体像元上的平均概率，以及"模棱两可"像元占比。"""
    prob = np.asarray(prob, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return {"mean_confidence": 0.0, "low_confidence_pct": 0.0, "edge_pixels": 0}
    vals = prob[mask]
    water_uncertain = np.count_nonzero(mask & (prob > 0.35) & (prob < 0.65))
    edge = np.count_nonzero((prob > 0.35) & (prob < 0.65))
    return {
        "mean_confidence": float(vals.mean()),
        "min_confidence": float(vals.min()),
        "low_confidence_pct": 100.0 * float(water_uncertain) / max(int(np.count_nonzero(mask)), 1),
        "edge_pixels": int(edge),
    }


def change_stats(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    pixel_size_m: float = 10.0,
) -> Dict[str, Any]:
    """双时相变化统计：新增淹没 / 退水 / 持续淹没。"""
    before = np.asarray(before_mask).astype(bool)
    after = np.asarray(after_mask).astype(bool)
    px_km2 = (float(pixel_size_m) ** 2) / 1_000_000.0
    new = after & ~before
    receded = before & ~after
    persistent = before & after
    return {
        "new_water_km2": float(np.count_nonzero(new)) * px_km2,
        "receded_water_km2": float(np.count_nonzero(receded)) * px_km2,
        "persistent_water_km2": float(np.count_nonzero(persistent)) * px_km2,
        "before_water_km2": float(np.count_nonzero(before)) * px_km2,
        "after_water_km2": float(np.count_nonzero(after)) * px_km2,
        "net_change_km2": float(np.count_nonzero(after) - np.count_nonzero(before)) * px_km2,
        "new_water_pct_of_after": 100.0 * float(np.count_nonzero(new)) / max(int(np.count_nonzero(after)), 1),
        "pixel_size_m": float(pixel_size_m),
        "masks": {"new": new, "receded": receded, "persistent": persistent},
    }


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.45,
    draw_contour: bool = True,
    contour_color: Tuple[int, int, int] = (255, 40, 40),
    thickness: int = 2,
) -> np.ndarray:
    """把水体掩膜半透明叠加到真彩影像上，并勾勒边界。返回 uint8 RGB。"""
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    out = rgb.astype(np.float32).copy()
    mask = np.asarray(mask).astype(bool)
    if mask.shape != out.shape[:2]:
        raise ValueError(f"掩膜尺寸 {mask.shape} 与影像 {out.shape[:2]} 不一致")

    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr.reshape(-1, 3)[0] * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    if draw_contour and mask.any():
        cv2 = _get_cv2()
        if cv2 is not None:
            cnts = extract_contours(mask, simplify=1.5)
            cv2.drawContours(out, cnts, -1, tuple(int(c) for c in contour_color), thickness)
    return out


def mask_to_rgba(mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """掩膜 -> RGBA 图（水体上色，背景透明），用于滑块叠加。"""
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask] = (*color, 200)
    return rgba


def change_map_rgb(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    base_rgb: np.ndarray,
) -> np.ndarray:
    """变化图：红=新增淹没，绿=退水，青=持续水体。"""
    before = np.asarray(before_mask).astype(bool)
    after = np.asarray(after_mask).astype(bool)
    base = np.asarray(base_rgb).astype(np.float32)
    out = base * 0.45
    palette = {
        "new": (255, 60, 60),
        "receded": (60, 230, 120),
        "persistent": (0, 200, 255),
    }
    for name, m in (("persistent", before & after), ("new", after & ~before), ("receded", before & ~after)):
        if m.any():
            out[m] = np.asarray(palette[name], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def side_by_side(images: List[np.ndarray], gap: int = 8, bg: int = 20) -> np.ndarray:
    """横向拼接多张图（统一高度），用于导出对比图。"""
    if not images:
        raise ValueError("没有可拼接的影像")
    imgs = [np.asarray(i) for i in images]
    h = max(i.shape[0] for i in imgs)
    resized = []
    for im in imgs:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            try:
                import cv2

                im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), h))
            except Exception:
                im = np.repeat(np.repeat(im, 2, 0), 2, 1)[:h, : im.shape[1] * 2]
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        resized.append(im.astype(np.uint8))
    total_w = sum(i.shape[1] for i in resized) + gap * (len(resized) - 1)
    canvas = np.full((h, total_w, 3), bg, dtype=np.uint8)
    x = 0
    for im in resized:
        canvas[:, x : x + im.shape[1]] = im[..., :3]
        x += im.shape[1] + gap
    return canvas
