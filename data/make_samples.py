"""
慧眼识灾 · 演示样本生成器
=========================

为什么需要它？
    真实数据 Sen1Floods11 有 4GB+，下载/解压/对齐要好几个小时；
    而项目第一天就要能演示、能跑通训练链路。

    本脚本用"地物光谱 + 地形 + 洪水淹没"的物理先验，合成 Sentinel-2 风格
    的 4 波段影像（蓝/绿/红/近红外，uint16，含地理坐标），并附带逐像元真值掩膜。
    它只用于**验证全流程可跑通**，绝不能用来宣称模型精度。

真实训练请执行：
    python scripts/download_data.py --dataset sen1floods11

生成产物（每个样本一组）：
    demoXX_pre.tif     灾前影像（4 波段 uint16 GeoTIFF）
    demoXX_post.tif    灾后影像（洪水淹没）
    demoXX_mask.png    灾后水体真值掩膜（白色=水）
    demoXX_preview.png 灾前/灾后真彩预览（横向拼接）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Windows 控制台默认 GBK，中文/上标字符会报 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

try:
    from scipy.ndimage import gaussian_filter, zoom as nd_zoom
except Exception:  # pragma: no cover
    gaussian_filter = None
    nd_zoom = None


# --------------------------------------------------------------------------
# 地物光谱库（0~1 反射率，顺序：蓝 绿 红 近红外）
# 参考 Sentinel-2 典型地物光谱，量级合理即可
# --------------------------------------------------------------------------

SPECTRA: Dict[str, np.ndarray] = {
    "water": np.array([0.055, 0.062, 0.040, 0.018], dtype=np.float32),
    "vegetation": np.array([0.030, 0.060, 0.038, 0.310], dtype=np.float32),
    "cropland": np.array([0.048, 0.082, 0.090, 0.255], dtype=np.float32),
    "bare_soil": np.array([0.120, 0.155, 0.195, 0.235], dtype=np.float32),
    "urban": np.array([0.140, 0.150, 0.162, 0.180], dtype=np.float32),
}

BAND_NAMES = ["B2(blue)", "B3(green)", "B4(red)", "B8(nir)"]


# --------------------------------------------------------------------------
# 噪声与地形
# --------------------------------------------------------------------------


def fractal_noise(shape: Tuple[int, int], octaves: int = 5, seed: int = 0, persistence: float = 0.55) -> np.ndarray:
    """多倍频随机噪声（值域 0~1），用来模拟地形/地物分布的自然纹理。"""
    rng = np.random.default_rng(seed)
    h, w = shape
    total = np.zeros(shape, dtype=np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        res = 2 ** (o + 2)
        small = rng.random((max(2, res), max(2, res))).astype(np.float32)
        if nd_zoom is not None:
            big = nd_zoom(small, (h / small.shape[0], w / small.shape[1]), order=3)
            big = big[:h, :w]
        else:  # 无 scipy 时用最近邻放大
            yi = (np.arange(h) * small.shape[0] / h).astype(int).clip(0, small.shape[0] - 1)
            xi = (np.arange(w) * small.shape[1] / w).astype(int).clip(0, small.shape[1] - 1)
            big = small[np.ix_(yi, xi)]
        if big.shape != shape:
            pad = np.zeros(shape, dtype=np.float32)
            pad[: big.shape[0], : big.shape[1]] = big[:h, :w]
            big = pad
        total += amp * big
        norm += amp
        amp *= persistence
    total /= max(norm, 1e-6)
    lo, hi = float(total.min()), float(total.max())
    return (total - lo) / max(hi - lo, 1e-6)


def river_distance(shape: Tuple[int, int], cfg: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (到河道中心线的距离场, 河道半宽场)。河道沿 x 方向蜿蜒。"""
    h, w = shape
    xs = np.arange(w, dtype=np.float32)
    yc = (
        cfg["y0"] * h
        + cfg["amp1"] * h * np.sin(2 * np.pi * xs / w * cfg["freq1"] + cfg["phase"])
        + cfg["amp2"] * h * np.sin(2 * np.pi * xs / w * cfg["freq2"] + cfg["phase"] * 1.7)
    )
    ys = np.arange(h, dtype=np.float32)[:, None]
    dist = np.abs(ys - yc[None, :])
    width = cfg["width"] * h * (1.0 + 0.35 * np.sin(2 * np.pi * xs / w * 3.0 + cfg["phase"]))
    return dist, np.repeat(width[None, :], h, axis=0)


def ellipse_mask(shape: Tuple[int, int], cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - cx * w) / (rx * w)) ** 2 + ((yy - cy * h) / (ry * h)) ** 2) <= 1.0


def soft_classes(masks: Dict[str, np.ndarray], sigma: float = 0.9) -> Dict[str, np.ndarray]:
    """把硬分类掩膜高斯模糊成"软隶属度"，模拟混合像元，再归一化。"""
    soft: Dict[str, np.ndarray] = {}
    for k, m in masks.items():
        s = gaussian_filter(m.astype(np.float32), sigma=sigma) if gaussian_filter is not None else m.astype(np.float32)
        soft[k] = np.clip(s, 0.0, 1.0)
    stack = np.stack(list(soft.values()), axis=0)
    norm = stack.sum(axis=0, keepdims=True)
    norm[norm < 1e-6] = 1.0
    stack = stack / norm
    return {k: stack[i] for i, k in enumerate(soft)}


def render_scene(
    classes: Dict[str, np.ndarray],
    rng: np.random.Generator,
    haze: float = 0.02,
) -> np.ndarray:
    """按地物隶属度合成 4 波段反射率，加传感器噪声与大气雾。"""
    h, w = next(iter(classes.values())).shape
    bands = np.zeros((4, h, w), dtype=np.float32)
    for name, members in classes.items():
        base = SPECTRA[name]
        # 同类地物的空间差异（农田长势、土壤湿度）
        variation = 1.0 + 0.25 * (fractal_noise((h, w), octaves=3, seed=int(rng.integers(1e6))) - 0.5)
        for b in range(4):
            bands[b] += members * base[b] * variation
    # 大气雾 + 传感器噪声
    bands = bands + haze * fractal_noise((h, w), octaves=2, seed=int(rng.integers(1e6)))
    bands += rng.normal(0.0, 0.006, size=bands.shape).astype(np.float32)
    return np.clip(bands, 0.0, 1.2)


def to_uint16(bands: np.ndarray) -> np.ndarray:
    """反射率 -> Sentinel-2 L2A 风格的 uint16（×10000）。"""
    return np.clip(bands * 10000.0, 1, 10000).astype(np.uint16)


# --------------------------------------------------------------------------
# 单个样本
# --------------------------------------------------------------------------


def make_sample(
    index: int,
    size: int = 640,
    seed: int = 2026,
    flood_min: float = 0.06,
    flood_max: float = 0.16,
) -> Dict[str, Any]:
    """合成一个"灾前 + 灾后 + 真值"样本。"""
    rng = np.random.default_rng(seed + index * 7919)
    shape = (size, size)

    # --- 地形：低洼处更容易被淹 ---
    elev = fractal_noise(shape, octaves=6, seed=seed + index * 13)
    if gaussian_filter is not None:
        elev = gaussian_filter(elev, sigma=size / 64.0)
    elev = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-6)

    # --- 河道参数（每个样本不一样） ---
    river_cfg = {
        "y0": 0.45 + 0.08 * rng.random(),
        "amp1": 0.10 + 0.06 * rng.random(),
        "amp2": 0.035 + 0.03 * rng.random(),
        "freq1": 0.6 + 0.5 * rng.random(),
        "freq2": 1.8 + 0.8 * rng.random(),
        "phase": 2 * np.pi * rng.random(),
        "width": 0.012 + 0.010 * rng.random(),
    }
    dist, width = river_distance(shape, river_cfg)

    # --- 灾前：河道 + 一个常年湖泊 ---
    river = dist < width
    lake_cx, lake_cy = 0.18 + 0.6 * rng.random(), 0.18 + 0.6 * rng.random()
    lake = ellipse_mask(shape, lake_cx, lake_cy, 0.05 + 0.03 * rng.random(), 0.04 + 0.02 * rng.random())
    water_pre = river | lake

    # --- 灾后：洪水沿河道漫入低洼区 ---
    target_frac = float(rng.uniform(flood_min, flood_max))
    floodplain = dist < (0.16 + 0.10 * rng.random())  # 洪泛区范围
    inundation_score = (1.0 - elev) + 0.35 * floodplain.astype(np.float32) + 0.08 * rng.random(shape).astype(np.float32)
    thr = np.quantile(inundation_score, 1.0 - target_frac)
    flood = inundation_score >= thr
    flood = flood | water_pre
    if gaussian_filter is not None:  # 去掉孤岛碎斑，让边界更自然
        flood = gaussian_filter(flood.astype(np.float32), sigma=1.2) > 0.5

    # --- 地物分类 ---
    builtup = fractal_noise(shape, octaves=4, seed=seed + index * 31) > 0.82
    veg_field = fractal_noise(shape, octaves=4, seed=seed + index * 37)
    crop_field = fractal_noise(shape, octaves=4, seed=seed + index * 41) > 0.55

    def landcover(water: np.ndarray) -> Dict[str, np.ndarray]:
        masks = {
            "water": water,
            "urban": (~water) & builtup,
            "bare_soil": (~water) & (~builtup) & (elev > 0.72) & (~crop_field),
            "cropland": (~water) & (~builtup) & crop_field,
            "vegetation": (~water) & (~builtup) & (~crop_field) & (~((elev > 0.72) & (~crop_field))),
        }
        # 保证互斥且并集为全图
        masks["vegetation"] = (~water) & (~builtup) & (~masks["bare_soil"]) & (~masks["cropland"])
        return masks

    bands_pre = render_scene(soft_classes(landcover(water_pre)), rng)
    bands_post = render_scene(soft_classes(landcover(flood)), rng)

    return {
        "index": index,
        "size": size,
        "pre": to_uint16(bands_pre),
        "post": to_uint16(bands_post),
        "mask_pre": water_pre,
        "mask_post": flood,
        "new_flood": flood & ~water_pre,
        "pixel_size_m": 10.0,
        "river_cfg": {k: float(v) for k, v in river_cfg.items()},
    }


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------


def write_geotiff(path: str, arr_chw: np.ndarray, pixel_size_m: float = 10.0, epsg: int = 32650) -> str:
    """写 4 波段 GeoTIFF（带地理坐标，单位米）。"""
    import rasterio
    from rasterio.transform import from_origin

    c, h, w = arr_chw.shape
    transform = from_origin(500000.0, 3000000.0, pixel_size_m, pixel_size_m)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "uint16",
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "nodata": 0,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr_chw.astype(np.uint16))
        for i, name in enumerate(BAND_NAMES[:c], start=1):
            ds.set_band_description(i, name)
    return path


def save_preview(path: str, arr_chw: np.ndarray, stretch: bool = True) -> str:
    from PIL import Image

    rgb = np.moveaxis(arr_chw[[2, 1, 0]].astype(np.float32), 0, -1)  # B4,B3,B2 -> RGB (H,W,3)
    if stretch:
        lo, hi = np.percentile(rgb, [2, 98])
        rgb = (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb / 3000.0 * 255, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def save_mask(path: str, mask: np.ndarray) -> str:
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)
    return path


def build_dataset(
    out_dir: str = "data/samples",
    n: int = 6,
    size: int = 640,
    seed: int = 2026,
    preview: bool = True,
) -> Dict[str, Any]:
    """生成整份演示数据集 + 清单文件。"""
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "note": "合成演示数据，仅用于验证流程，不可用于宣称模型精度。",
        "pixel_size_m": 10.0,
        "crs": "EPSG:32650",
        "bands": BAND_NAMES,
        "samples": [],
    }

    for i in range(1, n + 1):
        s = make_sample(i, size=size, seed=seed)
        stem = f"demo{i:02d}"
        pre_path = os.path.join(out_dir, f"{stem}_pre.tif")
        post_path = os.path.join(out_dir, f"{stem}_post.tif")
        mask_path = os.path.join(out_dir, f"{stem}_mask.png")
        pre_mask_path = os.path.join(out_dir, f"{stem}_pre_mask.png")

        write_geotiff(pre_path, s["pre"], s["pixel_size_m"])
        write_geotiff(post_path, s["post"], s["pixel_size_m"])
        save_mask(mask_path, s["mask_post"])
        save_mask(pre_mask_path, s["mask_pre"])

        if preview:
            rgb_pre = np.moveaxis(s["pre"][[2, 1, 0]].astype(np.float32), 0, -1)
            rgb_post = np.moveaxis(s["post"][[2, 1, 0]].astype(np.float32), 0, -1)
            lo, hi = np.percentile(np.concatenate([rgb_pre.ravel(), rgb_post.ravel()]), [2, 98])
            norm = lambda a: (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)  # noqa: E731
            gap = np.full((size, 6, 3), 255, dtype=np.uint8)
            from PIL import Image

            preview_arr = np.concatenate([norm(rgb_pre), gap, norm(rgb_post)], axis=1)
            im = Image.fromarray(preview_arr)
            if im.width > 900:  # 预览图压缩，避免仓库体积过大
                im = im.resize((900, int(im.height * 900 / im.width)), Image.BILINEAR)
            im.save(os.path.join(out_dir, f"{stem}_preview.jpg"), quality=82, optimize=True)

        px_km2 = (s["pixel_size_m"] ** 2) / 1e6
        manifest["samples"].append(
            {
                "id": stem,
                "pre": os.path.basename(pre_path),
                "post": os.path.basename(post_path),
                "mask": os.path.basename(mask_path),
                "pre_mask": os.path.basename(pre_mask_path),
                "preview": f"{stem}_preview.jpg" if preview else None,
                "size": [size, size],
                "pixel_size_m": s["pixel_size_m"],
                "flood_area_km2": float(np.count_nonzero(s["new_flood"]) * px_km2),
                "post_water_km2": float(np.count_nonzero(s["mask_post"]) * px_km2),
                "pre_water_km2": float(np.count_nonzero(s["mask_pre"]) * px_km2),
                "flood_fraction_pct": float(100.0 * np.count_nonzero(s["new_flood"]) / s["new_flood"].size),
                "description": f"合成场景 {i}：蜿蜒河道 + 低洼区漫滩淹没（10 m 分辨率）",
            }
        )
        print(
            f"[{i}/{n}] {stem}: 灾前水体 {np.count_nonzero(s['mask_pre']) * px_km2:6.2f} km² -> "
            f"灾后 {np.count_nonzero(s['mask_post']) * px_km2:6.2f} km²，新增淹没 "
            f"{np.count_nonzero(s['new_flood']) * px_km2:6.2f} km²"
        )

    manifest_path = os.path.join(out_dir, "samples.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\n清单已写入：{manifest_path}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="生成慧眼识灾演示样本（合成 Sentinel-2 风格影像）")
    ap.add_argument("--out", default="data/samples", help="输出目录")
    ap.add_argument("--n", type=int, default=6, help="样本数量")
    ap.add_argument("--size", type=int, default=640, help="影像边长（像元）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-preview", action="store_true", help="不生成 PNG 预览")
    args = ap.parse_args()
    build_dataset(args.out, n=args.n, size=args.size, seed=args.seed, preview=not args.no_preview)


if __name__ == "__main__":
    main()
