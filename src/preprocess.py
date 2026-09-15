"""
慧眼识灾 · 预处理模块
=====================

职责：把"卫星影像文件"变成"模型能吃的张量"。

流程：
    读取影像(GeoTIFF / PNG / JPG / NPY)
        -> 波段解析(蓝/绿/红/近红外，统一到 0~1 反射率)
        -> NDWI 水体指数计算
        -> 百分位拉伸(出图用) / 均值方差归一化(模型用)
        -> 512x512 切片(带重叠，避免拼接缝)

约定：内部统一波段顺序为 [blue, green, red, nir]，数值为 0~1 的反射率。
Sentinel-2 L2A 的 uint16 数据默认按 10000 缩放还原反射率。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

# 规范波段顺序
CANONICAL = ("blue", "green", "red", "nir")

# 常见卫星数据的波段排列预设（0-based 索引）
BAND_ORDERS: Dict[str, Dict[str, int]] = {
    # Sentinel-2 四波段导出：B2 B3 B4 B8
    "s2_bgr_nir": {"blue": 0, "green": 1, "red": 2, "nir": 3},
    # Sentinel-2 L2A 12/13 波段：B1,B2,B3,B4,... B8 在索引 7（12 波段 L2A 无 B10，B8 仍是第 8 个）
    "s2_l2a_13": {"blue": 1, "green": 2, "red": 3, "nir": 7},
    # 10 波段常见栈：B2 B3 B4 B5 B6 B7 B8 B8A B11 B12（无 B1）
    "s2_10band": {"blue": 0, "green": 1, "red": 2, "nir": 6},
    # 高分/资源系列常见顺序：蓝 绿 红 近红外
    "gfx_4band": {"blue": 0, "green": 1, "red": 2, "nir": 3},
    # 三波段可见光：红 绿 蓝
    "rgb": {"red": 0, "green": 1, "blue": 2},
    # 两波段：绿 + 近红外
    "g_nir": {"green": 0, "nir": 1},
}

# SCL 云/云影/卷云（Sentinel-2 场景分类）
SCL_CLOUD_CLASSES = (3, 8, 9, 10)
_SCL_RGB = {
    0: (0, 0, 0),
    1: (120, 120, 120),
    2: (60, 60, 60),
    3: (150, 150, 150),
    4: (60, 150, 60),
    5: (200, 180, 120),
    6: (30, 90, 200),
    7: (150, 200, 220),
    8: (235, 235, 235),
    9: (255, 255, 255),
    10: (210, 225, 245),
    11: (240, 240, 240),
}

# 模型输入归一化默认值（0~1 反射率尺度，训练脚本会用真实数据统计值覆盖）
DEFAULT_MEAN = (0.0850, 0.0950, 0.1050, 0.2450)
DEFAULT_STD = (0.0500, 0.0520, 0.0600, 0.1100)

EPS = 1e-8


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass
class Scene:
    """一景影像（已解析为规范波段）。"""

    bands: Dict[str, np.ndarray]  # 键取自 CANONICAL，值为 float32 (H, W)，反射率 0~1
    transform: Any = None  # rasterio Affine，用于计算像元尺寸/地理坐标
    crs: Any = None
    path: str = ""
    pixel_size_m: float = 10.0
    nodata_mask: Optional[np.ndarray] = None  # True 表示无效像元
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- 基础属性 ----------------------------------------------------------
    @property
    def height(self) -> int:
        return int(next(iter(self.bands.values())).shape[0])

    @property
    def width(self) -> int:
        return int(next(iter(self.bands.values())).shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def has_nir(self) -> bool:
        return "nir" in self.bands

    @property
    def channel_names(self) -> List[str]:
        """按规范顺序返回实际存在的波段名。"""
        return [b for b in CANONICAL if b in self.bands]

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0] if self.path else "scene"

    # -- 组装 --------------------------------------------------------------
    def stack(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        """堆叠为 (C, H, W) float32，用于喂给模型。"""
        names = list(names) if names else self.channel_names
        if not names:
            raise ValueError("影像没有任何可用波段")
        return np.stack([self.bands[n] for n in names], axis=0).astype(np.float32)

    def ndwi(self) -> np.ndarray:
        """归一化水体指数 NDWI = (Green - NIR) / (Green + NIR)。

        缺少近红外波段时退化为 (Green - Red) / (Green + Red) 代理指数，
        并在 meta 中标记 nir_available=False（精度会下降，仅作演示兜底）。
        """
        green = self.bands.get("green")
        if green is None:
            raise ValueError("缺少绿波段，无法计算 NDWI")
        if self.has_nir:
            nir = self.bands["nir"]
            idx = (green - nir) / (green + nir + EPS)
        else:
            red = self.bands.get("red")
            if red is None:
                raise ValueError("缺少近红外与红波段，无法计算水体指数")
            idx = (green - red) / (green + red + EPS)
        return np.clip(idx, -1.0, 1.0).astype(np.float32)

    def rgb(self) -> np.ndarray:
        """返回可视化用 RGB uint8 (H, W, 3)，2%~98% 百分位拉伸。"""
        r = self.bands.get("red")
        g = self.bands.get("green")
        b = self.bands.get("blue")
        if r is None or g is None or b is None:  # 波段不全时用灰度兜底
            gray = percentile_stretch(next(iter(self.bands.values())))
            return np.stack([gray, gray, gray], axis=-1)
        return percentile_stretch(np.stack([r, g, b], axis=-1))

    def to_rgba(self) -> np.ndarray:
        rgb = self.rgb()
        alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
        if self.nodata_mask is not None:
            alpha[self.nodata_mask, 0] = 0
        return np.concatenate([rgb, alpha], axis=-1)


# --------------------------------------------------------------------------
# 读取与波段解析
# --------------------------------------------------------------------------


def _is_tiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".tif", ".tiff", ".img", ".vrt")


def _read_tiff(path: str) -> Tuple[np.ndarray, Any, Any, Optional[np.ndarray], Dict[str, Any]]:
    """读取 GeoTIFF -> (H, W, C) float32 原始值，附带 scale/offset/描述。"""
    import rasterio

    extra: Dict[str, Any] = {}
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)  # (C, H, W)
        arr = np.moveaxis(arr, 0, -1)  # -> (H, W, C)
        transform, crs = ds.transform, ds.crs
        nodata = ds.nodata
        extra["scales"] = tuple(ds.scales) if ds.scales else ()
        extra["offsets"] = tuple(ds.offsets) if ds.offsets else ()
        extra["descriptions"] = tuple(ds.descriptions) if ds.descriptions else ()
        extra["tags"] = dict(ds.tags() or {})
        mask = None
        if nodata is not None:
            mask = np.all(np.isclose(arr, nodata), axis=-1)
        # 有些数据集用 0 表示无效
        if mask is None and np.issubdtype(ds.dtypes[0], np.integer):
            mask = np.all(arr == 0, axis=-1)
    return arr, transform, crs, mask, extra


def _read_plain_image(path: str) -> Tuple[np.ndarray, Any, Any, Optional[np.ndarray], Dict[str, Any]]:
    """读取 PNG/JPG -> (H, W, C) float32。"""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:  # 去掉 alpha
        arr = arr[..., :3]
    return arr.astype(np.float32), None, None, None, {}


def _read_npy(path: str) -> Tuple[np.ndarray, Any, Any, Optional[np.ndarray], Dict[str, Any]]:
    # allow_pickle=False：.npy 允许存放对象数组，反序列化即执行 pickle，
    # 对用户提供的样本文件等于任意代码执行入口。显式关闭，不依赖 numpy 版本默认值。
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[0] < arr.shape[-1]:  # 看起来像 (C,H,W)
        arr = np.moveaxis(arr, 0, -1)
    return arr.astype(np.float32), None, None, None, {}


def _apply_raster_scale_offset(
    arr: np.ndarray,
    scales: Sequence[Any],
    offsets: Sequence[Any],
) -> Tuple[np.ndarray, bool]:
    """按 GeoTIFF 每波段 scale/offset 还原物理量。全是 1/0 则原样返回。"""
    c = int(arr.shape[-1])
    applied = False
    out = arr
    for i in range(c):
        sc = 1.0
        off = 0.0
        if scales and i < len(scales) and scales[i] is not None:
            sc = float(scales[i])
            if not np.isfinite(sc) or sc == 0.0:
                sc = 1.0
        if offsets and i < len(offsets) and offsets[i] is not None:
            off = float(offsets[i])
            if not np.isfinite(off):
                off = 0.0
        if abs(sc - 1.0) <= 1e-12 and abs(off) <= 1e-12:
            continue
        if not applied:
            out = arr.copy()
            applied = True
        out[..., i] = arr[..., i] * sc + off
    return out, applied


def to_reflectance(arr: np.ndarray, meta: Dict[str, Any], allow_boa_heuristic: bool = True) -> np.ndarray:
    """把原始 DN 值转换为 0~1 反射率。

    - Sentinel-2 L2A 整型量化到 10000：除以 10000
    - 2022 年后部分 L2A 带 BOA −1000 偏移且未扣除：有效像元中位 DN≳1100 时自动减 1000
    - 8bit 影像：除以 255
    - 已经是 0~1 浮点：原样返回
    - GeoTIFF 元数据已应用 scale/offset 时传 allow_boa_heuristic=False，避免二次扣除
    """
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    vmax = float(finite.max())
    offset = 0.0
    if allow_boa_heuristic and 2000.0 < vmax <= 25000.0:
        valid = finite[finite > 1.0]
        if valid.size >= 64:
            p01 = float(np.percentile(valid, 0.1))
            med = float(np.median(valid))
            if p01 >= 600.0 and med >= 1500.0:
                offset = 1000.0
                arr = arr - offset
                finite = arr[np.isfinite(arr)]
                vmax = float(finite.max()) if finite.size else vmax
    if vmax <= 1.5:
        scale = 1.0
    elif vmax <= 255.0:
        scale = 255.0
    elif vmax <= 40000.0:
        # Sentinel-2 L1C/L2A 量化到 10000，云/饱和可到 2e4；不要误用 65535
        scale = 10000.0
    else:
        scale = 65535.0
    meta["dn_scale"] = scale
    meta["dn_max"] = vmax
    meta["boa_offset"] = -offset if offset else 0.0
    return np.clip(arr / scale, 0.0, 1.5).astype(np.float32)


def resolve_band_order_from_descriptions(
    descriptions: Optional[Sequence[Any]],
    n_channels: int,
) -> Optional[Dict[str, int]]:
    """从 GeoTIFF 波段描述解析蓝/绿/红/近红外。信息不够则返回 None。"""
    if not descriptions:
        return None
    mapping: Dict[str, int] = {}
    for i, raw in enumerate(list(descriptions)[:n_channels]):
        if raw is None:
            continue
        d = str(raw).strip().lower().replace(" ", "")
        if not d:
            continue
        if "nir" in d or "b08" in d or "b8(" in d or d in ("b8", "b08", "band8"):
            mapping["nir"] = i
        elif "green" in d or "b03" in d or "b3(" in d or d in ("b3", "b03", "band3"):
            mapping["green"] = i
        elif "red" in d or "b04" in d or "b4(" in d or d in ("b4", "b04", "band4"):
            mapping["red"] = i
        elif "blue" in d or "b02" in d or "b2(" in d or d in ("b2", "b02", "band2"):
            mapping["blue"] = i
    if "green" in mapping and ("nir" in mapping or "red" in mapping):
        return mapping
    return None


def resolve_band_order(
    n_channels: int,
    band_order: str,
    descriptions: Optional[Sequence[Any]] = None,
) -> Dict[str, int]:
    """根据通道数自动推断或使用指定预设的波段顺序。"""
    if band_order and band_order != "auto":
        if band_order not in BAND_ORDERS:
            raise KeyError(f"未知波段顺序 '{band_order}'，可选：{list(BAND_ORDERS)}")
        return dict(BAND_ORDERS[band_order])
    from_desc = resolve_band_order_from_descriptions(descriptions, n_channels)
    if from_desc:
        return from_desc
    if n_channels >= 12:
        return dict(BAND_ORDERS["s2_l2a_13"])
    if n_channels in (10, 11):
        return dict(BAND_ORDERS["s2_10band"])
    if n_channels >= 4:
        return dict(BAND_ORDERS["s2_bgr_nir"])
    if n_channels == 3:
        return dict(BAND_ORDERS["rgb"])
    if n_channels == 2:
        return dict(BAND_ORDERS["g_nir"])
    if n_channels == 1:
        return {"green": 0}
    return dict(BAND_ORDERS["s2_bgr_nir"])


def load_scene(
    path: str,
    band_order: str = "auto",
    nir_path: Optional[str] = None,
    pixel_size_m: Optional[float] = None,
) -> Scene:
    """读取影像并解析为规范波段。

    参数
    ----
    path : 影像路径，支持 .tif/.tiff/.png/.jpg/.npy
    band_order : "auto" 或 BAND_ORDERS 中的预设名
    nir_path : 当可见光影像没有近红外波段时，单独提供的近红外文件
    pixel_size_m : 像元分辨率（米），缺省时从 GeoTIFF transform 推断，否则 10m
    """
    ext = os.path.splitext(path)[1].lower()
    extra: Dict[str, Any] = {}
    if _is_tiff(path):
        raw, transform, crs, nodata_mask, extra = _read_tiff(path)
    elif ext == ".npy":
        raw, transform, crs, nodata_mask, extra = _read_npy(path)
    else:
        raw, transform, crs, nodata_mask, extra = _read_plain_image(path)

    meta: Dict[str, Any] = {"source_ext": ext, "n_channels": int(raw.shape[-1])}
    raw, scaled = _apply_raster_scale_offset(raw, extra.get("scales") or (), extra.get("offsets") or ())
    if scaled:
        meta["raster_scale_offset"] = True
        raw = to_reflectance(raw, meta, allow_boa_heuristic=False)
    else:
        raw = to_reflectance(raw, meta)

    mapping = resolve_band_order(raw.shape[-1], band_order, extra.get("descriptions"))
    bands: Dict[str, np.ndarray] = {}
    for name, idx in mapping.items():
        if 0 <= idx < raw.shape[-1]:
            bands[name] = np.ascontiguousarray(raw[..., idx])
    # 先确认主影像解析出了波段：否则下面 nir 分支里的 next(iter(bands.values()))
    # 会抛 StopIteration，把真正的错误原因（影像波段无法识别）掩盖掉。
    if not bands:
        raise ValueError(
            f"未能从影像解析出任何可用波段：{os.path.basename(path)}"
            f"（读到 {raw.shape[-1]} 个通道，波段映射 {mapping}）。"
            "请检查影像波段数或改用显式的 band_order。"
        )
    meta["band_order"] = band_order if band_order != "auto" else f"auto({len(mapping)}ch)"
    meta["band_map"] = mapping
    if extra.get("descriptions"):
        meta["band_descriptions"] = list(extra["descriptions"])

    # 单独的近红外文件
    if nir_path:
        if _is_tiff(nir_path):
            nir_raw, _, _, _, nir_extra = _read_tiff(nir_path)
        else:
            nir_raw, _, _, _, nir_extra = _read_plain_image(nir_path)
        nir_raw, nir_scaled = _apply_raster_scale_offset(
            nir_raw, nir_extra.get("scales") or (), nir_extra.get("offsets") or ()
        )
        nir_meta: Dict[str, Any] = {}
        nir_band = to_reflectance(nir_raw, nir_meta, allow_boa_heuristic=not nir_scaled)[..., 0]
        if nir_band.shape != next(iter(bands.values())).shape:
            raise ValueError("近红外影像尺寸与主影像不一致")
        bands["nir"] = nir_band
        meta["nir_source"] = os.path.basename(nir_path)

    meta["nir_available"] = "nir" in bands

    if pixel_size_m is None or float(pixel_size_m) <= 0:
        pixel_size_m = pixel_size_from_transform(transform, crs)
    meta["pixel_size_m"] = float(pixel_size_m)

    hw = next(iter(bands.values())).shape[:2]
    scl = _load_scl_sidecar(path, (int(hw[0]), int(hw[1])))
    if scl is not None:
        meta["scl"] = scl
        meta["scl_source"] = "sidecar"

    return Scene(
        bands=bands,
        transform=transform,
        crs=crs,
        path=path,
        pixel_size_m=float(pixel_size_m),
        nodata_mask=nodata_mask,
        meta=meta,
    )


def _scl_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """把彩色 SCL 预览图还原成类别编号。"""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for code, color in _SCL_RGB.items():
        match = (
            (rgb[..., 0] == color[0])
            & (rgb[..., 1] == color[1])
            & (rgb[..., 2] == color[2])
        )
        out[match] = np.uint8(code)
    return out


def _load_scl_sidecar(path: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """读取同名 _scl.tif / _scl.png。尺寸不一致则放弃，避免错位掩膜。"""
    stem, _ = os.path.splitext(path)
    for cand in (stem + "_scl.tif", stem + "_scl.tiff"):
        if not os.path.isfile(cand):
            continue
        try:
            arr, _, _, _, _ = _read_tiff(cand)
            if arr.ndim == 3:
                arr = arr[..., 0]
            scl = np.asarray(arr).astype(np.uint8)
            if scl.shape[:2] == shape:
                return scl
        except Exception:
            continue
    png = stem + "_scl.png"
    if os.path.isfile(png):
        try:
            from PIL import Image

            rgb = np.asarray(Image.open(png).convert("RGB"))
            if rgb.shape[:2] == shape:
                return _scl_from_rgb(rgb)
        except Exception:
            return None
    return None


def _looks_like_lonlat(transform: Any) -> bool:
    """没有 CRS 时：原点像经纬度、尺度像度，就按地理坐标处理。"""
    try:
        x0 = float(transform.c)
        y0 = float(transform.f)
        ax = abs(float(transform.a))
        return -180.0 <= x0 <= 180.0 and -90.0 <= y0 <= 90.0 and 1e-8 < ax < 0.05
    except Exception:
        return False


def pixel_size_from_transform(transform: Any, crs: Any = None, default: float = 10.0) -> float:
    """从 rasterio transform 读取像元尺寸（米）。

    地理坐标系（度）会按纬度换算成米，避免把 0.0001° 当成 0.0001 m 从而面积错十几个数量级。
    """
    if transform is None:
        return default
    try:
        a = float(getattr(transform, "a", 0.0))
        b = float(getattr(transform, "b", 0.0))
        d = float(getattr(transform, "d", 0.0))
        e = float(getattr(transform, "e", 0.0))
        sx = (a * a + d * d) ** 0.5
        sy = (b * b + e * e) ** 0.5
        if sx <= 0 and sy <= 0:
            return default
        geographic = False
        if crs is not None:
            try:
                from rasterio.crs import CRS

                c = crs if hasattr(crs, "is_geographic") else CRS.from_user_input(crs)
                geographic = bool(c.is_geographic)
            except Exception:
                geographic = False
        if not geographic and _looks_like_lonlat(transform):
            geographic = True
        if geographic:
            try:
                lat = float(transform.f)
            except Exception:
                lat = 0.0
            lat = max(-89.9, min(89.9, lat))
            m_lat = 111_320.0
            m_lon = 111_320.0 * math.cos(math.radians(lat))
            size = 0.5 * (sx * m_lon + sy * m_lat)
            return float(size) if size > 0 else default
        size = 0.5 * (sx + sy) if sy > 0 else sx
        return float(size) if size > 0 else default
    except Exception:  # pragma: no cover - 防御性分支
        return default


def estimate_cloud_mask(scene: "Scene") -> Optional[np.ndarray]:
    """优先用 SCL 真值；没有时用可见光高亮 + 光谱平坦（及近红外高值）兜底。"""
    scl = scene.meta.get("scl")
    if scl is not None:
        arr = np.asarray(scl)
        if arr.shape[:2] == scene.shape:
            return np.isin(arr, SCL_CLOUD_CLASSES)
    r = scene.bands.get("red")
    g = scene.bands.get("green")
    b = scene.bands.get("blue")
    if r is None or g is None or b is None:
        return None
    vis = (r + g + b) / 3.0
    flat = (np.abs(r - g) < 0.08) & (np.abs(g - b) < 0.08)
    bright = vis > 0.28
    if scene.has_nir:
        cloud = bright & flat & (scene.bands["nir"] > 0.22)
    else:
        cloud = bright & flat
    return np.asarray(cloud, dtype=bool)


def same_geo_grid(a: "Scene", b: "Scene", atol: float = 1e-4) -> bool:
    """两景是否已经在同一栅格上（同尺寸、同仿射、同 CRS）。"""
    if a.shape != b.shape or a.transform is None or b.transform is None:
        return False
    try:
        ta = [float(x) for x in a.transform[:6]]
        tb = [float(x) for x in b.transform[:6]]
    except Exception:
        return False
    if not np.allclose(ta, tb, atol=atol, rtol=1e-6):
        return False
    if a.crs is not None and b.crs is not None:
        try:
            from rasterio.crs import CRS

            ca = a.crs if hasattr(a.crs, "equals") else CRS.from_user_input(a.crs)
            cb = b.crs if hasattr(b.crs, "equals") else CRS.from_user_input(b.crs)
            if not bool(ca.equals(cb)):
                return False
        except Exception:
            if str(a.crs) != str(b.crs):
                return False
    return True


def reproject_array(
    src: np.ndarray,
    src_transform: Any,
    src_crs: Any,
    dst_transform: Any,
    dst_shape: Tuple[int, int],
    dst_crs: Any,
    resampling: str = "bilinear",
    src_nodata: Optional[float] = None,
    dst_nodata: float = 0.0,
) -> np.ndarray:
    """把二维数组重投影到目标网格。"""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    kind = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest, "cubic": Resampling.cubic}
    src = np.asarray(src)
    dst = np.full(dst_shape, dst_nodata, dtype=np.float32)
    reproject(
        source=src.astype(np.float32, copy=False),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=kind.get(resampling, Resampling.bilinear),
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )
    return dst


def reproject_scene_to(src: "Scene", dst_ref: "Scene") -> "Scene":
    """把 src 重投影到 dst_ref 的网格（尺寸 / transform / CRS）。"""
    if src.transform is None or dst_ref.transform is None:
        raise ValueError("无地理参考，无法重投影对齐")
    if src.crs is None or dst_ref.crs is None:
        raise ValueError("无坐标系，无法重投影对齐")
    shape = dst_ref.shape
    bands: Dict[str, np.ndarray] = {}
    first = next(iter(src.bands.values()))
    for name, band in src.bands.items():
        bands[name] = reproject_array(
            band,
            src.transform,
            src.crs,
            dst_ref.transform,
            shape,
            dst_ref.crs,
            resampling="bilinear",
            dst_nodata=0.0,
        ).astype(np.float32)

    # 必须显式产出"重投影后的无效像元掩膜"。
    # 波段在源覆盖范围外被填成 0.0，而 0 反射率本身是合法值，无法据此区分
    # "真实暗像元"和"没数据"。若下游只依赖 src.nodata_mask（S2 COG 常常为空），
    # 这些空白区会以 green=0、nir=0 进入 NDWI=(0-0)/eps=0，在浑浊水体那种
    # 略为负的全局阈值下被判成水体，产生大片假阳性淹没。
    src_valid = np.ones(first.shape[:2], dtype=np.float32)
    if src.nodata_mask is not None:
        src_valid = (~np.asarray(src.nodata_mask, dtype=bool)).astype(np.float32)
    covered = reproject_array(
        src_valid,
        src.transform,
        src.crs,
        dst_ref.transform,
        shape,
        dst_ref.crs,
        resampling="nearest",
        dst_nodata=0.0,
    )
    nodata = covered < 0.5
    meta = dict(src.meta)
    meta["reprojected_to"] = dst_ref.path or "reference"
    meta["reproject_nodata_fraction_pct"] = round(100.0 * float(nodata.mean()), 2)
    return Scene(
        bands=bands,
        transform=dst_ref.transform,
        crs=dst_ref.crs,
        path=src.path,
        pixel_size_m=float(dst_ref.pixel_size_m),
        nodata_mask=nodata,
        meta=meta,
    )


# --------------------------------------------------------------------------
# 归一化 / 拉伸
# --------------------------------------------------------------------------


def percentile_stretch(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """百分位拉伸 -> uint8。arr 可以是 (H,W) 或 (H,W,C)。"""
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low, high])
    if hi - lo < 1e-6:
        lo, hi = float(finite.min()), float(finite.max()) + 1e-6
    # 先把 NaN/Inf 归到 lo：astype(uint8) 对非有限值的转换属未定义行为
    # （新版本 numpy 会发 RuntimeWarning 并落到 0），无效像元会变成纯黑点。
    clean = np.where(np.isfinite(arr), arr, lo)
    out = np.clip((clean - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def normalize(
    chw: np.ndarray,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> np.ndarray:
    """(C,H,W) 反射率 -> 归一化张量。mean/std 按通道给出，长度不足时自动补齐。"""
    c = chw.shape[0]
    m = np.asarray(list(mean)[:c], dtype=np.float32).reshape(c, 1, 1)
    s = np.asarray(list(std)[:c], dtype=np.float32).reshape(c, 1, 1)
    s = np.where(s < 1e-6, 1e-6, s)
    return ((chw - m) / s).astype(np.float32)


# --------------------------------------------------------------------------
# 切片与拼接
# --------------------------------------------------------------------------


def iter_tiles(
    arr: np.ndarray,
    tile: int = 512,
    overlap: int = 64,
) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
    """把 (H,W,...) 影像切成带重叠的方块，yield (小块, (y0,x0))。

    重叠区在拼接时取平均，避免块与块之间的硬接缝。
    """
    h, w = arr.shape[:2]
    tile = int(tile)
    overlap = int(overlap)
    if tile <= 0:
        raise ValueError(f"tile 必须为正整数，收到 {tile}")
    if not 0 <= overlap < tile:
        # overlap >= tile 时 stride 退化成 1，切片数按 O(H*W) 爆炸
        # （512×512 影像会产出几十万块），内存和耗时都不可控。
        raise ValueError(f"overlap 必须满足 0 <= overlap < tile，收到 overlap={overlap}, tile={tile}")
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, h - tile + 1), stride)) or [0]
    xs = list(range(0, max(1, w - tile + 1), stride)) or [0]
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            yield arr[y0:y1, x0:x1], (y0, x0)


def stitch_tiles(
    tiles: Sequence[np.ndarray],
    coords: Sequence[Tuple[int, int]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """把切片结果加权平均拼回原尺寸。"""
    h, w = shape
    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for tile, (y0, x0) in zip(tiles, coords):
        th, tw = tile.shape[:2]
        acc[y0 : y0 + th, x0 : x0 + tw] += tile.astype(np.float32)
        weight[y0 : y0 + th, x0 : x0 + tw] += 1.0
    weight[weight == 0] = 1.0
    return acc / weight


def pad_to_tile(arr: np.ndarray, tile: int, mode: str = "reflect") -> Tuple[np.ndarray, Tuple[int, int]]:
    """把影像补边到 tile 的整数倍，返回 (补边后影像, (原始高, 原始宽))。"""
    h, w = arr.shape[:2]
    ph = (tile - h % tile) % tile
    pw = (tile - w % tile) % tile
    if ph == 0 and pw == 0:
        return arr, (h, w)
    # reflect 模式在任一维长度为 1 时 numpy 会直接报错；
    # 单行/单列影像（或裁剪到 1 像素）退化为 edge 复制。
    if mode == "reflect" and (h == 1 or w == 1):
        mode = "edge"
    pad_width = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad_width, mode=mode), (h, w)


def pixel_area_km2(pixel_size_m: float) -> float:
    """单个像元的面积（平方公里）。"""
    return (float(pixel_size_m) ** 2) / 1_000_000.0


def mask_area_km2(mask: np.ndarray, pixel_size_m: float) -> float:
    """掩膜面积（平方公里）。"""
    return float(np.count_nonzero(mask)) * pixel_area_km2(pixel_size_m)
