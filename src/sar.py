"""
慧眼识灾 · 雷达（SAR）处理模块
==============================

为什么需要它？
    光学卫星（NDWI 那套）在汛期会被云挡住 —— 2024 洞庭湖团洲垸决口时，
    AOI 窗口云量最低的一景仍有 33.5% 被云遮挡。雷达能穿云，是洪水监测的刚需。

本模块处理 **Sentinel-1 IW GRD**（哨兵一号 干涉宽幅 地距探测）数据：

    SAFE 产品
      → 读标定查找表（sigmaNought）        ← DN 只是相对亮度，必须标定成后向散射系数 σ0
      → σ0(dB) = 10*log10((DN/A)^2)
      → 斑点滤波（中值/均值）               ← SAR 固有相干斑噪声
      → 水体提取（低后向散射）              ← 平静水面像镜子，几乎不反射回雷达
      → 形态学 + 连通域 + 面积统计

与光学的差异（务必理解，否则会用错）：
    · 光学看"颜色"（NDWI 用绿/近红外），雷达看"回波强度"（σ0，单位 dB）
    · 水体在光学里是"亮"的（NDWI 高），在雷达里是"暗"的（σ0 低，通常 < -16 dB）
    · 雷达影像没有仿射变换，靠 GCP（地面控制点）定位 —— 本模块用 GCP 拟合仿射变换
    · 不同日期的两景存在几何偏移（实测 147 m），变化检测前必须先对齐

典型用法：
    from src.sar import load_s1_scene, detect_water_sar
    sc = load_s1_scene("...SAFE", polarization="VV")
    db = sc.read_db(row=8000, col=7700, size=4096)
    mask, prob, meta = detect_water_sar(db)
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-10

# Sentinel-1 IW GRD 常用水体阈值（dB）
DEFAULT_WATER_DB = {"VV": -16.0, "VH": -19.0}

# 仅作兜底：河南/华北。正常路径按 GCP 经纬度选带。
DEFAULT_UTM = "EPSG:32649"


def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """由经纬度选 UTM 带（尼泊尔 45N、河南 49N）。"""
    import math
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    zone = math.floor((lon + 180.0) / 6.0) + 1
    zone = min(max(int(zone), 1), 60)
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def pixel_size_m_from_affine(transform: Any) -> float:
    """仿射矩阵的地面像元尺寸（米），含旋转项。"""
    try:
        ax, bx = float(transform.a), float(transform.b)
        dx, ey = float(transform.d), float(transform.e)
        sx = (ax ** 2 + dx ** 2) ** 0.5
        sy = (bx ** 2 + ey ** 2) ** 0.5
        size = 0.5 * (sx + sy)
        return float(size) if size > 0 else 10.0
    except Exception:
        return 10.0


# --------------------------------------------------------------------------
# 标定查找表
# --------------------------------------------------------------------------


@dataclass
class CalibrationLUT:
    """Sentinel-1 标定查找表：DN -> σ0。"""

    lines: np.ndarray  # (n_vec,) 方位向行号
    pixels: np.ndarray  # (n_vec, n_pix) 距离向列号
    sigma: np.ndarray  # (n_vec, n_pix) sigmaNought 标定常数

    def at(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """插值出 (len(rows), len(cols)) 的标定常数。"""
        rows = np.asarray(rows, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64)
        # 先沿距离向插值
        ri = np.stack([np.interp(cols, self.pixels[i], self.sigma[i]) for i in range(len(self.lines))])
        # 再沿方位向插值
        out = np.empty((len(rows), len(cols)), dtype=np.float64)
        for j in range(len(cols)):
            out[:, j] = np.interp(rows, self.lines, ri[:, j])
        return out


def calibration_xml_candidates(safe_dir: str, polarization: str = "vv") -> List[str]:
    """标定 XML，排除同目录下的 noise-*.xml。"""
    pol = polarization.lower()
    for sub in ("annotation/calibration", "annotation", ""):
        folder = os.path.join(safe_dir, *sub.split("/")) if sub else safe_dir
        hits = sorted(glob.glob(os.path.join(folder, f"calibration-*{pol}*.xml")))
        if hits:
            return hits
        all_xml = glob.glob(os.path.join(folder, f"*{pol}*.xml"))
        cands = sorted(p for p in all_xml if not os.path.basename(p).lower().startswith("noise"))
        if cands:
            return cands
    return []


def parse_calibration(safe_dir: str, polarization: str = "vv") -> CalibrationLUT:
    """解析 SAFE 里的标定 XML（annotation/calibration/calibration-*-{pol}-*.xml）。"""
    pol = polarization.lower()
    cands = calibration_xml_candidates(safe_dir, pol)
    if not cands:
        raise FileNotFoundError(f"找不到 {pol} 极化标定文件：{safe_dir}")
    with open(cands[0], encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()

    lines: List[int] = []
    pixels: List[np.ndarray] = []
    sigmas: List[np.ndarray] = []
    for block in re.findall(r"<calibrationVector>(.*?)</calibrationVector>", txt, re.S):
        m_line = re.search(r"<line>(\d+)</line>", block)
        m_pix = re.search(r"<pixel[^>]*>([^<]+)</pixel>", block)
        m_sig = re.search(r"<sigmaNought[^>]*>([^<]+)</sigmaNought>", block)
        if not (m_line and m_pix and m_sig):
            continue
        lines.append(int(m_line.group(1)))
        pixels.append(np.fromstring(m_pix.group(1), sep=" "))
        sigmas.append(np.fromstring(m_sig.group(1), sep=" "))
    if not lines:
        raise ValueError(f"标定文件解析失败：{cands[0]}")
    return CalibrationLUT(np.asarray(lines), np.asarray(pixels), np.asarray(sigmas))


def dn_to_db(dn: np.ndarray, lut_at: np.ndarray) -> np.ndarray:
    """DN -> σ0(dB)：σ0 = (DN / A)^2，再取 10log10。"""
    dn = np.asarray(dn, dtype=np.float32)
    sigma0 = (dn / np.maximum(lut_at, EPS)) ** 2
    return (10.0 * np.log10(np.maximum(sigma0, EPS))).astype(np.float32)


# --------------------------------------------------------------------------
# GCP 仿射定位
# --------------------------------------------------------------------------


@dataclass
class GcpAffine:
    """由 GCP 拟合的 (col,row) -> (x,y) 仿射变换。"""

    matrix: np.ndarray  # 3x3
    crs: str
    residual_m: float
    n_gcps: int

    def pixel_to_xy(self, col: np.ndarray, row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        col = np.asarray(col, dtype=np.float64)
        row = np.asarray(row, dtype=np.float64)
        x = self.matrix[0, 0] * col + self.matrix[0, 1] * row + self.matrix[0, 2]
        y = self.matrix[1, 0] * col + self.matrix[1, 1] * row + self.matrix[1, 2]
        return x, y

    def xy_to_pixel(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inv = np.linalg.inv(self.matrix)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        col = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        row = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
        return col, row


def find_safe_dir(path: str) -> str:
    """接受 SAFE 目录、其父目录或 zip 解压目录，返回 .SAFE 路径。"""
    path = os.path.abspath(path)
    if path.endswith(".SAFE") and os.path.isdir(path):
        return path
    for pattern in ("*.SAFE", "*/*.SAFE", "*/*/*.SAFE"):
        found = sorted(glob.glob(os.path.join(path, pattern)))
        if found:
            return found[0]
    raise FileNotFoundError(f"在 {path} 下找不到 .SAFE 目录")


def find_measurement(safe_dir: str, polarization: str = "vv") -> str:
    pol = polarization.lower()
    cands = glob.glob(os.path.join(safe_dir, "measurement", f"*{pol}*.tiff")) or \
            glob.glob(os.path.join(safe_dir, "measurement", f"*{pol}*.tif"))
    if not cands:
        raise FileNotFoundError(f"找不到 {pol} 极化测量数据：{safe_dir}")
    return sorted(cands)[0]


def fit_gcp_affine(tif_path: str, dst_crs: Optional[str] = None) -> GcpAffine:
    """用影像自带的 GCP 拟合仿射变换（GRD 没有仿射变换，必须自己拟合）。

    dst_crs 缺省时按 GCP 平均经纬度选 UTM 带。
    返回的 residual_m 是拟合残差（米），反映地形引起的几何畸变程度。
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(tif_path) as src:
        gcps, gcp_crs = src.gcps
    if not gcps:
        raise ValueError(f"{tif_path} 没有 GCP，无法定位")

    rows = np.array([g.row for g in gcps], dtype=np.float64)
    cols = np.array([g.col for g in gcps], dtype=np.float64)
    lons = np.array([g.x for g in gcps], dtype=np.float64)
    lats = np.array([g.y for g in gcps], dtype=np.float64)
    if dst_crs is None:
        dst_crs = utm_epsg_from_lonlat(float(np.mean(lons)), float(np.mean(lats)))
    xs, ys = warp_transform(str(gcp_crs or "EPSG:4326"), dst_crs, lons.tolist(), lats.tolist())
    xs, ys = np.asarray(xs), np.asarray(ys)

    A = np.column_stack([cols, rows, np.ones_like(cols)])
    coef_x, *_ = np.linalg.lstsq(A, xs, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, ys, rcond=None)
    matrix = np.array(
        [
            [coef_x[0], coef_x[1], coef_x[2]],
            [coef_y[0], coef_y[1], coef_y[2]],
            [0.0, 0.0, 1.0],
        ]
    )
    pred_x = A @ coef_x
    pred_y = A @ coef_y
    residual = float(np.sqrt(np.mean((pred_x - xs) ** 2 + (pred_y - ys) ** 2)))
    return GcpAffine(matrix=matrix, crs=dst_crs, residual_m=residual, n_gcps=len(gcps))


# --------------------------------------------------------------------------
# 场景读取
# --------------------------------------------------------------------------


@dataclass
class S1Scene:
    """一景 Sentinel-1 IW GRD。"""

    safe_dir: str
    polarization: str
    tif_path: str
    lut: CalibrationLUT
    affine: GcpAffine
    width: int = 0
    height: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.safe_dir).replace(".SAFE", "")

    @property
    def datetime(self) -> str:
        m = re.search(r"_(\d{8}T\d{6})_", self.name)
        return m.group(1) if m else ""

    def read_dn(self, col0: int, row0: int, width: int, height: int) -> np.ndarray:
        """读原始 DN 窗口。"""
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(self.tif_path) as src:
            win = Window(col0, row0, width, height)
            return src.read(1, window=win).astype(np.float32)

    def read_db(self, col0: int, row0: int, width: int, height: int) -> np.ndarray:
        """读窗口并标定为 σ0(dB)。"""
        dn = self.read_dn(col0, row0, width, height)
        rows = np.arange(row0, row0 + dn.shape[0], dtype=np.float64)
        cols = np.arange(col0, col0 + dn.shape[1], dtype=np.float64)
        lut_at = self.lut.at(rows, cols)
        return dn_to_db(dn, lut_at)

    def read_db_decimated(self, out_shape: Tuple[int, int]) -> np.ndarray:
        """整景降采样并标定（用于快速定位）。"""
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(self.tif_path) as src:
            dn = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(np.float32)
            h, w = src.height, src.width
        rows = np.linspace(0, h - 1, out_shape[0], dtype=np.float64)
        cols = np.linspace(0, w - 1, out_shape[1], dtype=np.float64)
        return dn_to_db(dn, self.lut.at(rows, cols))

    def window_for_xy(self, x: float, y: float, size: int) -> Tuple[int, int]:
        """给定 UTM 坐标，返回以它为中心的窗口左上角（自动裁剪到影像范围）。"""
        col, row = self.affine.xy_to_pixel(x, y)
        col0 = int(round(float(col) - size / 2))
        row0 = int(round(float(row) - size / 2))
        col0 = max(0, min(col0, self.width - size))
        row0 = max(0, min(row0, self.height - size))
        return col0, row0

    def transform_for_window(self, col0: int, row0: int) -> Any:
        """窗口左上角对应的仿射变换（可直接写进 GeoTIFF）。"""
        from rasterio.transform import Affine

        m = self.affine.matrix
        return Affine(m[0, 0], m[0, 1], m[0, 0] * col0 + m[0, 1] * row0 + m[0, 2],
                      m[1, 0], m[1, 1], m[1, 0] * col0 + m[1, 1] * row0 + m[1, 2])


def load_s1_scene(path: str, polarization: str = "VV", dst_crs: Optional[str] = None) -> S1Scene:
    """加载一景 Sentinel-1 GRD（path 可以是 SAFE 目录或其父目录）。

    dst_crs 用于把两景钉到同一投影（变化检测时把灾后跟灾前走同一带）。
    """
    import rasterio

    safe_dir = find_safe_dir(path)
    tif = find_measurement(safe_dir, polarization)
    lut = parse_calibration(safe_dir, polarization)
    affine = fit_gcp_affine(tif, dst_crs=dst_crs)
    with rasterio.open(tif) as src:
        width, height = src.width, src.height
    return S1Scene(
        safe_dir=safe_dir,
        polarization=polarization.upper(),
        tif_path=tif,
        lut=lut,
        affine=affine,
        width=width,
        height=height,
        meta={"source": os.path.basename(safe_dir), "polarization": polarization.upper()},
    )


# --------------------------------------------------------------------------
# 水体提取
# --------------------------------------------------------------------------


def speckle_filter(db: np.ndarray, size: int = 3) -> np.ndarray:
    """斑点滤波（中值），压制 SAR 相干斑噪声。"""
    if size <= 1:
        return db
    try:
        import cv2

        return cv2.medianBlur(db.astype(np.float32), int(size))
    except Exception:  # pragma: no cover
        from scipy import ndimage  # 源码环境兜底

        return ndimage.median_filter(db, size=int(size))


def _otsu(values: np.ndarray, nbins: int = 256) -> float:
    v = values[np.isfinite(values)]
    if v.size < 64:
        return float("nan")
    try:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(v, nbins=nbins))
    except Exception:
        hist, edges = np.histogram(v, bins=nbins)
        hist = hist.astype(np.float64)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = hist.sum()
        w0 = np.cumsum(hist)
        w1 = total - w0
        valid = (w0 > 0) & (w1 > 0)
        mu = np.cumsum(hist * centers)
        mu_t = mu[-1]
        mu0 = np.divide(mu, w0, out=np.zeros_like(mu), where=w0 > 0)
        mu1 = np.divide(mu_t - mu, w1, out=np.zeros_like(mu), where=w1 > 0)
        between = w0 * w1 * (mu0 - mu1) ** 2
        between[~valid] = -1.0
        return float(centers[int(np.argmax(between))])


def detect_water_sar(
    db: np.ndarray,
    polarization: str = "VV",
    threshold_db: Optional[float] = None,
    method: str = "fixed",
    speckle: int = 3,
    min_area_px: int = 100,
    softness_db: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """从 σ0(dB) 影像提取水体。

    原理：平静水面近似镜面反射，几乎不把能量反射回雷达，因此后向散射极低。
         VV 极化水体通常 < -16 dB，VH 通常 < -19 dB。

    参数
    ----
    method : "fixed" 用经验阈值（默认，稳健）；"otsu" 自动阈值（水面占比大时可用）
    threshold_db : 指定阈值，覆盖默认值
    """
    db = np.asarray(db, dtype=np.float32)
    if threshold_db is None:
        threshold_db = DEFAULT_WATER_DB.get(polarization.upper(), -16.0)

    filtered = speckle_filter(db, speckle) if speckle and speckle > 1 else db

    meta: Dict[str, Any] = {
        "polarization": polarization.upper(),
        "method": method,
        "speckle": int(speckle),
    }
    if method == "otsu":
        t = _otsu(filtered)
        if not np.isfinite(t) or not (-30.0 < t < 0.0):
            t = float(threshold_db)
            meta["threshold_source"] = "fallback"
        else:
            meta["threshold_source"] = "otsu"
        threshold = float(t)
    else:
        threshold = float(threshold_db)
        meta["threshold_source"] = "fixed"
    meta["threshold_db"] = threshold

    # 软概率：越低于阈值越确信是水
    prob = 1.0 / (1.0 + np.exp((filtered - threshold) / max(softness_db, 0.1)))
    mask = prob > 0.5

    # 后处理：形态学 + 连通域
    from .postprocess import clean_mask

    mask = clean_mask(mask, open_radius=1, close_radius=2, min_area_px=min_area_px, fill_holes=True)
    meta["water_fraction_raw"] = float(mask.mean())
    return mask.astype(bool), prob.astype(np.float32), meta


# --------------------------------------------------------------------------
# 变化检测
# --------------------------------------------------------------------------


def sar_change(
    pre_db: np.ndarray,
    post_db: np.ndarray,
    polarization: str = "VV",
    threshold_db: Optional[float] = None,
    drop_db: float = 3.0,
    speckle: int = 3,
    min_area_px: int = 100,
) -> Dict[str, Any]:
    """双时相变化检测：新增淹没 / 退水 / 持续水体。

    同时用两个条件，避免把"本来就是水"和"噪声"算成新增：
        · 绝对条件：灾后 σ0 低于水体阈值
        · 相对条件：灾后比灾前暗了 drop_db 以上
    """
    pre_m, pre_p, pre_meta = detect_water_sar(pre_db, polarization, threshold_db, "fixed", speckle, min_area_px)
    post_m, post_p, post_meta = detect_water_sar(post_db, polarization, threshold_db, "fixed", speckle, min_area_px)
    thr = post_meta["threshold_db"]

    pre_f = speckle_filter(pre_db, speckle)
    post_f = speckle_filter(post_db, speckle)
    drop = post_f - pre_f

    new = post_m & (drop < -drop_db) & (~pre_m)
    receded = pre_m & (~post_m)
    persistent = pre_m & post_m
    return {
        "pre_mask": pre_m,
        "post_mask": post_m,
        "new": new,
        "receded": receded,
        "persistent": persistent,
        "drop_db": drop,
        "threshold_db": thr,
        "pre_meta": pre_meta,
        "post_meta": post_meta,
    }


def area_km2(mask: np.ndarray, pixel_size_m: float = 10.0) -> float:
    return float(np.count_nonzero(mask)) * (pixel_size_m ** 2) / 1e6


# --------------------------------------------------------------------------
# 坐标与重投影
# --------------------------------------------------------------------------


def lonlat_to_utm(lon: float, lat: float, dst_crs: Optional[str] = None) -> Tuple[float, float]:
    from rasterio.warp import transform as warp_transform

    if dst_crs is None:
        dst_crs = utm_epsg_from_lonlat(lon, lat)
    xs, ys = warp_transform("EPSG:4326", dst_crs, [lon], [lat])
    return float(xs[0]), float(ys[0])


def utm_to_lonlat(x: float, y: float, src_crs: str = DEFAULT_UTM) -> Tuple[float, float]:
    from rasterio.warp import transform as warp_transform

    lons, lats = warp_transform(src_crs, "EPSG:4326", [x], [y])
    return float(lons[0]), float(lats[0])


def reproject_decimated(scene: S1Scene, out_shape: Tuple[int, int], grid, grid_shape: Tuple[int, int]) -> np.ndarray:
    """把整景降采样后按 GCP 仿射重投影到公共 UTM 网格。

    cv2.remap 需要"每个目标像元对应的源像元坐标"，所以由网格坐标反算回本景像素。
    """
    import cv2

    db = scene.read_db_decimated(out_shape)
    gh, gw = grid_shape
    grow, gcol = np.meshgrid(np.arange(gh, dtype=np.float64), np.arange(gw, dtype=np.float64), indexing="ij")
    xs = grid.c + grid.a * gcol + grid.b * grow
    ys = grid.f + grid.d * gcol + grid.e * grow
    src_col, src_row = scene.affine.xy_to_pixel(xs, ys)
    src_col = src_col * (out_shape[1] / scene.width)
    src_row = src_row * (out_shape[0] / scene.height)
    return cv2.remap(db, src_col.astype(np.float32), src_row.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-35.0)


def locate_change_area(
    pre: S1Scene,
    post: S1Scene,
    polarization: str = "VV",
    threshold_db: Optional[float] = None,
    drop_db: float = 3.0,
    coarse: int = 1536,
    progress=None,
) -> Dict[str, Any]:
    """自动定位洪水：两景降到公共网格，找"低回波 + 明显变暗"的最大连通区。"""
    import cv2
    from rasterio.transform import from_origin

    if threshold_db is None:
        threshold_db = DEFAULT_WATER_DB.get(polarization.upper(), -16.0)

    corners = []
    for sc in (pre, post):
        for c, r in ((0, 0), (sc.width, 0), (0, sc.height), (sc.width, sc.height)):
            cx, cy = sc.affine.pixel_to_xy(c, r)
            corners.append((float(cx), float(cy)))
    xs = np.array([c[0] for c in corners])
    ys = np.array([c[1] for c in corners])
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    res = max(30.0, (x1 - x0) / 4500.0)
    gw, gh = int((x1 - x0) / res), int((y1 - y0) / res)
    grid = from_origin(x0, y1, res, res)
    if progress:
        progress(f"公共网格 {gw}×{gh} @ {res:.0f} m")

    pre_g = reproject_decimated(pre, (coarse, int(coarse * pre.width / pre.height)), grid, (gh, gw))
    post_g = reproject_decimated(post, (coarse, int(coarse * post.width / post.height)), grid, (gh, gw))
    if pre_g.shape != post_g.shape:
        h = min(pre_g.shape[0], post_g.shape[0])
        w = min(pre_g.shape[1], post_g.shape[1])
        pre_g, post_g = pre_g[:h, :w], post_g[:h, :w]

    drop = post_g - pre_g
    abs_th = threshold_db + 3.0  # 定位放宽 3 dB：降采样会稀释窄水体
    valid = (pre_g > -34) & (post_g > -34)
    cand = ((post_g < abs_th) & (drop < -drop_db) & valid).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, _labels, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        return {"found": False, "reason": "没有找到明显的新增淹没信号"}
    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    cx, cy = cents[idx]
    ux = grid.c + grid.a * cx
    uy = grid.f + grid.e * cy
    lon, lat = utm_to_lonlat(ux, uy, src_crs=pre.affine.crs)
    area = float(stats[idx, cv2.CC_STAT_AREA]) * res * res / 1e6
    return {
        "found": True, "x": float(ux), "y": float(uy), "lon": lon, "lat": lat,
        "area_km2": area, "grid_res_m": res, "candidate_frac": float(cand.mean()),
        "source": "auto",
    }


# --------------------------------------------------------------------------
# 落盘与出图
# --------------------------------------------------------------------------


def save_db_geotiff(path: str, db: np.ndarray, transform, crs: str) -> str:
    """保存 σ0(dB) 为 int16（×100），带 CRS/transform。"""
    import rasterio

    arr = np.clip(np.round(np.asarray(db) * 100.0), -32768, 32767).astype(np.int16)
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
        "dtype": "int16", "crs": crs, "transform": transform, "compress": "deflate",
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
        ds.set_band_description(1, "sigma0_dB_x100")
    return path


def read_db_geotiff(path: str) -> Tuple[np.ndarray, Any, str]:
    """读回 σ0(dB) GeoTIFF（自动除以 100）。"""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32) / 100.0
        return arr, src.transform, str(src.crs)


def db_to_gray(db: np.ndarray, lo: float = -25.0, hi: float = 0.0) -> np.ndarray:
    """dB -> 灰度（越黑=回波越弱=越可能是水）。"""
    return (np.clip((np.asarray(db) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _save_rgb(path: str, arr: np.ndarray, max_size: int = 2048) -> str:
    """保存可视化 PNG；超过 max_size 自动等比缩小（GeoTIFF 仍是全分辨率）。"""
    from PIL import Image

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    im = Image.fromarray(a[..., :3])
    if max_size and max(im.size) > max_size:
        scale = max_size / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path)
    return path


def process_sar_pair(
    pre_path: str,
    post_path: str,
    polarization: str = "VV",
    out_dir: str = "outputs/sar",
    sample_id: Optional[str] = None,
    size: int = 4096,
    threshold_db: Optional[float] = None,
    drop_db: float = 3.0,
    min_area_px: int = 100,
    aoi: Optional[Tuple[float, float]] = None,
    progress=None,
) -> Dict[str, Any]:
    """一条龙：加载 → 定位 → 裁剪 → 水体提取 → 变化检测 → 出图落盘。

    返回 dict（含 paths / stats / provenance），可直接给界面或脚本用。
    """
    def say(msg: str) -> None:
        if progress:
            progress(msg)

    pol = polarization.upper()
    thr = threshold_db if threshold_db is not None else DEFAULT_WATER_DB.get(pol, -16.0)

    say(f"加载灾前景：{os.path.basename(os.path.abspath(pre_path))}")
    pre = load_s1_scene(pre_path, pol)
    say(f"加载灾后景：{os.path.basename(os.path.abspath(post_path))}")
    post = load_s1_scene(post_path, pol, dst_crs=pre.affine.crs)

    # 定位
    if aoi is not None:
        lon, lat = aoi
        ux, uy = lonlat_to_utm(lon, lat, dst_crs=pre.affine.crs)
        loc = {"found": True, "x": ux, "y": uy, "lon": lon, "lat": lat, "area_km2": None, "source": "manual"}
        say(f"使用指定 AOI：经纬({lon},{lat})")
    else:
        say("自动定位洪水（降采样 + 公共网格 + 变化聚类）...")
        loc = locate_change_area(pre, post, pol, thr, drop_db, progress=progress)
        if not loc.get("found"):
            raise RuntimeError(loc.get("reason", "自动定位失败，请用 AOI 手动指定"))

    max_win = min(pre.width, pre.height, post.width, post.height)
    if size > max_win:
        size = int(max_win)
        say(f"窗口大于影像，已缩小为 {size}")

    # 裁剪
    pre_col0, pre_row0 = pre.window_for_xy(loc["x"], loc["y"], size)
    post_col0, post_row0 = post.window_for_xy(loc["x"], loc["y"], size)
    say(f"裁剪窗口 {size}×{size}（灾前 {pre_col0},{pre_row0} | 灾后 {post_col0},{post_row0}）")
    pre_db = pre.read_db(pre_col0, pre_row0, size, size)
    post_db = post.read_db(post_col0, post_row0, size, size)
    pre_tf = pre.transform_for_window(pre_col0, pre_row0)
    post_tf = post.transform_for_window(post_col0, post_row0)
    # 全分辨率必须重投影到同一网格，不能像素对像素相减
    if pre_db.shape != post_db.shape or pre.affine.crs != post.affine.crs:
        need_warp = True
    else:
        try:
            need_warp = not np.allclose(list(pre_tf)[:6], list(post_tf)[:6], atol=1e-2, rtol=1e-4)
        except Exception:
            need_warp = True
    if need_warp:
        say("将灾后窗口重投影到灾前网格…")
        from .preprocess import reproject_array

        post_db = reproject_array(
            post_db, post_tf, post.affine.crs, pre_tf, pre_db.shape, pre.affine.crs,
            resampling="bilinear", dst_nodata=-35.0,
        )
        post_tf = pre_tf

    # 提取
    say("水体提取 + 变化检测...")
    chg = sar_change(pre_db, post_db, pol, thr, drop_db, speckle=3, min_area_px=min_area_px)
    px = pixel_size_m_from_affine(pre_tf)
    stats = {
        "pre_water_km2": area_km2(chg["pre_mask"], px),
        "post_water_km2": area_km2(chg["post_mask"], px),
        "new_water_km2": area_km2(chg["new"], px),
        "receded_km2": area_km2(chg["receded"], px),
        "persistent_km2": area_km2(chg["persistent"], px),
        "window_km2": (size * px / 1000.0) ** 2,
        "pixel_size_m": px,
        "threshold_db": chg["threshold_db"],
        "polarization": pol,
    }

    # 落盘
    os.makedirs(out_dir, exist_ok=True)
    sid = sample_id or (f"henan_{post.datetime[:8]}" if post.datetime else "sar_sample")
    from .postprocess import change_map_rgb, overlay_mask

    pre_tif = save_db_geotiff(os.path.join(out_dir, f"{sid}_pre_{pol.lower()}.tif"), pre_db,
                              pre_tf, pre.affine.crs)
    post_tif = save_db_geotiff(os.path.join(out_dir, f"{sid}_post_{pol.lower()}.tif"), post_db,
                               post_tf, pre.affine.crs)

    pre_gray = np.stack([db_to_gray(pre_db)] * 3, axis=-1)
    post_gray = np.stack([db_to_gray(post_db)] * 3, axis=-1)
    pre_ov = overlay_mask(pre_gray, chg["pre_mask"], color=(0, 255, 255), alpha=0.55)
    post_ov = overlay_mask(post_gray, chg["post_mask"], color=(0, 255, 255), alpha=0.55)
    cmap = change_map_rgb(chg["pre_mask"], chg["post_mask"], np.stack([db_to_gray(pre_db)] * 3, -1))
    p_pre = _save_rgb(os.path.join(out_dir, f"{sid}_pre_overlay.png"), pre_ov)
    p_post = _save_rgb(os.path.join(out_dir, f"{sid}_post_overlay.png"), post_ov)
    p_change = _save_rgb(os.path.join(out_dir, f"{sid}_change.png"), cmap)
    _save_rgb(os.path.join(out_dir, f"{sid}_post_mask.png"),
              (chg["post_mask"].astype(np.uint8) * 255))

    preview = _make_preview(os.path.join(out_dir, f"{sid}_preview.jpg"),
                            [p_pre, p_post, p_change],
                            [f"灾前 {pre.datetime[:8]}", f"灾后 {post.datetime[:8]}",
                             "红=新增淹没 绿=退水 青=持续"])

    result = {
        "id": sid,
        "paths": {
            "preview": preview, "pre_tif": pre_tif, "post_tif": post_tif,
            "pre_overlay": p_pre, "post_overlay": p_post, "change": p_change,
        },
        "stats": stats,
        "masks": {"pre": chg["pre_mask"], "post": chg["post_mask"],
                  "new": chg["new"], "receded": chg["receded"]},
        "db": {"pre": pre_db, "post": post_db},
        "provenance": {
            "sensor": "Sentinel-1 IW GRD",
            "pre_scene": pre.name, "post_scene": post.name,
            "polarization": pol, "threshold_db": chg["threshold_db"], "drop_db": drop_db,
            "aoi_source": loc.get("source"), "aoi_lonlat": [round(loc["lon"], 5), round(loc["lat"], 5)],
            "window": [size, size],
            "pixel_size_m": round(px, 2),
            "crs": pre.affine.crs,
            "coregistered": bool(need_warp),
            "gcp_residual_m": {"pre": round(pre.affine.residual_m, 1),
                               "post": round(post.affine.residual_m, 1)},
            "note": "σ0(dB) 存为 int16（×100）；水体 = 低回波 + 相对变暗；变化检测前已配准到灾前网格",
        },
    }
    say(f"完成：灾前 {stats['pre_water_km2']:.2f} km² → 灾后 {stats['post_water_km2']:.2f} km²，"
        f"新增 {stats['new_water_km2']:.2f} km²")
    return result


def _make_preview(path: str, panels: List[str], captions: List[str], max_width: int = 1400) -> str:
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        if os.path.isfile(f):
            try:
                font = ImageFont.truetype(f, 20)
                break
            except Exception:
                font = None

    imgs = [Image.open(p).convert("RGB") for p in panels]
    h = max(i.height for i in imgs)
    imgs = [i if i.height == h else i.resize((int(i.width * h / i.height), h)) for i in imgs]
    gap, bar = 8, 34
    W = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (W, h + bar), (18, 30, 48))
    d = ImageDraw.Draw(canvas)
    x = 0
    for im, cap in zip(imgs, captions):
        canvas.paste(im, (x, bar))
        try:
            d.text((x + 8, 8), cap, fill=(255, 255, 255), font=font)
        except Exception:
            d.text((x + 8, 8), cap.encode("ascii", "replace").decode(), fill=(255, 255, 255))
        x += im.width + gap
    if canvas.width > max_width:
        canvas = canvas.resize((max_width, int(canvas.height * max_width / canvas.width)), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    canvas.save(path, quality=88, optimize=True)
    return path
