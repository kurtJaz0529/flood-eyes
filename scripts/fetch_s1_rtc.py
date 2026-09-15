"""
慧眼识灾 · 抓取 Sentinel-1 RTC（免注册、地形校正）并提取洪水
==============================================================

为什么用 RTC？
    · **免注册**：微软 Planetary Computer 公开数据，只需一个 SAS token（脚本自动取）
    · **地形校正**：山区不会像 GRD 那样有严重的叠掩/透视收缩，直接带 UTM 坐标
    · **已经标定**：像素值就是线性 σ0，转 dB 即可（不用解析 GRD 的标定查找表）
    · **同一轨道对齐**：相隔 12 天的两景几何几乎一致，变化检测不用额外配准

数据来源：Microsoft Planetary Computer `sentinel-1-rtc`
    https://planetarycomputer.microsoft.com/api/stac/v1

用法：
    # 尼泊尔 Rasuwa 山洪（2026-08-26）
    python scripts/fetch_s1_rtc.py --lon 85.368 --lat 28.247 ^
        --before 2026-08-16 --after 2026-08-28 --out data/nepal2026 --size 2048

    # 其它事件同理，只要给中心经纬度 + 灾前/灾后日期
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src.sar import (  # noqa: E402
    _assert_within,
    _make_preview,
    _save_rgb,
    area_km2,
    db_to_gray,
    detect_water_sar,
    normalize_polarization,
    safe_sid,
    sar_change,
    save_db_geotiff,
)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
COLLECTION = "sentinel-1-rtc"

_GDAL_ENV = {
    "GDAL_HTTP_TIMEOUT": "300",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_HTTP_MULTIPLEX": "NO",
    "GDAL_HTTP_VERSION": "1",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "AWS_NO_SIGN_REQUEST": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "67108864",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stac_search(bbox: List[float], dt: str, limit: int = 60) -> List[Dict[str, Any]]:
    body = {"collections": [COLLECTION], "bbox": bbox, "datetime": dt, "limit": limit,
            "sortby": [{"field": "properties.datetime", "direction": "asc"}]}
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("features", [])


def get_sas_token() -> str:
    req = urllib.request.Request(SAS, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["token"]


def pick_candidates(items: List[Dict[str, Any]], date: str, n: int = 6) -> List[Dict[str, Any]]:
    """按日期接近程度排序返回前 n 个候选（同一轨道有多个切片，需要逐个试）。"""
    def dist(it: Dict[str, Any]) -> float:
        return abs((np.datetime64(it["properties"]["datetime"][:10]) - np.datetime64(date)) / np.timedelta64(1, "D"))

    return sorted(items, key=dist)[:n]


class WindowNotCovered(RuntimeError):
    """窗口不在该切片覆盖范围内（不重试，直接换下一个候选）。"""


def read_rtc_window(href: str, token: str, lon: float, lat: float, size_px: int,
                    retries: int = 3) -> Tuple[np.ndarray, Any, str, np.ndarray]:
    """读 RTC 窗口（10m），返回 (线性 σ0 数组, transform, crs, 有效像元掩膜)。

    第 4 个返回值标记真实读到的像元。补边区必须由调用方排除：
    该数组是线性 σ0，补边用的 0 换算成 dB 约 -60 dB，远低于水体阈值，
    若不加掩膜会被判成"水"，让窗口边缘出现大片假淹没、面积统计虚高。
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform as warp_transform

    url = "/vsicurl/" + href + "?" + token
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            log(f"    读取 RTC 窗口 {size_px}×{size_px}（第 {attempt + 1}/{retries} 次，可能需 1–3 分钟）…")
            with rasterio.Env(**_GDAL_ENV):
                with rasterio.open(url) as src:
                    xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                    half = size_px * 5.0  # 10 m 像元 -> 米
                    win = from_bounds(xs[0] - half, ys[0] - half, xs[0] + half, ys[0] + half, src.transform)
                    arr = src.read(1, window=win).astype(np.float32)
                    valid = np.isfinite(arr) & (arr > 0)
                    if arr.shape != (size_px, size_px):
                        h, w = arr.shape
                        if h < size_px * 0.7 or w < size_px * 0.7:
                            raise WindowNotCovered(f"窗口尺寸 {arr.shape}，该切片不覆盖 AOI")
                        kh, kw = min(h, size_px), min(w, size_px)
                        pad = np.zeros((size_px, size_px), dtype=np.float32)
                        pad_valid = np.zeros((size_px, size_px), dtype=bool)
                        pad[:kh, :kw] = arr[:kh, :kw]
                        pad_valid[:kh, :kw] = valid[:kh, :kw]
                        arr, valid = pad, pad_valid
                    finite = arr[valid]
                    if finite.size == 0 or float(np.median(np.maximum(finite, 0))) <= 0:
                        raise WindowNotCovered("窗口无有效值（可能不覆盖 AOI）")
                    return arr, src.window_transform(win), str(src.crs), valid
        except WindowNotCovered:
            raise
        except Exception as exc:
            last = exc
            log(f"    读取失败({type(exc).__name__})，{3 * (attempt + 1)} 秒后重试")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"RTC 窗口读取失败：{last}")


def run_rtc_flood(
    lon: float,
    lat: float,
    before: str,
    after: str,
    out_dir: str,
    sid: Optional[str] = None,
    size: int = 1536,
    threshold_db: float = -16.0,
    drop_db: float = 3.5,
    pol: str = "VV",
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """检索 Sentinel-1 RTC → 裁窗 → 水体提取。供 CLI 和软件界面共用。"""
    def say(msg: str) -> None:
        log(msg)
        if progress:
            progress(msg)

    pad = 0.4
    bbox = [lon - pad, lat - pad, lon + pad, lat + pad]
    d0 = (np.datetime64(before) - np.timedelta64(6, "D")).astype(str)
    d1 = (np.datetime64(after) + np.timedelta64(6, "D")).astype(str)
    say(f"检索 {COLLECTION}：AOI({lon},{lat}) {d0} ~ {d1}")
    items = stac_search(bbox, f"{d0}T00:00:00Z/{d1}T23:59:59Z")
    if not items:
        raise RuntimeError("没有找到 RTC 数据（该区域可能不在覆盖范围内）")
    token = get_sas_token()
    say("已获取 SAS token（有效期约 1 小时）")
    size = int(size)
    if not 64 <= size <= 4096:
        # size 直接来自命令行/界面；过大（如 100000）会申请数十 GB 级数组把内存打爆。
        raise ValueError(f"窗口边长 size 需在 64~4096 之间，收到 {size}")
    pol = normalize_polarization(pol)
    asset = "vv" if pol == "VV" else "vh"

    def _try_read(tag: str, date: str):
        for it in pick_candidates(items, date):
            href = (it.get("assets") or {}).get(asset, {}).get("href")
            if not href:
                continue
            try:
                arr, tr, crs_, valid = read_rtc_window(href, token, lon, lat, size)
                say(f"{tag}景选定：{it['id']}  {it['properties']['datetime'][:16]}  CRS={crs_}")
                return it, arr, tr, crs_, valid
            except Exception as exc:
                say(f"  {tag}候选 {it['id'][:46]}… 不可用（{type(exc).__name__}），试下一个")
        raise RuntimeError(f"{tag}景：所有候选切片都读不到（可能该区域不在 RTC 覆盖内）")

    pre_item, pre_lin, transform, crs, pre_valid = _try_read("灾前", before)
    post_item, post_lin, transform2, _, post_valid = _try_read("灾后", after)
    say(f"{pol} 读取完成 {pre_lin.shape}")

    pre_db = 10.0 * np.log10(np.maximum(pre_lin, 1e-6))
    post_db = 10.0 * np.log10(np.maximum(post_lin, 1e-6))

    pre_m, _, _pre_meta = detect_water_sar(pre_db, pol, threshold_db, "fixed", 3, 100)
    post_m, _, _post_meta = detect_water_sar(post_db, pol, threshold_db, "fixed", 3, 100)
    chg = sar_change(pre_db, post_db, pol, threshold_db, drop_db, 3, 100)
    # 补边/无效像元一律不算水体，也不参与新增/退水/持续判定：
    # 它们的 dB 值（约 -60）天然低于阈值，不掩膜会被整体计入淹水面。
    pre_m = np.asarray(pre_m, dtype=bool) & pre_valid
    post_m = np.asarray(post_m, dtype=bool) & post_valid
    both_valid = pre_valid & post_valid
    for key in ("new", "receded", "persistent"):
        chg[key] = np.asarray(chg[key], dtype=bool) & both_valid
    px = 10.0
    stats = {
        "pre_water_km2": area_km2(pre_m, px),
        "post_water_km2": area_km2(post_m, px),
        "new_water_km2": area_km2(chg["new"], px),
        "receded_km2": area_km2(chg["receded"], px),
        "persistent_km2": area_km2(chg["persistent"], px),
        "window_km2": (size * px / 1000.0) ** 2,
        "threshold_db": chg["threshold_db"],
        "polarization": pol,
    }
    say(f"灾前水体 {stats['pre_water_km2']:.2f} km² → 灾后 {stats['post_water_km2']:.2f} km²"
        f"（新增 {stats['new_water_km2']:.2f} km²）")

    os.makedirs(out_dir, exist_ok=True)
    # sid 可能来自命令行 --id，必须清洗后再拼文件名，否则 "..\..\x" 可把成果
    # 写到 out_dir 之外（save_db_geotiff/_save_rgb 会自动建目录，破坏范围更大）。
    sid = safe_sid(sid, f"rtc_{after.replace('-', '')}")
    pre_tif = save_db_geotiff(_assert_within(out_dir, os.path.join(out_dir, f"{sid}_pre_{pol.lower()}.tif")), pre_db, transform, crs)
    post_tif = save_db_geotiff(_assert_within(out_dir, os.path.join(out_dir, f"{sid}_post_{pol.lower()}.tif")), post_db, transform2, crs)

    from src.postprocess import change_map_rgb, overlay_mask

    pre_gray = np.stack([db_to_gray(pre_db)] * 3, -1)
    post_gray = np.stack([db_to_gray(post_db)] * 3, -1)
    p1 = _save_rgb(os.path.join(out_dir, f"{sid}_pre_overlay.png"),
                   overlay_mask(pre_gray, pre_m, (0, 255, 255), 0.55))
    p2 = _save_rgb(os.path.join(out_dir, f"{sid}_post_overlay.png"),
                   overlay_mask(post_gray, post_m, (0, 255, 255), 0.55))
    p3 = _save_rgb(os.path.join(out_dir, f"{sid}_change.png"),
                   change_map_rgb(pre_m, post_m, pre_gray))
    preview = _make_preview(os.path.join(out_dir, f"{sid}_preview.jpg"), [p1, p2, p3],
                            [f"灾前 {pre_item['properties']['datetime'][:10]}",
                             f"灾后 {post_item['properties']['datetime'][:10]}",
                             "红=新增淹没 绿=退水 青=持续"])

    provenance = {
        "sensor": "Sentinel-1 RTC (Planetary Computer, 地形校正)",
        "pre_scene": pre_item["id"], "post_scene": post_item["id"],
        "pre_datetime": pre_item["properties"]["datetime"],
        "post_datetime": post_item["properties"]["datetime"],
        "polarization": pol, "threshold_db": chg["threshold_db"], "drop_db": drop_db,
        "aoi_lonlat": [lon, lat], "window": [size, size],
        "aoi_source": "界面全流程",
        "source": "Microsoft Planetary Computer / sentinel-1-rtc",
        "license": "Copernicus Sentinel Data Terms (free and open)",
    }
    manifest = {
        "id": sid,
        "label": f"Sentinel-1 RTC 洪水提取（{pol}）",
        "pixel_size_m": px,
        "pre": os.path.basename(pre_tif), "post": os.path.basename(post_tif),
        "preview": os.path.basename(preview),
        "change": os.path.basename(p3),
        "stats": stats,
        "provenance": provenance,
    }
    mpath = _assert_within(out_dir, os.path.join(out_dir, "samples.json"))
    samples = []
    if os.path.isfile(mpath):
        try:
            with open(mpath, encoding="utf-8") as fh:
                samples = json.load(fh).get("samples", [])
        except Exception:
            samples = []
    samples = [s for s in samples if s.get("id") != sid] + [manifest]
    # 原子替换：直接覆写若中途崩溃/断电会留下截断 JSON，下次加载只能清空整个清单。
    # 先写同目录临时文件再 os.replace；路径边界由 _assert_within 兜住。
    from pathlib import Path

    tmp_file = Path(_assert_within(out_dir, mpath + ".tmp"))
    tmp_file.write_text(
        json.dumps({"note": "Sentinel-1 RTC 处理成果（σ0 dB）", "samples": samples},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(str(tmp_file), mpath)

    say(f"✓ 成果写入 {out_dir}")
    return {
        "id": sid,
        "stats": stats,
        "provenance": provenance,
        "paths": {
            "preview": preview,
            "change": p3,
            "pre_overlay": p1,
            "post_overlay": p2,
            "pre": pre_tif,
            "post": post_tif,
            "manifest": mpath,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 Sentinel-1 RTC 并提取洪水（免注册）")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--before", required=True, help="灾前日期 YYYY-MM-DD")
    ap.add_argument("--after", required=True, help="灾后日期 YYYY-MM-DD")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rtc"))
    ap.add_argument("--id", default=None)
    ap.add_argument("--size", type=int, default=2048, help="窗口边长（像元，10m）")
    ap.add_argument("--threshold-db", type=float, default=-16.0)
    ap.add_argument("--drop-db", type=float, default=3.5)
    ap.add_argument("--pol", default="VV", choices=["VV", "VH"])
    args = ap.parse_args()
    try:
        run_rtc_flood(
            args.lon, args.lat, args.before, args.after, args.out,
            sid=args.id, size=args.size, threshold_db=args.threshold_db,
            drop_db=args.drop_db, pol=args.pol,
        )
        return 0
    except Exception as exc:
        log(f"✗ {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
