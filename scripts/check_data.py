"""
慧眼识灾 · 数据体检器
=====================

不管你从哪里下载了什么卫星数据，先跑一下这个脚本，它会告诉你：
    · 这是什么数据（光学/雷达、什么卫星、几个波段、什么量纲）
    · 能不能直接用软件处理
    · 不能直接用的，应该跑哪条命令

用法：
    python scripts/check_data.py "F:\\遥感河南"
    python scripts/check_data.py "F:\\某处\\S2A_MSIL2A_xxx.SAFE"
    python scripts/check_data.py "F:\\某处\\image.tif"
    python scripts/check_data.py "F:\\某处\\data.zip"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def shell_safe_path(path: str) -> str:
    """把路径放进"可复制命令"提示里前先净化。

    这些提示是让用户直接复制粘贴执行的；若路径含引号、反引号或换行，
    复制后命令会被改写甚至注入额外语句。这里只保留路径语义、剔除引号类字符。
    """
    return (
        str(path)
        .replace('"', "")
        .replace("'", "")
        .replace("`", "")
        .replace("$", "")
        .replace("\n", " ")
        .replace("\r", " ")
    )


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

LINE = "=" * 76


# --------------------------------------------------------------------------
# 识别逻辑
# --------------------------------------------------------------------------


def _safe_kind(safe_dir: str) -> Tuple[str, Dict[str, Any]]:
    """判断一个 .SAFE 是 Sentinel-1 还是 Sentinel-2。"""
    info: Dict[str, Any] = {"path": safe_dir}
    if glob.glob(os.path.join(safe_dir, "measurement", "*.tiff")) or \
       glob.glob(os.path.join(safe_dir, "measurement", "*.tif")):
        pols = sorted({os.path.basename(p).split("-")[3].upper()
                       for p in glob.glob(os.path.join(safe_dir, "measurement", "*.tif*"))
                       if len(os.path.basename(p).split("-")) > 3})
        info["polarizations"] = pols or ["VV/VH"]
        return "sentinel1", info
    if glob.glob(os.path.join(safe_dir, "IMG_DATA", "**", "*.jp2"), recursive=True):
        jp2 = glob.glob(os.path.join(safe_dir, "IMG_DATA", "**", "*.jp2"), recursive=True)
        info["bands_jp2"] = len(jp2)
        info["is_l2a"] = "MSIL2A" in os.path.basename(safe_dir)
        return "sentinel2", info
    if glob.glob(os.path.join(safe_dir, "GRANULE", "**", "*.jp2"), recursive=True):
        info["bands_jp2"] = len(glob.glob(os.path.join(safe_dir, "GRANULE", "**", "*.jp2"), recursive=True))
        return "sentinel2", info
    return "unknown_safe", info


def _inspect_geotiff(path: str) -> Dict[str, Any]:
    """检查一个 GeoTIFF / 影像文件的波段、量纲、范围。"""
    import numpy as np
    import rasterio

    out: Dict[str, Any] = {"path": path}
    with rasterio.open(path) as src:
        out.update({
            "bands": src.count,
            "size": [src.width, src.height],
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else None,
            "has_transform": not src.transform.is_identity,
            "descriptions": [d for d in src.descriptions if d],
            "n_gcps": len(src.gcps[0]) if src.gcps and src.gcps[0] else 0,
        })
        # 采样一个窗口看数值范围
        win = rasterio.windows.Window(max(0, src.width // 2 - 256), max(0, src.height // 2 - 256),
                                      min(512, src.width), min(512, src.height))
        arr = src.read(window=win)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            out["value_range"] = [float(finite.min()), float(np.median(finite)), float(finite.max())]
    # 猜测量纲
    lo, mid, hi = out.get("value_range", [0, 0, 0])
    if out["dtype"] in ("int16",) and -4000 <= lo and hi <= 1000:
        out["guess"] = "SAR σ0(dB)×100"
    elif out["dtype"] in ("uint16", "int16") and hi > 2000:
        out["guess"] = "反射率×10000（或原始 DN）"
    elif out["dtype"] in ("uint8",):
        out["guess"] = "8bit 影像"
    elif out["dtype"] in ("float32", "float64"):
        out["guess"] = "浮点（反射率或 dB）"
    else:
        out["guess"] = "未知"
    return out


def _zip_kind(path: str) -> Dict[str, Any]:
    """看 zip 里装的是什么。"""
    info: Dict[str, Any] = {"path": path}
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            info["entries"] = len(names)
            info["has_safe"] = any(n.endswith(".SAFE/") or ".SAFE/" in n for n in names)
            info["has_jp2"] = any(n.lower().endswith(".jp2") for n in names)
            info["has_measurement_tiff"] = any("/measurement/" in n.lower() and n.lower().endswith((".tif", ".tiff")) for n in names)
            info["has_tiff"] = any(n.lower().endswith((".tif", ".tiff")) for n in names)
            if info["has_measurement_tiff"]:
                info["kind"] = "sentinel1"
            elif info["has_jp2"]:
                info["kind"] = "sentinel2"
            elif info["has_safe"]:
                info["kind"] = "safe"
            elif info["has_tiff"]:
                info["kind"] = "tiff"
            else:
                info["kind"] = "other"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def report_optical(info: Dict[str, Any], path: str) -> None:
    bands = info.get("bands", 0)
    print(f"  类型      : 光学影像（{bands} 波段）")
    print(f"  尺寸/类型 : {info.get('size')}  {info.get('dtype')}")
    print(f"  坐标系    : {info.get('crs') or '无（按像元处理，需手动填分辨率）'}")
    print(f"  数值      : {info.get('value_range')}  →  {info.get('guess')}")
    print()
    if bands >= 4:
        print("  ✅ 可以直接用：打开软件 → 页签①「单时相识别」→ 上传这个文件")
        print("     软件会自动按 4 波段 = 蓝、绿、红、近红外 解析并计算 NDWI")
        if bands >= 13:
            print("     （13 波段 Sentinel-2 会自动按 B2/B3/B4/B8 取蓝绿红近红外）")
    elif bands == 3:
        print("  ⚠️ 只有 RGB（缺近红外）：软件会用 (绿-红)/(绿+红) 代理指数，精度下降")
        print("     建议下载 Sentinel-2 的 B08（近红外）一起上传，或直接用 4 波段产品")
    else:
        print("  ⚠️ 单波段影像无法算 NDWI，只能当灰度图看")
    print()
    print("  命令行方式：")
    print(f'     python -c "from src import FloodDetector; '
          f'print(FloodDetector(mode=\'baseline\').detect(r\'{shell_safe_path(path)}\').summary_text())"')


def report_sentinel1(path: str, safe_dirs: List[str], zips: List[str]) -> None:
    print("  类型      : Sentinel-1 雷达影像（IW GRD，双极化 VV/VH）")
    print("  说明      : 雷达能穿云，适合汛期；但**不能**用 NDWI（那是光学方法）")
    print("  需要      : 先标定成 σ0(dB) 再按低回波提取水体")
    print()
    if len(safe_dirs) >= 2:
        pre, post = safe_dirs[0], safe_dirs[-1]
        print("  ✅ 检测到两景（灾前 + 灾后），直接跑：")
        print(f'     python scripts/prepare_s1_sar.py --pre "{shell_safe_path(pre)}" '
              f'--post "{shell_safe_path(post)}" --pol VV --out data/my_flood --size 4096')
        print()
        print("  也可以先看数据信息：")
        print(f'     python scripts/prepare_s1_sar.py --pre "{shell_safe_path(pre)}" '
              f'--post "{shell_safe_path(post)}" --info')
    elif len(safe_dirs) == 1:
        print("  ⚠️ 只有一景：单时相也能提取水体范围，但没法算「新增淹没」")
        print("     建议再下载一景（灾前或灾后）做对比")
    if zips:
        print()
        print("  注意：还有未解压的 zip：")
        for z in zips:
            print(f"     {z}")
        print("     先右键解压（或 Expand-Archive），再跑上面的命令")


def report_sentinel2(path: str, safe_dirs: List[str], zips: List[str]) -> None:
    print("  类型      : Sentinel-2 光学影像（SAFE / JP2 格式）")
    print("  说明      : 这是光学数据，本系统用 NDWI 提取水体；但 SAFE 里的 JP2 需要先合成 4 波段 GeoTIFF")
    print()
    print("  【方法一】用 Copernicus Browser 导出 4 波段 GeoTIFF（最省事）")
    print("     1. 打开 https://browser.dataspace.copernicus.eu/")
    print("     2. 找到同一景 → 下载 → 选 'Analytical' 或 'Bands'，勾选 B02 B03 B04 B08")
    print("     3. 导出 GeoTIFF，然后直接在软件页签①上传")
    print()
    print("  【方法二】用命令行从 SAFE 里合成（需要 rasterio）")
    if safe_dirs:
        safe = safe_dirs[0]
        print(f"     SAFE: {safe}")
        print("     取 GRANULE 里的 B02/B03/B04/B08（10m 分辨率）四个 JP2，")
        print("     按 蓝、绿、红、近红外 顺序堆叠成一个 4 波段 GeoTIFF，再上传。")
    print()
    print("  提示：如果只需要看单景水体，3 波段 RGB 也能用（软件会用代理指数）")


def main() -> int:
    ap = argparse.ArgumentParser(description="慧眼识灾 · 数据体检器")
    ap.add_argument("path", help="文件或目录路径")
    args = ap.parse_args()
    p = os.path.abspath(args.path)

    print(LINE)
    print(f"数据体检：{p}")
    print(LINE)
    if not os.path.exists(p):
        print("✗ 路径不存在")
        return 2

    # ---- 单文件 ----
    if os.path.isfile(p):
        ext = os.path.splitext(p)[1].lower()
        if ext == ".zip":
            info = _zip_kind(p)
            print(f"  类型      : ZIP 压缩包（{info.get('entries')} 个条目）")
            if info.get("kind") == "sentinel1":
                print("  内容      : Sentinel-1 雷达 SAFE（未解压）")
                print()
                print("  ⚠️ 先解压再处理：")
                print(f'     Expand-Archive "{p}" -DestinationPath "{os.path.dirname(p)}"')
                print("     然后：python scripts/check_data.py \"<解压后的目录>\"")
            elif info.get("kind") == "sentinel2":
                print("  内容      : Sentinel-2 光学 SAFE（未解压）")
                print("  → 解压后用 Copernicus Browser 导出 4 波段 GeoTIFF，或按下方说明合成")
            elif info.get("kind") == "tiff":
                print("  内容      : 普通 GeoTIFF（未解压）→ 解压后直接在软件上传")
            else:
                print("  内容      : 其他文件")
            return 0
        if ext in (".tif", ".tiff", ".img"):
            info = _inspect_geotiff(p)
            if info.get("guess", "").startswith("SAR"):
                print("  类型      : 雷达 σ0(dB) 影像（单波段）")
                print("  ✅ 可以用 src.sar 处理：")
                print(f'     python -c "import rasterio, numpy as np; '
                      f'from src.sar import detect_water_sar; '
                      f'from src.postprocess import area_stats; '
                      f'a=rasterio.open(r\'{p}\').read(1)/100.0; '
                      f'm,_,_=detect_water_sar(a); print(area_stats(m))"')
            else:
                report_optical(info, p)
            return 0
        if ext in (".png", ".jpg", ".jpeg"):
            from PIL import Image

            with Image.open(p) as im:
                print(f"  类型      : 普通图片 {im.mode} {im.size}")
            print("  ✅ 可以直接在软件页签①上传（RGB 三波段，会用代理指数）")
            return 0
        if ext == ".jp2":
            print("  类型      : JPEG2000（Sentinel-2 单波段）")
            print("  ⚠️ 单波段不能算 NDWI；请把 B02/B03/B04/B08 四个波段合成为 4 波段 GeoTIFF")
            return 0
        print(f"  类型      : 未识别（{ext}）")
        return 1

    # ---- 目录 ----
    safe_dirs = [d for d in glob.glob(os.path.join(p, "**", "*.SAFE"), recursive=True)
                 if os.path.isdir(d) and os.path.basename(d).lower() != "manifest.safe"]

    def _date_key(d: str) -> str:
        import re

        m = re.search(r"(\d{8}T\d{6})", os.path.basename(d))
        return m.group(1) if m else os.path.basename(d)

    safe_dirs.sort(key=_date_key)
    zips = sorted(glob.glob(os.path.join(p, "**", "*.zip"), recursive=True))
    tifs = sorted(glob.glob(os.path.join(p, "**", "*.tif"), recursive=True) +
                  glob.glob(os.path.join(p, "**", "*.tiff"), recursive=True))
    jp2s = sorted(glob.glob(os.path.join(p, "**", "*.jp2"), recursive=True))

    print(f"  目录内容  : {len(safe_dirs)} 个 SAFE | {len(zips)} 个 zip | {len(tifs)} 个 tif | {len(jp2s)} 个 jp2")
    print()

    kinds = [_safe_kind(s)[0] for s in safe_dirs]
    if "sentinel1" in kinds:
        report_sentinel1(p, safe_dirs, zips)
    elif "sentinel2" in kinds or jp2s:
        report_sentinel2(p, safe_dirs, zips)
    elif tifs:
        # 逐个体检（最多 6 个）
        print("  检测到 GeoTIFF：")
        for t in tifs[:6]:
            info = _inspect_geotiff(t)
            print(f"    {os.path.relpath(t, p)}  {info['bands']} 波段  {info['dtype']}  "
                  f"{info['size']}  量纲猜测: {info['guess']}")
        print()
        report_optical(_inspect_geotiff(tifs[0]), tifs[0])
    elif zips:
        print("  只有压缩包，先解压：")
        for z in zips[:3]:
            print(f'    Expand-Archive "{z}" -DestinationPath "{os.path.dirname(z)}"')
    else:
        print("  ⚠️ 没找到可识别的卫星数据（支持 .SAFE / .zip / .tif / .png / .npy）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
