"""
慧眼识灾 · Sentinel-1 SAR 数据准备与洪水提取（命令行入口）
=========================================================

把从欧空局下载的 Sentinel-1 IW GRD（SAFE 格式）变成"能直接看、能算面积"的成果。
核心处理链在 `src/sar.py::process_sar_pair`，与软件界面「③ 雷达 SAR 识别」页签共用同一套代码。

用法：
    # 自动定位洪水并处理（推荐）
    python scripts/prepare_s1_sar.py --pre "F:\\遥感河南\\S1A_..._20210715...\\S1A_....SAFE" ^
        --post "F:\\遥感河南\\S1A_..._20210727...\\S1A_....SAFE" --pol VV --out data/henan2021

    # 指定 AOI（经度,纬度）
    python scripts/prepare_s1_sar.py --pre ... --post ... --aoi 114.42,34.06

    # 只看数据信息
    python scripts/prepare_s1_sar.py --pre ... --post ... --info

不确定数据是什么？先跑：python scripts/check_data.py "你的数据目录"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src.sar import DEFAULT_WATER_DB, load_s1_scene, process_sar_pair  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sentinel-1 SAR 洪水提取")
    ap.add_argument("--pre", required=True, help="灾前 SAFE 目录（或其父目录）")
    ap.add_argument("--post", required=True, help="灾后 SAFE 目录（或其父目录）")
    ap.add_argument("--pol", default="VV", choices=["VV", "VH", "vv", "vh"])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sar"))
    ap.add_argument("--id", default=None, help="样本 id（默认用灾后景日期）")
    ap.add_argument("--size", type=int, default=4096, help="裁剪窗口边长（像元，10m）")
    ap.add_argument("--aoi", default=None, help="指定 AOI：经度,纬度（默认自动定位）")
    ap.add_argument("--threshold-db", type=float, default=None, help="水体阈值 dB（VV 默认 -16 / VH -19）")
    ap.add_argument("--drop-db", type=float, default=3.0, help="变化检测要求的最小变暗幅度")
    ap.add_argument("--min-area-px", type=int, default=100)
    ap.add_argument("--info", action="store_true", help="只打印数据信息")
    args = ap.parse_args()

    pol = args.pol.upper()
    thr = args.threshold_db if args.threshold_db is not None else DEFAULT_WATER_DB.get(pol, -16.0)

    if args.info:
        pre = load_s1_scene(args.pre, pol)
        post = load_s1_scene(args.post, pol)
        print(json.dumps({
            "pre": {"scene": pre.name, "size": [pre.width, pre.height],
                    "gcps": pre.affine.n_gcps, "residual_m": round(pre.affine.residual_m, 1)},
            "post": {"scene": post.name, "size": [post.width, post.height],
                     "gcps": post.affine.n_gcps, "residual_m": round(post.affine.residual_m, 1)},
            "polarization": pol, "threshold_db": thr,
        }, ensure_ascii=False, indent=2))
        return 0

    aoi = None
    if args.aoi:
        lon, lat = [float(v) for v in args.aoi.split(",")]
        aoi = (lon, lat)

    result = process_sar_pair(
        args.pre, args.post, pol, out_dir=args.out, sample_id=args.id,
        size=args.size, threshold_db=thr, drop_db=args.drop_db,
        min_area_px=args.min_area_px, aoi=aoi, progress=log,
    )

    # 追加到清单
    mpath = os.path.join(args.out, "samples.json")
    samples = []
    if os.path.isfile(mpath):
        try:
            samples = json.load(open(mpath, encoding="utf-8")).get("samples", [])
        except Exception:
            samples = []
    entry = {
        "id": result["id"],
        "label": f"Sentinel-1 SAR 洪水提取（{pol}）",
        "pixel_size_m": 10.0,
        "pre": os.path.basename(result["paths"]["pre_tif"]),
        "post": os.path.basename(result["paths"]["post_tif"]),
        "preview": os.path.basename(result["paths"]["preview"]),
        "change": os.path.basename(result["paths"]["change"]),
        "stats": result["stats"],
        "provenance": result["provenance"],
    }
    samples = [s for s in samples if s.get("id") != entry["id"]] + [entry]
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump({"note": "Sentinel-1 SAR 处理成果（σ0 dB）", "samples": samples}, fh,
                  ensure_ascii=False, indent=2)

    log(f"✓ 成果已写入 {args.out}")
    for k, v in result["paths"].items():
        log(f"   {k:12s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
