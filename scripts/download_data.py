"""
慧眼识灾 · 真实训练数据准备
===========================

演示用合成样本 30 秒就能生成，但**竞赛的精度指标必须来自真实数据**。
本脚本负责把公开数据集的下载/解压/整理流程讲清楚，能自动做的就自动做。

用法：
    python scripts/download_data.py --list                 # 看有哪些数据集
    python scripts/download_data.py --dataset sen1floods11 # 打印下载与整理指引
    python scripts/download_data.py --dataset sen1floods11 --check   # 检查本地是否已就绪

推荐数据集（按"免费 + 带人工标注 + 有洪水"筛选）：
    1. Sen1Floods11  —— 446 景 Sentinel-1/2 影像 + 人工标注，洪水分割的经典基准
       https://github.com/cloudtostreet/Sen1Floods11
    2. WorldFloods  —— 全球洪水标注数据集（Sentinel-2）
       https://github.com/spaceml-org/ml4floods
    3. SpaceNet 8   —— 洪水/基础设施，含高分辨率光学影像
    4. Copernicus EMS Rapid Mapping —— 应急制图产品（可作独立验证集）

为什么不能直接帮你下载？
    这些数据集动辄 4~20 GB，且需要 Kaggle/Google Drive 账号授权。
    脚本会给出**可直接复制粘贴**的命令和校验步骤，避免你踩坑。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS: Dict[str, Dict[str, object]] = {
    "sen1floods11": {
        "name": "Sen1Floods11",
        "size": "≈4 GB（含 Sentinel-1 与 Sentinel-2）",
        "url": "https://github.com/cloudtostreet/Sen1Floods11",
        "note": "446 景全球洪水影像 + 人工标注，洪水语义分割最常用的公开基准",
        "steps": [
            "git clone https://github.com/cloudtostreet/Sen1Floods11.git",
            "cd Sen1Floods11",
            "python download_data.py --s2   # 只下 Sentinel-2 光学影像（约 1.2 GB）",
            "python download_data.py --s1   # 需要雷达数据再下这个",
        ],
        "layout": [
            "把 Sen1Floods11/flood_events/HandLabeled/S2Hand/ 下的 *_S2Hand.tif 复制到 data/sen1floods11/",
            "把 Sen1Floods11/flood_events/HandLabeled/LabelHand/ 下的 *_LabelHand.tif 复制过来并改名：",
            "    xxx_S2Hand.tif  ->  xxx_post.tif",
            "    xxx_LabelHand.tif  ->  xxx_mask.png   （0/1 二值，>127 视为水）",
        ],
        "check": ["data/sen1floods11"],
    },
    "worldfloods": {
        "name": "WorldFloods",
        "size": "≈20 GB",
        "url": "https://github.com/spaceml-org/ml4floods",
        "note": "全球洪水标注数据集，含多时相 Sentinel-2，适合做跨区域泛化实验",
        "steps": [
            "git clone https://github.com/spaceml-org/ml4floods.git",
            "按仓库 README 从 Google Drive / Zenodo 下载 train/val/test 切片",
        ],
        "layout": ["整理成 data/worldfloods/*_post.tif + *_mask.png 的配对结构"],
        "check": ["data/worldfloods"],
    },
    "synthetic": {
        "name": "内置合成样本",
        "size": "≈27 MB",
        "url": "本地生成，无需下载",
        "note": "仅用于验证全流程可跑通，不可用于宣称精度",
        "steps": ["python data/make_samples.py --n 6 --size 640"],
        "layout": ["输出到 data/samples/"],
        "check": ["data/samples"],
    },
}


def _print_dataset(key: str) -> None:
    d = DATASETS[key]
    print("=" * 74)
    print(f"数据集：{d['name']}   体积：{d['size']}")
    print(f"主页：{d['url']}")
    print(f"说明：{d['note']}")
    print("-" * 74)
    print("【下载步骤】")
    for i, s in enumerate(d["steps"], 1):  # type: ignore[arg-type]
        print(f"  {i}. {s}")
    print("【整理为本项目可识别的结构】")
    for s in d["layout"]:  # type: ignore[arg-type]
        print(f"  · {s}")
    print("  · 目录形如：data/sen1floods11/xxx_post.tif  +  xxx_mask.png")
    print("【训练】")
    print("  python train.py --data data/sen1floods11 --epochs 60 --img-size 512 \\")
    print("      --arch smp --encoder resnet34 --encoder-weights imagenet --batch-size 8")
    print("=" * 74)


def _check(key: str) -> bool:
    d = DATASETS[key]
    ok = True
    for rel in d["check"]:  # type: ignore[arg-type]
        path = os.path.join(ROOT, rel)
        # 用 with 管理 ScandirIterator：原实现 any(os.scandir(path)) 会泄漏目录句柄，
        # 且随后又对同一目录重复 scandir。
        names: List[str] = []
        if os.path.isdir(path):
            with os.scandir(path) as it:
                names = [e.name for e in it if e.is_file()]
        exists = bool(names)
        print(f"  [{'✓' if exists else '✗'}] {rel}")
        if exists:
            tifs = [f for f in names if f.endswith((".tif", ".tiff"))]
            masks = [f for f in names if f.endswith((".png", ".tif")) and "mask" in f.lower()]
            print(f"        影像 {len(tifs)} 个，掩膜 {len(masks)} 个")
        ok = ok and exists
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="慧眼识灾 · 训练数据准备指引")
    ap.add_argument("--dataset", default="sen1floods11", choices=list(DATASETS))
    ap.add_argument("--list", action="store_true", help="列出所有数据集")
    ap.add_argument("--check", action="store_true", help="检查本地数据是否就绪")
    args = ap.parse_args()

    if args.list:
        print("可选数据集：")
        for k, v in DATASETS.items():
            print(f"  {k:16s} {v['name']:16s} {v['size']:12s} {v['note']}")
        return

    if args.check:
        print(f"检查 {args.dataset} 本地数据：")
        if _check(args.dataset):
            print("\n✅ 数据已就绪，可以直接训练：")
            print(f"   python train.py --data data/{args.dataset} --epochs 60 --img-size 512")
        else:
            print("\n⚠️ 数据尚未就绪，执行下面的命令查看下载指引：")
            print(f"   python scripts/download_data.py --dataset {args.dataset}")
        return

    _print_dataset(args.dataset)


if __name__ == "__main__":
    main()
