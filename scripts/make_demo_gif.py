"""
慧眼识灾 · 演示 GIF 生成器
===========================

README 和路演 PPT 都需要一段"会动的门面"。本脚本用真实推理结果生成动图：

    灾前真彩色  →  擦除式揭示洪水范围  →  水体掩膜  →  循环

用法：
    python scripts/make_demo_gif.py                        # 用 demo04，自动选模型
    python scripts/make_demo_gif.py --sample demo02 --mode baseline
    python scripts/make_demo_gif.py --out docs/assets/demo.gif --width 720
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src.infer import FloodDetector  # noqa: E402


def build_frames(pre_rgb: np.ndarray, overlay: np.ndarray, mask: np.ndarray, width: int = 720, colors: int = 64) -> list:
    """构造动画帧：擦除揭示 + 停留 + 掩膜。输出调色板量化帧以压缩体积。"""
    from PIL import Image

    h, w = pre_rgb.shape[:2]
    scale = width / w
    size = (width, max(1, int(round(h * scale))))
    frames = []

    def _im(arr: np.ndarray) -> "Image.Image":
        im = Image.fromarray(arr).resize(size, Image.BILINEAR)
        return im.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)

    # 1) 灾前真彩色停留
    frames += [_im(pre_rgb)] * 3
    # 2) 从左到右擦除式揭示识别结果
    steps = 10
    for i in range(steps + 1):
        cut = int(w * i / steps)
        frame = pre_rgb.copy()
        frame[:, :cut] = overlay[:, :cut]
        frames.append(_im(frame))
    # 3) 结果停留
    frames += [_im(overlay)] * 4
    # 4) 纯掩膜（白=水）
    mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=-1)
    frames += [_im(mask_rgb)] * 3
    # 5) 掩膜 -> 叠加，回到结果
    for i in range(2):
        frames.append(_im(overlay if i % 2 else mask_rgb))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="生成慧眼识灾演示 GIF")
    ap.add_argument("--sample", default="demo04", help="样本 id（对应 data/samples/demoXX_*）")
    ap.add_argument("--mode", default="auto", choices=["auto", "baseline", "unet"])
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assets", "demo.gif"))
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--colors", type=int, default=64, help="调色板颜色数，越小体积越小")
    ap.add_argument("--duration", type=int, default=110, help="每帧毫秒")
    args = ap.parse_args()

    samples_dirs = [os.path.join(ROOT, "data", "real"), os.path.join(ROOT, "data", "samples")]
    pre_path = post_path = ""
    for d in samples_dirs:
        p1 = os.path.join(d, f"{args.sample}_pre.tif")
        p2 = os.path.join(d, f"{args.sample}_post.tif")
        if os.path.isfile(p1) and os.path.isfile(p2):
            pre_path, post_path = p1, p2
            break
    if not pre_path:
        raise SystemExit(
            f"未找到样本 {args.sample}\n"
            "· 合成样本：python data/make_samples.py\n"
            "· 真实影像：python scripts/fetch_real_samples.py --event all"
        )

    det = FloodDetector(mode=args.mode)
    pre = det.detect(pre_path)
    post = det.detect(post_path)
    print(f"[gif] 样本 {args.sample}｜模型 {post.meta.get('model_label')}｜灾后水体 {post.area_km2:.2f} km²")

    frames = build_frames(pre.rgb, post.overlay, post.mask, width=args.width, colors=args.colors)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"[gif] 已生成 {args.out}（{len(frames)} 帧，{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
