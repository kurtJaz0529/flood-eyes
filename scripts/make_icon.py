"""
慧眼识灾 · 应用图标生成
=======================

生成 Windows 应用图标（.ico，含 16/32/48/64/128/256 多尺寸）。
图案：深蓝渐变圆角底 + 白色水滴 + 青色雷达弧 —— 一眼看出"遥感 + 洪水"。

用法：
    python scripts/make_icon.py --out docs/assets/app_icon.ico
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _gradient(size: int, top: tuple, bottom: tuple) -> "object":
    from PIL import Image

    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        color = tuple(int(round(_lerp(top[i], bottom[i], t))) for i in range(3))
        for x in range(size):
            px[x, y] = color
    return img


def make_icon(size: int = 256) -> "object":
    from PIL import Image, ImageDraw

    # 背景渐变 + 圆角
    bg = _gradient(size, (11, 61, 145), (30, 111, 217))  # #0b3d91 -> #1e6fd9
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=255)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(bg, (0, 0), mask)

    d = ImageDraw.Draw(img)
    cx, cy = size / 2, size * 0.54
    r = size * 0.26

    # 雷达弧（三条，模拟卫星扫描）
    for i, rr in enumerate((0.40, 0.52, 0.64)):
        rad = size * rr
        box = [cx - rad, cy - rad, cx + rad, cy + rad]
        d.arc(box, start=205, end=335, fill=(120, 220, 255, int(210 - i * 55)), width=max(2, int(size * 0.028)))

    # 水滴
    drop_top = (cx, cy - r * 1.55)
    left = (cx - r * 0.92, cy + r * 0.05)
    right = (cx + r * 0.92, cy + r * 0.05)
    d.polygon([drop_top, right, left], fill=(255, 255, 255, 255))
    d.ellipse([cx - r * 0.92, cy - r * 0.78, cx + r * 0.92, cy + r * 0.92], fill=(255, 255, 255, 255))

    # 水滴内部的一道水波
    d.arc(
        [cx - r * 0.62, cy - r * 0.10, cx + r * 0.62, cy + r * 0.62],
        start=200,
        end=340,
        fill=(30, 111, 217, 255),
        width=max(2, int(size * 0.045)),
    )

    # 右上角卫星点 + 信号弧
    sx, sy = size * 0.76, size * 0.22
    d.ellipse([sx - size * 0.035, sy - size * 0.035, sx + size * 0.035, sy + size * 0.035], fill=(255, 214, 102, 255))
    for i, rr in enumerate((0.06, 0.095)):
        d.arc(
            [sx - size * rr, sy - size * rr, sx + size * rr, sy + size * rr],
            start=180,
            end=270,
            fill=(255, 214, 102, 200 - i * 70),
            width=max(1, int(size * 0.014)),
        )
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="生成慧眼识灾应用图标")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assets", "app_icon.ico"))
    ap.add_argument("--png", default=os.path.join(ROOT, "docs", "assets", "app_icon.png"))
    args = ap.parse_args()

    base = make_icon(256)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base.save(args.png)
    base.save(args.out, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"[icon] 已生成 {args.out} 与 {args.png}")


if __name__ == "__main__":
    main()
