"""
慧眼识灾 · 应用图标生成
=======================

**优先从品牌 logo 生成**（`docs/assets/logo_source.jpg`），取其中的眼睛图形
做成 Windows 应用图标；原图缺失时退回程序化合成的备选图标。

产出：
    docs/assets/app_icon.ico    多尺寸图标（16/32/48/64/128/256），exe 与安装程序用
    docs/assets/app_icon.png    256px PNG，界面 favicon / 文档用
    docs/assets/logo.png        完整字标（图形 + 中文 + 英文），界面与文档用

用法：
    python scripts/make_icon.py
    python scripts/make_icon.py --logo docs/assets/logo_source.jpg
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

ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

# 从原图切出图形区时的判定阈值（低于该灰度视为"有内容"）
_INK_THRESHOLD = 200
# 图标内图形四周留白比例（相对于图标边长）
_PADDING_RATIO = 0.09
# 图标底色圆角半径比例
_RADIUS_RATIO = 0.20


# --------------------------------------------------------------------------
# 主路径：从品牌 logo 生成
# --------------------------------------------------------------------------


def split_segments(img):
    """按纵向空白把 logo 切成若干内容段。

    品牌字标自上而下是「眼睛图形 / 中文名 / 英文名」三段，
    用逐行墨迹量自动分段，避免把坐标写死。
    """
    import numpy as np

    gray = np.asarray(img.convert("L"))
    rows = np.nonzero((gray < _INK_THRESHOLD).sum(axis=1) > 2)[0]
    if rows.size == 0:
        return []
    segments = []
    start = prev = rows[0]
    for r in rows[1:]:
        if r - prev > 8:          # 超过 8 行空白视为分段
            segments.append((int(start), int(prev)))
            start = r
        prev = r
    segments.append((int(start), int(prev)))
    return segments


def _ink_bbox(img, box=None):
    """返回内容的最小外接矩形（可限定在 box 内）。"""
    import numpy as np

    region = img if box is None else img.crop(box)
    gray = np.asarray(region.convert("L"))
    mask = gray < _INK_THRESHOLD
    rows = np.nonzero(mask.sum(axis=1) > 1)[0]
    cols = np.nonzero(mask.sum(axis=0) > 1)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    x0, y0 = int(cols.min()), int(rows.min())
    x1, y1 = int(cols.max()), int(rows.max())
    if box is not None:
        x0 += box[0]
        x1 += box[0]
        y0 += box[1]
        y1 += box[1]
    return x0, y0, x1, y1


def build_icon_from_logo(logo_path: str, size: int = 256):
    """从 logo 取"眼睛图形"做成应用图标（白底圆角 + 图形居中）。

    注意不要为了凑正方形而把裁剪框垂直撑开——那样会把下一段（中文字标）
    的顶部一起裁进图标。这里按图形自身的长宽比缩放，居中贴入即可。
    """
    from PIL import Image, ImageDraw

    src = Image.open(logo_path).convert("RGB")
    segments = split_segments(src)
    if not segments:
        raise ValueError("无法从 logo 中找到内容区域")

    # 第一段即眼睛图形；裁剪范围严格限定在本段内，避免带入相邻文字
    top0, top1 = segments[0]
    band = (0, max(0, top0 - 5), src.width, min(src.height, top1 + 6))
    bbox = _ink_bbox(src, band)
    if bbox is None:
        raise ValueError("无法定位 logo 中的图形区域")
    x0, y0, x1, y1 = bbox
    symbol = src.crop((x0, y0, x1 + 1, y1 + 1))     # 保持原长宽比

    # 图标底：白色圆角方块（图形是深蓝+亮蓝，白底在任何主题下都清晰）
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * _RADIUS_RATIO), fill=255
    )
    canvas.paste(Image.new("RGBA", (size, size), (255, 255, 255, 255)), (0, 0), mask)

    # 等比缩放到留白框内并居中
    inner = int(size * (1 - 2 * _PADDING_RATIO))
    scale = min(inner / symbol.width, inner / symbol.height)
    w2 = max(1, int(round(symbol.width * scale)))
    h2 = max(1, int(round(symbol.height * scale)))
    art = symbol.convert("RGBA").resize((w2, h2), Image.LANCZOS)
    canvas.alpha_composite(art, ((size - w2) // 2, (size - h2) // 2))
    return canvas


def build_lockup_from_logo(logo_path: str):
    """整幅字标（图形 + 中文 + 英文），裁掉四周空白，供界面/文档使用。"""
    from PIL import Image

    src = Image.open(logo_path).convert("RGB")
    bbox = _ink_bbox(src)
    if bbox is None:
        raise ValueError("无法定位 logo 内容区域")
    x0, y0, x1, y1 = bbox
    pad = int(max(x1 - x0, y1 - y0) * 0.04)
    box = (max(0, x0 - pad), max(0, y0 - pad),
           min(src.width, x1 + pad), min(src.height, y1 + pad))
    return src.crop(box)


# --------------------------------------------------------------------------
# 备选路径：程序化合成（原实现，logo 缺失时使用）
# --------------------------------------------------------------------------


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


def make_icon_fallback(size: int = 256) -> "object":
    """程序化合成的备选图标：深蓝渐变圆角底 + 白色水滴 + 雷达弧 + 卫星点。"""
    from PIL import Image, ImageDraw

    bg = _gradient(size, (11, 61, 145), (30, 111, 217))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1],
                                           radius=int(size * 0.22), fill=255)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(bg, (0, 0), mask)

    d = ImageDraw.Draw(img)
    cx, cy = size / 2, size * 0.54
    r = size * 0.26

    for i, rr in enumerate((0.40, 0.52, 0.64)):
        rad = size * rr
        d.arc([cx - rad, cy - rad, cx + rad, cy + rad], start=205, end=335,
              fill=(120, 220, 255, int(210 - i * 55)), width=max(2, int(size * 0.028)))

    drop_top = (cx, cy - r * 1.55)
    d.polygon([drop_top, (cx + r * 0.92, cy + r * 0.05), (cx - r * 0.92, cy + r * 0.05)],
              fill=(255, 255, 255, 255))
    d.ellipse([cx - r * 0.92, cy - r * 0.78, cx + r * 0.92, cy + r * 0.92],
              fill=(255, 255, 255, 255))
    d.arc([cx - r * 0.62, cy - r * 0.10, cx + r * 0.62, cy + r * 0.62],
          start=200, end=340, fill=(30, 111, 217, 255), width=max(2, int(size * 0.045)))

    sx, sy = size * 0.76, size * 0.22
    d.ellipse([sx - size * 0.035, sy - size * 0.035, sx + size * 0.035, sy + size * 0.035],
              fill=(255, 214, 102, 255))
    for i, rr in enumerate((0.06, 0.095)):
        d.arc([sx - size * rr, sy - size * rr, sx + size * rr, sy + size * rr],
              start=180, end=270, fill=(255, 214, 102, 200 - i * 70),
              width=max(1, int(size * 0.014)))
    return img


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="生成慧眼识灾应用图标")
    ap.add_argument("--logo", default=os.path.join(ROOT, "docs", "assets", "logo_source.jpg"),
                    help="品牌 logo 原图；缺失时退回程序化合成图标")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assets", "app_icon.ico"))
    ap.add_argument("--png", default=os.path.join(ROOT, "docs", "assets", "app_icon.png"))
    ap.add_argument("--lockup", default=os.path.join(ROOT, "docs", "assets", "logo.png"))
    ap.add_argument("--no-lockup", action="store_true", help="不导出完整字标")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.png)), exist_ok=True)

    source = "程序化合成（未找到 logo 原图）"
    base = None
    if os.path.isfile(args.logo):
        try:
            base = build_icon_from_logo(args.logo, 256)
            source = f"品牌 logo（{os.path.relpath(args.logo, ROOT)}）"
        except Exception as exc:  # noqa: BLE001
            print(f"[icon] 从 logo 生成失败（{type(exc).__name__}: {exc}），改用合成图标")
    if base is None:
        base = make_icon_fallback(256)

    base.save(args.png)
    base.save(args.out, format="ICO", sizes=ICO_SIZES)
    print(f"[icon] 图标来源：{source}")
    print(f"[icon] 已生成 {os.path.relpath(args.out, ROOT)} 与 {os.path.relpath(args.png, ROOT)}")

    if not args.no_lockup and os.path.isfile(args.logo):
        try:
            lockup = build_lockup_from_logo(args.logo)
            lockup.save(args.lockup)
            print(f"[icon] 已生成完整字标 {os.path.relpath(args.lockup, ROOT)}（{lockup.width}×{lockup.height}）")
        except Exception as exc:  # noqa: BLE001
            print(f"[icon] 字标导出失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
