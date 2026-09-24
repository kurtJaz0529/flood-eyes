"""
慧眼识灾 · 本地 GeoTIFF 多时相光谱监测命令行
===============================================

在本地 GeoTIFF 上计算植被/水体指数，并按相邻两景统计指数变化。不联网、
不改动洪水识别流水线，只读取本地文件并写出栅格与 ``summary.json``。

    python scripts/run_spectral.py --index ndvi \
        --images a_20200101.tif b_20200201.tif --out-dir outputs/spectral

    python scripts/run_spectral.py --index ndwi \
        --images a.tif b.tif --min-valid-pct 15 \
        --dates 2020-01-01 2020-02-01

支持的指数（严格要求真实波段，禁止无 NIR 代理）：

    ndvi = (nir - red) / (nir + red)
    savi = 1.5 * (nir - red) / (nir + red + 0.5)
    ndwi = (green - nir) / (green + nir)

日期只能由 ``--dates`` 显式提供，否则所有场景标记为 ``unverified``；
**绝不从文件名推断日期**。全部场景相对首景对齐；缺 CRS/transform 会直接报错。
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:  # Windows 控制台默认 GBK，中文提示会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - 依赖运行环境
    pass

from src.spectral_monitor import (  # noqa: E402
    format_summary_text,
    run_monitor,
    supported_indices,
)

_EXAMPLE = (
    "示例：\n"
    "  python scripts/run_spectral.py --index ndvi --images a.tif b.tif "
    "--out-dir outputs/spectral\n"
    "  python scripts/run_spectral.py --index savi --images a.tif b.tif c.tif "
    "--min-valid-pct 15\n"
    "  python scripts/run_spectral.py --index ndwi --images a.tif b.tif "
    "--dates 2020-01-01 2020-02-01\n"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "本地 GeoTIFF 多时相光谱监测：计算 ndvi/savi/ndwi 并按相邻两景统计变化。"
        ),
        epilog=_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index",
        required=True,
        choices=list(supported_indices()),
        help="光谱指数：ndvi / savi / ndwi",
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        metavar="GEOTIFF",
        help="按时间先后给出的本地 GeoTIFF 路径（首个作为参考网格）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="本次运行的新输出目录；缺省自动生成唯一目录（防止覆盖旧成果）",
    )
    parser.add_argument(
        "--min-valid-pct",
        type=float,
        default=10.0,
        help="单景及相邻共同有效比例门禁（0~100，低于阈值报告 missing/null，默认 10）",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        metavar="DATE",
        help=(
            "可选，与 --images 等长的日期；支持空格或逗号分隔。"
            "缺省则所有场景标记为 unverified（不臆测日期）"
        ),
    )
    parser.add_argument(
        "--band-order",
        default="auto",
        help="传给 load_scene 的波段顺序（默认 auto，按波段描述/通道数解析）",
    )
    return parser


def _parse_dates(raw) -> "list[str] | None":
    if not raw:
        return None
    out = []
    for item in raw:
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out or None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    images = [os.path.abspath(os.fspath(p)) for p in args.images]
    dates = _parse_dates(args.dates)
    out_dir = args.out_dir or os.path.join(
        ROOT, "outputs", "spectral_monitor",
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8],
    )

    try:
        summary = run_monitor(
            images,
            args.index,
            os.path.abspath(out_dir),
            min_valid_pct=args.min_valid_pct,
            dates=dates,
            band_order=args.band_order,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 必须给出可读错误
        print(f"光谱监测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(format_summary_text(summary))
    print(f"摘要文件：{summary['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
