"""
慧眼识灾 · 成果导出模块
=======================

把一次识别结果打包成应急部门能直接用的东西：
    · 水体掩膜 PNG / 概率热力图 PNG / 叠加图 PNG
    · 统计指标 JSON
    · 中文 PDF 简报（含影像、面积表、模型信息、免责声明）
    · 全部打成一个 zip，一键下载
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 图像工具
# --------------------------------------------------------------------------

# 简易蓝->青->黄->红 色带，用于概率热力图（不依赖 matplotlib）
_HEAT_STOPS = np.array(
    [
        [0.05, 0.10, 0.45],
        [0.00, 0.55, 0.85],
        [0.30, 0.85, 0.55],
        [0.98, 0.85, 0.20],
        [0.85, 0.15, 0.10],
    ],
    dtype=np.float32,
)


def heatmap_rgb(prob: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """概率图 -> RGB 热力图 uint8。"""
    p = np.asarray(prob, dtype=np.float32)
    p = np.clip((p - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    n = len(_HEAT_STOPS) - 1
    pos = p * n
    lo = np.floor(pos).astype(np.int32)
    lo = np.clip(lo, 0, n - 1)
    frac = (pos - lo)[..., None]
    rgb = _HEAT_STOPS[lo] * (1.0 - frac) + _HEAT_STOPS[lo + 1] * frac
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def save_rgb(path: str, arr: np.ndarray) -> str:
    """保存 RGB/RGBA/灰度数组为 PNG。"""
    from PIL import Image

    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def save_mask(path: str, mask: np.ndarray, scale: int = 255) -> str:
    """保存二值掩膜 PNG（白色=水）。"""
    from PIL import Image

    m = (np.asarray(mask).astype(bool) * scale).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(m, mode="L").save(path)
    return path


# --------------------------------------------------------------------------
# PDF 简报
# --------------------------------------------------------------------------


def _register_cjk_font() -> str:
    """注册 PDF 内置中文 CID 字体，无需外部字体文件。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    name = "STSong-Light"
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(name))
    return name


def _fit_image(path: str, max_w: float, max_h: float) -> Tuple[float, float]:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h, 1.0)
    return w * scale, h * scale


def build_pdf_report(
    out_path: str,
    title: str,
    metrics: Sequence[Tuple[str, str]],
    images: Sequence[Tuple[str, str]],  # (图片路径, 图注)
    notes: Optional[Sequence[str]] = None,
    footer: str = "",
    max_image_width: float = 480.0,
) -> str:
    """生成中文 PDF 简报。images 中的图片路径必须已存在。"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font = _register_cjk_font()
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1cn", parent=styles["Title"], fontName=font, fontSize=18, leading=24, spaceAfter=6)
    h2 = ParagraphStyle("h2cn", parent=styles["Heading2"], fontName=font, fontSize=12, leading=18, spaceBefore=8)
    body = ParagraphStyle("bodycn", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=15)
    small = ParagraphStyle("smallcn", parent=body, fontSize=8, textColor=colors.HexColor("#666666"))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="慧眼识灾 · 遥感 AI 洪水识别系统",
    )

    story: List[Any] = [Paragraph(title, h1)]
    story.append(Paragraph(f"生成时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Spacer(1, 6))

    if metrics:
        story.append(Paragraph("一、关键指标", h2))
        rows = [[Paragraph(k, body), Paragraph(str(v), body)] for k, v in metrics]
        table = Table(rows, colWidths=[60 * mm, 100 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F6FA")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(table)

    if images:
        story.append(Paragraph("二、识别成果", h2))
        for path, caption in images:
            if not os.path.isfile(path):
                continue
            w, h = _fit_image(path, max_image_width, 320)
            story.append(Image(path, width=w, height=h))
            story.append(Paragraph(caption, small))
            story.append(Spacer(1, 6))

    if notes:
        story.append(Paragraph("三、说明", h2))
        for n in notes:
            story.append(Paragraph(f"· {n}", body))

    if footer:
        story.append(Spacer(1, 10))
        story.append(Paragraph(footer, small))

    doc.build(story)
    return out_path


# --------------------------------------------------------------------------
# 打包
# --------------------------------------------------------------------------


def export_bundle(
    result: Any,
    out_dir: str = "outputs",
    basename: Optional[str] = None,
    include_pdf: bool = True,
    report_title: str = "洪水淹没范围识别简报",
) -> Dict[str, Any]:
    """把 FloodResult 导出为文件包，返回 {zip, files, report}。"""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = basename or f"{result.meta.get('source', 'result')}_{stamp}"
    workdir = os.path.join(out_dir, name)
    os.makedirs(workdir, exist_ok=True)

    files: Dict[str, str] = {}
    files["mask"] = save_mask(os.path.join(workdir, "water_mask.png"), result.mask)
    files["overlay"] = save_rgb(os.path.join(workdir, "overlay.png"), result.overlay)
    files["rgb"] = save_rgb(os.path.join(workdir, "true_color.png"), result.rgb)
    files["heatmap"] = save_rgb(os.path.join(workdir, "probability_heatmap.png"), heatmap_rgb(result.prob))
    files["comparison"] = save_rgb(
        os.path.join(workdir, "before_after.png"),
        _make_comparison_image(result),
    )

    stats = dict(result.stats)
    stats.pop("components", None)
    stats_payload = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": result.meta.get("source"),
        "model": result.meta.get("model_label"),
        "pixel_size_m": result.meta.get("pixel_size_m"),
        "elapsed_s": round(result.elapsed_s, 3),
        "stats": stats,
        "largest_components": result.stats.get("components", [])[:5],
        "meta": {k: v for k, v in result.meta.items() if k not in ("params",)},
    }
    if result.change:
        stats_payload["change"] = {
            k: v for k, v in result.change.items() if not isinstance(v, np.ndarray) and k not in ("before_result", "after_result")
        }
    stats_path = os.path.join(workdir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, ensure_ascii=False, indent=2, default=str)
    files["stats"] = stats_path

    report_path = ""
    if include_pdf:
        metrics = [
            ("数据源", str(result.meta.get("source", "-"))),
            ("识别模型", str(result.meta.get("model_label", "-"))),
        ]
        # 影像溯源：应急研判需要能追到"用的哪几景、什么时候、云量多少"。
        # 原实现只导出统计量，拿到成果包无法回溯原始影像。
        prov = result.meta.get("provenance")
        if isinstance(prov, dict) and prov:
            for tag, name in (("pre", "灾前影像"), ("post", "灾后影像")):
                scene = prov.get(f"{tag}_scene")
                if scene:
                    when = str(prov.get(f"{tag}_datetime", ""))[:10]
                    metrics.append((name, f"{scene}　{when}".strip()))
            clouds = []
            for tag, name in (("pre", "灾前"), ("post", "灾后")):
                value = prov.get(f"{tag}_window_cloud_pct")
                if value is not None:
                    clouds.append(f"{name} {value}%")
            if clouds:
                metrics.append(("窗口云量", "　".join(clouds)))
            if prov.get("source"):
                metrics.append(("影像来源", str(prov["source"])))

        matched = result.meta.get("matched_event")
        if isinstance(matched, dict) and matched.get("label"):
            metrics.append(("匹配事件", str(matched["label"])))

        metrics += [
            ("水体面积", f"{result.stats.get('water_area_km2', 0):,.3f} km²"),
            ("占影像比例", f"{result.stats.get('water_fraction_pct', 0):.2f}%"),
            ("连通水域个数", f"{result.stats.get('n_components', 0)}"),
            ("最大连通水域", f"{result.stats.get('largest_area_km2', 0):,.3f} km²"),
            ("平均置信度", f"{result.stats.get('mean_confidence', 0):.3f}"),
            ("像元分辨率", f"{result.stats.get('pixel_size_m', 0):g} m"),
            ("处理耗时", f"{result.elapsed_s:.2f} s"),
        ]
        if result.change:
            c = result.change
            metrics += [
                ("灾前水体", f"{c['before_water_km2']:,.3f} km²"),
                ("灾后水体", f"{c['after_water_km2']:,.3f} km²"),
                ("新增淹没", f"{c['new_water_km2']:,.3f} km²"),
                ("退水面积", f"{c['receded_water_km2']:,.3f} km²"),
            ]
        images = [
            (files["overlay"], "图 1  洪水淹没范围（青色为水体，红色为边界）"),
            (files["heatmap"], "图 2  模型置信度热力图（越红表示越确信为水体）"),
        ]
        if result.change:
            images.append((files["comparison"], "图 3  灾前 / 灾后 / 变化检测（红=新增淹没，绿=退水，青=持续水体）"))
        notes = [
            "本简报由「慧眼识灾」遥感 AI 洪水识别系统自动生成，供应急研判参考。",
            "水体范围由卫星影像自动提取，受云层、阴影、薄云和影像时相影响，边界存在一定误差。",
            "面积统计基于影像像元分辨率推算，未做地形坡度改正，山区结果需结合实地核查。",
        ]
        if result.meta.get("warnings"):
            notes.append("系统提示：" + "；".join(result.meta["warnings"]))
        report_path = build_pdf_report(
            os.path.join(workdir, "report.pdf"),
            report_title,
            metrics,
            images,
            notes,
            footer="慧眼识灾 · 基于深度学习的高分辨率遥感影像洪水识别系统",
        )
        files["report"] = report_path

    zip_path = os.path.join(out_dir, f"{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files.values():
            zf.write(path, arcname=os.path.basename(path))
    return {"zip": zip_path, "files": files, "workdir": workdir, "report": report_path}


def _make_comparison_image(result: Any) -> np.ndarray:
    """灾前/灾后/变化图 三连拼图；没有双时相时输出 原图/叠加图 二连。"""
    from . import postprocess

    if result.change:
        return postprocess.side_by_side(
            [result.change["before_overlay"], result.change["after_overlay"], result.change["change_map"]]
        )
    return postprocess.side_by_side([result.rgb, result.overlay])
