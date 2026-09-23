"""
慧眼识灾 · 界面组件
===================

把"模型输出"翻译成"评委/应急人员一眼看懂的东西"：
    · 统计面板（HTML 卡片，面积/占比/连通域/置信度）
    · 原图 ↔ 识别结果 对比滑块（Gradio ImageSlider，老版本自动降级为左右两图）
    · 一键导出（PNG + JSON + PDF 简报 + zip）
    · 示例样本清单读取
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # Gradio 是可选依赖，纯脚本调用时不需要
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None  # type: ignore[assignment]

HAS_IMAGE_SLIDER = bool(gr is not None and hasattr(gr, "ImageSlider"))

from src.paths import bundle_root, real_dir, samples_dir  # noqa: E402

ROOT = bundle_root()
SAMPLES_DIR = samples_dir()
REAL_DIR = real_dir()


def _esc(value: Any) -> str:
    """转义插入 HTML 的文本。

    场景名、模型标签、溯源字段等可能来自远端 STAC 元数据或权重文件，
    直接 f-string 拼接会被 gr.HTML 按 innerHTML 渲染，构成注入面。
    """
    return html.escape(str(value), quote=True)


def _num(value: Any, default: float = 0.0) -> float:
    """把可能是 None / 字符串 / nan 的数值安全转成 float。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


# --------------------------------------------------------------------------
# 示例样本
# --------------------------------------------------------------------------


def load_manifest(dirs: Any = None) -> List[Dict[str, Any]]:
    """合并加载 data/samples（合成）与 data/real（真实）的样本清单。

    每个样本会带上 `_dir`（所在目录）和 `_real`（是否真实影像）两个内部字段。
    """
    if dirs is None:
        dirs = [SAMPLES_DIR, REAL_DIR]
    elif isinstance(dirs, str):
        dirs = [dirs]
    out: List[Dict[str, Any]] = []
    for d in dirs:
        path = os.path.join(d, "samples.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            samples = payload.get("samples", []) if isinstance(payload, dict) else []
            if not isinstance(samples, list):
                samples = []
        except (OSError, ValueError) as exc:
            # 半写入/损坏的清单不能让整个应用起不来：
            # main.py 在模块级就调 sample_choices()，这里抛异常会直接崩溃启动。
            print(f"[warn] 跳过无法解析的样本清单 {path}：{type(exc).__name__}: {exc}")
            continue
        for s in samples:
            if not isinstance(s, dict) or "id" not in s:
                continue
            s = dict(s)
            s["_dir"] = d
            s["_real"] = os.path.normcase(d) == os.path.normcase(REAL_DIR)
            out.append(s)
    return out


def sample_choices(dirs: Any = None) -> List[Tuple[str, str]]:
    """返回 [(界面显示文本, 样本id)]。合成样本显示真值面积，真实样本显示事件与日期。"""
    out: List[Tuple[str, str]] = []
    for s in load_manifest(dirs):
        if s.get("_real"):
            prov = s.get("provenance", {})
            d0 = str(prov.get("pre_datetime", ""))[:10]
            d1 = str(prov.get("post_datetime", ""))[:10]
            label = f"【真实】{s.get('label', s['id'])}（{d0} → {d1}）"
        else:
            label = (
                f"【合成】{s['id']}｜灾前 {s.get('pre_water_km2', 0):.1f} km² → "
                f"灾后 {s.get('post_water_km2', 0):.1f} km²（新增 {s.get('flood_area_km2', 0):.1f} km²）"
            )
        out.append((label, s["id"]))
    return out


def sample_paths(sample_id: str, dirs: Any = None) -> Dict[str, str]:
    for s in load_manifest(dirs):
        if s["id"] != sample_id:
            continue
        base = s["_dir"]
        paths = {
            "pre": os.path.join(base, s["pre"]),
            "post": os.path.join(base, s["post"]),
            "pixel_size_m": s.get("pixel_size_m", 10.0),
            "real": s.get("_real", False),
        }
        for key in ("mask", "pre_mask", "preview", "pre_scl", "post_scl"):
            if s.get(key):
                paths[key] = os.path.join(base, s[key])
        return paths
    raise KeyError(f"未找到样本 {sample_id}")


# --------------------------------------------------------------------------
# 统计面板
# --------------------------------------------------------------------------

_CARD_CSS = ""  # 视觉令牌与组件样式集中在 apple.css


def empty_stats_html(hint: str = "识别结果将在这里显示") -> str:
    return f'<div class="heye-empty">{hint}</div>'


def stats_html(result: Any) -> str:
    """识别结果统计面板。"""
    s = result.stats
    m = result.meta
    area = _num(s.get("water_area_km2"))
    frac = _num(s.get("water_fraction_pct"))
    warn = m.get("warnings") or []
    tags = [
        f'<span class="heye-tag info">{_esc(m.get("model_label", "-"))}</span>',
        f'<span class="heye-tag">像元 {_num(s.get("pixel_size_m"), 10.0):g} m</span>',
        f'<span class="heye-tag">耗时 {_num(result.elapsed_s):.2f} s</span>',
    ]
    if warn:
        # 警告文本可能带权重元数据/远端字段，需转义
        tags.append(f'<span class="heye-tag warn">{_esc(warn[0])}</span>')

    if result.change:
        c = result.change
        # 一律用 .get 取值：change 缺字段时直接下标会抛 KeyError，
        # 而此时成果包已经导出，界面却报"失败"，用户拿不到任何结论。
        before_km2 = _num(c.get("before_water_km2"))
        after_km2 = _num(c.get("after_water_km2"))
        new_km2 = _num(c.get("new_water_km2"))
        receded_km2 = _num(c.get("receded_water_km2"))
        persistent_km2 = _num(c.get("persistent_water_km2", after_km2 - new_km2))
        bar = min(100.0, 100.0 * new_km2 / max(after_km2, 1e-6))
        grid = [
            ("灾前水体", f'{before_km2:.2f}<small> km²</small>', ""),
            ("灾后水体", f'{after_km2:.2f}<small> km²</small>', ""),
            ("持续水体", f'{persistent_km2:.2f}<small> km²</small>', ""),
            ("退水面积", f'{receded_km2:.2f}<small> km²</small>', "color:var(--success)"),
        ]
        card_html = "".join(
            f'<div class="heye-card"><div class="k">{_esc(k)}</div><div class="v" style="{st}">{v}</div></div>'
            for k, v, st in grid
        )
        return f"""
<div class="heye-stats">
  <div class="heye-hero">
    <div class="t">新增淹没面积</div>
    <div class="n danger">{new_km2:,.2f}<small> km²</small></div>
    <div class="heye-bar"><i style="width:{bar:.1f}%"></i></div>
  </div>
  <div>{''.join(tags)}</div>
  <div class="heye-grid">{card_html}</div>
</div>"""

    cards = [
        ("占影像比例", f'{frac:.2f}<small> %</small>'),
        ("连通水域", f'{_num(s.get("n_components")):.0f}<small> 处</small>'),
        ("最大水域", f'{_num(s.get("largest_area_km2")):.2f}<small> km²</small>'),
        ("平均置信度", f'{_num(s.get("mean_confidence")):.3f}'),
    ]
    card_html = "".join(
        f'<div class="heye-card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards
    )
    return f"""
<div class="heye-stats">
  <div class="heye-hero">
    <div class="t">洪水淹没面积</div>
    <div class="n">{area:,.2f}<small> km²</small></div>
    <div class="heye-bar"><i style="width:{min(max(frac, 0.0), 100.0):.1f}%"></i></div>
  </div>
  <div>{''.join(tags)}</div>
  <div class="heye-grid">{card_html}</div>
</div>"""


def pipeline_html(model_info: Dict[str, Any], dirs: Any = None) -> str:
    """"关于系统"页：流程 + 当前模型状态 + 数据来源。"""
    all_samples = load_manifest(dirs)
    n = len(all_samples)
    n_real = sum(1 for s in all_samples if s.get("_real"))
    n_syn = n - n_real
    mode = model_info.get("mode", "baseline")
    if mode == "unet":
        # val_iou 可能缺字段或为 None，直接 :.4f 会抛 TypeError
        iou = _num(model_info.get("val_iou"), float("nan"))
        model_line = (
            f"当前已加载 <b>{_esc(model_info.get('label'))}</b>，"
            f"验证集 IoU <b>{iou:.4f}</b>，"
            f"训练数据：{_esc(model_info.get('trained_on', '-'))}（{_esc(model_info.get('data_note', '-'))}），"
            f"设备：{_esc(model_info.get('device', '-'))}"
        )
    else:
        warn = model_info.get("warning")
        extra = f"（{_esc(warn)}）" if warn else ""
        model_line = (
            f"当前运行 <b>{_esc(model_info.get('label'))}</b>（本机计算，无需大模型、无需联网）{extra}。"
            "U-Net 需在识别参数里显式选择；合成样本权重在「自动」模式下不会启用。"
        )
    return f"""
<div class="heye-stats heye-about" style="max-width:720px;margin:0 auto">
<div class="heye-hero">
  <div class="t">Huiyan · Flood Intelligence</div>
  <div class="n" style="font-size:28px;letter-spacing:-0.03em">分钟级 · 像元级 · 本机计算</div>
</div>
<div class="heye-flow">
  <div class="n">输入</div><div class="a">→</div>
  <div class="n">预处理</div><div class="a">→</div>
  <div class="n">识别</div><div class="a">→</div>
  <div class="n">后处理</div>
</div>
<div class="heye-card"><div class="k">模型</div><div class="v" style="font-size:15px;font-weight:400;letter-spacing:-0.01em">{model_line}</div></div>
<div class="heye-honesty">自带权重仅在合成样本上训练，不能当作竞赛精度。真实指标需用 Sen1Floods11 等公开数据训练后评测。</div>
<p>示例　{n_syn} 景合成影像（含真值）＋ {n_real} 组真实 Sentinel-2。<br>
训练　<code>python scripts/download_data.py --dataset sen1floods11</code></p>
</div>"""


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------


def export_result(result: Any, out_dir: Optional[str] = None) -> str:
    """导出成果包，返回 zip 路径。

    out_dir 默认取 src.paths.outputs_dir()（用户可写目录）。原实现写死
    ROOT/outputs，那是打包后的只读资源目录——装到 Program Files 这类位置时
    导出必然抛 OSError，而界面只回一句"失败"，用户拿不到任何文件。
    """
    from src.paths import outputs_dir
    from src.report import export_bundle

    target = out_dir or outputs_dir()
    try:
        out = export_bundle(result, out_dir=target)
    except OSError as exc:
        # 目标目录不可写时退到临时目录，至少保证成果包能拿到
        import tempfile

        print(f"[warn] 导出到 {target} 失败({type(exc).__name__})，改用临时目录")
        out = export_bundle(result, out_dir=tempfile.mkdtemp(prefix="huiyan_"))
    return out["zip"]


def compare_figure(result: Any) -> np.ndarray:
    """双时相变化图（红=新增淹没，绿=退水，青=持续水体）。"""
    if result.change and "change_map" in result.change:
        return result.change["change_map"]
    return result.overlay


def image_slider_component(label: str = "拖动滑块对比", height: int = 420) -> Any:
    """对比滑块组件（Gradio 版本较老时降级为两张图）。"""
    if HAS_IMAGE_SLIDER:
        return gr.ImageSlider(label=label, height=height)  # type: ignore[union-attr]
    return gr.Image(label=label, height=height)
