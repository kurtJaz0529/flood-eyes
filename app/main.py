"""
慧眼识灾 · Gradio 演示入口
===========================

启动：
    python app/main.py                 # http://127.0.0.1:7860
    python app/main.py --share         # 生成公网链接（路演/远程评审用）
    python app/main.py --port 8080 --baseline-only

单页：地图选点 + 灾前/灾后时间范围 → 自动下载卫星数据 → 识别并对比。
"""

from __future__ import annotations

import argparse
import functools
import html
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

import gradio as gr  # noqa: E402

from app.components import (  # noqa: E402
    HAS_IMAGE_SLIDER,
    compare_figure,
    empty_stats_html,
    export_result,
    image_slider_component,
    load_manifest,
    pipeline_html,
    sample_choices,
    sample_paths,
    stats_html,
)
from src import __version__ as APP_VERSION  # noqa: E402
from src.infer import FloodDetector, available_weights  # noqa: E402
from src.paths import bundle_root, outputs_dir, samples_dir, weights_dirs  # noqa: E402
from src.preprocess import load_scene  # noqa: E402
from src.report import export_bundle  # noqa: E402

ROOT = bundle_root()
SAMPLES_DIR = samples_dir()
OUT_DIR = outputs_dir()

# --------------------------------------------------------------------------
# Gradio 版本兼容层（5.x / 6.x 的 API 有差异，保证两代都能跑）
# --------------------------------------------------------------------------

GRADIO_MAJOR = int(gr.__version__.split(".")[0])


def _esc(value: Any) -> str:
    """转义插入 HTML 的文本（gr.HTML 按 innerHTML 渲染）。"""
    return html.escape(str(value), quote=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _load_apple_css() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apple.css")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return ".gradio-container {max-width: 1280px !important}"


CSS = _load_apple_css()


def _apple_theme() -> Any:
    try:
        return gr.themes.Soft(
            font=["ui-sans-serif", "system-ui", "sans-serif"],
            primary_hue="blue",
            neutral_hue="slate",
            spacing_size="md",
            radius_size="lg",
        )
    except Exception:
        return gr.themes.Soft()


def _textbox(**kwargs: Any) -> Any:
    """Gradio 6 用 buttons=[...] 取代了 show_copy_button。"""
    kw = dict(kwargs)
    if "show_copy_button" in kw:
        copy = bool(kw.pop("show_copy_button"))
        if GRADIO_MAJOR >= 6:
            if copy:
                try:
                    return gr.Textbox(**kw, buttons=["copy"])
                except Exception:
                    pass
        else:
            kw["show_copy_button"] = copy
    return gr.Textbox(**kw)


_THEME_JS = """
() => {
  try {
    const t = localStorage.getItem('heye-theme');
    if (t === 'dark') {
      document.documentElement.classList.add('heye-dark');
      document.documentElement.classList.remove('heye-light');
    }
  } catch (e) {}
}
"""


def _blocks_kwargs(title: str) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"title": title}
    if GRADIO_MAJOR < 6:  # 6.x 起 theme/css/js 移到 launch()
        kw.update(theme=_apple_theme(), css=CSS, js=_THEME_JS)
    else:
        try:  # 6.x 默认把内容限制在 ~1168px，横向布局需要更宽的画布
            import inspect

            if "fill_width" in inspect.signature(gr.Blocks.__init__).parameters:
                kw["fill_width"] = True
        except Exception:
            pass
    return kw


@functools.lru_cache(maxsize=1)
def _logo_path() -> Optional[str]:
    """应用图标文件（浏览器的 favicon 与顶栏 logo 共用同一张）。

    位于 docs/assets/app_icon.png，由 scripts/make_icon.py 从品牌 logo 生成。
    打包时该文件随 datas 一起进包，因此 source / frozen 两种模式都能取到。
    """
    candidate = os.path.join(bundle_root(), "docs", "assets", "app_icon.png")
    return candidate if os.path.isfile(candidate) else None


@functools.lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    """顶栏 logo 的 data URI。

    内联而不走外部请求：避免额外文件路径依赖，离线与打包环境都稳定。
    图标缺失时返回空串，顶栏退回原来的文字占位。
    """
    path = _logo_path()
    if not path:
        return ""
    try:
        import base64
        from pathlib import Path as _Path

        return "data:image/png;base64," + base64.b64encode(_Path(path).read_bytes()).decode("ascii")
    except OSError:
        return ""


def _launch_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if GRADIO_MAJOR >= 6:
        kw.update(theme=_apple_theme(), css=CSS, js=_THEME_JS)
    icon = _logo_path()
    if icon:
        kw["favicon_path"] = icon      # 浏览器标签页图标
    return kw

CHOICES = sample_choices()
DEFAULT_SAMPLE = CHOICES[0][1] if CHOICES else None
DEFAULT_SAMPLE2 = CHOICES[1][1] if len(CHOICES) > 1 else DEFAULT_SAMPLE


_HEADER_TEMPLATE = """
<div class="heye-nav">
  <div class="heye-nav-left">
    __LOGO__
    <div>
      <h1>慧眼识灾</h1>
      <p>地图选点 · 自动下载 · 灾前灾后对比</p>
    </div>
  </div>
  <button type="button" class="heye-theme" title="深浅色" onclick="(function(){var r=document.documentElement;var d=r.classList.toggle('heye-dark');r.classList.toggle('heye-light',!d);try{localStorage.setItem('heye-theme',d?'dark':'light')}catch(e){}})()">◐</button>
</div>
"""


def _header_html() -> str:
    """顶栏：品牌 logo + 标题 + 深浅色切换。

    刻意用占位符 + replace 而不是 f-string：下面的按钮 onclick 里满是 `{` `}`，
    放进 f-string 会被当成替换字段而语法报错。
    """
    uri = _logo_data_uri()
    if uri:
        logo = f'<div class="heye-logo"><img src="{uri}" alt="慧眼识灾"></div>'
    else:
        logo = '<div class="heye-logo">眼</div>'   # 图标缺失时的降级
    return _HEADER_TEMPLATE.replace("__LOGO__", logo)


FOOTER = f"""
<p class="heye-foot">
  Huiyan {APP_VERSION}　本机计算，不经过大模型<br>
  默认 NDWI 基线　合成权重不可作为竞赛精度
</p>
"""


# --------------------------------------------------------------------------
# 回调
# --------------------------------------------------------------------------


def _pick_source(upload: Optional[Any], sample_id: Optional[str], which: str = "post") -> Tuple[str, List[str]]:
    """优先使用上传文件，否则用示例样本。返回 (路径, 提示列表)。"""
    notes: List[str] = []
    path = ""
    if upload:
        path = upload if isinstance(upload, str) else getattr(upload, "name", str(upload))
    if not path:
        if not sample_id:
            raise gr.Error("请上传影像，或在下拉框中选择一个示例样本")
        paths = sample_paths(sample_id)
        path = paths[which]
        notes.append(f"使用示例样本：{os.path.basename(path)}")
    if not os.path.isfile(path):
        raise gr.Error(f"文件不存在：{path}")
    return path, notes


def _resolve_weights(weights: Optional[str]) -> Optional[str]:
    """把界面下拉框的值转成真实权重路径；占位符 -> None（自动选择）。"""
    if not weights or weights.startswith("（"):
        return None
    if os.path.isabs(weights) and os.path.isfile(weights):
        return weights
    for d in weights_dirs():  # 用户目录 -> 打包资源目录
        probe = os.path.join(d, os.path.basename(weights))
        if os.path.isfile(probe):
            return probe
    # 用户显式选了权重却找不到文件时不能静默返回 None：
    # 那样界面仍显示 U-Net，实际跑的是基线，属于"假成功"。
    raise gr.Error(f"找不到所选权重文件：{weights}（已搜索：{weights_dirs()}）")


def _coerce_pixel_size(value: Any) -> Optional[float]:
    """0 / 空 = 从影像读取分辨率；非法值必须报错，不能静默改用默认值。"""
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise gr.Error(f"像元大小必须是数字，收到：{value!r}") from None
    if not np.isfinite(x) or x < 0:
        raise gr.Error(f"像元大小必须是非负数字，收到：{value!r}")
    return None if x == 0 else x


def _make_detector(
    mode_label: str,
    pixel_size_m: float,
    min_area_px: float,
    threshold: float,
    tta: bool,
    weights: Optional[str],
) -> FloodDetector:
    mode = {"自动（非合成权重才用 U-Net）": "auto", "NDWI + Otsu 基线": "baseline", "U-Net 深度模型": "unet"}.get(
        mode_label, "baseline"
    )
    return FloodDetector(
        mode=mode,
        weights=_resolve_weights(weights),
        pixel_size_m=_coerce_pixel_size(pixel_size_m),
        min_area_px=int(min_area_px),
        threshold=float(threshold),
        tta=bool(tta),
    )


def run_single(
    upload: Optional[Any],
    nir_upload: Optional[Any],
    sample_id: Optional[str],
    mode_label: str,
    pixel_size_m: float,
    min_area_px: float,
    threshold: float,
    tta: bool,
    weights: Optional[str],
):
    try:
        path, notes = _pick_source(upload, sample_id, which="post")
        nir_path = None
        if nir_upload:
            nir_path = nir_upload if isinstance(nir_upload, str) else getattr(nir_upload, "name", str(nir_upload))
        det = _make_detector(mode_label, pixel_size_m, min_area_px, threshold, tta, weights)
        result = det.detect(path, nir_path=nir_path)

        overlay = result.overlay
        slider_value = (result.rgb, overlay) if HAS_IMAGE_SLIDER else result.rgb
        log = "\n".join(
            notes
            + [
                f"输入：{os.path.basename(path)}  {result.scene.shape[0]}×{result.scene.shape[1]}  "
                f"{len(result.scene.channel_names)} 波段（近红外：{'有' if result.scene.has_nir else '无'}）",
                f"模型：{result.meta.get('model_label')}",
                f"耗时：{result.elapsed_s:.2f} s",
                result.summary_text(),
            ]
        )
        zip_path = export_result(result, OUT_DIR)
        return slider_value, overlay, stats_html(result), result.summary_text(), zip_path, log
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"识别失败：{type(exc).__name__}: {exc}") from exc


def run_compare(
    pre_upload: Optional[Any],
    post_upload: Optional[Any],
    sample_id: Optional[str],
    mode_label: str,
    pixel_size_m: float,
    min_area_px: float,
    threshold: float,
    weights: Optional[str],
):
    try:
        pre_path, notes = _pick_source(pre_upload, sample_id, which="pre")
        post_path, _ = _pick_source(post_upload, sample_id, which="post")
        det = _make_detector(mode_label, pixel_size_m, min_area_px, threshold, False, weights)
        result = det.compare(pre_path, post_path)

        change = result.change
        slider_value = (change["before_overlay"], change["after_overlay"]) if HAS_IMAGE_SLIDER else change["after_overlay"]
        log = "\n".join(
            notes
            + [
                f"灾前：{os.path.basename(pre_path)}",
                f"灾后：{os.path.basename(post_path)}",
                f"模型：{result.meta.get('model_label')}",
                f"耗时：{result.elapsed_s:.2f} s",
                result.summary_text(),
            ]
        )
        zip_path = export_result(result, OUT_DIR)
        return slider_value, compare_figure(result), stats_html(result), result.summary_text(), zip_path, log
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"对比失败：{type(exc).__name__}: {exc}") from exc


def show_model_info(mode_label: str, weights: Optional[str]) -> str:
    det = _make_detector(mode_label, 10.0, 120, 0.5, False, weights)
    info = det.model_info()
    return pipeline_html(info)


# --------------------------------------------------------------------------
# 雷达（SAR）回调
# --------------------------------------------------------------------------

_SAR_CSS = ""


@functools.lru_cache(maxsize=None)
def _load_script(name: str) -> Any:
    """按路径加载 scripts/*.py，打包后也能找到。

    结果按名字缓存：原实现每次调用都 exec_module，而 _preset_events /
    _nearest_preset / _preset_map_points 会被频繁调用，脚本模块级代码
    （含耗时或写盘逻辑）会被反复执行。
    """
    import importlib.util

    path = os.path.join(ROOT, "scripts", f"{name}.py")
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到脚本：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def _preset_events() -> Dict[str, Dict[str, Any]]:
    """预设事件表。

    优先用内置事件库（data/events/flood_events.json，含 17 起历史洪灾），
    再补上 fetch_real_samples.EVENTS 里事件库没有的条目。
    返回结构与原来完全一致（label/aoi/pre/post/size/note），下游无需改动；
    事件 id 保持不变，因此本地缓存影像（data/real/{id}_pre.tif）仍能命中。
    """
    merged: Dict[str, Dict[str, Any]] = {}
    try:
        from src.events import templates_for

        merged.update(templates_for())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 事件库不可用，回退到内置脚本事件：{type(exc).__name__}: {exc}")
    try:
        for key, cfg in _load_script("fetch_real_samples").EVENTS.items():
            merged.setdefault(key, cfg)
    except Exception as exc:  # noqa: BLE001
        if not merged:
            print(f"[warn] 内置脚本事件也读取失败：{type(exc).__name__}: {exc}")
    return merged


def _sar_stats_html(result: Dict[str, Any]) -> str:
    s = result["stats"]
    p = result["provenance"]
    lonlat = p.get("aoi_lonlat") or [0, 0]
    lon, lat = _safe_float(lonlat[0]), _safe_float(lonlat[1])
    # pre_scene / aoi_source 等字段直接来自远端 STAC 元数据，必须转义后再拼 HTML。
    src = _esc(p.get("aoi_source") or p.get("source") or "-")
    new_km2 = _safe_float(s.get("new_water_km2"))
    window = p.get("window") or [0, 0]
    bar = min(100.0, 100.0 * new_km2 / max(_safe_float(s.get("window_km2")), 1.0))
    return f"""
<div class="heye-stats">
  <div class="heye-hero">
    <div class="t">新增淹没面积</div>
    <div class="n danger">{new_km2:,.2f}<small> km²</small></div>
    <div class="heye-bar"><i style="width:{bar:.1f}%"></i></div>
  </div>
  <div>
    <span class="heye-tag info">Sentinel-1 {_esc(p.get('polarization', '-'))}</span>
    <span class="heye-tag">阈值 {_safe_float(s.get('threshold_db')):.1f} dB</span>
    <span class="heye-tag">窗口 {int(_safe_float(window[0]))}×{int(_safe_float(window[1]))}</span>
    <span class="heye-tag">定位 {src} @ {lon},{lat}</span>
  </div>
  <div class="heye-grid">
    <div class="heye-card"><div class="k">灾前水体</div><div class="v">{_safe_float(s.get('pre_water_km2')):.2f}<small> km²</small></div></div>
    <div class="heye-card"><div class="k">灾后水体</div><div class="v">{_safe_float(s.get('post_water_km2')):.2f}<small> km²</small></div></div>
    <div class="heye-card"><div class="k">持续水体</div><div class="v">{_safe_float(s.get('persistent_km2')):.2f}<small> km²</small></div></div>
    <div class="heye-card"><div class="k">退水面积</div><div class="v" style="color:var(--success)">{_safe_float(s.get('receded_km2')):.2f}<small> km²</small></div></div>
  </div>
  <div style="font-size:12px;color:#5b6b7f;margin-top:6px">
    灾前景 {_esc(str(p.get('pre_scene', '-'))[:46])}…<br>灾后景 {_esc(str(p.get('post_scene', '-'))[:46])}…
  </div>
</div>"""


def run_sar(pre_path: str, post_path: str, polarization: str, threshold_db: float, size: int):
    """雷达双时相处理：本地 SAFE 目录 → 水体范围 + 变化统计 + 成果包。"""
    import json
    import zipfile

    from src.sar import process_sar_pair

    logs: List[str] = []

    def prog(msg: str) -> None:
        logs.append(msg)
        print(f"[SAR] {msg}")

    try:
        for tag, p in (("灾前", pre_path), ("灾后", post_path)):
            if not p or not os.path.exists(p.strip()):
                raise gr.Error(f"{tag} SAFE 路径不存在：{p}")
        out_dir = os.path.join(OUT_DIR, "sar")
        result = process_sar_pair(
            pre_path.strip(), post_path.strip(), polarization,
            out_dir=out_dir, size=int(size), threshold_db=float(threshold_db), progress=prog,
        )
        sid = result["id"]
        zip_path = os.path.join(out_dir, f"{sid}_SAR成果包.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in result["paths"].values():
                zf.write(p, os.path.basename(p))
            zf.writestr(f"{sid}_info.json", json.dumps(
                {"id": sid, "stats": result["stats"], "provenance": result["provenance"]},
                ensure_ascii=False, indent=2))
        logs.append(f"成果包：{zip_path}")
        return result["paths"]["preview"], _sar_stats_html(result), zip_path, "\n".join(logs)
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"雷达处理失败：{type(exc).__name__}: {exc}") from exc


def _zip_paths(zip_path: str, files: Dict[str, str], extra: Optional[Dict[str, Any]] = None) -> str:
    import json
    import zipfile

    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files.values():
            if p and os.path.isfile(p):
                zf.write(p, os.path.basename(p))
        if extra:
            zf.writestr("info.json", json.dumps(extra, ensure_ascii=False, indent=2, default=str))
    return zip_path


def _local_optical_pair(sid: str) -> Optional[Tuple[str, str]]:
    """预设事件若本地已有 data/real/{id}_pre/post.tif，直接用，免再下载。"""
    real = os.path.join(ROOT, "data", "real")
    pre = os.path.join(real, f"{sid}_pre.tif")
    post = os.path.join(real, f"{sid}_post.tif")
    if os.path.isfile(pre) and os.path.isfile(post):
        return pre, post
    return None


class _LogTee:
    """把脚本 print 实时灌进界面日志。"""

    def __init__(self, emit, also) -> None:
        self._emit = emit
        self._also = also
        self._buf = ""

    def write(self, data: str) -> int:
        if self._also is not None:
            try:
                self._also.write(data)
            except Exception:
                pass
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._emit(line)
        return len(data)

    def flush(self) -> None:
        if self._also is not None:
            try:
                self._also.flush()
            except Exception:
                pass


def _as_local_path(upload: Any, typed: Optional[str] = None) -> Optional[str]:
    raw = (typed or "").strip().strip('"')
    if raw and os.path.isfile(raw):
        return raw
    if upload:
        path = upload if isinstance(upload, str) else getattr(upload, "name", "")
        if path and os.path.isfile(str(path)):
            return str(path)
    return None


def _parse_ymd(value: Any, label: str) -> str:
    s = str(value or "").strip()[:10]
    if len(s) != 10 or s[4] != "-" or s[7] != "-":
        raise gr.Error(f"请填写{label}（YYYY-MM-DD）")
    try:
        np.datetime64(s)
    except Exception as exc:
        raise gr.Error(f"{label} 不是有效日期：{s}") from exc
    return s


def _slider_pair(before: Any, after: Any) -> Any:
    if HAS_IMAGE_SLIDER:
        return (before, after)
    return after


def _imread_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _nearest_preset(lon: float, lat: float, max_deg: float = 0.25) -> Optional[str]:
    try:
        events = _preset_events()
    except Exception:
        return None
    best_key = None
    best_d = max_deg
    for key, cfg in events.items():
        elon, elat = cfg["aoi"]
        dist = ((float(lon) - float(elon)) ** 2 + (float(lat) - float(elat)) ** 2) ** 0.5
        if dist < best_d:
            best_d = dist
            best_key = str(key)
    return best_key


def _fallback_local_optical(lon: float, lat: float) -> Optional[Tuple[str, str, str]]:
    """下载失败时，用距离最近的本地真实影像把流程跑完。"""
    try:
        events = _preset_events()
    except Exception:
        events = {}
    ranked: List[Tuple[float, str]] = []
    for key, cfg in events.items():
        pair = _local_optical_pair(key)
        if not pair:
            continue
        elon, elat = cfg["aoi"]
        dist = ((float(lon) - float(elon)) ** 2 + (float(lat) - float(elat)) ** 2) ** 0.5
        ranked.append((dist, key))
    if not ranked:
        for key in ("poyang2020", "zhuozhou2023"):
            pair = _local_optical_pair(key)
            if pair:
                return pair[0], pair[1], key
        return None
    ranked.sort()
    key = ranked[0][1]
    pair = _local_optical_pair(key)
    if not pair:
        return None
    return pair[0], pair[1], key


def _pipeline_body(
    lon: Optional[float], lat: Optional[float],
    pre_start: Optional[str], pre_end: Optional[str],
    post_start: Optional[str], post_end: Optional[str],
    win_size: int, prog,
    terrain_profile: str = "unspecified", dem_path: Optional[str] = None,
) -> Tuple[Any, Any, str, str, str]:
    """Use the same validated processing core as persistent batch jobs."""
    from types import SimpleNamespace
    from src.task_contracts import PipelineRequest
    from src.pipeline import run_pipeline

    request = PipelineRequest(
        lon=lon, lat=lat, pre_start=pre_start, pre_end=pre_end,
        post_start=post_start, post_end=post_end, size=int(win_size or 768),
        terrain_profile=terrain_profile or "unspecified",
    )
    outcome = run_pipeline(
        request, os.path.join(OUT_DIR, "automation"), dem_path=dem_path,
        progress=lambda message, stage=None: prog(message),
    )
    result = SimpleNamespace(
        stats=outcome["stats"], meta=outcome["meta"], change=outcome.get("change"),
        elapsed_s=outcome["elapsed_s"],
    )
    files = outcome["files"]
    slider = _slider_pair(files["before_overlay"], files["overlay"])
    return slider, files["comparison"], stats_html(result), outcome["summary"], outcome["zip"]


def run_full_pipeline(
    lon: Optional[float],
    lat: Optional[float],
    pre_start: Optional[str],
    pre_end: Optional[str],
    post_start: Optional[str],
    post_end: Optional[str],
    win_size: int,
    terrain_profile: str = "unspecified",
    dem_path: Optional[str] = None,
):
    """地图选点 → 下载 → 灾前/灾后对比。后台线程跑，日志持续刷新。"""
    import threading
    import time

    logs: List[str] = []
    lock = threading.Lock()
    box: Dict[str, Any] = {"done": False, "err": None, "final": None}

    def prog(msg: str) -> None:
        with lock:
            logs.append(str(msg))
        try:
            sys.__stdout__.write(f"[全流程] {msg}\n")
            sys.__stdout__.flush()
        except Exception:
            pass

    def snapshot():
        with lock:
            text = "\n".join(logs[-80:]) or "启动中…"
        final = box.get("final")
        if final:
            slider, preview, html, summary, zpath = final
            return slider, preview, html, summary, zpath, text
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), text

    def work() -> None:
        try:
            box["final"] = _pipeline_body(
                lon, lat, pre_start, pre_end, post_start, post_end, win_size, prog,
                terrain_profile, dem_path,
            )
        except Exception as exc:
            box["err"] = exc
            traceback.print_exc()
            prog(f"失败：{type(exc).__name__}: {exc}")
        finally:
            box["done"] = True

    threading.Thread(target=work, daemon=True).start()
    n_seen = 0
    last_beat = time.time()
    yield snapshot()
    while not box["done"]:
        time.sleep(0.4)
        with lock:
            n = len(logs)
        now = time.time()
        if n != n_seen:
            n_seen = n
            last_beat = now
            yield snapshot()
        elif now - last_beat >= 8:
            prog("仍在下载/处理，请稍候（网络慢时单景窗口可能要 1–3 分钟）…")
            last_beat = now
            yield snapshot()

    err = box["err"]
    if err is not None:
        if isinstance(err, gr.Error):
            raise err
        raise gr.Error(f"全流程失败：{type(err).__name__}: {err}\n" + "\n".join(logs[-12:])) from err
    yield snapshot()


# --------------------------------------------------------------------------
# 地图选点（默认高德，免密钥；点选输出 WGS84 供卫星检索）
# --------------------------------------------------------------------------


def _preset_map_points() -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    try:
        for key, cfg in _preset_events().items():
            lon, lat = cfg["aoi"]
            pre = cfg.get("pre") or ("", "", "")
            post = cfg.get("post") or ("", "", "")
            pts.append({
                "key": key,
                "lon": float(lon),
                "lat": float(lat),
                "label": cfg.get("label", key),
                "pre_start": pre[0],
                "pre_end": pre[1],
                "post_start": post[0],
                "post_end": post[1],
            })
    except Exception:
        pass
    return pts


def _map_engine_js() -> str:
    """无外部 Leaflet：用高德瓦片在页面内画地图（国内可访问）。"""
    import json

    presets = json.dumps(_preset_map_points(), ensure_ascii=False)
    # 防止预设文案里的 "</script>" 之类把注入点截断
    presets = presets.replace("</", "<\\/")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heye_map.js")
    try:
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
    except OSError as exc:
        # 地图脚本缺失不应该让整个应用起不来：退化为一段提示，其余功能照常。
        print(f"[warn] 地图脚本 heye_map.js 读取失败({type(exc).__name__})，地图将不可用")
        body = (
            "(function(){var r=element.querySelector('#heye-map')||element;"
            "r.innerHTML='<div class=\"heye-empty\">地图脚本缺失，请手动填写经纬度</div>';})();"
        )
    return f"window.__HEYE_PRESETS = {presets};\n{body}"


def _map_html(lon: float = 116.30, lat: float = 29.15, zoom: int = 7) -> str:
    """占位：真实地图由 js_on_load 绘制，避免 iframe/CDN 被拦截。"""
    _ = (lon, lat, zoom)
    return '<div id="heye-map" class="heye-map"></div>'


def _dates_from_preset(cfg: Dict[str, Any]) -> Tuple[str, str, str, str]:
    pre = cfg.get("pre") or ("2020-05-08", "2020-06-05", "2020-05-20")
    post = cfg.get("post") or ("2020-07-10", "2020-07-30", "2020-07-13")
    return str(pre[0]), str(pre[1]), str(post[0]), str(post[1])


def geocode_place(query: str):
    """先匹配内置事件名，再尝试地名检索。"""
    import json
    import urllib.parse
    import urllib.request

    q = (query or "").strip()
    if not q:
        raise gr.Error("请输入地名后再点「定位此地」")
    try:
        events = _preset_events()
    except Exception:
        events = {}
    qn = q.lower()
    for key, cfg in events.items():
        blob = f"{key} {cfg.get('label', '')}".lower()
        if qn in blob or key in qn:
            lon, lat = cfg["aoi"]
            a, b, c, d = _dates_from_preset(cfg)
            return (
                float(lon), float(lat),
                f"已定位：{cfg.get('label', key)}，时间范围已填入",
                a, b, c, d,
            )
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": 1}
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "HuiyanShizai/0.2 (flood-eyes mapping)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        raise gr.Error("地名检索不可用，请直接在地图上点击选点") from None
    if not data:
        raise gr.Error(f"找不到地点：{q}。请在地图上点击。")
    lon = float(data[0]["lon"])
    lat = float(data[0]["lat"])
    label = data[0].get("display_name") or q
    return (
        lon, lat,
        f"已定位：{label}",
        gr.update(), gr.update(), gr.update(), gr.update(),
    )


def apply_map_point(
    lon: Optional[float],
    lat: Optional[float],
    pre_start: Optional[str],
    pre_end: Optional[str],
    post_start: Optional[str],
    post_end: Optional[str],
):
    if lon is None or lat is None:
        raise gr.Error("请先在地图上点一下")
    lon_v, lat_v = float(lon), float(lat)
    if not (-180.0 <= lon_v <= 180.0 and -90.0 <= lat_v <= 90.0):
        raise gr.Error("地图坐标无效")
    note = f"已写入 {lon_v:.4f}, {lat_v:.4f}，可改时间范围后开始分析"
    nearby = _nearest_preset(lon_v, lat_v)
    if nearby:
        try:
            note = f"已写入 {_preset_events()[nearby].get('label', nearby)}  {lon_v:.4f}, {lat_v:.4f}"
        except Exception:
            pass
    else:
        # 不在预设点上时，告诉用户附近有哪些已知洪灾事件——
        # 否则很容易出现"随便点一个位置 + 默认日期"导致结果为 0 的困惑。
        note += _nearby_event_note(lon_v, lat_v)
    return lon_v, lat_v, note, pre_start, pre_end, post_start, post_end


def _nearby_event_note(lon: float, lat: float, max_km: float = 300.0) -> str:
    """给"附近有哪些已知洪灾事件"的提示文案（无则返回空串）。"""
    try:
        from src.events import nearest_events

        hits = nearest_events(lon, lat, max_km=max_km, limit=2)
    except Exception:  # noqa: BLE001
        return ""
    if not hits:
        return ""
    parts = [f"{e.label}（{e.event_date}，约 {d:.0f} km）" for d, e in hits]
    return "\n\n💡 附近已知洪灾事件：" + "；".join(parts) + \
           "\n　 想分析这些事件，请点地图上的圆点或在上方「检索历史洪灾事件」里载入——" \
           "直接点空白处时间范围可能与该地实际汛情不符，容易得出 0 km²。"


# --------------------------------------------------------------------------
# 事件库：检索 + 一键填表
# --------------------------------------------------------------------------


def search_event_library(query: str):
    """检索内置事件库，返回（下拉选项, 提示）。"""
    try:
        from src.events import load_registry, search_events

        events = load_registry()
        hits = search_events(query or "", events=events, limit=50)
    except Exception as exc:  # noqa: BLE001
        return gr.update(choices=[], value=None), \
            f"事件库不可用：{type(exc).__name__}: {exc}"
    if not hits:
        return gr.update(choices=[], value=None), (
            "没有匹配的事件。可以换关键词（如「鄱阳湖」「湖南」「2024」），"
            "或用命令行联网发现新事件：\n"
            "`python scripts/events.py --online --from 2024-06-01 --to 2024-12-31 --save`"
        )
    choices = [(f"{e.event_date}｜{e.label}", e.id) for e in hits]
    return gr.update(choices=choices, value=choices[0][1]), \
        f"匹配到 {len(hits)} 起事件，选一个后点「载入并填表」。"


def load_event_template(event_id: Optional[str]):
    """把事件模板填进界面：坐标 + 灾前/灾后四个日期，并给出事件背景。"""
    if not event_id:
        raise gr.Error("请先搜索并选择一个事件")
    try:
        from src.events import build_template, load_registry, template_hint
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"事件库不可用：{type(exc).__name__}: {exc}") from exc

    event = next((e for e in load_registry() if e.id == event_id), None)
    if event is None:
        raise gr.Error(f"事件库里没有找到：{event_id}")

    tpl = build_template(event)
    lon, lat = tpl["aoi"]
    pre, post = tpl["pre"], tpl["post"]
    hint = template_hint(event)
    if not tpl["analysable"]:
        hint += "　⚠️ 该事件早于 Sentinel-2 可用日期（2015-06-23），本项目取不到影像。"
    if tpl["precision"] != "aoi":
        hint += "　⚠️ 坐标为事件区域近似中心，建议在地图上确认具体受淹区。"
    return (float(lon), float(lat), pre[0], pre[1], post[0], post[1], hint)


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------


def build_ui(baseline_only: bool = False) -> gr.Blocks:
    _ = baseline_only
    with gr.Blocks(**_blocks_kwargs("慧眼识灾 · 遥感 AI 洪水识别系统")) as demo:
        gr.HTML(_header_html())
        gr.Markdown(
            "在地图上点选任意地点后点击‘使用此地点’（或搜索地名），填好灾前/灾后时间范围后一键分析。"
            "系统按所选地点下载 Sentinel-2 公开影像并对比，首次下载约 1–3 分钟，同一地点再次分析走本地缓存。"
        )

        # —— 第一区（横向）：地图选点 ＋ 参数设置 ——
        with gr.Row():
            with gr.Column(scale=7, min_width=420, elem_classes=["heye-map-col"]):
                map_view = gr.HTML(
                    value=_map_html(),
                    elem_id="heye-map-host",
                    min_height=420,
                    js_on_load=_map_engine_js(),
                )
            with gr.Column(scale=5, min_width=340, elem_classes=["heye-panel"]):
                gr.HTML('<div class="heye-sec-title">① 选点与时间</div>')
                # —— 事件库：检索历史洪灾事件 → 一键填表 ——
                with gr.Row():
                    event_q = gr.Textbox(
                        label="检索历史洪灾事件",
                        placeholder="如：鄱阳湖 / 湖南 / 2024 / 郑州",
                        scale=7,
                        elem_classes=["heye-mini"],
                    )
                    event_search_btn = gr.Button("搜索事件", scale=3, min_width=92)
                with gr.Row():
                    event_pick = gr.Dropdown(
                        label="匹配到的事件（坐标与时间将自动填入）",
                        choices=[],
                        value=None,
                        scale=8,
                    )
                    event_load_btn = gr.Button("载入并填表", variant="primary", scale=4,
                                               min_width=104, elem_classes=["heye-primary"])
                with gr.Row():
                    place_q = gr.Textbox(
                        label="或按地名定位",
                        placeholder="例如：鄱阳湖、涿州、洞庭湖",
                        scale=7,
                        elem_classes=["heye-mini"],
                    )
                    search_btn = gr.Button("定位此地", scale=3, min_width=92)
                with gr.Row():
                    lon_in = gr.Number(label="经度（WGS84）", value=116.30, precision=4, scale=4)
                    lat_in = gr.Number(label="纬度（WGS84）", value=29.15, precision=4, scale=4)
                    pick_btn = gr.Button("使用此地点", scale=3, min_width=104)
                with gr.Row():
                    pre_start = gr.Textbox(label="灾前 · 开始", value="2020-05-08", placeholder="YYYY-MM-DD", elem_classes=["heye-mini"])
                    pre_end = gr.Textbox(label="灾前 · 结束", value="2020-06-05", placeholder="YYYY-MM-DD", elem_classes=["heye-mini"])
                with gr.Row():
                    post_start = gr.Textbox(label="灾后 · 开始", value="2020-07-10", placeholder="YYYY-MM-DD", elem_classes=["heye-mini"])
                    post_end = gr.Textbox(label="灾后 · 结束", value="2020-07-30", placeholder="YYYY-MM-DD", elem_classes=["heye-mini"])
                with gr.Row():
                    win_size = gr.Dropdown(
                        [512, 768, 1024],
                        value=512,
                        label="窗口边长（像元，10 m → 5/8/10 km）",
                        scale=5,
                    )
                    go_btn = gr.Button("开始分析", variant="primary", size="lg", scale=5, elem_classes=["heye-primary"])
                search_status = gr.Markdown("右侧地图点击选点。坐标转为 WGS84，供卫星检索。")
                terrain_profile = gr.Dropdown(
                    choices=[("未指定", "unspecified"), ("平原河湖", "plain"),
                             ("丘陵", "hilly"), ("山地", "mountain"),
                             ("城市建成区", "urban"), ("沿海与河口", "coastal")],
                    value="unspecified", label="地貌场景（由使用者选择）",
                )
                dem_input = gr.File(label="可选 DEM 高程影像（高程单位：米）",
                                    file_types=[".tif", ".tiff"], type="filepath")
                gr.Markdown("地貌用于提示适用性；提供 DEM 后另附坡度风险图，供复核使用。")

        # —— 第二区（横向）：灾前↔灾后 ＋ 变化检测 ——
        with gr.Row():
            with gr.Column(scale=6, min_width=360, elem_classes=["heye-panel"]):
                gr.HTML('<div class="heye-sec-title">② 灾前 ↔ 灾后对比</div>')
                slider = image_slider_component("拖动滑块：灾前 ↔ 灾后")
            with gr.Column(scale=6, min_width=360, elem_classes=["heye-panel"]):
                gr.HTML('<div class="heye-sec-title">③ 变化检测图</div>')
                change_img = gr.Image(label="变化检测图", height=346, show_label=False)
                gr.HTML(
                    '<div class="heye-legend">'
                    '<span class="heye-tag" style="background:rgba(255,59,48,.16);color:#ff3b30">红 · 新增淹没</span>'
                    '<span class="heye-tag" style="background:rgba(52,199,89,.16);color:#248a3d">绿 · 退水</span>'
                    '<span class="heye-tag info">青 · 持续水体</span></div>'
                )

        # —— 第三区（横向）：统计 ＋ 结论 ＋ 成果包 ——
        with gr.Row():
            with gr.Column(scale=6, min_width=360, elem_classes=["heye-panel"]):
                pipe_stats = gr.HTML(value=empty_stats_html("分析完成后显示灾前/灾后对比"), label="统计面板")
            with gr.Column(scale=3, min_width=240, elem_classes=["heye-panel"]):
                pipe_summary = _textbox(label="对比结论", lines=7, show_copy_button=True)
            with gr.Column(scale=3, min_width=220, elem_classes=["heye-panel"]):
                pipe_zip = gr.File(label="成果包", interactive=False)

        # —— 日志（通栏，可折叠）——
        with gr.Accordion("流程日志（下载/识别进度）", open=True, elem_classes=["heye-log-acc"]):
            pipe_log = _textbox(label="", lines=6, show_copy_button=True, elem_classes=["heye-log"], container=False)

        _MAP_JS = (
            "(lon, lat, a, b, c, d) => ["
            "window._heyeLon ?? lon, window._heyeLat ?? lat, "
            "window._heyePreStart ?? a, window._heyePreEnd ?? b, "
            "window._heyePostStart ?? c, window._heyePostEnd ?? d]"
        )
        # 事件库：搜索 → 选一个 → 一键把坐标与时间填进表单
        event_search_btn.click(
            search_event_library,
            inputs=event_q,
            outputs=[event_pick, search_status],
        )
        event_q.submit(
            search_event_library,
            inputs=event_q,
            outputs=[event_pick, search_status],
        )
        event_load_btn.click(
            load_event_template,
            inputs=event_pick,
            outputs=[lon_in, lat_in, pre_start, pre_end, post_start, post_end, search_status],
        )
        event_pick.change(
            load_event_template,
            inputs=event_pick,
            outputs=[lon_in, lat_in, pre_start, pre_end, post_start, post_end, search_status],
        )
        search_btn.click(
            geocode_place,
            inputs=place_q,
            outputs=[lon_in, lat_in, search_status, pre_start, pre_end, post_start, post_end],
        )
        place_q.submit(
            geocode_place,
            inputs=place_q,
            outputs=[lon_in, lat_in, search_status, pre_start, pre_end, post_start, post_end],
        )
        pick_btn.click(
            apply_map_point,
            inputs=[lon_in, lat_in, pre_start, pre_end, post_start, post_end],
            outputs=[lon_in, lat_in, search_status, pre_start, pre_end, post_start, post_end],
            js=_MAP_JS,
        )
        lon_in.change(
            lambda lon, lat: None,
            inputs=[lon_in, lat_in],
            js="(lon, lat) => { if (window._heyeFly && lon != null && lat != null) window._heyeFly(Number(lat), Number(lon)); }",
        )
        go_btn.click(
            run_full_pipeline,
            inputs=[lon_in, lat_in, pre_start, pre_end, post_start, post_end, win_size,
                    terrain_profile, dem_input],
            outputs=[slider, change_img, pipe_stats, pipe_summary, pipe_zip, pipe_log],
        )
        from app.automation import build_automation_ui
        build_automation_ui(os.path.join(OUT_DIR, "automation"))
        gr.HTML(FOOTER)
    return demo

def stats_html_preview(sample_id: Optional[str]) -> str:
    """未识别时先展示参考信息：合成样本给真值，真实样本给数据溯源。"""
    if not sample_id:
        return empty_stats_html()
    try:
        s = next(x for x in load_manifest() if x["id"] == sample_id)
    except Exception:
        return empty_stats_html()
    if s.get("_real"):
        prov = s.get("provenance", {})
        rows = [
            ("灾前景", f"{prov.get('pre_scene', '-')}　{str(prov.get('pre_datetime', ''))[:10]}"),
            ("灾后景", f"{prov.get('post_scene', '-')}　{str(prov.get('post_datetime', ''))[:10]}"),
            ("AOI 窗口云量", f"灾前 {prov.get('pre_window_cloud_pct', '-')}%　灾后 {prov.get('post_window_cloud_pct', '-')}%"),
            ("数据来源", "Sentinel-2 L2A · AWS 公开 COG"),
            ("许可", "Copernicus Sentinel Data Terms（免费开放）"),
        ]
        body = "".join(
            f'<div class="k" style="margin:6px 0 2px">{k}</div><div style="font-size:13px">{v}</div>'
            for k, v in rows
        )
        return f"""
<div class="heye-card" style="background:var(--accent-soft);box-shadow:none">
  <div class="k">数据溯源</div>
  <div class="v" style="font-size:15px;font-weight:600">真实卫星影像（无逐像元真值）</div>
  {body}
  <p style="font-size:12px;color:var(--text-2);margin:10px 0 0">请与灾害报道交叉验证，勿把演示数字当作精度。</p>
</div>"""
    return f"""
<div class="heye-stats">
  <div class="heye-hero">
    <div class="t">样本真值参考</div>
    <div class="n danger">{s.get('flood_area_km2', 0):.2f}<small> km²</small></div>
  </div>
  <div class="heye-grid">
    <div class="heye-card"><div class="k">灾前水体</div><div class="v">{s.get('pre_water_km2', 0):.2f}<small> km²</small></div></div>
    <div class="heye-card"><div class="k">灾后水体</div><div class="v">{s.get('post_water_km2', 0):.2f}<small> km²</small></div></div>
  </div>
</div>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="慧眼识灾 Gradio 演示界面")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成公网分享链接")
    ap.add_argument("--baseline-only", action="store_true", help="只暴露基线模型（无权重时）")
    args = ap.parse_args()

    weights = available_weights()
    print("=" * 68)
    print(f"慧眼识灾 · 遥感 AI 洪水识别系统（演示版 v{APP_VERSION}）")
    print(f"示例样本：{len(CHOICES)} 景   |   权重：{len(weights)} 个" + (f" -> {os.path.basename(weights[0])}" if weights else "（将使用 NDWI 基线）"))
    print(f"访问地址：http://{args.host}:{args.port}")
    print("=" * 68)

    launch_kw = dict(_launch_kwargs())
    if args.share:
        # --share 会把服务发布到公网，且本界面包含上传与本地路径输入参数，
        # 默认无鉴权等于把这些能力开放给任何人。强制要求账号口令才允许分享。
        share_user = os.environ.get("HUIYAN_SHARE_USER")
        share_password = os.environ.get("HUIYAN_SHARE_PASSWORD")
        if not (share_user and share_password):
            raise SystemExit(
                "已拒绝 --share：该选项会把服务发布到公网，而本界面可上传文件、填写本地路径，"
                "无鉴权开放存在风险。\n"
                "如确需分享，请先设置环境变量 HUIYAN_SHARE_USER 与 HUIYAN_SHARE_PASSWORD，"
                "启动时会自动启用用户名/密码保护。"
            )
        launch_kw["auth"] = (share_user, share_password)
        print("[warn] --share 已启用公网分享，并已开启口令保护")

    demo = build_ui(baseline_only=args.baseline_only)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=False,
        quiet=False,
        **launch_kw,
    )


if __name__ == "__main__":
    main()
