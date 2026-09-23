"""
慧眼识灾 · 单次处理流水线（任务 D）
===================================

把"一个 `PipelineRequest`"变成"一个可交付的成果包"，固定三段：

    acquire（获取灾前/灾后影像） → detect（识别 + 变化 + 质量门禁） → export（导出成果包）

设计约束：

* 全流程持有进程级 ``PIPELINE_LOCK``：``scripts/fetch_real_samples`` 用模块级
  全局变量保存下载超时/快速模式，单跑与批处理并发会互相污染，因此整段串行。
* 每次运行一个独立 ``run_id`` 目录，互不覆盖；目录内原子写 ``manifest.json``，
  记录请求身份、本地输入/DEM 哈希、获取到的产物、完成结论与成果文件哈希。
* 同一 ``run_id`` 只有在"请求身份 + 本地输入/DEM 哈希 + 成果文件哈希"全部吻合时
  才直接复用；否则拒绝覆盖，避免把两次不同请求的产物混在一起。
* 获取走 ``fetch_real_samples.fetch_event``（模块只加载一次），按请求指纹落到确定
  性缓存目录，``allow_cloudy=False``、``fast=False``，不做"附近演示数据"兜底。
* 识别后先过共同有效观测门禁：低于 ``request.min_valid_pct`` 或为 0 时抛
  ``DataQualityError``，明确说"数据不足"，绝不给出"零淹没"的假结论。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .jobs import JobCancelled
from .report import export_bundle, save_rgb
from .task_contracts import PipelineRequest
from .terrain import assess_dem, terrain_context

__all__ = ["PIPELINE_LOCK", "PIPELINE_SCHEMA", "DataQualityError", "run_pipeline", "run_dir_for"]

#: 整个处理（获取 + 识别 + 导出）期间持有的进程级互斥锁。
PIPELINE_LOCK = threading.RLock()
PIPELINE_SCHEMA = 1
#: 下载脚本模块级全局（超时/快速模式）不允许并发，因此整段串行。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_ARTIFACT_ROLES = ("pre", "post", "preview", "pre_scl", "post_scl")

_FETCH_MODULE: Any = None


class DataQualityError(RuntimeError):
    """共同有效观测不足，无法给出淹没结论（不是"零淹没"）。"""


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """严格 JSON 安全：剔除 ndarray（绝不字符串化），numpy 标量还原为 Python 值。"""
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, np.ndarray):
                continue  # 栅格数组由 export_bundle 写成 GeoTIFF，不进 JSON
            out[str(key)] = _jsonable(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value if not isinstance(item, np.ndarray)]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def _atomic_write_json(path: str, payload: Any) -> None:
    """原子写 JSON：先写同目录临时文件再替换，避免半截清单被当成有效记录。"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _load_manifest(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _progress(progress: Optional[Callable[..., Any]], message: str, stage: Optional[str] = None) -> None:
    if progress is None:
        return
    try:
        progress(str(message), stage=stage)
    except TypeError:  # 兼容只接受单个位置参数的旧回调
        progress(str(message))


def _check_cancel(is_cancelled: Optional[Callable[[], bool]]) -> None:
    """安全边界检查取消标志；确认取消就抛 ``JobCancelled``。"""
    if is_cancelled is not None and is_cancelled():
        raise JobCancelled("任务在流水线安全边界被取消")


def _safe_run_id(run_id: Any) -> str:
    text = str(run_id if run_id is not None else "").strip()
    if not _RUN_ID_RE.fullmatch(text) or text in (".", ".."):
        raise ValueError(f"非法 run_id：{run_id!r}（只允许字母数字._-，长度 1..64）")
    return text


def run_dir_for(out_root: str, run_id: str) -> str:
    """某次运行的产物目录（run_id 先校验，杜绝路径穿越）。"""
    return os.path.join(os.path.abspath(os.fspath(out_root)), "runs", _safe_run_id(run_id))


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------


def _inspect_raster(path: str, label: str) -> None:
    try:
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(path) as ds:
            if ds.count < 1 or ds.width < 1 or ds.height < 1:
                raise ValueError(f"{label}影像为空（{ds.count} 波段 {ds.width}×{ds.height}）：{path}")
            ds.read(1, window=Window(0, 0, min(16, ds.width), min(16, ds.height)))
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取{label}影像（{type(exc).__name__}: {exc}）：{path}") from exc


def _validate_local_pair(local_pair: Any) -> Optional[Tuple[str, str]]:
    """可选的两份显式本地影像；路径必须可读，否则显式报错，不做静默降级。"""
    if local_pair is None:
        return None
    if isinstance(local_pair, (str, os.PathLike)):
        raise ValueError("local_pair 需要 (灾前, 灾后) 两个文件路径")
    try:
        items = list(local_pair)
    except TypeError as exc:
        raise ValueError("local_pair 需要 (灾前, 灾后) 两个文件路径") from exc
    if len(items) != 2:
        raise ValueError(f"local_pair 需要正好两个文件，收到 {len(items)} 个")
    paths: List[str] = []
    for label, item in zip(("灾前", "灾后"), items):
        path = os.path.abspath(os.fspath(item))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label}本地影像不存在：{path}")
        if os.path.getsize(path) <= 0:
            raise ValueError(f"{label}本地影像为空文件：{path}")
        _inspect_raster(path, label)
        paths.append(path)
    if os.path.normcase(paths[0]) == os.path.normcase(paths[1]):
        raise ValueError("灾前/灾后本地影像不能是同一个文件")
    return paths[0], paths[1]


def _validate_dem(dem_path: Any) -> Optional[str]:
    """DEM 可选；一旦提供就必须能读且有有效高程，否则显式失败。"""
    if dem_path is None or (isinstance(dem_path, str) and not dem_path.strip()):
        return None
    path = os.path.abspath(os.fspath(dem_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DEM 文件不存在：{path}")
    try:
        import rasterio

        with rasterio.open(path) as ds:
            if ds.count < 1 or ds.width < 1 or ds.height < 1:
                raise ValueError(f"DEM 为空（{ds.count} 波段 {ds.width}×{ds.height}）：{path}")
            if ds.crs is None or ds.transform is None:
                raise ValueError(f"DEM 缺少 CRS/仿射变换，无法重投影到影像网格：{path}")
            shape = (min(64, ds.height), min(64, ds.width))
            sample = ds.read(1, out_shape=shape)
            masks = ds.read_masks(1, out_shape=shape)
            if not bool(np.any((masks > 0) & np.isfinite(sample))):
                raise ValueError(f"DEM 全部为无数据/非有限值，无法评估坡度：{path}")
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取 DEM（{type(exc).__name__}: {exc}）：{path}") from exc
    return path


def _build_identity(
    request: PipelineRequest,
    local_pair: Optional[Tuple[str, str]],
    dem: Optional[str],
    synthetic: bool,
) -> Dict[str, Any]:
    """完成复用的身份：请求指纹 + 本地输入/DEM 内容哈希。"""
    return {
        "request_key": request.cache_key,
        "local_pre_sha256": _sha256_file(local_pair[0]) if local_pair else None,
        "local_post_sha256": _sha256_file(local_pair[1]) if local_pair else None,
        "dem_sha256": _sha256_file(dem) if dem else None,
        "synthetic": bool(synthetic),
    }


# --------------------------------------------------------------------------
# 获取（acquire）
# --------------------------------------------------------------------------


def _fetch_module() -> Any:
    """按路径加载 ``scripts/fetch_real_samples.py``，进程内只加载一次。"""
    global _FETCH_MODULE
    if _FETCH_MODULE is None:
        import importlib.util

        path = os.path.join(_ROOT, "scripts", "fetch_real_samples.py")
        if not os.path.isfile(path):
            raise RuntimeError(f"找不到影像获取脚本：{path}")
        spec = importlib.util.spec_from_file_location("_flood_eyes_fetch_real_samples", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载影像获取脚本：{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FETCH_MODULE = module
    return _FETCH_MODULE


def _acquire(request: PipelineRequest, out_root: str) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """按请求指纹到确定性缓存目录抓取一对影像，返回 (pre, post, entry, 产物哈希)。"""
    cache_dir = os.path.join(out_root, "acquisition_cache", request.cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    key = request.cache_key
    cfg = {
        "label": f"自定义 {request.lon:.4f},{request.lat:.4f}",
        "aoi": (float(request.lon), float(request.lat)),
        "pre": (request.pre_start, request.pre_end, request.pre_end),
        "post": (request.post_start, request.post_end, request.post_start),
        "size": int(request.size),
        "note": "按请求坐标与时间窗口自动获取 Sentinel-2 L2A 公开影像",
    }
    module = _fetch_module()
    entry = module.fetch_event(
        key, cfg, cache_dir, size=int(request.size),
        max_cloud=float(request.max_cloud_pct), allow_cloudy=False, fast=False, budget_s=120,
    )
    if not isinstance(entry, dict):
        raise RuntimeError(
            "未能在指定的时间窗口获取到满足云量要求的卫星影像；"
            "请调整时间范围/云量上限，或改用本地影像输入。"
        )
    paths: Dict[str, str] = {}
    for role in _ARTIFACT_ROLES:
        name = entry.get(role)
        if not name:
            continue
        # 获取清单中的文件名必须留在该请求自己的缓存目录内。
        # 拒绝绝对路径和 ../，避免损坏或被改写的清单串用其他请求的影像。
        name = str(name)
        if os.path.basename(name) != name or name in (".", ".."):
            raise RuntimeError(f"获取结果的{role}文件名无效：{name}")
        full = os.path.join(cache_dir, name)
        if not os.path.isfile(full):
            if role in ("pre", "post"):
                raise RuntimeError(f"获取结果缺少{role}影像：{full}")
            continue
        paths[role] = full
    for role in ("pre", "post"):
        if role not in paths:
            raise RuntimeError(f"获取结果缺少{role}影像，无法进入识别阶段。")
    artifacts = {
        role: {"name": os.path.basename(path), "bytes": os.path.getsize(path),
               "sha256": _sha256_file(path)}
        for role, path in paths.items()
    }
    return paths["pre"], paths["post"], entry, {"cache_dir": cache_dir, "artifacts": artifacts}


# --------------------------------------------------------------------------
# 识别（detect）
# --------------------------------------------------------------------------


def _run_detection(request: PipelineRequest, pre_path: str, post_path: str) -> Any:
    """灾前/灾后识别与变化统计。基线模型（不依赖权重），云掩膜按请求云量上限。"""
    from .infer import FloodDetector

    detector = FloodDetector(mode="baseline", mask_clouds=True,
                             max_cloud_pct=float(request.max_cloud_pct))
    return detector.compare(pre_path, post_path)


def _common_valid_stats(result: Any) -> Dict[str, Any]:
    """从 B 的 change/common_valid 字段读取共同有效观测。

    注意：``result.valid_mask`` 只是"灾后单景"的有效掩膜，不能当共同有效区用；
    优先取 ``change['masks']['common_valid']``（灾前∩灾后），仅在其缺失时兜底。
    """
    change = getattr(result, "change", None) or {}
    masks = change.get("masks") if isinstance(change.get("masks"), dict) else {}
    valid = masks.get("common_valid")
    if valid is None:
        valid = getattr(result, "valid_mask", None)
    pixels = change.get("common_valid_pixels")
    fraction = change.get("common_valid_fraction_pct")
    if valid is not None:
        arr = np.asarray(valid, dtype=bool)
        if pixels is None:
            pixels = int(np.count_nonzero(arr))
        if fraction is None:
            fraction = 100.0 * int(np.count_nonzero(arr)) / max(int(arr.size), 1)
    return {
        "pixels": int(pixels or 0),
        "fraction_pct": float(fraction) if fraction is not None else 0.0,
        "quality_status": change.get("quality_status") or result.stats.get("quality_status"),
    }


def _attach_terrain(result: Any, request: PipelineRequest, dem: Optional[str]) -> Dict[str, Any]:
    """附加用户声明的地形背景；有 DEM 时做坡度筛查（仅供复核，不改掩膜）。"""
    context = terrain_context(request.terrain_profile)
    warnings = list(result.meta.get("warnings") or [])
    warnings.append(f"地形背景为用户声明：{context['label']}；仅提示复核重点，不修改水体掩膜。")
    result.meta["terrain"] = context
    if dem is not None:
        risk, summary = assess_dem(dem, result.scene)
        result.meta["terrain_risk_mask"] = risk
        result.meta["terrain_risk_summary"] = summary
        context["dem_assessed"] = True
        context["dem_summary"] = summary
        warnings.append(
            f"已用本地 DEM 做坡度筛查（阈值 {summary.get('screening_threshold_deg')}°，"
            f"覆盖 {summary.get('coverage_pct')}%）；坡度标记仅供人工复核，不是淹没判据。"
        )
    else:
        warnings.append("未提供本地 DEM，本次未做坡度筛查（地形背景仅为用户声明）。")
    result.meta["warnings"] = warnings
    return context


def _attach_provenance(
    result: Any,
    *,
    synthetic: bool,
    local_pair: Optional[Tuple[str, str]],
    entry: Optional[Dict[str, Any]],
) -> None:
    """写入来源；本地输入明确标注"未核验观测日期"，绝不假装有真实成像时间。"""
    if synthetic:
        provenance: Dict[str, Any] = {
            "source": "合成演示数据，不可代表真实精度",
            "note": "本次为显式合成演示输入，请勿用于精度评估。",
        }
        if local_pair:
            provenance["files"] = [os.path.basename(p) for p in local_pair]
    elif local_pair is not None:
        provenance = {
            "source": "本地输入，未核验观测日期",
            "note": "用户显式提供的本地影像，系统未核验其真实观测日期与地理配准精度。",
            "pre_file": os.path.basename(local_pair[0]),
            "post_file": os.path.basename(local_pair[1]),
        }
    else:
        provenance = dict((entry or {}).get("provenance") or {})
        provenance.setdefault("source", "Sentinel-2 L2A 公开影像（自动获取）")
    result.meta["provenance"] = provenance


# --------------------------------------------------------------------------
# 导出（export）
# --------------------------------------------------------------------------


def _export(result: Any, run_dir: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """导出成果包，并额外保存 before/after 叠加图供手动界面的对比滑块使用。"""
    change = result.change or {}
    before_overlay = change.get("before_overlay")
    if before_overlay is None:
        before_result = change.get("before_result")
        before_overlay = getattr(before_result, "overlay", None)
    extra: Dict[str, str] = {}
    if before_overlay is not None:
        before_path = os.path.join(run_dir, "before_overlay.png")
        save_rgb(before_path, np.asarray(before_overlay))
        extra["before_overlay"] = before_path
    after_path = os.path.join(run_dir, "after_overlay.png")
    save_rgb(after_path, np.asarray(result.overlay))
    extra["after_overlay"] = after_path

    bundled = export_bundle(result, out_dir=run_dir, basename="bundle")
    files = dict(bundled.get("files") or {})
    files.update(extra)
    return bundled, files


# --------------------------------------------------------------------------
# 完成复用校验
# --------------------------------------------------------------------------


def _verify_completed(payload: Dict[str, Any], run_dir: str) -> Optional[Dict[str, Any]]:
    """校验已完成清单里的全部成果哈希；全部吻合才允许直接复用。"""
    if not payload.get("completed"):
        return None
    result = payload.get("result")
    artifacts = payload.get("artifacts")
    if not isinstance(result, dict) or not isinstance(artifacts, dict):
        return None
    files = result.get("files")
    if not isinstance(files, dict) or set(artifacts) != set(files) | {"zip"}:
        return None
    run_base = os.path.normcase(os.path.realpath(run_dir))
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            return None
        path = record.get("path")
        expected = result.get("zip") if name == "zip" else files.get(name)
        if not isinstance(path, str) or path != expected or not os.path.isfile(path):
            return None
        try:
            if os.path.commonpath([run_base, os.path.normcase(os.path.realpath(path))]) != run_base:
                return None
            if int(record.get("bytes", -1)) != os.path.getsize(path):
                return None
            if record.get("sha256") != _sha256_file(path):
                return None
        except (OSError, ValueError):
            return None
    if not result.get("zip") or not artifacts.get("zip"):
        return None
    return result


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def run_pipeline(
    request: PipelineRequest,
    out_root: str,
    run_id: Optional[str] = None,
    local_pair: Any = None,
    dem_path: Any = None,
    progress: Optional[Callable[..., Any]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    synthetic: bool = False,
) -> Dict[str, Any]:
    """执行一次完整处理，返回 JSON 安全的成果字典。

    返回键：run_id, request_key, zip, files, summary, stats, meta, change, elapsed_s。
    """
    if not isinstance(request, PipelineRequest):
        raise TypeError("request 必须是 PipelineRequest")
    out_root = os.path.abspath(os.fspath(out_root))
    run_id = _safe_run_id(run_id if run_id is not None else request.new_run_id())
    run_dir = run_dir_for(out_root, run_id)
    manifest_path = os.path.join(run_dir, "manifest.json")

    # 获取脚本用模块级全局保存超时/快速模式，整段处理必须串行，不能让批处理与单跑互踩。
    with PIPELINE_LOCK:
        started = time.perf_counter()
        os.makedirs(run_dir, exist_ok=True)
        manifest: Dict[str, Any] = {}
        try:
            local = _validate_local_pair(local_pair)
            dem = _validate_dem(dem_path)
            identity = _build_identity(request, local, dem, synthetic)

            existing = _load_manifest(manifest_path)
            if existing is not None:
                if existing.get("identity") != identity:
                    raise ValueError(
                        f"run_id {run_id} 已存在且对应不同请求/输入，拒绝覆盖。"
                        "请换一个新的 run_id，或删除该运行目录后重试。"
                    )
                resumed = _verify_completed(existing, run_dir)
                if resumed is not None:
                    _progress(progress, "检测到已完成的成果包，直接复用（不重复处理）", "resume")
                    return _jsonable(resumed)

            manifest = {
                "schema": PIPELINE_SCHEMA,
                "run_id": run_id,
                "created_at": (existing or {}).get("created_at") or _now(),
                "updated_at": _now(),
                "request": request.to_dict(),
                "request_key": request.cache_key,
                "identity": identity,
                "inputs": {
                    "local_pair": list(local) if local else None,
                    "dem_path": dem,
                },
                "stages": {},
                "completed": False,
                "outcome": None,
                "artifacts": {},
                "result": None,
            }
            _atomic_write_json(manifest_path, manifest)

            # ① acquire
            _check_cancel(is_cancelled)
            source = "network"
            entry: Optional[Dict[str, Any]] = None
            acquire_info: Dict[str, Any] = {}
            if local is not None:
                pre_path, post_path = local
                source = "local"
                _progress(progress, "使用本地输入的灾前/灾后影像（未核验观测日期）", "acquire")
            else:
                _progress(progress, "正在获取灾前/灾后卫星影像（命中请求缓存则直接复用）…", "acquire")
                pre_path, post_path, entry, acquire_info = _acquire(request, out_root)
            manifest["stages"]["acquire"] = {
                "status": "completed",
                "source": source,
                "pre": pre_path,
                "post": post_path,
                "entry": _jsonable({k: v for k, v in (entry or {}).items() if k != "provenance"}),
                "provenance": _jsonable((entry or {}).get("provenance")),
                **acquire_info,
            }
            manifest["updated_at"] = _now()
            _atomic_write_json(manifest_path, manifest)
            _check_cancel(is_cancelled)

            # ② detect
            _progress(progress, "正在识别灾前/灾后影像并计算变化…", "detect")
            result = _run_detection(request, pre_path, post_path)
            stats = _common_valid_stats(result)
            if stats["pixels"] <= 0 or stats["fraction_pct"] < float(request.min_valid_pct):
                raise DataQualityError(
                    "数据不足：灾前/灾后共同有效观测 "
                    f"{stats['fraction_pct']:.1f}%（{stats['pixels']} 像元），"
                    f"低于请求下限 {request.min_valid_pct:g}%。本次不给出淹没范围与面积结论，"
                    "请补充无云影像、放宽云量上限或调整时相窗口后重试。"
                )
            result.meta["common_valid_pixels"] = stats["pixels"]
            result.meta["common_valid_fraction_pct"] = round(stats["fraction_pct"], 2)
            context = _attach_terrain(result, request, dem)
            _attach_provenance(result, synthetic=synthetic, local_pair=local, entry=entry)
            if synthetic:
                result.meta.setdefault("warnings", []).append(
                    "合成演示数据，不可代表真实精度（请勿用于精度评估）。"
                )
            manifest["stages"]["detect"] = {
                "status": "completed",
                "common_valid_pixels": stats["pixels"],
                "common_valid_fraction_pct": round(stats["fraction_pct"], 2),
                "quality_status": stats["quality_status"],
                "terrain_profile": context.get("profile"),
                "dem_assessed": bool(context.get("dem_assessed")),
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
            manifest["updated_at"] = _now()
            _atomic_write_json(manifest_path, manifest)
            _check_cancel(is_cancelled)

            # ③ export
            _progress(progress, "正在导出成果包（PNG / GeoTIFF / 简报 / ZIP）…", "export")
            bundled, files = _export(result, run_dir)

            payload: Dict[str, Any] = {
                "run_id": run_id,
                "request_key": request.cache_key,
                "zip": bundled["zip"],
                "files": files,
                "summary": result.summary_text(),
                "stats": _jsonable(result.stats),
                "meta": _jsonable(result.meta),
                "change": _jsonable({
                    key: value for key, value in (result.change or {}).items()
                    if key not in ("before_result", "after_result", "masks") and not isinstance(value, np.ndarray)
                }),
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
            payload = _jsonable(payload)
            artifacts: Dict[str, Any] = {}
            for name, path in payload["files"].items():
                artifacts[name] = {"path": path, "bytes": os.path.getsize(path),
                                   "sha256": _sha256_file(path)}
            artifacts["zip"] = {"path": payload["zip"], "bytes": os.path.getsize(payload["zip"]),
                                "sha256": _sha256_file(payload["zip"])}
            manifest["stages"]["export"] = {
                "status": "completed",
                "zip": payload["zip"],
                "files": sorted(payload["files"]),
                "workdir": bundled.get("workdir"),
            }
            manifest["artifacts"] = artifacts
            manifest["result"] = payload
            manifest["completed"] = True
            manifest["outcome"] = {"status": "completed", "finished_at": _now()}
            manifest["updated_at"] = _now()
            _atomic_write_json(manifest_path, manifest)
            _progress(progress, f"处理完成：{payload['zip']}", "done")
            return payload

        except JobCancelled:
            manifest["completed"] = False
            manifest["outcome"] = {"status": "cancelled", "finished_at": _now()}
            manifest["updated_at"] = _now()
            if manifest.get("run_id"):
                _atomic_write_json(manifest_path, manifest)
            raise
        except Exception as exc:  # noqa: BLE001 - 记录失败结论后原样抛出，由上层决定重试
            manifest["completed"] = False
            manifest["outcome"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _now(),
            }
            manifest["updated_at"] = _now()
            if manifest.get("run_id"):
                _atomic_write_json(manifest_path, manifest)
            raise
