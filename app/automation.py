"""
慧眼识灾 · 批量任务界面（任务 D）
=================================

在现有单页界面里加一个可折叠的「批量处理」区：上传/粘贴 CSV（或 JSON）→
逐行校验 → 加入本地任务队列 → 串行处理 → 表格里看进度/质量 → 下载成果包。

分工：

* ``src/jobs.py`` 负责 SQLite 持久化与串行执行；
* ``src/pipeline.py`` 负责单次处理；
* 本模块只做「表格 → 请求 → 队列」的翻译和界面回调，并暴露
  ``load_batch / import_batch / make_runner / get_store`` 给脚本与测试复用。

界面文案一律用业务语言（获取影像 / 识别 / 导出 / 地形背景），不出现内部实现术语。
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # Gradio 只在真正建界面时需要，脚本复用本模块的 helper 时可不装
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None  # type: ignore[assignment]

from src.jobs import RUNNING, JobStore, QueueRunner
from src.paths import samples_dir
from src.pipeline import run_pipeline
from src.task_contracts import PipelineRequest

__all__ = [
    "ALLOWED_COLUMNS", "REQUIRED_COLUMNS", "DEFAULT_CSV",
    "load_batch", "import_batch", "make_runner", "get_store",
    "recover_interrupted_jobs", "build_automation_ui", "build_automation_blocks",
]

#: 允许的输入列：请求字段 + 可选的本地影像 / DEM 路径。
ALLOWED_COLUMNS = tuple(PipelineRequest.__dataclass_fields__) + ("local_pre", "local_post", "dem_path")
REQUIRED_COLUMNS = ("lon", "lat", "pre_start", "pre_end", "post_start", "post_end")
_NUMERIC = ("lon", "lat", "max_cloud_pct", "min_valid_pct", "index_threshold", "slope_threshold_deg")

DEFAULT_CSV = (
    "lon,lat,pre_start,pre_end,post_start,post_end,size,terrain_profile,"
    "max_cloud_pct,min_valid_pct\n"
    "116.30,29.15,2020-05-08,2020-06-05,2020-07-10,2020-07-30,512,plain,35,50\n"
)

_STATUS_CN = {
    "pending": "等待中", "running": "处理中", "succeeded": "已完成",
    "failed": "失败", "cancelled": "已取消", "interrupted": "已中断",
}
_STAGE_CN = {
    "acquire": "获取影像", "detect": "识别", "export": "导出",
    "resume": "复用成果", "done": "完成", "terrain": "地形核查",
}
_HISTORY_HEADERS = ["任务编号", "状态", "阶段", "地形背景", "质量 / 错误"]

_STORES: Dict[str, JobStore] = {}
_STORES_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# 任务存储 / 执行器
# --------------------------------------------------------------------------


def get_store(out_root: str) -> JobStore:
    """按输出目录缓存 JobStore（同一 jobs.sqlite，界面与脚本共享）。"""
    key = os.path.abspath(os.fspath(out_root))
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = JobStore(os.path.join(key, "jobs.sqlite"))
            _STORES[key] = store
        return store


def _payload_pair(payload: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    pre, post = payload.get("local_pre"), payload.get("local_post")
    if pre and post:
        return str(pre), str(post)
    if pre or post:
        raise ValueError("本地影像必须同时提供灾前与灾后两张（local_pre / local_post）")
    return None


def make_runner(out_root: str) -> Callable[..., Any]:
    """构造队列执行函数；``run_id`` 固定为 job_id，重试时可校验复用已完成成果。"""
    root = os.path.abspath(os.fspath(out_root))

    def runner(payload: Any, job_id: str, progress: Callable[..., Any],
               is_cancelled: Callable[[], bool]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("任务内容必须是键值表（JSON 对象）")
        request_payload = payload.get("request") if isinstance(payload.get("request"), dict) else payload
        request = PipelineRequest.from_dict(request_payload)

        def report(message: str, stage: Optional[str] = None) -> None:
            try:
                progress(message, stage)
            except TypeError:
                progress(message)

        return run_pipeline(
            request, root, run_id=str(job_id),
            local_pair=_payload_pair(payload), dem_path=payload.get("dem_path") or None,
            progress=report, is_cancelled=is_cancelled,
            synthetic=bool(payload.get("synthetic")),
        )

    return runner


# --------------------------------------------------------------------------
# 批量输入解析与校验
# --------------------------------------------------------------------------


def _clean_key(key: Any) -> str:
    return str(key).strip().lstrip("\ufeff")


def load_batch(source: Any) -> List[Dict[str, Any]]:
    """把 CSV/JSON 文件路径或原始文本解析成行列表。

    支持：CSV 文本 / CSV 文件；JSON 数组、``{"rows": [...]}`` 或单个请求对象。
    """
    text: str
    hint = ""
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str) and "\n" not in source and os.path.isfile(source):
        with open(source, encoding="utf-8-sig") as fh:
            text = fh.read()
        hint = os.path.splitext(source)[1].lower()
    elif isinstance(source, str):
        text = source
    else:
        raise ValueError("批量输入需要 CSV/JSON 文件路径或文本内容")

    stripped = text.strip()
    if not stripped:
        raise ValueError("批量输入为空")

    if hint == ".json" or stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except ValueError as exc:
            raise ValueError(f"JSON 解析失败：{exc}") from exc
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if isinstance(payload, dict) and rows is None:
            rows = [payload]  # 单个请求对象
        if not isinstance(rows, list):
            raise ValueError("JSON 需要是行数组，或带 rows 数组的对象")
        out: List[Dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError("JSON 每一行都必须是键值对象")
            out.append({_clean_key(k): v for k, v in item.items()})
        return out

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV 缺少列名（首行）")
    out: List[Dict[str, Any]] = []
    for row in reader:
        if not row or not any(str(v).strip() for v in row.values() if v is not None):
            continue
        extras = [str(v) for v in (row.get(None) or []) if v is not None and str(v).strip()]
        item = {_clean_key(key): value for key, value in row.items() if key is not None}
        if extras:  # 数据列多于表头：显式标记，校验时按行报错，不静默丢弃
            item["__extra__"] = ";".join(extras)
        out.append(item)
    return out


def _coerce_scalar(field: str, value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"{field} 不能是布尔值")
    if field in _NUMERIC:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"{field} 需要数字，收到 {value!r}") from exc
    if field == "size":
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        text = str(value).strip()
        if not text.lstrip("-").isdigit():
            raise ValueError(f"size 需要整数，收到 {value!r}")
        return int(text)
    return value


def _validate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """校验一行，返回可直接入队的 payload；非法则抛 ``ValueError``。"""
    if "__extra__" in row:
        raise ValueError(f"该行列数多于表头（多出：{row['__extra__']}）")
    unknown = sorted(set(row) - set(ALLOWED_COLUMNS))
    if unknown:
        raise ValueError(f"未知列：{unknown}；允许的列：{list(ALLOWED_COLUMNS)}")
    clean = {k: v for k, v in row.items() if v is not None and str(v).strip() != ""}
    missing = [name for name in REQUIRED_COLUMNS if name not in clean]
    if missing:
        raise ValueError(f"缺少必填列：{missing}")
    request_payload = {
        field: _coerce_scalar(field, clean[field])
        for field in PipelineRequest.__dataclass_fields__
        if field in clean
    }
    request = PipelineRequest.from_dict(request_payload)

    payload: Dict[str, Any] = {"request": request.to_dict()}
    pre, post = clean.get("local_pre"), clean.get("local_post")
    if bool(pre) != bool(post):
        raise ValueError("本地影像必须同时提供灾前与灾后两张（local_pre / local_post）")
    if pre and post:
        for label, path in (("灾前", pre), ("灾后", post)):
            if not os.path.isfile(str(path)):
                raise FileNotFoundError(f"{label}本地影像不存在：{path}")
        payload["local_pre"], payload["local_post"] = str(pre), str(post)
    dem = clean.get("dem_path")
    if dem:
        if not os.path.isfile(str(dem)):
            raise FileNotFoundError(f"DEM 文件不存在：{dem}")
        payload["dem_path"] = str(dem)
    return payload


def import_batch(store: JobStore, rows: Any) -> Dict[str, Any]:
    """逐行独立校验并入队；坏行只记错误，不影响好行。"""
    if isinstance(rows, (str, os.PathLike)):
        rows = load_batch(rows)
    if not isinstance(rows, (list, tuple)):
        raise ValueError("import_batch 需要行列表或 CSV/JSON 文本")

    queued: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append({"row": index, "error": "该行不是键值对象"})
            continue
        try:
            payload = _validate_row(row)
            queued.append({"row": index, "job_id": store.enqueue(payload)})
        except Exception as exc:  # noqa: BLE001 - 单行失败不影响其它行
            errors.append({"row": index, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "total": len(rows), "queued": queued, "errors": errors,
        "queued_count": len(queued), "error_count": len(errors),
    }


def format_import_result(summary: Dict[str, Any]) -> str:
    lines = [f"已加入队列 {summary['queued_count']} 个任务；{summary['error_count']} 行未通过校验。"]
    if summary["queued"]:
        ids = "、".join(item["job_id"][:8] for item in summary["queued"][:6])
        lines.append(f"任务编号（前 8 位）：{ids}")
    for item in summary["errors"]:
        lines.append(f"第 {item['row']} 行：{item['error']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 队列状态展示
# --------------------------------------------------------------------------


def _request_of(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("request"), dict):
            return payload["request"]
        if "lon" in payload:
            return payload
    return {}


def _terrain_label(profile: Any) -> str:
    try:
        from src.terrain import terrain_context

        return str(terrain_context(str(profile or "unspecified"))["label"])
    except Exception:  # noqa: BLE001
        return str(profile or "未声明地形背景")


def _quality_text(job: Dict[str, Any]) -> str:
    if job.get("error"):
        return str(job["error"])[:160]
    result = job.get("result") or {}
    change = result.get("change") or {}
    stats = result.get("stats") or {}
    status = change.get("quality_status") or stats.get("quality_status")
    if status is None:
        return ""
    fraction = change.get("common_valid_fraction_pct")
    if fraction is None:
        return str(status)
    return f"{status}；共同有效观测 {fraction}%"


def history_rows(store: JobStore) -> List[List[str]]:
    rows: List[List[str]] = []
    for job in store.list_jobs():
        request = _request_of(job.get("payload"))
        stage = job.get("stage") or ""
        rows.append([
            job["job_id"],
            _STATUS_CN.get(job["status"], job["status"]),
            _STAGE_CN.get(stage, stage),
            _terrain_label(request.get("terrain_profile")),
            _quality_text(job),
        ])
    return rows


def _selected_view(store: JobStore, job_id: Optional[str]) -> Tuple[str, Optional[str]]:
    text = str(job_id or "").strip()
    if not text:
        return "在表格中复制任务编号填入上方文本框，即可查看结果或取消/重试。", None
    job = store.get(text)
    if job is None:
        return f"未找到任务编号 {text}。", None
    request = _request_of(job.get("payload"))
    lines = [
        f"任务编号：{job['job_id']}",
        f"状态：{_STATUS_CN.get(job['status'], job['status'])}　阶段：{_STAGE_CN.get(job.get('stage') or '', job.get('stage') or '-')}",
        f"地形背景：{_terrain_label(request.get('terrain_profile'))}（用户声明，仅提示复核重点）",
        f"尝试次数：{job.get('attempt')}",
    ]
    if job.get("error"):
        lines.append(f"失败原因：{job['error']}")
    result = job.get("result") or {}
    if result.get("summary"):
        lines.append("识别结论：")
        lines.append(str(result["summary"]))
        quality = _quality_text(job)
        if quality:
            lines.append(f"数据质量：{quality}")
        lines.append(f"成果包：{result.get('zip')}")
    logs = job.get("logs") or []
    if logs:
        lines.append("最近进度：")
        for entry in logs[-8:]:
            stage = _STAGE_CN.get(entry.get("stage") or "", entry.get("stage") or "")
            lines.append(f"· [{stage}] {entry.get('message')}")
    zip_path = result.get("zip") if isinstance(result, dict) else None
    if not (isinstance(zip_path, str) and os.path.isfile(zip_path)):
        zip_path = None
    return "\n".join(lines), zip_path


# --------------------------------------------------------------------------
# 队列操作（供界面/脚本复用）
# --------------------------------------------------------------------------


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def recover_interrupted_jobs(out_root: str, stale_after_s: float = 900.0) -> Dict[str, Any]:
    """显式恢复中断任务；持有执行权的进程即使长时间无进度也不会被打断。"""
    store = get_store(out_root)
    with store.worker_lock() as acquired:
        if not acquired:
            running = [job["job_id"] for job in store.list_jobs() if job.get("status") == RUNNING]
            return {"recovered": 0, "running": running,
                    "message": "检测到队列正在执行，已跳过恢复以免打断正在运行的任务。"}
        now = datetime.now()
        live: List[str] = []
        for job in store.list_jobs():
            if job.get("status") != RUNNING:
                continue
            stamp = _parse_ts(job.get("updated_at") or job.get("started_at"))
            age = (now - stamp).total_seconds() if stamp else 0.0
            if age < float(stale_after_s):
                live.append(job["job_id"])
        if live:
            return {"recovered": 0, "running": live,
                    "message": f"检测到 {len(live)} 个近期任务，已跳过恢复；若原进程已退出，请稍后重试。"}
        count = store.recover_interrupted()
        return {"recovered": count, "running": [],
                "message": f"已把 {count} 个中断任务标记为可重试。"}


# --------------------------------------------------------------------------
# Gradio 界面
# --------------------------------------------------------------------------

_TERRAIN_HELP = """
**地形背景（可选，用户声明）**：原基线仅提示复核；`detection_strategy=adaptive` 启用实验场景适配。
`unspecified` 未声明｜`plain` 平原河湖｜`hilly` 丘陵｜`mountain` 山地｜`urban` 城市｜`coastal` 海岸｜`wetland` 湿地稻田｜`arid` 干旱裸地。

**本地 DEM（可选）**：提供米制投影的 GeoTIFF 后，系统会把坡度 ≥15° 的区域标为“需复核”，
并随成果包输出地形风险栅格；坡度只是筛查标记，不是淹没判据。不提供 DEM 时不做坡度筛查。
"""


def _as_path(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return getattr(value, "name", None)


def build_automation_ui(out_root: str) -> Any:
    """在调用方的 Blocks 上下文里插入「批量处理」折叠区，返回该 Accordion。

    只登记回调，不读文件、不建数据库；所有文件读写都发生在按钮点击时。
    """
    if gr is None:  # pragma: no cover
        raise RuntimeError("当前环境未安装 Gradio，无法构建批量处理界面")

    with gr.Accordion("批量处理（上传 CSV / JSON，排队执行）", open=False) as accordion:
        gr.Markdown(_TERRAIN_HELP)
        gr.Markdown(
            "输入列：`lon, lat, pre_start, pre_end, post_start, post_end`"
            "（必需），可选 `size, terrain_profile, max_cloud_pct, min_valid_pct, "
            "local_pre, local_post, dem_path, detection_strategy, water_index, "
            "index_threshold, slope_threshold_deg, band_order`。日期用 `YYYY-MM-DD`；"
            "填了本地影像就不会联网下载，且成果会标注“本地输入，未核验观测日期”。"
        )
        with gr.Row():
            batch_file = gr.File(
                label="上传 CSV / JSON 文件", file_types=[".csv", ".json"],
                type="filepath", scale=1,
            )
            dem_file = gr.File(
                label="本地 DEM（可选，GeoTIFF）", file_types=[".tif", ".tiff"],
                type="filepath", scale=1,
            )
        batch_text = gr.Textbox(
            label="或直接编辑表格内容（逗号分隔，首行为列名）",
            value=DEFAULT_CSV, lines=5,
        )
        with gr.Row():
            import_btn = gr.Button("导入并校验", variant="primary")
            demo_btn = gr.Button("加入最小演示任务（合成样本）")
        import_status = gr.Textbox(label="导入结果 / 校验错误", lines=4, interactive=False)

        gr.Markdown("#### 任务进度")
        history = gr.Dataframe(
            headers=_HISTORY_HEADERS, value=[], interactive=False,
            label="任务列表", wrap=True,
        )
        with gr.Row():
            run_btn = gr.Button("开始处理队列", variant="primary")
            refresh_btn = gr.Button("刷新")
            recover_btn = gr.Button("恢复中断任务")
        with gr.Row():
            job_id = gr.Textbox(label="选中任务编号（从表格复制）", scale=3)
            cancel_btn = gr.Button("取消选中任务", scale=1)
            retry_btn = gr.Button("重试选中任务", scale=1)
        queue_status = gr.Textbox(label="队列状态", lines=2, interactive=False)
        selected_summary = gr.Textbox(label="选中任务结果", lines=8, interactive=False)
        result_zip = gr.File(label="成果包下载", interactive=False)

        def do_import(upload: Any, dem: Any, text: str) -> Tuple[List[List[str]], str]:
            store = get_store(out_root)
            try:
                upload_path = _as_path(upload)
                if upload_path and not os.path.isfile(upload_path):
                    return history_rows(store), f"上传文件不可读：{upload_path}"
                source: Any = upload_path or text
                rows = load_batch(source)
                dem_path = _as_path(dem)
                if dem_path:
                    for row in rows:
                        if isinstance(row, dict) and not row.get("dem_path"):
                            row["dem_path"] = dem_path
                summary = import_batch(store, rows)
            except Exception as exc:  # noqa: BLE001
                return history_rows(store), f"导入失败：{type(exc).__name__}: {exc}"
            return history_rows(store), format_import_result(summary)

        def do_queue_demo() -> Tuple[List[List[str]], str]:
            store = get_store(out_root)
            pre = os.path.join(samples_dir(), "demo01_pre.tif")
            post = os.path.join(samples_dir(), "demo01_post.tif")
            if not (os.path.isfile(pre) and os.path.isfile(post)):
                return history_rows(store), f"示例样本缺失：{pre} / {post}"
            request = PipelineRequest(
                lon=116.30, lat=29.15,
                pre_start="2020-05-08", pre_end="2020-06-05",
                post_start="2020-07-10", post_end="2020-07-30",
                size=512, terrain_profile="plain",
            )
            store.enqueue({
                "request": request.to_dict(), "local_pre": pre, "local_post": post,
                "synthetic": True, "label": "合成演示（不可代表真实精度）",
            })
            return history_rows(store), "已加入 1 个合成演示任务（不可代表真实精度）。"

        def do_refresh(selected: str) -> Tuple[List[List[str]], str, str, Optional[str]]:
            store = get_store(out_root)
            summary, zip_path = _selected_view(store, selected)
            return history_rows(store), "已刷新任务状态。", summary, zip_path

        def do_run(selected: str):
            store = get_store(out_root)
            runner = QueueRunner(store, make_runner(out_root))
            yield history_rows(store), "队列开始处理…", *_selected_view(store, selected)
            for event in runner.run_pending():
                job = event.get("job") or {}
                text = f"{_STATUS_CN.get(job.get('status'), job.get('status'))}：{job.get('job_id', '')[:8]}（{event.get('event')}）"
                yield history_rows(store), text, *_selected_view(store, selected)
            yield history_rows(store), "队列已处理完毕。", *_selected_view(store, selected)

        def do_cancel(selected: str) -> Tuple[List[List[str]], str, str, Optional[str]]:
            store = get_store(out_root)
            text = str(selected or "").strip()
            if not text:
                return history_rows(store), "请先填写任务编号。", *_selected_view(store, None)
            if store.request_cancel(text) is None:
                return history_rows(store), f"未找到任务编号 {text}。", *_selected_view(store, text)
            return history_rows(store), "已请求取消；正在执行的任务会在安全边界停止。", *_selected_view(store, text)

        def do_retry(selected: str) -> Tuple[List[List[str]], str, str, Optional[str]]:
            store = get_store(out_root)
            text = str(selected or "").strip()
            try:
                store.retry(text)
            except KeyError:
                return history_rows(store), f"未找到任务编号 {text}。", *_selected_view(store, text)
            except ValueError as exc:
                return history_rows(store), str(exc), *_selected_view(store, text)
            return history_rows(store), "已重新排队，可点击“开始处理队列”。", *_selected_view(store, text)

        def do_recover() -> Tuple[List[List[str]], str, str, Optional[str]]:
            store = get_store(out_root)
            outcome = recover_interrupted_jobs(out_root)
            return history_rows(store), outcome["message"], *_selected_view(store, None)

        import_btn.click(do_import, inputs=[batch_file, dem_file, batch_text],
                         outputs=[history, import_status])
        demo_btn.click(do_queue_demo, outputs=[history, import_status])
        run_btn.click(do_run, inputs=[job_id],
                      outputs=[history, queue_status, selected_summary, result_zip])
        refresh_btn.click(do_refresh, inputs=[job_id],
                          outputs=[history, queue_status, selected_summary, result_zip],
                          queue=False)
        cancel_btn.click(do_cancel, inputs=[job_id],
                         outputs=[history, queue_status, selected_summary, result_zip],
                         queue=False)
        retry_btn.click(do_retry, inputs=[job_id],
                        outputs=[history, queue_status, selected_summary, result_zip])
        recover_btn.click(do_recover,
                          outputs=[history, queue_status, selected_summary, result_zip])
    return accordion


def build_automation_blocks(out_root: str) -> Any:
    """独立可运行的批量界面（便于本地演示/自测）；主界面请用 ``build_automation_ui``。"""
    if gr is None:  # pragma: no cover
        raise RuntimeError("当前环境未安装 Gradio，无法构建批量处理界面")
    with gr.Blocks(title="慧眼识灾 · 批量处理") as demo:
        gr.Markdown("### 慧眼识灾 · 批量处理")
        build_automation_ui(out_root)
    return demo
