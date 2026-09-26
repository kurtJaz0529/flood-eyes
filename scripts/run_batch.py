"""
慧眼识灾 · 批量任务命令行（任务 D）
===================================

不开界面，直接操作与界面同一套队列和流水线：

    python scripts/run_batch.py jobs.csv                 # 只导入并逐行校验（不执行）
    python scripts/run_batch.py jobs.csv --run           # 导入后执行队列中的等待任务
    python scripts/run_batch.py jobs.json --run          # 也支持 JSON 行数组
    python scripts/run_batch.py --list                   # 查看任务状态
    python scripts/run_batch.py --retry <任务编号>        # 重新排队失败/取消/中断的任务
    python scripts/run_batch.py --cancel <任务编号>       # 取消等待或请求取消正在执行的任务
    python scripts/run_batch.py --recover                # 显式恢复中断任务（不打断正在运行的任务）

输入列与界面一致：必需 `lon,lat,pre_start,pre_end,post_start,post_end`，可选
`size,terrain_profile,max_cloud_pct,min_valid_pct,local_pre,local_post,dem_path`。
场景适配另支持 `detection_strategy,water_index,index_threshold,slope_threshold_deg,band_order`。
只有加 `--run` 才会真正执行；只给输入文件时仅导入为等待任务。
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:  # Windows 控制台默认 GBK，中文提示会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from app.automation import (  # noqa: E402
    format_import_result, get_store, history_rows, import_batch, load_batch,
    make_runner, recover_interrupted_jobs,
)
from src.jobs import QueueRunner  # noqa: E402
from src.paths import outputs_dir  # noqa: E402

_EXAMPLE = (
    "示例：\n"
    "  python scripts/run_batch.py jobs.csv\n"
    "  python scripts/run_batch.py jobs.csv --run\n"
    "  python scripts/run_batch.py jobs.json --run --out-dir outputs\n"
    "  python scripts/run_batch.py --list\n"
    "  python scripts/run_batch.py --retry 1a2b3c4d\n"
    "  python scripts/run_batch.py --cancel 1a2b3c4d\n"
    "  python scripts/run_batch.py --recover\n"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="慧眼识灾批量任务：导入 CSV/JSON、查看队列、显式执行。",
        epilog=_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="待导入的 CSV 或 JSON 文件路径")
    parser.add_argument("--out-dir", default=outputs_dir(), help="任务库与成果包目录（默认 outputs）")
    parser.add_argument("--run", action="store_true", help="执行队列中的等待任务（不加则只导入）")
    parser.add_argument("--list", action="store_true", help="列出任务状态")
    parser.add_argument("--retry", metavar="ID", help="把失败/取消/中断的任务重新排队")
    parser.add_argument("--cancel", metavar="ID", help="取消等待中的任务，或请求取消正在执行的任务")
    parser.add_argument("--recover", action="store_true", help="显式恢复中断任务（不打断正在运行的任务）")
    return parser


def _print_history(store) -> None:
    rows = history_rows(store)
    if not rows:
        print("（暂无任务）")
        return
    print(f"{'任务编号':<34}{'状态':<8}{'阶段':<10}{'地形背景':<16}质量/错误")
    for job_id, status, stage, terrain, quality in rows:
        print(f"{job_id:<34}{status:<8}{stage:<10}{terrain:<16}{quality}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    store = get_store(out_dir)
    did_something = False
    had_errors = False

    if args.input:
        did_something = True
        try:
            summary = import_batch(store, load_batch(args.input))
        except Exception as exc:  # noqa: BLE001
            print(f"导入失败：{type(exc).__name__}: {exc}")
            return 1
        print(format_import_result(summary))
        had_errors = had_errors or summary["error_count"] > 0

    if args.retry:
        did_something = True
        try:
            job = store.retry(args.retry)
            print(f"已重新排队：{job['job_id']}（第 {job['attempt']} 次）")
        except KeyError:
            print(f"未找到任务编号：{args.retry}")
            had_errors = True
        except ValueError as exc:
            print(str(exc))
            had_errors = True

    if args.cancel:
        did_something = True
        job = store.request_cancel(args.cancel)
        if job is None:
            print(f"未找到任务编号：{args.cancel}")
            had_errors = True
        elif job["status"] == "cancelled":
            print(f"已取消等待中的任务：{args.cancel}")
        else:
            print(f"已请求取消：{args.cancel}（执行中的任务会在安全边界停止）")

    if args.recover:
        did_something = True
        print(recover_interrupted_jobs(out_dir)["message"])

    if args.list or args.run or not did_something:
        _print_history(store)
        did_something = did_something or args.list or args.run

    if args.run:
        print("开始处理队列…")
        runner = QueueRunner(store, make_runner(out_dir))
        for event in runner.run_pending():
            job = event.get("job") or {}
            print(f"[{event['event']}] {job.get('job_id', '')} {job.get('status', '')}"
                  + (f" 错误：{job.get('error')}" if job.get("error") else ""))
        print("队列处理完毕。")
        _print_history(store)

    if not did_something:
        build_parser().print_help()
        return 0
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
