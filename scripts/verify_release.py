"""Verify the desktop UI with isolated synthetic fixtures, from EXE or source."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]


def write_raster(path, data, names=None):
    data = np.asarray(data)
    with rasterio.open(path, "w", driver="GTiff", count=len(data), width=data.shape[2],
                       height=data.shape[1], dtype=data.dtype, crs="EPSG:32650",
                       transform=from_origin(500000, 3200000, 10, 10)) as ds:
        ds.write(data)
        if names:
            ds.descriptions = names
    return str(path)


def table_rows(table):
    return table.get("data", []) if isinstance(table, dict) else table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default=str(ROOT / "dist/慧眼识灾/慧眼识灾.exe"))
    ap.add_argument("--source", action="store_true", help="使用当前 Python 启动源码作对照")
    ap.add_argument("--deep", action="store_true", help="Also verify explicit U-Net single/pair UI and bundled weights")
    ap.add_argument("--online", action="store_true", help="Fetch a real 512px Sentinel pair through the running UI")
    ap.add_argument("--port", type=int, default=7873)
    ap.add_argument("--timeout", type=float, default=90, help="启动及每个 API 任务的超时秒数")
    ap.add_argument("--report", default=str(ROOT / "outputs/release_v0.5.0/verification.json"))
    args = ap.parse_args(argv)
    if args.timeout <= 0:
        ap.error("--timeout 必须大于 0")
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    exe = Path(args.exe).resolve()
    command = [sys.executable, str(ROOT / "app/desktop.py")] if args.source else [str(exe)]
    report = {"mode": "source" if args.source else "frozen", "command": command,
              "synthetic_only": True, "checks": [], "success": False, "cleanup_errors": []}
    proc = client = temporary = launch_log = None
    folder = data_dir = None
    saved_env = {key: os.environ.get(key) for key in ("NO_PROXY", "no_proxy", "GRADIO_ANALYTICS_ENABLED")}

    def check(name, valid):
        report["checks"].append({"name": name, "passed": bool(valid)})
        print(f"{'PASS' if valid else 'FAIL'} {name}", flush=True)
        if not valid:
            raise AssertionError(name)

    def predict(*values, api_name):
        job = client.submit(*values, api_name=api_name)
        try:
            return job.result(timeout=args.timeout)
        except BaseException:
            try:
                job.cancel()
            except Exception as cancel_error:
                report.setdefault("cancellation_errors", []).append(f"{api_name}: {cancel_error}")
            raise

    try:
        if not args.source and not exe.is_file():
            raise FileNotFoundError(exe)
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError("验收端口已占用，拒绝连接其他实例")
        existing = os.environ.get("NO_PROXY", "") + "," + os.environ.get("no_proxy", "")
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = existing + ",127.0.0.1,localhost,::1"
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        temporary = tempfile.TemporaryDirectory(prefix="flood_release_")
        folder = Path(temporary.name)
        data_dir = folder / "app_data"
        data_dir.mkdir()
        report["isolated_data_dir"] = str(data_dir)
        launch_log = (folder / "launcher.log").open("wb")
        proc = subprocess.Popen(command + ["--headless", "--port", str(args.port), "--data-dir", str(data_dir)],
            cwd=ROOT if args.source else exe.parent, stdout=launch_log, stderr=launch_log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        from gradio_client import Client, handle_file
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                client = Client(f"http://127.0.0.1:{args.port}", verbose=False, analytics_enabled=False,
                                download_files=folder / "downloads",
                                httpx_kwargs={"timeout": min(3, args.timeout), "trust_env": False})
                break
            except Exception as startup_error:
                report["last_startup_error"] = f"{type(startup_error).__name__}: {startup_error}"
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        check("source_server_ready" if args.source else "frozen_server_ready", client is not None)
        # Startup requests use a short timeout; a running job may take longer to emit data.
        client.httpx_kwargs["timeout"] = args.timeout
        api = client.view_api(return_format="dict", print_info=False)["named_endpoints"]
        if args.online:
            report["synthetic_only"] = False
            out = predict(116.3, 29.15, "2020-05-19", "2020-05-19", "2020-07-15", "2020-07-15",
                          512, "unspecified", None, "baseline", None, None, "auto", "auto",
                          api_name="/run_full_pipeline")
            check("online_sentinel_bundle", bool(out[4]) and os.path.isfile(out[4]))
            with zipfile.ZipFile(out[4]) as online_zip:
                check("online_gis_pdf", {"water_mask.tif", "report.pdf", "stats.json"} <= set(online_zip.namelist()))
            report["online_summary"] = out[3]
            report["online_log"] = out[5]
        check("new_ui_endpoints", all(x in api for x in (
            "/run_full_pipeline", "/run_spectral", "/do_queue_demo", "/do_run", "/do_refresh")))
        names = ("blue", "green", "red", "nir", "swir1", "swir2")
        pre = np.stack([np.full((64, 64), v, np.float32) for v in (.08,.1,.12,.3,.25,.2)])
        post = pre.copy()
        for i, value in enumerate((.03,.08,.02,.02,.005,.005)):
            post[i,16:48,16:48] = value
        a = write_raster(folder / "pre.tif", pre, names)
        b = write_raster(folder / "post.tif", post, names)
        if args.deep:
            check("deep_ui_endpoints", all(k in api for k in ("/run_single", "/run_compare", "/show_model_info")))
            parameters = api["/run_single"]["parameters"]
            report["deep_api_parameters"] = parameters
            weight = next(p["parameter_default"] for p in parameters if p["parameter_name"] == "weights")
            out = predict(handle_file(b), "U-Net 深度模型", 0.5, weight, api_name="/run_single")
            check("deep_single_bundle", bool(out[4]) and os.path.isfile(out[4]))
            check("deep_single_used_unet", "U-Net (tiny_unet)" in out[5])
            out = predict(handle_file(a), handle_file(b), "U-Net 深度模型", 0.5, weight, api_name="/run_compare")
            check("deep_pair_bundle", bool(out[4]) and os.path.isfile(out[4]))
            check("deep_pair_used_unet", "U-Net (tiny_unet)" in out[5])
        dem = write_raster(folder / "dem.tif", np.tile(np.arange(64, dtype=np.float32)*10, (1,64,1)))
        for profile in ("plain", "hilly", "mountain", "urban", "coastal", "wetland", "arid"):
            out = predict(116.3, 29.15, "2020-01-01", "2020-01-03", "2020-02-01", "2020-02-03",
                512, profile, handle_file(dem) if profile == "mountain" else None, "adaptive",
                handle_file(a), handle_file(b), "s2_6band", "auto", api_name="/run_full_pipeline")
            check(profile + "_bundle", bool(out[4]) and os.path.isfile(out[4]))
            with zipfile.ZipFile(out[4]) as zf:
                required = {"water_mask.tif", "change.tif", "review.tif", "before_review.tif", "report.pdf", "stats.json"}
                check(profile + "_gis_pdf", required <= set(zf.namelist()))
                payload = json.loads(zf.read("stats.json"))
                check(profile + "_recipe", payload["meta"]["adaptive"]["terrain_profile"] == profile)
                check(profile + "_pdf", zf.read("report.pdf").startswith(b"%PDF"))
                if profile == "mountain":
                    for name, value in (("review.tif",2),("change.tif",255)):
                        with rasterio.MemoryFile(zf.read(name)) as mem, mem.open() as ds:
                            check(name + "_unknown_semantics", int(ds.read(1)[30,30]) == value)
        for index in ("ndvi", "savi", "ndwi", "mndwi", "ndmi", "nbr"):
            out = predict([handle_file(a),handle_file(b)], index, "2020-01-01 2020-02-01",
                          "s2_6band", 50, api_name="/run_spectral")
            with zipfile.ZipFile(out[1]) as zf:
                payload = json.loads(zf.read("summary.json"))
                check(index + "_monitor", payload["index"] == index and payload["changes"][0]["status"] == "ok")
                check(index + "_direction", payload["change_direction"] == "later_minus_earlier")
        initial = predict("", api_name="/do_refresh")
        initial_ids = {row[0] for row in table_rows(initial[0])}
        check("queue_isolated_empty", not initial_ids)
        queued = predict(api_name="/do_queue_demo")
        check("queue_import", "已加入" in queued[1])
        rows = table_rows(queued[0])
        new_ids = {row[0] for row in rows} - initial_ids
        check("queue_one_new_job", len(rows) == 1 and len(new_ids) == 1)
        job_id = next(iter(new_ids))
        report["queue_job_id"] = job_id
        predict(job_id, api_name="/do_run")
        result = predict(job_id, api_name="/do_refresh")
        db = data_dir / "outputs/automation/jobs.sqlite"
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=3)) as connection:
            state = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        report["queue_state"] = state[0] if state else None
        check("queue_succeeded", report["queue_state"] == "succeeded")
        check("queue_export", bool(result[3]) and os.path.isfile(result[3]))
        pred = write_raster(folder / "pred.tif", np.array([[[1,255,0]]], np.uint8))
        ref = write_raster(folder / "ref.tif", np.array([[[1,1,0]]], np.uint8))
        out_json = folder / "eval.json"
        call = subprocess.run(command + ["--evaluate", pred, ref, "--evaluation-out", str(out_json),
                                         "--data-dir", str(data_dir)],
            cwd=folder, timeout=args.timeout, stdout=launch_log, stderr=launch_log,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        check("source_evaluation" if args.source else "exe_evaluation", call.returncode == 0 and out_json.is_file())
        check("evaluation_coverage", json.loads(out_json.read_text(encoding="utf-8"))["recall_over_all_labelled_water"] == .5)
        report["success"] = True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                report["process_returncode"] = proc.returncode
            except Exception as exc:
                report["cleanup_errors"].append(f"process: {exc}")
        if client is not None:
            try:
                client.close()
                for name in ("executor", "helper_executor", "stream_executor"):
                    executor = getattr(client, name, None)
                    if executor is not None:
                        executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                report["cleanup_errors"].append(f"client: {exc}")
        if launch_log is not None:
            try:
                launch_log.close()
            except Exception as exc:
                report["cleanup_errors"].append(f"launcher_log: {exc}")
        if folder is not None:
            for label, path in (("desktop_log_tail", data_dir / "logs/desktop.log"),
                                ("launcher_log_tail", folder / "launcher.log")):
                if path.is_file():
                    try:
                        report[label] = path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                    except OSError as exc:
                        report["cleanup_errors"].append(f"log: {exc}")
        if temporary is not None:
            try:
                temporary.cleanup()
                report["temporary_directory_removed"] = not folder.exists()
            except Exception as exc:
                report["cleanup_errors"].append(f"temporary_directory: {exc}")
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if report["cleanup_errors"]:
            report["success"] = False
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{sum(c['passed'] for c in report['checks'])}/{len(report['checks'])} checks passed; report: {report_path}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
