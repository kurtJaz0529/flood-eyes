"""Exercise installed/frozen calculations without starting the local web server."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import rasterio

from verify_release import write_raster


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    exe = Path(args.exe).resolve()
    report = {"exe": str(exe), "synthetic_only": True, "success": False, "checks": []}
    def check(name, valid):
        report["checks"].append({"name": name, "passed": bool(valid)})
        print(f"{'PASS' if valid else 'FAIL'} {name}", flush=True)
        if not valid:
            raise AssertionError(name)
    try:
        with tempfile.TemporaryDirectory(prefix="flood_frozen_cli_") as temp:
            folder = Path(temp)
            def run(*command, expected=0):
                result = subprocess.run([str(exe), "--data-dir", str(folder), *map(str, command)],
                    cwd=folder, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode != expected:
                    log = folder / "logs/desktop.log"
                    if log.exists():
                        print(log.read_text(encoding="utf-8",errors="replace")[-5000:])
                check("command_" + str(command[0]), result.returncode == expected)
            names = ("blue", "green", "red", "nir", "swir1", "swir2")
            pre = np.stack([np.full((64,64), value, np.float32) for value in (.08,.1,.12,.3,.25,.2)])
            post = pre.copy()
            for i,value in enumerate((.03,.08,.02,.02,.005,.005)):
                post[i,16:48,16:48] = value
            a = write_raster(folder / "pre.tif", pre, names)
            b = write_raster(folder / "post.tif", post, names)
            dem = write_raster(folder / "dem.tif", np.tile(np.arange(64,dtype=np.float32)*10, (1,64,1)))
            profiles = ("plain","hilly","mountain","urban","coastal","wetland","arid")
            rows = [{"lon":116.3,"lat":29.15,"pre_start":"2020-01-01","pre_end":"2020-01-03",
                "post_start":"2020-02-01","post_end":"2020-02-03","size":64,
                "terrain_profile":profile,"detection_strategy":"adaptive","band_order":"s2_6band",
                "local_pre":a,"local_post":b, **({"dem_path":dem} if profile=="mountain" else {})}
                for profile in profiles]
            manifest = folder / "jobs.json"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            out = folder / "batch"
            run("--batch", manifest, "--run", "--out-dir", out)
            with closing(sqlite3.connect(out / "jobs.sqlite")) as db:
                jobs = db.execute("SELECT status,result_json,error FROM jobs ORDER BY rowid").fetchall()
            check("all_seven_jobs_succeeded", len(jobs)==7 and all(j[0]=="succeeded" for j in jobs))
            for profile,job in zip(profiles,jobs):
                payload = json.loads(job[1])
                with zipfile.ZipFile(payload["zip"]) as archive:
                    required = {"water_mask.tif","change.tif","review.tif","before_review.tif","report.pdf","stats.json"}
                    check(profile+"_exports", required <= set(archive.namelist()))
                    stats = json.loads(archive.read("stats.json"))
                    check(profile+"_recipe", stats["meta"]["adaptive"]["terrain_profile"]==profile)
                    check(profile+"_pdf", archive.read("report.pdf").startswith(b"%PDF"))
                    if profile=="mountain":
                        for name,value in (("review.tif",2),("change.tif",255)):
                            with rasterio.MemoryFile(archive.read(name)) as mem,mem.open() as ds:
                                check(name+"_unknown", ds.read(1)[30,30]==value and ds.crs.to_epsg()==32650)
            for index in ("ndvi","savi","ndwi","mndwi","ndmi","nbr"):
                spectral_out = folder / index
                run("--spectral","--index",index,"--images",a,b,"--dates","2020-01-01","2020-02-01",
                    "--band-order","s2_6band","--out-dir",spectral_out)
                summary = json.loads((spectral_out / "summary.json").read_text(encoding="utf-8"))
                check(index+"_monitor", summary["index"]==index and summary["changes"][0]["status"]=="ok")
                check(index+"_direction", summary["change_direction"]=="later_minus_earlier")
            pred = write_raster(folder / "pred.tif",np.array([[[1,255,0]]],np.uint8))
            ref = write_raster(folder / "ref.tif",np.array([[[1,1,0]]],np.uint8))
            evaluation = folder / "eval.json"
            run("--evaluate",pred,ref,"--evaluation-out",evaluation)
            check("unknown_penalizes_recall",json.loads(evaluation.read_text(encoding="utf-8"))["recall_over_all_labelled_water"]==.5)
            bad = rows[0].copy()
            bad["local_pre"] = str(folder / "missing.tif")
            manifest.write_text(json.dumps([bad]),encoding="utf-8")
            run("--batch",manifest,"--run","--out-dir",folder / "failure-case",expected=1)
            report["success"] = True
    except Exception as exc:
        report["success"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    raise SystemExit(main())
