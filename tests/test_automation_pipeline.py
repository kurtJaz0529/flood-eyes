"""
慧眼识灾 · 任务 D 回归测试：流水线 + 批量队列 + 界面辅助函数
============================================================

运行：
    python -m pytest tests/test_automation_pipeline.py -q

覆盖任务卡 D 的验收点：

    · 本地小场景全流程：识别 → 地形元数据 → 成果包（含 before/after 叠加图）；
    · 同请求新 run_id 输出互不覆盖；同 run_id 仅在身份 + 文件哈希全吻合时复用；
      DEM / 输入变化会让旧完成记录失效；
    · 共同有效观测不足时抛 DataQualityError（明确“数据不足”，不给零淹没结论）；
    · 获取过程用假 fetch_event 模块，校验请求缓存参数与溯源；
    · CSV 两行有效 + 一行非法：有效行入队、非法行只报自己的错；
    · 混合队列失败隔离：失败任务不影响前后任务。

仅使用临时目录、mock 与本地小 GeoTIFF，不联网、不装依赖。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

import app.automation as automation  # noqa: E402
import src.pipeline as pipeline  # noqa: E402
from src.jobs import QueueRunner  # noqa: E402
from src.pipeline import DataQualityError, run_pipeline  # noqa: E402
from src.task_contracts import PipelineRequest  # noqa: E402

BANDS = ("blue", "green", "red", "nir")
_LAND = {"blue": 400, "green": 600, "red": 800, "nir": 1500}
_WATER = {"blue": 350, "green": 1400, "red": 400, "nir": 150}

BASE_REQUEST = {
    "lon": 112.66, "lat": 29.36,
    "pre_start": "2024-06-01", "pre_end": "2024-06-30",
    "post_start": "2024-07-06", "post_end": "2024-07-25",
    "size": 64, "terrain_profile": "plain",
    "max_cloud_pct": 35.0, "min_valid_pct": 50.0,
}


def write_scene(path: str, water_frac: float = 0.05, nodata_frac: float = 0.0) -> str:
    """写一小景 64×64、四波段、带 EPSG:32650 地理参考的 GeoTIFF。"""
    size = 64
    water = np.zeros(size * size, dtype=bool)
    water[: int(round(size * size * water_frac))] = True
    water = water.reshape(size, size)
    nodata = np.zeros(size * size, dtype=bool)
    if nodata_frac > 0:
        count = int(round(size * size * nodata_frac))
        nodata[size * size - count:] = True
    nodata = nodata.reshape(size, size)

    data = np.zeros((4, size, size), dtype=np.uint16)
    for index, name in enumerate(BANDS):
        band = np.full((size, size), _LAND[name], dtype=np.uint16)
        band[water] = _WATER[name]
        band[nodata] = 0
        data[index] = band
    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 4, "dtype": "uint16",
        "crs": "EPSG:32650", "nodata": 0, "transform": from_origin(500000, 3000000, 10, 10),
        "compress": "deflate",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data)
        for index, name in enumerate(BANDS):
            ds.set_band_description(index + 1, name)
    return path


def write_dem(path: str, slope_per_px: float = 5.0) -> str:
    """写一小块米制投影 DEM；坡度默认约 26.6°（用于触发“需复核”）。"""
    size = 64
    row = np.arange(size, dtype=np.float32)[None, :]
    data = (100.0 + slope_per_px * row).astype(np.float32)
    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 1, "dtype": "float32",
        "crs": "EPSG:32650", "nodata": -9999.0,
        "transform": from_origin(500000, 3000000, 10, 10), "compress": "deflate",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)
    return path


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="huiyan_pipeline_")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.out, exist_ok=True)
        self.pre = write_scene(os.path.join(self.tmp, "pre.tif"), 0.05)
        self.post = write_scene(os.path.join(self.tmp, "post.tif"), 0.15)
        self.pre_alt = write_scene(os.path.join(self.tmp, "pre_alt.tif"), 0.30)
        self.low_pre = write_scene(os.path.join(self.tmp, "low_pre.tif"), 0.01, 0.90)
        self.low_post = write_scene(os.path.join(self.tmp, "low_post.tif"), 0.02, 0.90)
        self.dem1 = write_dem(os.path.join(self.tmp, "dem1.tif"), 5.0)
        self.dem2 = write_dem(os.path.join(self.tmp, "dem2.tif"), 2.0)
        self.request = PipelineRequest(**BASE_REQUEST)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestLocalPipeline(PipelineTestCase):
    def test_full_local_run_exports_and_terrain_metadata(self):
        events = []

        def progress(message, stage=None):
            events.append((stage, message))

        result = run_pipeline(
            self.request, self.out, run_id="full1",
            local_pair=(self.pre, self.post), dem_path=self.dem1, progress=progress,
        )
        # 返回结构必须是 JSON 安全的，且不把栅格数组字符串化
        json.dumps(result)
        self.assertNotIn("terrain_risk_mask", result["meta"])

        self.assertTrue(os.path.isfile(result["zip"]))
        self.assertTrue(os.path.isfile(result["files"]["before_overlay"]))
        self.assertTrue(os.path.isfile(result["files"]["after_overlay"]))
        self.assertEqual(result["meta"]["provenance"]["source"], "本地输入，未核验观测日期")
        self.assertIn("未核验", result["meta"]["provenance"]["note"])

        terrain = result["meta"]["terrain"]
        self.assertEqual(terrain["profile"], "plain")
        self.assertTrue(terrain["dem_assessed"])
        self.assertTrue(result["meta"]["terrain_risk_summary"]["dem_assessed"])
        self.assertIn("terrain_risk_tif", result["files"])
        self.assertTrue(os.path.isfile(result["files"]["terrain_risk_tif"]))

        # 成果包内不得把栅格数组字符串化：stats.json 里不应出现 masks / terrain_risk_mask
        with open(result["files"]["stats"], encoding="utf-8") as fh:
            stats_payload = json.load(fh)
        self.assertNotIn("masks", stats_payload.get("change", {}))
        self.assertNotIn("terrain_risk_mask", stats_payload.get("meta", {}))
        import zipfile

        with zipfile.ZipFile(result["zip"]) as bundle:
            names = bundle.namelist()
        self.assertIn("terrain_risk.tif", names)
        self.assertIn("stats.json", names)
        self.assertIn("【识别结论】", result["summary"])
        self.assertIn("【双时相】", result["summary"])

        stages = [stage for stage, _ in events]
        for expected in ("acquire", "detect", "export", "done"):
            self.assertIn(expected, stages)

        with open(os.path.join(pipeline.run_dir_for(self.out, "full1"), "manifest.json"),
                  encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertTrue(manifest["completed"])
        self.assertEqual(manifest["request"], self.request.to_dict())
        self.assertEqual(manifest["outcome"]["status"], "completed")
        self.assertEqual(manifest["stages"]["acquire"]["source"], "local")
        for stage in ("acquire", "detect", "export"):
            self.assertEqual(manifest["stages"][stage]["status"], "completed")
        self.assertEqual(manifest["identity"]["dem_sha256"], pipeline._sha256_file(self.dem1))
        json.dumps(manifest)  # 清单本身也必须 JSON 安全

    def test_terrain_context_attached_without_dem(self):
        result = run_pipeline(
            PipelineRequest(**{**BASE_REQUEST, "terrain_profile": "urban"}),
            self.out, run_id="nod1", local_pair=(self.pre, self.post),
        )
        terrain = result["meta"]["terrain"]
        self.assertEqual(terrain["profile"], "urban")
        self.assertFalse(terrain["dem_assessed"])
        self.assertNotIn("terrain_risk_mask", result["meta"])
        self.assertTrue(any("未提供本地 DEM" in warning for warning in result["meta"]["warnings"]))

    def test_new_run_id_keeps_independent_output(self):
        first = run_pipeline(self.request, self.out, run_id="run_a",
                             local_pair=(self.pre, self.post))
        second = run_pipeline(self.request, self.out, run_id="run_b",
                              local_pair=(self.pre, self.post))
        self.assertNotEqual(first["zip"], second["zip"])
        self.assertTrue(os.path.isfile(first["zip"]))
        self.assertTrue(os.path.isfile(second["zip"]))
        self.assertNotEqual(pipeline.run_dir_for(self.out, "run_a"),
                            pipeline.run_dir_for(self.out, "run_b"))

    def test_resume_only_when_identity_and_hashes_match(self):
        first = run_pipeline(self.request, self.out, run_id="resume1",
                             local_pair=(self.pre, self.post))
        with mock.patch.object(pipeline, "_run_detection",
                               side_effect=AssertionError("不允许重新进入识别")):
            second = run_pipeline(self.request, self.out, run_id="resume1",
                                  local_pair=(self.pre, self.post))
        self.assertEqual(second["zip"], first["zip"])

        # 成果文件被改动后，完成记录必须失效（不能盲目复用）
        with open(first["files"]["after_overlay"], "ab") as fh:
            fh.write(b"tampered")
        with mock.patch.object(pipeline, "_run_detection",
                               side_effect=AssertionError("成果哈希不符时不应复用")):
            with self.assertRaises(AssertionError):
                run_pipeline(self.request, self.out, run_id="resume1",
                             local_pair=(self.pre, self.post))

    def test_changed_dem_or_input_invalidates_completed_run(self):
        run_pipeline(self.request, self.out, run_id="ident1",
                     local_pair=(self.pre, self.post), dem_path=self.dem1)
        with self.assertRaises(ValueError):
            run_pipeline(self.request, self.out, run_id="ident1",
                         local_pair=(self.pre, self.post), dem_path=self.dem2)
        with self.assertRaises(ValueError):
            run_pipeline(self.request, self.out, run_id="ident1",
                         local_pair=(self.pre_alt, self.post), dem_path=self.dem1)

    def test_bad_common_quality_raises_data_quality_error(self):
        with self.assertRaises(DataQualityError) as ctx:
            run_pipeline(self.request, self.out, run_id="quality1",
                         local_pair=(self.low_pre, self.low_post))
        self.assertIn("数据不足", str(ctx.exception))
        with open(os.path.join(pipeline.run_dir_for(self.out, "quality1"), "manifest.json"),
                  encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertFalse(manifest["completed"])
        self.assertEqual(manifest["outcome"]["status"], "failed")


class TestAcquisitionIsMocked(PipelineTestCase):
    def test_acquisition_rejects_file_outside_request_cache(self):
        fake_module = types.SimpleNamespace(fetch_event=lambda *args, **kwargs: {
            "pre": "../other_request.tif", "post": "fake_post.tif",
        })
        with mock.patch.object(pipeline, "_FETCH_MODULE", fake_module):
            with self.assertRaisesRegex(RuntimeError, "文件名无效"):
                run_pipeline(self.request, self.out, run_id="bad_cache_entry")

    def test_mocked_acquisition_uses_request_cache_and_records_provenance(self):
        calls = []

        def fake_fetch_event(key, cfg, out_dir, size=None, max_cloud=None,
                             allow_cloudy=None, fast=None, budget_s=None):
            calls.append({
                "key": key, "cfg": cfg, "out_dir": out_dir, "size": size,
                "max_cloud": max_cloud, "allow_cloudy": allow_cloudy,
                "fast": fast, "budget_s": budget_s,
            })
            write_scene(os.path.join(out_dir, "fake_pre.tif"), 0.05)
            write_scene(os.path.join(out_dir, "fake_post.tif"), 0.15)
            return {
                "pre": "fake_pre.tif", "post": "fake_post.tif",
                "provenance": {"source": "假数据源（测试）", "pre_scene": "FAKE_PRE",
                               "post_scene": "FAKE_POST"},
            }

        fake_module = types.SimpleNamespace(fetch_event=fake_fetch_event)
        with mock.patch.object(pipeline, "_FETCH_MODULE", fake_module):
            result = run_pipeline(self.request, self.out, run_id="acq1")

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["key"], self.request.cache_key)
        self.assertEqual(call["size"], self.request.size)
        self.assertEqual(call["max_cloud"], self.request.max_cloud_pct)
        self.assertIs(call["allow_cloudy"], False)
        self.assertIs(call["fast"], False)
        self.assertEqual(call["budget_s"], 120)
        self.assertEqual(call["out_dir"],
                         os.path.join(os.path.abspath(self.out), "acquisition_cache",
                                      self.request.cache_key))
        self.assertEqual(call["cfg"]["aoi"], (self.request.lon, self.request.lat))
        self.assertEqual(call["cfg"]["pre"], (self.request.pre_start, self.request.pre_end,
                                              self.request.pre_end))
        self.assertEqual(result["meta"]["provenance"]["source"], "假数据源（测试）")


class TestBatchHelpers(PipelineTestCase):
    def test_csv_two_valid_one_bad_keeps_valid_and_reports_bad(self):
        store = automation.get_store(self.out)
        rows = [
            {**{k: str(v) for k, v in BASE_REQUEST.items()},
             "local_pre": self.pre, "local_post": self.post},
            {"lon": "112.66", "lat": "不是数字",
             "pre_start": "2024-06-01", "pre_end": "2024-06-30",
             "post_start": "2024-07-06", "post_end": "2024-07-25"},
            {**{k: str(v) for k, v in BASE_REQUEST.items()},
             "local_pre": self.pre, "local_post": self.post},
        ]
        summary = automation.import_batch(store, rows)
        self.assertEqual(summary["queued_count"], 2)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["errors"][0]["row"], 2)
        self.assertIn("lat", summary["errors"][0]["error"])
        self.assertEqual(len(store.list_jobs()), 2)
        self.assertIn("1 行未通过校验", automation.format_import_result(summary))

    def test_load_batch_rejects_unknown_columns_and_extra_fields(self):
        header = ("lon,lat,pre_start,pre_end,post_start,post_end,completely_unknown\n"
                  "112.66,29.36,2024-06-01,2024-06-30,2024-07-06,2024-07-25,x\n")
        rows = automation.load_batch(header)
        self.assertIn("completely_unknown", rows[0])
        # 未知列按行报错，而不是整体失败
        summary = automation.import_batch(automation.get_store(self.out), rows)
        self.assertEqual(summary["queued_count"], 0)
        self.assertIn("未知列", summary["errors"][0]["error"])

        extra = ("lon,lat,pre_start,pre_end,post_start,post_end\n"
                 "112.66,29.36,2024-06-01,2024-06-30,2024-07-06,2024-07-25,多余的一列\n")
        rows2 = automation.load_batch(extra)
        self.assertIn("__extra__", rows2[0])
        summary2 = automation.import_batch(automation.get_store(self.out), rows2)
        self.assertEqual(summary2["queued_count"], 0)
        self.assertIn("列数多于表头", summary2["errors"][0]["error"])

    def test_mixed_queue_failure_isolation(self):
        store = automation.get_store(self.out)
        good = {"request": self.request.to_dict(),
                "local_pre": self.pre, "local_post": self.post}
        bad = {"request": self.request.to_dict(),
               "local_pre": os.path.join(self.tmp, "missing.tif"), "local_post": self.post}
        ids = [store.enqueue(good), store.enqueue(bad), store.enqueue(good)]
        events = list(QueueRunner(store, automation.make_runner(self.out)).run_pending())
        self.assertEqual([event["event"] for event in events],
                         ["started", "succeeded", "started", "failed", "started", "succeeded"])

        jobs = {job["job_id"]: job for job in store.list_jobs()}
        self.assertEqual(jobs[ids[0]]["status"], "succeeded")
        self.assertEqual(jobs[ids[1]]["status"], "failed")
        self.assertIn("FileNotFoundError", jobs[ids[1]]["error"])
        self.assertEqual(jobs[ids[2]]["status"], "succeeded")
        first_zip = jobs[ids[0]]["result"]["zip"]
        third_zip = jobs[ids[2]]["result"]["zip"]
        self.assertTrue(os.path.isfile(first_zip) and os.path.isfile(third_zip))
        self.assertNotEqual(first_zip, third_zip)

    def test_synthetic_demo_payload_is_labeled(self):
        store = automation.get_store(self.out)
        job_id = store.enqueue({
            "request": self.request.to_dict(),
            "local_pre": self.pre, "local_post": self.post, "synthetic": True,
        })
        events = list(QueueRunner(store, automation.make_runner(self.out)).run_pending())
        self.assertEqual(events[-1]["event"], "succeeded")
        result = store.get(job_id)["result"]
        self.assertEqual(result["meta"]["provenance"]["source"], "合成演示数据，不可代表真实精度")
        self.assertTrue(any("不可代表真实精度" in warning
                            for warning in result["meta"]["warnings"]))

    def test_build_automation_ui_does_not_touch_disk(self):
        import gradio as gr

        with gr.Blocks():
            accordion = automation.build_automation_ui(self.out)
        self.assertIsNotNone(accordion)
        # 只登记回调：构建页面不应创建任务库或读取输入文件
        self.assertFalse(os.path.exists(os.path.join(self.out, "jobs.sqlite")))

    def test_recover_refuses_while_a_job_is_live(self):
        store = automation.get_store(self.out)
        job_id = store.enqueue({"request": self.request.to_dict()})
        store.claim_next()  # 模拟另一个进程正在执行（running 且刚更新）
        outcome = automation.recover_interrupted_jobs(self.out)
        self.assertEqual(outcome["recovered"], 0)
        self.assertEqual(outcome["running"], [job_id])
        self.assertEqual(store.get(job_id)["status"], "running")

    def test_recover_keeps_long_running_worker(self):
        store = automation.get_store(self.out)
        job_id = store.enqueue({"request": self.request.to_dict()})
        with store.worker_lock() as acquired:
            self.assertTrue(acquired)
            store.claim_next()
            outcome = automation.recover_interrupted_jobs(self.out, stale_after_s=0)
            self.assertEqual(outcome["recovered"], 0)
            self.assertEqual(outcome["running"], [job_id])
            self.assertEqual(store.get(job_id)["status"], "running")


if __name__ == "__main__":
    unittest.main()
