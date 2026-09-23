"""任务 A 回归：PipelineRequest 指纹 + fetch_event 的 key 级缓存清单。

只跑本文件：python -m pytest tests/test_task_cache.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load(name, rel):
    """按文件路径加载被测模块，绕开 src/__init__.py 的无关导入。"""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 依赖 __module__ 能在 sys.modules 里解析
    spec.loader.exec_module(mod)
    return mod


PipelineRequest = _load("task_contracts_under_test", "src/task_contracts.py").PipelineRequest
FRS = _load("frs_cache_under_test", "scripts/fetch_real_samples.py")
SIZE = 64
BASE = {"lon": 112.66, "lat": 29.36, "pre_start": "2024-06-01", "pre_end": "2024-06-30",
        "post_start": "2024-07-06", "post_end": "2024-07-25"}


def _cfg():
    return {"label": "单测事件", "aoi": (112.66, 29.36), "size": SIZE, "note": "本地夹具",
            "pre": ("2024-06-01", "2024-06-30", "2024-06-20"),
            "post": ("2024-07-06", "2024-07-25", "2024-07-08")}


def _write_tif(path, size, value):
    import rasterio
    from rasterio.transform import from_origin
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {"driver": "GTiff", "height": size, "width": size, "count": 4, "dtype": "uint16",
               "crs": "EPSG:32649", "nodata": 0, "transform": from_origin(500000, 3000000, 10, 10)}
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.full((4, size, size), value, dtype=np.uint16))


def _fake_fetch_tag(calls):
    def fake(tag, cfg, lon, lat, bbox, size, half_km, max_cloud, allow_cloudy, out_dir, entry):
        calls.append(tag)
        _write_tif(os.path.join(out_dir, entry[tag]), size, 1000 if tag == "pre" else 900)
        when = "2024-06-01T00:00:00Z" if tag == "pre" else "2024-07-08T00:00:00Z"
        item = {"id": f"S2_{tag}", "properties": {"datetime": when,
                "eo:cloud_cover": 5.0, "proj:epsg": 32649}}
        data = {"scl": np.full((size, size), 6, dtype=np.uint8), "offset_applied": False,
                "arr": np.full((4, size, size), 1000, dtype=np.uint16), "offset_method": "夹具"}
        return item, {"cloud_pct": 5.0, "water_pct": 1.0}, (500000.0, 3000000.0, 0.5), data, "pc"
    return fake


class TestPipelineRequest(unittest.TestCase):
    def test_cache_key_changes_with_every_field(self):
        base = PipelineRequest(**BASE)
        self.assertEqual(base.cache_key, PipelineRequest(**BASE).cache_key)
        self.assertNotEqual(base.cache_key, PipelineRequest.new_run_id())
        for delta in ({"lon": 112.6601}, {"lat": 29.3601}, {"pre_start": "2024-06-02"},
                      {"pre_end": "2024-06-29"}, {"post_start": "2024-07-07"},
                      {"post_end": "2024-07-24"}, {"size": 1024},
                      {"terrain_profile": "urban"}, {"max_cloud_pct": 50.0},
                      {"min_valid_pct": 60.0}):
            self.assertNotEqual(base.cache_key,
                                PipelineRequest(**{**BASE, **delta}).cache_key, delta)

    def test_rejects_invalid_values(self):
        for delta in ({"lon": float("nan")}, {"lat": float("inf")}, {"lon": 181.0},
                      {"lat": -91.0}, {"pre_start": "2024-02-30"}, {"pre_start": "2024/06/01"},
                      {"pre_end": "2024-08-01"}, {"post_start": "2024-06-30"}, {"size": 32},
                      {"size": 8192}, {"size": True}, {"terrain_profile": "swamp"},
                      {"max_cloud_pct": -1.0}, {"min_valid_pct": 101.0}):
            with self.assertRaises(ValueError, msg=delta):
                PipelineRequest(**{**BASE, **delta})

    def test_dict_roundtrip_and_rejects_unknown_keys(self):
        req = PipelineRequest(**BASE, size=512, terrain_profile="hilly")
        self.assertEqual(PipelineRequest.from_dict(req.to_dict()), req)
        with self.assertRaises(ValueError):
            PipelineRequest.from_dict({**req.to_dict(), "surprise": 1})
        with self.assertRaises(ValueError):
            PipelineRequest.from_dict({k: v for k, v in req.to_dict().items() if k != "lon"})


class TestFetchEventCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="huiyan_cache_")
        self.calls = []
        self._orig = FRS._fetch_tag
        FRS._fetch_tag = _fake_fetch_tag(self.calls)

    def tearDown(self):
        FRS._fetch_tag = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fetch(self, size=SIZE, cfg=None):
        payload = _cfg()
        payload.update(cfg or {})
        return FRS.fetch_event("unit", payload, self.tmp, size=size,
                               max_cloud=35.0, allow_cloudy=False, fast=False)

    def test_valid_reuse_preserves_provenance(self):
        first = self._fetch()
        self.assertIsNotNone(first)
        self.assertEqual(self.calls, ["pre", "post"])
        with open(os.path.join(self.tmp, "unit.cache.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        blob = json.dumps(manifest)
        self.assertNotIn("_arr", blob)
        self.assertNotIn("http", blob.lower())
        self.assertEqual(manifest["request"]["aoi"], [112.66, 29.36])
        self.assertEqual(manifest["request"]["pre"], ["2024-06-01", "2024-06-30", "2024-06-20"])
        self.calls.clear()
        second = self._fetch()
        self.assertEqual(self.calls, [], "相同请求必须命中缓存")
        self.assertEqual(second["provenance"], first["provenance"])
        self.assertEqual(second["provenance"]["pre_scene"], "S2_pre")

    def test_changed_request_misses(self):
        self._fetch()
        self.calls.clear()
        self._fetch(cfg={"pre": ("2024-06-02", "2024-06-30", "2024-06-20")})
        self.assertIn("pre", self.calls, "灾前日期变化必须失效")
        self.calls.clear()
        self._fetch(size=128)
        self.assertIn("post", self.calls, "窗口尺寸变化必须失效")

    def test_corrupt_or_missing_post_misses(self):
        self._fetch()
        self.calls.clear()
        with open(os.path.join(self.tmp, "unit_post.tif"), "ab") as fh:
            fh.write(b"corrupt")
        self.assertIsNotNone(self._fetch())
        self.assertIn("post", self.calls, "POST 内容被改动必须失效")
        self.calls.clear()
        os.remove(os.path.join(self.tmp, "unit.cache.json"))
        self.assertIsNotNone(self._fetch())
        self.assertIn("pre", self.calls, "缺少 key 级清单的旧数据必须视为未命中")

    def test_cached_pair_requires_request(self):
        self._fetch()
        self.assertIsNone(FRS._cached_pair(self.tmp, "unit", SIZE))


if __name__ == "__main__":
    unittest.main()
