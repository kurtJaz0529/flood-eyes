"""
慧眼识灾 · 全流程自检
=====================

运行：
    python tests/test_pipeline.py           # 全部用例
    python -m pytest tests -q               # 有 pytest 也行

覆盖：波段解析 / NDWI / 基线 / 后处理面积换算 / 双时相 / 模型前向 /
      分块推理一致性 / 权重存取 / 成果导出 / 端到端识别精度
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src import FloodDetector, build_model, load_checkpoint, load_scene, save_checkpoint  # noqa: E402
from src import baseline, postprocess, preprocess, report  # noqa: E402
from src.model_unet import DiceBCELoss, predict_tiled, segmentation_metrics  # noqa: E402

SAMPLES = os.path.join(ROOT, "data", "samples")
POST = os.path.join(SAMPLES, "demo04_post.tif")
PRE = os.path.join(SAMPLES, "demo04_pre.tif")
MASK = os.path.join(SAMPLES, "demo04_mask.png")
REAL_DIR = os.path.join(ROOT, "data", "real")
REAL_MANIFEST = os.path.join(REAL_DIR, "samples.json")


def _read_mask(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path)) > 127


def _has_samples() -> bool:
    return os.path.isfile(POST) and os.path.isfile(MASK)


class TestPreprocess(unittest.TestCase):
    def test_band_parsing(self):
        sc = load_scene(POST)
        self.assertEqual(sc.channel_names, ["blue", "green", "red", "nir"])
        self.assertTrue(sc.has_nir)
        self.assertEqual(sc.shape, (640, 640))
        self.assertAlmostEqual(sc.pixel_size_m, 10.0, places=3)

    def test_reflectance_range(self):
        sc = load_scene(POST)
        for name, band in sc.bands.items():
            self.assertGreaterEqual(float(band.min()), 0.0, name)
            self.assertLessEqual(float(band.max()), 1.5, name)

    def test_ndwi_range_and_water_signal(self):
        sc = load_scene(POST)
        ndwi = sc.ndwi()
        self.assertLessEqual(abs(float(ndwi.min())), 1.0)
        self.assertLessEqual(abs(float(ndwi.max())), 1.0)
        if _has_samples():
            gt = _read_mask(MASK)
            self.assertGreater(float(ndwi[gt].mean()), float(ndwi[~gt].mean()) + 0.3,
                               "水体 NDWI 应显著高于陆地")

    def test_no_nir_fallback(self):
        sc = load_scene(POST)
        sc3 = load_scene(POST, band_order="rgb")
        self.assertFalse(sc3.has_nir)
        idx = sc3.ndwi()  # 应退化为 (G-R)/(G+R) 且不报错
        self.assertEqual(idx.shape, sc.shape)
        self.assertTrue(np.isfinite(idx).all())

    def test_tiling_roundtrip(self):
        arr = np.random.rand(300, 500).astype(np.float32)
        tiles, coords = [], []
        for t, c in preprocess.iter_tiles(arr, tile=128, overlap=32):
            tiles.append(t)
            coords.append(c)
        self.assertGreater(len(tiles), 4)
        rec = preprocess.stitch_tiles(tiles, coords, arr.shape)
        self.assertEqual(rec.shape, arr.shape)
        self.assertLess(float(np.abs(rec - arr).max()), 1e-5, "拼接应能无损还原")

    def test_pixel_area(self):
        self.assertAlmostEqual(preprocess.pixel_area_km2(10.0), 1e-4, places=12)

    def test_geographic_pixel_size_in_meters(self):
        from rasterio.crs import CRS
        from rasterio.transform import from_origin

        t = from_origin(114.0, 34.0, 0.0001, 0.0001)
        ps = preprocess.pixel_size_from_transform(t, CRS.from_epsg(4326))
        self.assertGreater(ps, 8.0)
        self.assertLess(ps, 13.0)
        t2 = from_origin(500000.0, 3000000.0, 10.0, 10.0)
        ps2 = preprocess.pixel_size_from_transform(t2, CRS.from_epsg(32650))
        self.assertAlmostEqual(ps2, 10.0, places=5)

    def test_boa_offset_when_dn_looks_like_raw_l2a(self):
        arr = np.zeros((16, 16, 4), dtype=np.float32)
        arr[2:, 2:] = 2500.0
        meta: dict = {}
        out = preprocess.to_reflectance(arr, meta)
        self.assertEqual(meta.get("boa_offset"), -1000.0)
        self.assertAlmostEqual(float(out[8, 8, 1]), 0.15, places=3)

    def test_boa_not_applied_to_already_corrected(self):
        arr = np.full((16, 16, 4), 589.0, dtype=np.float32)
        arr[0, 0] = 4000.0
        meta: dict = {}
        preprocess.to_reflectance(arr, meta)
        self.assertEqual(meta.get("boa_offset"), 0.0)

    def test_boa_heuristic_ignores_water(self):
        arr = np.full((16, 16, 4), 2500.0, dtype=np.float32)
        arr[0, :3] = 200.0
        meta: dict = {}
        preprocess.to_reflectance(arr, meta)
        self.assertEqual(meta.get("boa_offset"), 0.0)
        
        arr2 = np.full((16, 16, 4), 2500.0, dtype=np.float32)
        meta2: dict = {}
        preprocess.to_reflectance(arr2, meta2)
        self.assertEqual(meta2.get("boa_offset"), -1000.0)

    def test_resolve_band_order_12_channels(self):
        mapping = preprocess.resolve_band_order(12, "auto")
        self.assertEqual(mapping["blue"], 1)
        self.assertEqual(mapping["green"], 2)
        self.assertEqual(mapping["red"], 3)
        self.assertEqual(mapping["nir"], 7)

    def test_resolve_band_order_10_channels(self):
        mapping = preprocess.resolve_band_order(10, "auto")
        self.assertEqual(mapping["blue"], 0)
        self.assertEqual(mapping["nir"], 6)

    def test_resolve_band_order_from_descriptions(self):
        mapping = preprocess.resolve_band_order(
            4, "auto", descriptions=("B2(blue)", "B3(green)", "B4(red)", "B8(nir)")
        )
        self.assertEqual(mapping, {"blue": 0, "green": 1, "red": 2, "nir": 3})

    def test_s2_high_dn_uses_10000_scale(self):
        arr = np.full((8, 8, 4), 21000.0, dtype=np.float32)
        meta: dict = {}
        out = preprocess.to_reflectance(arr, meta)
        self.assertEqual(meta["dn_scale"], 10000.0)
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.5, places=5)

    def test_boa_heuristic_skipped_when_metadata_offset_applied(self):
        arr = np.full((16, 16, 4), 2500.0, dtype=np.float32)
        meta: dict = {}
        out = preprocess.to_reflectance(arr, meta, allow_boa_heuristic=False)
        self.assertEqual(meta.get("boa_offset"), 0.0)
        self.assertAlmostEqual(float(out[8, 8, 1]), 0.25, places=3)

    def test_geotiff_scale_offset_not_double_counted(self):
        import tempfile

        import rasterio
        from rasterio.transform import from_origin

        arr = np.full((16, 16, 4), 2500.0, dtype=np.float32)
        path = os.path.join(tempfile.gettempdir(), "_hysz_scale_offset.tif")
        profile = {
            "driver": "GTiff",
            "height": 16,
            "width": 16,
            "count": 4,
            "dtype": "float32",
            "crs": "EPSG:32650",
            "transform": from_origin(500000, 3000000, 10, 10),
        }
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(np.moveaxis(arr, -1, 0))
            ds.scales = (0.0001, 0.0001, 0.0001, 0.0001)
            ds.offsets = (0.0, 0.0, 0.0, 0.0)
            for i, name in enumerate(("B2(blue)", "B3(green)", "B4(red)", "B8(nir)"), start=1):
                ds.set_band_description(i, name)
        try:
            sc = load_scene(path)
            self.assertTrue(sc.meta.get("raster_scale_offset"))
            self.assertAlmostEqual(float(sc.bands["green"][8, 8]), 0.25, places=4)
            self.assertEqual(sc.meta.get("boa_offset"), 0.0)
            self.assertEqual(sc.channel_names, ["blue", "green", "red", "nir"])
        finally:
            os.remove(path)

    def test_scl_sidecar_used_for_cloud_mask(self):
        import tempfile

        import rasterio
        from PIL import Image
        from rasterio.transform import from_origin

        from src.preprocess import estimate_cloud_mask

        td = tempfile.mkdtemp(prefix="hysz_scl_")
        tif = os.path.join(td, "scene.tif")
        png = os.path.join(td, "scene_scl.png")
        data = np.full((12, 12, 4), 0.08, dtype=np.float32)
        data[..., 3] = 0.30
        profile = {
            "driver": "GTiff",
            "height": 12,
            "width": 12,
            "count": 4,
            "dtype": "float32",
            "crs": "EPSG:32650",
            "transform": from_origin(500000, 3000000, 10, 10),
        }
        with rasterio.open(tif, "w", **profile) as ds:
            ds.write(np.moveaxis(data, -1, 0))
        rgb = np.zeros((12, 12, 3), dtype=np.uint8)
        rgb[:, :] = (60, 150, 60)  # SCL 4 植被
        rgb[2:6, 2:6] = (255, 255, 255)  # SCL 9 高云
        Image.fromarray(rgb).save(png)
        try:
            sc = load_scene(tif)
            self.assertEqual(sc.meta.get("scl_source"), "sidecar")
            cloud = estimate_cloud_mask(sc)
            self.assertIsNotNone(cloud)
            self.assertTrue(bool(cloud[3, 3]))
            self.assertFalse(bool(cloud[0, 0]))
        finally:
            os.remove(tif)
            os.remove(png)
            os.rmdir(td)

    def test_rotated_pixel_size(self):
        from rasterio.crs import CRS
        from rasterio.transform import Affine

        t = Affine(0.0, -10.0, 0.0, 10.0, 0.0, 0.0)
        ps = preprocess.pixel_size_from_transform(t, CRS.from_epsg(32650))
        self.assertAlmostEqual(ps, 10.0, places=5)


class TestBaseline(unittest.TestCase):
    def test_otsu_on_bimodal(self):
        rng = np.random.default_rng(0)
        v = np.concatenate([rng.normal(-0.5, 0.05, 5000), rng.normal(0.5, 0.05, 5000)])
        t = baseline.otsu_threshold(v)
        # Otsu 的最优解是"两类之间的间隙"，落点取决于实现（首个极值/平台中点）
        self.assertTrue(-0.35 < t < 0.35, f"阈值应落在两峰之间，实际 {t}")
        self.assertLess(abs(float(np.mean(v < t)) - 0.5), 0.05, "阈值应把样本大致一分为二")

    def test_predict_shapes(self):
        ndwi = np.random.uniform(-1, 1, (128, 128)).astype(np.float32)
        mask, prob, meta = baseline.predict(ndwi)
        self.assertEqual(mask.shape, ndwi.shape)
        self.assertEqual(prob.shape, ndwi.shape)
        self.assertTrue(0.0 <= float(prob.min()) and float(prob.max()) <= 1.0)
        self.assertIn("threshold_global", meta)

    def test_uniform_image_fallback(self):
        ndwi = np.full((64, 64), -0.2, dtype=np.float32)  # 全陆地，Otsu 无意义
        mask, prob, meta = baseline.predict(ndwi)
        self.assertEqual(int(mask.sum()), 0, "全陆地影像不应提取出水体")
        self.assertEqual(meta["threshold_source"], "no_signal")
        self.assertIn("无空间变化", meta["note"])

    def test_uniform_all_water(self):
        ndwi = np.full((64, 64), 0.55, dtype=np.float32)  # 整景都是水
        mask, _, meta = baseline.predict(ndwi)
        self.assertEqual(int(mask.sum()), 64 * 64)
        self.assertEqual(meta["threshold_source"], "no_signal")

    def test_nir_gate_suppresses_urban(self):
        """城市像元 NDWI 略高但近红外也高，应被 NIR 闸门压掉。"""
        rng = np.random.default_rng(1)
        ndwi = (-0.09 + 0.002 * rng.standard_normal((16, 16))).astype(np.float32)
        nir = np.full((16, 16), 0.18, dtype=np.float32)  # 城市近红外
        m1, _, _ = baseline.predict(ndwi, method="otsu", fixed_threshold=-0.15)
        m2, _, meta = baseline.predict(ndwi, method="otsu", fixed_threshold=-0.15, nir=nir)
        self.assertGreater(int(m1.sum()), 0)
        self.assertEqual(int(m2.sum()), 0)
        self.assertIn("nir_gate", meta)

    def test_local_threshold_is_clamped(self):
        ndwi = np.random.uniform(-1, 1, (256, 256)).astype(np.float32)
        _, _, meta = baseline.predict(ndwi, method="hybrid")
        lo, hi = meta["threshold_local_range"]
        t = meta["threshold_global"]
        self.assertGreaterEqual(lo, t - 0.101)
        self.assertLessEqual(hi, t + 0.101)


class TestPostprocess(unittest.TestCase):
    def test_area_math(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:10, :10] = True  # 100 像元
        stats = postprocess.area_stats(mask, pixel_size_m=10.0)
        self.assertEqual(stats["water_pixels"], 100)
        self.assertAlmostEqual(stats["water_area_km2"], 100 * 1e-4, places=9)
        self.assertEqual(stats["n_components"], 1)
        self.assertAlmostEqual(stats["water_fraction_pct"], 1.0, places=6)

    def test_area_stats_excludes_nodata_from_denominator(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[:10, :5] = True
        nodata = np.zeros((10, 10), dtype=bool)
        nodata[:, 5:] = True
        stats = postprocess.area_stats(mask, pixel_size_m=10.0, nodata_mask=nodata)
        self.assertEqual(stats["total_pixels"], 50)
        self.assertAlmostEqual(stats["water_fraction_pct"], 100.0, places=5)

    def test_clean_mask_removes_specks(self):
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:150, 50:150] = True  # 主水体 10000 像元
        mask[5, 5] = True  # 噪点
        cleaned = postprocess.clean_mask(mask, min_area_px=50)
        self.assertFalse(cleaned[5, 5], "孤立噪点应被剔除")
        self.assertGreater(int(cleaned.sum()), 9000)

    def test_fill_holes(self):
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True
        mask[25:35, 25:35] = False  # 内部空洞 10×10=100 px
        filled = postprocess.clean_mask(mask, open_radius=0, close_radius=0, min_area_px=0, fill_holes=True)
        self.assertTrue(bool(filled[30, 30]), "内部空洞应被填充")

    def test_large_island_not_filled(self):
        mask = np.zeros((80, 80), dtype=bool)
        mask[5:75, 5:75] = True
        mask[20:60, 20:60] = False  # 40×40=1600 px 岛屿
        filled = postprocess.clean_mask(
            mask, open_radius=0, close_radius=0, min_area_px=0, fill_holes=True, max_hole_px=500
        )
        self.assertFalse(bool(filled[40, 40]), "大岛屿不应被填成水")

    def test_low_confidence_pct_is_among_water_pixels(self):
        prob = np.full((20, 20), 0.1, dtype=np.float32)
        mask = np.zeros((20, 20), dtype=bool)
        mask[:10, :10] = True
        prob[:10, :5] = 0.4
        prob[:10, 5:10] = 0.9
        s = postprocess.confidence_stats(prob, mask)
        self.assertAlmostEqual(s["low_confidence_pct"], 50.0, places=3)

    def test_change_stats(self):
        before = np.zeros((100, 100), dtype=bool)
        before[:20, :20] = True  # 400 像元
        after = np.zeros((100, 100), dtype=bool)
        after[:30, :20] = True  # 600 像元，新增 200
        c = postprocess.change_stats(before, after, pixel_size_m=10.0)
        self.assertAlmostEqual(c["new_water_km2"], 200 * 1e-4, places=9)
        self.assertAlmostEqual(c["receded_water_km2"], 0.0, places=9)
        self.assertAlmostEqual(c["persistent_water_km2"], 400 * 1e-4, places=9)

    def test_overlay_shape_and_color(self):
        rgb = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True
        out = postprocess.overlay_mask(rgb, mask)
        self.assertEqual(out.shape, rgb.shape)
        self.assertGreater(int(out[25, 25, 1]), int(out[5, 5, 1]), "水体区域应被上色")


class TestModel(unittest.TestCase):
    def test_build_and_forward(self):
        model, info = build_model(arch="tiny", in_channels=4, base=8)
        import torch

        x = torch.randn(2, 4, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertEqual(info["arch"], "tiny_unet")

    def test_loss_decreases_on_perfect_prediction(self):
        import torch

        crit = DiceBCELoss()
        gt = torch.zeros(1, 1, 32, 32)
        gt[..., 8:24, 8:24] = 1.0
        good = torch.where(gt > 0.5, torch.tensor(6.0), torch.tensor(-6.0))
        bad = torch.zeros_like(gt)
        self.assertLess(float(crit(good, gt)), float(crit(bad, gt)))

    def test_metrics_known_values(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[:5, :5] = True
        gt = np.zeros((10, 10), dtype=bool)
        gt[:5, :5] = True
        m = segmentation_metrics(pred, gt)
        self.assertAlmostEqual(m["iou"], 1.0, places=6)
        m2 = segmentation_metrics(np.zeros((10, 10), bool), gt)
        self.assertAlmostEqual(m2["iou"], 0.0, places=6)
        self.assertAlmostEqual(m2["recall"], 0.0, places=6)

    def test_checkpoint_roundtrip(self):
        import torch

        from src.model_unet import normalize_arch

        self.assertEqual(normalize_arch("smp_unet"), "smp")
        self.assertEqual(normalize_arch("tiny_unet"), "tiny")

        model, _ = build_model(arch="tiny", in_channels=4, base=8)
        model.eval()  # BatchNorm 在 train/eval 下统计量不同，比较前统一
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        path = os.path.join(ROOT, "logs", "_unit_test_model.pt")
        save_checkpoint(path, model, {"arch": "tiny", "in_channels": 4, "base": 8, "mean": [0.1] * 4, "std": [0.1] * 4})
        model2, meta = load_checkpoint(path, device="cpu")
        self.assertEqual(meta["arch"], "tiny")
        self.assertEqual(meta["arch_key"], "tiny")
        with torch.no_grad():
            x = torch.randn(1, 4, 32, 32)
            self.assertTrue(bool(torch.allclose(model(x), model2(x), atol=1e-6)))
        os.remove(path)

    def test_tiny_unet_alias_loads(self):
        model, _ = build_model(arch="tiny", in_channels=4, base=8)
        model.eval()
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        path = os.path.join(ROOT, "logs", "_unit_test_tiny_alias.pt")
        save_checkpoint(path, model, {"arch": "tiny_unet", "in_channels": 4, "base": 8})
        _m2, meta = load_checkpoint(path, device="cpu")
        self.assertEqual(meta["arch_key"], "tiny")
        os.remove(path)

    def test_load_rejects_architecture_mismatch(self):
        model, _ = build_model(arch="tiny", in_channels=4, base=8)
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        path = os.path.join(ROOT, "logs", "_unit_test_mismatch.pt")
        save_checkpoint(path, model, {"arch": "tiny", "in_channels": 4, "base": 16})
        with self.assertRaises(RuntimeError):
            load_checkpoint(path, device="cpu")
        os.remove(path)

    def test_tiled_matches_full(self):
        """分块推理与整图推理应基本一致（边界处允许小差异）。"""
        import torch

        model, _ = build_model(arch="tiny", in_channels=4, base=8)
        chw = np.random.rand(4, 128, 128).astype(np.float32)
        prob_tiled, _, _ = predict_tiled(model, chw, tile=64, overlap=32, batch_size=2, device="cpu")
        with torch.no_grad():
            logits = model(torch.from_numpy(preprocess.normalize(chw)[None]))
            prob_full = torch.sigmoid(logits)[0, 0].numpy()
        self.assertEqual(prob_tiled.shape, prob_full.shape)
        self.assertLess(float(np.abs(prob_tiled - prob_full).mean()), 0.02)


class TestEndToEnd(unittest.TestCase):
    @unittest.skipUnless(_has_samples(), "需要先运行 python data/make_samples.py")
    def test_baseline_beats_naive_threshold(self):
        result = FloodDetector(mode="baseline").detect(POST)
        gt = _read_mask(MASK)
        m = segmentation_metrics(result.mask, gt)
        self.assertGreater(m["iou"], 0.80, f"基线 IoU 偏低：{m}")
        self.assertGreater(result.area_km2, 0.0)
        self.assertEqual(result.mask.shape, gt.shape)
        self.assertEqual(result.overlay.shape[:2], gt.shape)

    @unittest.skipUnless(_has_samples(), "需要先运行 python data/make_samples.py")
    def test_comparison_mode(self):
        result = FloodDetector(mode="baseline").compare(PRE, POST)
        self.assertIsNotNone(result.change)
        c = result.change
        self.assertGreater(c["new_water_km2"], 0.0)
        self.assertAlmostEqual(c["after_water_km2"] - c["before_water_km2"], c["net_change_km2"], places=6)
        self.assertIn("change_map", c)

    @unittest.skipUnless(_has_samples(), "需要先运行 python data/make_samples.py")
    def test_auto_mode_falls_back_to_baseline_without_weights(self):
        det = FloodDetector(mode="auto", weights=os.path.join(ROOT, "weights", "__not_exist__.pt"))
        info = det.model_info()
        self.assertEqual(info["mode"], "baseline")

    def test_auto_mode_skips_synthetic_weights(self):
        model, _ = build_model(arch="tiny", in_channels=4, base=8)
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        path = os.path.join(ROOT, "logs", "_unit_test_synthetic.pt")
        save_checkpoint(
            path,
            model,
            {"arch": "tiny", "in_channels": 4, "base": 8, "data_note": "synthetic demo data"},
        )
        det = FloodDetector(mode="auto", weights=path)
        self.assertEqual(det.resolved_mode, "baseline")
        info = det.model_info()
        self.assertIn("合成", info.get("warning") or "")
        os.remove(path)

    def test_compare_reprojects_shifted_grids(self):
        from rasterio.crs import CRS
        from rasterio.transform import from_origin

        from src.preprocess import Scene

        def toy(xoff: float) -> Scene:
            g = np.full((48, 48), 0.08, dtype=np.float32)
            n = np.full((48, 48), 0.30, dtype=np.float32)
            g[16:32, 16:32] = 0.04
            n[16:32, 16:32] = 0.02
            return Scene(
                bands={"blue": g.copy(), "green": g, "red": g.copy(), "nir": n},
                transform=from_origin(xoff, 3_000_000.0, 10.0, 10.0),
                crs=CRS.from_epsg(32650),
                pixel_size_m=10.0,
            )

        result = FloodDetector(mode="baseline", mask_clouds=False).compare(
            before_scene=toy(500_000.0), after_scene=toy(500_040.0)
        )
        self.assertTrue(result.meta["comparison"]["reprojected"])
        self.assertTrue(any("重投影" in w for w in result.meta.get("warnings", [])))

    @unittest.skipUnless(_has_samples(), "需要先运行 python data/make_samples.py")
    def test_export_bundle(self):
        result = FloodDetector(mode="baseline").detect(POST)
        out = report.export_bundle(result, out_dir=os.path.join(ROOT, "outputs"), basename="_unit_test_export")
        for key in ("mask", "overlay", "rgb", "heatmap", "comparison", "stats", "report"):
            self.assertIn(key, out["files"])
            self.assertTrue(os.path.isfile(out["files"][key]), key)
        self.assertTrue(os.path.isfile(out["zip"]))
        self.assertGreater(os.path.getsize(out["zip"]), 1000)
        self.assertGreater(os.path.getsize(out["files"]["report"]), 5000, "PDF 简报不应为空")

    @unittest.skipUnless(_has_samples(), "需要先运行 python data/make_samples.py")
    def test_summary_and_json(self):
        result = FloodDetector(mode="baseline").detect(POST)
        self.assertIn("km²", result.summary_text())
        self.assertIn("water_area_km2", result.to_json())


class TestRealImagery(unittest.TestCase):
    """真实 Sentinel-2 影像（data/real/）。未抓取时自动跳过。

    重点防回归：BOA 偏移量处理错误会让整景反射率变负，NDWI 直接失效。
    """

    def setUp(self):
        if not os.path.isfile(REAL_MANIFEST):
            self.skipTest("未抓取真实影像：python scripts/fetch_real_samples.py --event all")
        with open(REAL_MANIFEST, encoding="utf-8") as fh:
            self.samples = json.load(fh).get("samples", [])
        if not self.samples:
            self.skipTest("data/real/samples.json 为空")

    def test_manifest_has_full_provenance(self):
        for s in self.samples:
            p = s.get("provenance", {})
            for key in ("pre_scene", "post_scene", "pre_datetime", "post_datetime", "source", "license"):
                self.assertIn(key, p, f"{s.get('id')} 缺少溯源字段 {key}")
            self.assertIn("Sentinel-2", p["source"])
            self.assertIn("Copernicus", p["license"])

    def test_reflectance_is_physical(self):
        """绿波段不应整体为负 —— 这是 BOA 偏移被二次扣除的典型症状。"""
        for s in self.samples:
            for tag in ("pre", "post"):
                sc = load_scene(os.path.join(REAL_DIR, s[tag]))
                self.assertEqual(sc.channel_names, ["blue", "green", "red", "nir"])
                green = sc.bands["green"]
                self.assertGreater(float(np.median(green)), 0.0, f"{s['id']}-{tag} 绿波段中位反射率为负")
                self.assertLess(float((green < 0).mean()), 0.2, f"{s['id']}-{tag} 负值像元过多")
                self.assertAlmostEqual(sc.pixel_size_m, 10.0, places=3)

    def test_scene_geometry_consistent(self):
        for s in self.samples:
            pre = load_scene(os.path.join(REAL_DIR, s["pre"]))
            post = load_scene(os.path.join(REAL_DIR, s["post"]))
            self.assertEqual(pre.shape, post.shape, "灾前/灾后影像尺寸应一致")
            self.assertEqual(pre.crs, post.crs, "灾前/灾后影像坐标系应一致")

    def test_real_comparison_runs(self):
        for s in self.samples:
            result = FloodDetector(mode="baseline").compare(
                os.path.join(REAL_DIR, s["pre"]), os.path.join(REAL_DIR, s["post"])
            )
            self.assertGreater(result.area_km2, 0.0)
            self.assertIsNotNone(result.change)
            self.assertGreaterEqual(result.change["new_water_km2"], 0.0)
            self.assertEqual(result.mask.shape, result.rgb.shape[:2])


class TestSar(unittest.TestCase):
    """雷达（Sentinel-1）处理链的合成数据回归测试。"""

    def test_utm_zone_from_lonlat(self):
        from src.sar import lonlat_to_utm, utm_epsg_from_lonlat, utm_to_lonlat

        self.assertEqual(utm_epsg_from_lonlat(85.0, 28.0), "EPSG:32645")  # 尼泊尔
        self.assertEqual(utm_epsg_from_lonlat(113.7, 34.8), "EPSG:32649")  # 河南
        self.assertEqual(utm_epsg_from_lonlat(179.9, 0.0), "EPSG:32660")
        self.assertEqual(utm_epsg_from_lonlat(-180.0, 0.0), "EPSG:32601")
        self.assertEqual(utm_epsg_from_lonlat(-51.31, -30.02), "EPSG:32722")  # 巴西南部

        crs = utm_epsg_from_lonlat(85.0, 28.0)
        x, y = lonlat_to_utm(85.0, 28.0, dst_crs=crs)
        lon, lat = utm_to_lonlat(x, y, src_crs=crs)
        self.assertAlmostEqual(lon, 85.0, places=4)
        self.assertAlmostEqual(lat, 28.0, places=4)
        wrong_lon, _wrong_lat = utm_to_lonlat(x, y)  # 默认河南 49N，不能拿来反算尼泊尔
        self.assertGreater(abs(wrong_lon - 85.0), 1.0)

    def test_calibration_xml_skips_noise(self):
        import tempfile

        from src.sar import calibration_xml_candidates

        with tempfile.TemporaryDirectory() as td:
            cal = os.path.join(td, "annotation", "calibration")
            os.makedirs(cal)
            want = os.path.join(cal, "calibration-s1a-iw-grd-vv-x.xml")
            with open(os.path.join(cal, "noise-s1a-iw-grd-vv-x.xml"), "w", encoding="utf-8") as fh:
                fh.write("<noise/>")
            with open(want, "w", encoding="utf-8") as fh:
                fh.write("<calibration/>")
            hits = calibration_xml_candidates(td, "vv")
            self.assertEqual(hits, [want])

    def test_detect_water_sar_fixed_threshold(self):
        from src.sar import detect_water_sar

        rng = np.random.default_rng(0)
        db = rng.normal(-8.0, 1.5, (200, 200)).astype(np.float32)
        db[60:140, 60:140] = rng.normal(-22.0, 1.5, (80, 80)).astype(np.float32)
        mask, prob, meta = detect_water_sar(db, "VV", speckle=3, min_area_px=50)
        self.assertTrue(bool(mask[100, 100]), "低回波区应判为水体")
        self.assertFalse(bool(mask[5, 5]), "高回波区不应判为水体")
        self.assertAlmostEqual(meta["threshold_db"], -16.0, places=3)
        self.assertEqual(mask.shape, db.shape)

    def test_vh_default_threshold(self):
        from src.sar import detect_water_sar

        mask, _prob, meta = detect_water_sar(np.full((64, 64), -25.0, np.float32), "VH", min_area_px=10)
        self.assertAlmostEqual(meta["threshold_db"], -19.0, places=3)
        self.assertTrue(bool(mask.all()), "全图远低于 VH 阈值时应全判为水")

    def test_sar_change_finds_new_water(self):
        from src.sar import sar_change

        rng = np.random.default_rng(1)
        pre = rng.normal(-8.0, 1.2, (200, 200)).astype(np.float32)
        post = pre + rng.normal(0.0, 0.3, (200, 200)).astype(np.float32)
        post[80:120, 20:180] = rng.normal(-22.0, 1.2, (40, 160)).astype(np.float32)
        chg = sar_change(pre, post, "VV", drop_db=3.0, min_area_px=50)
        self.assertGreater(int(np.count_nonzero(chg["new"][85:115, 30:170])), 1000)
        self.assertLess(float(chg["new"].mean()), 0.2, "新增水体不应超过全图 20%")
        self.assertGreater(int(np.count_nonzero(chg["post_mask"])), int(np.count_nonzero(chg["pre_mask"])))

    def test_db_geotiff_roundtrip(self):
        from rasterio.transform import from_origin

        from src.sar import db_to_gray, read_db_geotiff, save_db_geotiff

        db = np.linspace(-25, 0, 64 * 64).reshape(64, 64).astype(np.float32)
        path = os.path.join(ROOT, "logs", "_unit_test_sar_db.tif")
        save_db_geotiff(path, db, from_origin(0, 64, 10, 10), "EPSG:32650")
        back, transform, crs = read_db_geotiff(path)
        self.assertEqual(back.shape, db.shape)
        self.assertLess(float(np.abs(back - db).max()), 0.01, "int16×100 往返误差应小于 0.01 dB")
        self.assertAlmostEqual(transform.a, 10.0, places=6)
        self.assertIn("32650", crs)
        gray = db_to_gray(db)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertLess(int(gray[0, 0]), int(gray[-1, -1]), "越弱回波应越黑（数值越小）")
        os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
