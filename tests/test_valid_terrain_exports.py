"""
任务 B 回归：单景有效区 / 双时相共同有效区变化统计 / GIS 成果导出 / 地形筛查
=============================================================================

运行：
    python -m pytest tests/test_valid_terrain_exports.py -q

覆盖的既有缺陷：
    · compare 直接比较两个布尔水体掩膜，云下/无数据被当成"陆地"，
      于是"灾前云、灾后水"变成假新增，"灾前水、灾后云"变成假退水。
    · 没有有效观测时仍输出 0.00 km²，容易被误读成"无洪水"。
    · 成果包只有 PNG，没有带 CRS/transform 的 GIS GeoTIFF。
    · 元数据里的栅格数组被 default=str 字符串化，stats.json 不可解析。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import Affine, from_origin  # noqa: E402

from src import FloodDetector, FloodResult, postprocess, report, terrain  # noqa: E402
from src.preprocess import Scene  # noqa: E402

TRANSFORM = from_origin(500000.0, 3000000.0, 10.0, 10.0)
CRS_UTM = "EPSG:32650"


# --------------------------------------------------------------------------
# 构造辅助
# --------------------------------------------------------------------------


def _land(shape, value: float = 0.2, nir: float = 0.25):
    return {
        "blue": np.full(shape, value, np.float32),
        "green": np.full(shape, value, np.float32),
        "red": np.full(shape, value, np.float32),
        "nir": np.full(shape, nir, np.float32),
    }


def _paint(bands, block, value, nir=None):
    for name, arr in bands.items():
        arr[block] = value
    if nir is not None:
        bands["nir"][block] = nir


def _scene(bands, transform=TRANSFORM, crs=CRS_UTM, nodata=None, pixel_size_m=10.0):
    return Scene(
        bands=bands,
        transform=transform,
        crs=crs,
        pixel_size_m=pixel_size_m,
        nodata_mask=nodata,
        meta={},
    )


def _detector(**kwargs):
    base = dict(mode="baseline", mask_clouds=False, min_area_px=0,
                open_radius=0, close_radius=0, fill_holes=False)
    base.update(kwargs)
    return FloodDetector(**base)


def _write_tif(path, arr, transform=TRANSFORM, crs=CRS_UTM, nodata=None):
    import rasterio

    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = float(nodata)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype(np.float32), 1)
    return str(path)


# --------------------------------------------------------------------------
# 1. 共同有效区：未知不等于陆地
# --------------------------------------------------------------------------


def test_water_under_before_cloud_is_not_new():
    shape = (20, 20)
    before = np.zeros(shape, bool)
    after = np.zeros(shape, bool)
    after[2:6, 2:6] = True
    before_valid = np.ones(shape, bool)
    after_valid = np.ones(shape, bool)
    before_valid[2:6, 2:6] = False  # 灾前该处被云遮挡 -> 不可观测
    common = before_valid & after_valid

    c = postprocess.change_stats(before, after, 10.0, valid_mask=common)
    assert c["new_water_km2"] == 0.0
    assert c["masks"]["new"].sum() == 0
    assert c["masks"]["unknown"][2:6, 2:6].all()

    # 反证：不传共同有效掩膜时，同一场景会被算成假新增
    naive = postprocess.change_stats(before, after, 10.0)
    assert naive["new_water_km2"] > 0.0


def test_water_under_after_cloud_is_not_receded():
    shape = (20, 20)
    before = np.zeros(shape, bool)
    before[5:9, 5:9] = True
    after = np.zeros(shape, bool)
    after_valid = np.ones(shape, bool)
    after_valid[5:9, 5:9] = False  # 灾后该处被云遮挡 -> 不可观测

    c = postprocess.change_stats(before, after, 10.0, valid_mask=after_valid)
    assert c["receded_water_km2"] == 0.0
    assert c["masks"]["receded"].sum() == 0
    assert postprocess.change_stats(before, after, 10.0)["receded_water_km2"] > 0.0


def test_no_common_valid_is_data_insufficient_and_summary_says_so():
    shape = (8, 8)
    c = postprocess.change_stats(
        np.zeros(shape, bool), np.ones(shape, bool), 10.0, valid_mask=np.zeros(shape, bool)
    )
    assert c["quality_status"] == "data_insufficient"
    assert c["common_valid_pixels"] == 0
    assert c["new_water_km2"] == 0.0

    result = FloodResult(
        mask=np.zeros(shape, bool),
        prob=np.zeros(shape, np.float32),
        rgb=np.zeros(shape + (3,), np.uint8),
        overlay=np.zeros(shape + (3,), np.uint8),
        stats={"quality_status": "data_insufficient", "pixel_size_m": 10.0},
        meta={},
        change={"quality_status": "data_insufficient"},
    )
    text = result.summary_text()
    assert "数据不足" in text
    assert "检测到水体面积" not in text, "无有效观测时不得输出 0.00 km² 的淹没结论"


def test_change_map_marks_unknown_neutral_gray():
    before = np.zeros((6, 6), bool)
    after = np.zeros((6, 6), bool)
    valid = np.ones((6, 6), bool)
    valid[0, :] = False
    rgb = postprocess.change_map_rgb(before, after, np.zeros((6, 6, 3), np.uint8), valid_mask=valid)
    assert (rgb[0] == 128).all()


def test_compare_ignores_water_under_cloud_both_directions():
    shape = (32, 32)
    block = (slice(4, 14), slice(4, 14))

    cloud = _land(shape)
    _paint(cloud, block, 0.5)  # 亮且光谱平坦 -> 云
    water = _land(shape)
    _paint(water, block, 0.05, nir=0.01)  # 高 NDWI -> 水

    det = _detector(mask_clouds=True)
    # 灾前云、灾后水：不得算作新增
    fwd = det.compare(
        before_scene=_scene(cloud), after_scene=_scene(water)
    )
    assert fwd.meta["comparison"]["reprojected"] is False
    assert fwd.change["new_water_km2"] == 0.0, "云下未知区被当成了新增淹没"
    assert fwd.change["masks"]["unknown"][block].all()
    assert fwd.change["common_valid_pixels"] > 0
    assert fwd.change["quality_status"] == "ok"

    # 灾前水、灾后云：不得算作退水
    rev = det.compare(before_scene=_scene(water), after_scene=_scene(cloud))
    assert rev.change["receded_water_km2"] == 0.0, "云下未知区被当成了退水"


def test_single_scene_all_nodata_reports_data_insufficient():
    shape = (16, 16)
    scene = _scene(_land(shape), nodata=np.ones(shape, bool))
    result = _detector().detect_scene(scene)
    assert result.valid_mask is not None and not result.valid_mask.any()
    assert result.stats["quality_status"] == "data_insufficient"
    text = result.summary_text()
    assert "数据不足" in text
    assert "检测到水体面积" not in text


# --------------------------------------------------------------------------
# 2. 成果包：GIS GeoTIFF 往返与 ZIP
# --------------------------------------------------------------------------


def _compare_result_with_terrain(shape=(32, 32)):
    before_bands = _land(shape)
    after_bands = _land(shape)
    _paint(before_bands, (slice(4, 12), slice(4, 12)), 0.05, nir=0.01)   # 持续水
    _paint(after_bands, (slice(4, 12), slice(4, 12)), 0.05, nir=0.01)
    _paint(after_bands, (slice(4, 12), slice(16, 24)), 0.05, nir=0.01)  # 新增水
    nodata = np.zeros(shape, bool)
    nodata[24:30, 24:30] = True  # 灾后无数据 -> 水体/有效区/变化三份栅格都应为无效

    result = _detector().compare(
        before_scene=_scene(before_bands),
        after_scene=_scene(after_bands, nodata=nodata),
    )
    risk = np.zeros(shape, np.uint8)
    risk[0:4, :] = 1
    risk[28:, :] = 255
    result.meta["terrain"] = terrain.terrain_context("plain")
    result.meta["terrain_risk_mask"] = risk
    result.meta["terrain_risk_summary"] = {
        "dem_assessed": True,
        "screening_threshold_deg": 15.0,
        "risk_pct_of_assessed": 12.5,
        "warnings": ["坡度标记仅供人工复核。"],
    }
    return result, nodata


def test_export_bundle_writes_gis_tiffs_and_zip():
    pytest.importorskip("reportlab")
    import rasterio

    result, nodata = _compare_result_with_terrain()
    with tempfile.TemporaryDirectory() as td:
        out = report.export_bundle(result, out_dir=td, basename="gis_test", include_pdf=True)
        files = out["files"]
        for key in ("mask", "overlay", "rgb", "heatmap", "comparison",
                    "water_mask_tif", "valid_mask_tif", "change_tif", "terrain_risk_tif",
                    "stats", "report"):
            assert key in files, f"缺少导出项 {key}"
            assert os.path.isfile(files[key]), key
        assert out["gis"]["skipped"] is False

        with rasterio.open(files["water_mask_tif"]) as ds:
            assert ds.crs == CRS.from_user_input(result.scene.crs)
            assert ds.transform.a == pytest.approx(10.0) and ds.transform.e == pytest.approx(-10.0)
            assert (ds.height, ds.width) == result.scene.shape
            assert ds.nodata == 255
            water = ds.read(1)
            assert set(np.unique(water).tolist()) <= {0, 1, 255}
            assert (water[nodata] == 255).all(), "无数据区必须写 255"

        with rasterio.open(files["valid_mask_tif"]) as ds:
            assert ds.nodata is None, "0=无效是真实类别，不能声明为 nodata"
            valid = ds.read(1)
            assert set(np.unique(valid).tolist()) <= {0, 1}
            assert (valid[nodata] == 0).all()

        with rasterio.open(files["change_tif"]) as ds:
            assert ds.nodata == 255
            change = ds.read(1)
            assert set(np.unique(change).tolist()) <= {0, 1, 2, 3, 255}
            assert 2 in set(np.unique(change).tolist()), "应检出新增淹没像元"
            assert (change[nodata] == 255).all(), "共同有效区外必须写 255"

        with rasterio.open(files["terrain_risk_tif"]) as ds:
            assert ds.nodata == 255
            assert set(np.unique(ds.read(1)).tolist()) <= {0, 1, 255}

        with zipfile.ZipFile(out["zip"]) as zf:
            names = set(zf.namelist())
        for name in ("water_mask.tif", "valid_mask.tif", "change.tif",
                     "terrain_risk.tif", "overlay.png", "water_mask.png", "report.pdf"):
            assert name in names, f"ZIP 缺少 {name}"


def test_stats_json_excludes_raster_arrays():
    result, _ = _compare_result_with_terrain()
    with tempfile.TemporaryDirectory() as td:
        out = report.export_bundle(result, out_dir=td, basename="json_test", include_pdf=False)
        with open(out["files"]["stats"], encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["terrain"]["profile"] == "plain"
        assert "terrain_risk_mask" not in payload["meta"], "栅格数组不得写进 JSON"
        assert "masks" not in payload.get("change", {})
        text = json.dumps(payload, ensure_ascii=False)
        assert "terrain_risk_mask" not in text
        assert len(text) < 100_000, "stats.json 不应出现被字符串化的整幅数组"

    text = result.to_json()
    assert "terrain_risk_mask" not in text
    assert '"masks"' not in text
    json.loads(text)  # 必须是可解析的 JSON


def test_export_without_georef_skips_gis_with_warning():
    shape = (12, 12)
    scene = _scene(_land(shape), transform=None, crs=None)
    result = _detector().detect_scene(scene)
    with tempfile.TemporaryDirectory() as td:
        out = report.export_bundle(result, out_dir=td, basename="nogeo", include_pdf=False)
        assert "water_mask_tif" not in out["files"], "缺少地理参考时不得伪造 GeoTIFF"
        assert out["gis"]["skipped"] is True
        assert out["gis"]["warnings"]
        assert os.path.isfile(out["files"]["overlay"])
        with open(out["files"]["stats"], encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["gis_export"]["warnings"]


# --------------------------------------------------------------------------
# 3. 地形背景与 DEM 坡度筛查
# --------------------------------------------------------------------------


def test_terrain_context_profiles_and_no_mask_mutation():
    shape = (16, 16)
    bands = _land(shape)
    _paint(bands, (slice(4, 10), slice(4, 10)), 0.05, nir=0.01)
    result = _detector().detect_scene(_scene(bands))
    mask_before = result.mask.copy()

    for profile in ("unspecified", "plain", "hilly", "mountain", "urban", "coastal"):
        ctx = terrain.terrain_context(profile)
        assert ctx["profile"] == profile
        assert ctx["dem_assessed"] is False
        assert ctx["warnings"], f"{profile} 应有背景提示"
        json.dumps(ctx, ensure_ascii=False)  # JSON 安全

    assert terrain.terrain_context()["profile"] == "unspecified"
    with pytest.raises(ValueError):
        terrain.terrain_context("volcano")

    assert np.array_equal(result.mask, mask_before), "地形背景不得改动水体掩膜"


def _dem_scene(shape=(20, 20), transform=TRANSFORM, crs=CRS_UTM):
    return _scene(_land(shape), transform=transform, crs=crs)


def test_assess_dem_flat_is_low_and_slope_is_risk(tmp_path):
    shape = (20, 20)
    flat = np.full(shape, 100.0, np.float32)
    flat_path = _write_tif(tmp_path / "flat.tif", flat)
    risk, summary = terrain.assess_dem(flat_path, _dem_scene(shape))
    assert risk.dtype == np.uint8
    assert set(np.unique(risk).tolist()) == {0}
    assert summary["dem_assessed"] is True
    assert summary["risk_pixels"] == 0
    assert summary["coverage_pct"] == 100.0
    assert summary["slope_max_deg"] == pytest.approx(0.0, abs=1e-3)
    assert "warnings" in summary
    json.dumps(summary, ensure_ascii=False)

    # 10 m 高差 / 10 m 像元 = 45°，应全部标记为需复核
    plane = np.tile(np.arange(shape[1]) * 10.0, (shape[0], 1)).astype(np.float32)
    slope_path = _write_tif(tmp_path / "slope.tif", plane)
    risk2, summary2 = terrain.assess_dem(slope_path, _dem_scene(shape), slope_threshold_deg=15.0)
    assert int(risk2.min()) == 1
    assert summary2["slope_max_deg"] == pytest.approx(45.0, abs=0.5)
    assert summary2["risk_pct_of_assessed"] == 100.0


def test_assess_dem_nodata_and_outside_are_unassessed(tmp_path):
    shape = (20, 20)
    z = np.full(shape, 100.0, np.float32)
    z[0:5, :] = -9999.0
    path = _write_tif(tmp_path / "dem_nd.tif", z, nodata=-9999.0)
    risk, summary = terrain.assess_dem(path, _dem_scene(shape))
    assert (risk[0:5, :] == 255).all(), "DEM 无数据区不得被当成 0° 缓坡"
    assert risk[10, 10] == 0
    assert summary["coverage_pct"] < 100.0
    assert summary["unassessed_pixels"] > 0

    # DEM 只覆盖影像左上角：范围外必须未评估
    small = np.full((10, 10), 100.0, np.float32)
    small_path = _write_tif(tmp_path / "small.tif", small, transform=from_origin(500000.0, 3000000.0, 10.0, 10.0))
    risk_small, summary_small = terrain.assess_dem(small_path, _dem_scene(shape))
    assert (risk_small[15:, 15:] == 255).all(), "DEM 覆盖范围外不得被当成缓坡"
    assert summary_small["assessed_pixels"] < shape[0] * shape[1]


def test_assess_dem_rejects_bad_grid_and_threshold(tmp_path):
    shape = (10, 10)
    path = _write_tif(tmp_path / "dem.tif", np.full(shape, 100.0, np.float32))

    ll_scene = _dem_scene(shape, transform=from_origin(85.0, 28.0, 0.0001, 0.0001), crs="EPSG:4326")
    with pytest.raises(ValueError):
        terrain.assess_dem(path, ll_scene)

    rotated = _dem_scene(shape, transform=Affine(0.0, -10.0, 0.0, 10.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        terrain.assess_dem(path, rotated)

    with pytest.raises(ValueError):
        terrain.assess_dem(path, _dem_scene(shape), slope_threshold_deg=120.0)
    with pytest.raises(ValueError):
        terrain.assess_dem(path, _dem_scene(shape), slope_threshold_deg=float("nan"))
    with pytest.raises(FileNotFoundError):
        terrain.assess_dem(str(tmp_path / "missing.tif"), _dem_scene(shape))


def test_assess_dem_rejects_unreadable_file(tmp_path):
    bad = tmp_path / "not_a_dem.tif"
    bad.write_text("not a raster", encoding="utf-8")
    with pytest.raises(ValueError):
        terrain.assess_dem(str(bad), _dem_scene((8, 8)))
