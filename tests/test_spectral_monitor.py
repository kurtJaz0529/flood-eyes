"""
慧眼识灾 · 多时相光谱监测回归测试
====================================

运行：
    python -m pytest tests/test_spectral_monitor.py -q

覆盖：
    · ndvi / savi / ndwi 公式
    · 严格真实波段：缺 NIR 的 RGB 不得用无 NIR 代理算 ndvi/ndwi
    · 云（SCL）/NoData 不产生假变化
    · 错位配准相对首景对齐（同 CRS 平移 + 跨 CRS）
    · 低于 min_valid_pct / 共同有效区为空 -> null，绝不写 0
    · 输出栅格 nodata、网格、面积（米制投影 km² / 地理坐标 null）
    · 日期只能用户显式提供，不从文件名臆测；输入 SHA256 记录
    · CLI 端到端与参数校验
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

from src import spectral_monitor as sm  # noqa: E402
from src.preprocess import Scene, load_scene  # noqa: E402

TRANSFORM = from_origin(500000.0, 3000000.0, 10.0, 10.0)
CRS_UTM = "EPSG:32650"
BAND_ORDER = ("blue", "green", "red", "nir")

#: 默认"陆地"反射率：ndvi = (0.6 - 0.2) / (0.6 + 0.2) = 0.5
LAND = {"blue": 0.10, "green": 0.15, "red": 0.20, "nir": 0.60}


# --------------------------------------------------------------------------
# 构造辅助
# --------------------------------------------------------------------------


def land_bands(shape):
    return {name: np.full(shape, value, np.float32) for name, value in LAND.items()}


def _write_scl(path, scl, transform, crs):
    import rasterio

    arr = np.asarray(scl, dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)


def write_scene(path, bands, *, transform=TRANSFORM, crs=CRS_UTM, nodata=None, scl=None):
    """写一个带真实 CRS/transform 的多波段 GeoTIFF（可带 SCL sidecar）。"""
    import rasterio

    names = [name for name in BAND_ORDER if name in bands]
    stack = np.stack([np.asarray(bands[name], dtype=np.float32) for name in names], axis=0)
    profile = {
        "driver": "GTiff",
        "height": int(stack.shape[1]),
        "width": int(stack.shape[2]),
        "count": len(names),
        "dtype": "float32",
        "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = float(nodata)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(stack)
        ds.descriptions = tuple(names)
    if scl is not None:
        _write_scl(os.path.splitext(str(path))[0] + "_scl.tif", scl, transform, crs)
    return str(path)


def _read_tif(path, band=1):
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(band)


# --------------------------------------------------------------------------
# 1. 公式与严格波段
# --------------------------------------------------------------------------


def test_index_formulas_match_spec():
    red = np.array([[0.2]], np.float32)
    nir = np.array([[0.6]], np.float32)
    green = np.array([[0.3]], np.float32)

    num, den = sm.index_terms({"red": red, "nir": nir}, "ndvi")
    assert num[0, 0] == pytest.approx(0.4, abs=1e-6)
    assert den[0, 0] == pytest.approx(0.8, abs=1e-6)
    assert sm.compute_index({"red": red, "nir": nir}, "ndvi")[0, 0] == pytest.approx(0.5, abs=1e-6)

    num, den = sm.index_terms({"red": red, "nir": nir}, "savi")
    assert num[0, 0] == pytest.approx(0.6, abs=1e-6)
    assert den[0, 0] == pytest.approx(1.3, abs=1e-6)
    assert sm.compute_index({"red": red, "nir": nir}, "savi")[0, 0] == pytest.approx(
        1.5 * 0.4 / 1.3, rel=1e-6
    )

    num, den = sm.index_terms({"green": green, "nir": nir}, "ndwi")
    assert num[0, 0] == pytest.approx(-0.3, abs=1e-6)
    assert den[0, 0] == pytest.approx(0.9, abs=1e-6)
    assert sm.compute_index({"green": green, "nir": nir}, "ndwi")[0, 0] == pytest.approx(
        -1.0 / 3.0, rel=1e-6
    )


def test_zero_denominator_is_not_a_value():
    zero = np.zeros((2, 2), np.float32)
    idx = sm.compute_index({"red": zero, "nir": zero}, "ndvi")
    assert np.isnan(idx).all(), "分母为零不得产出 0 这种'看似有效'的值"

    scene = Scene(bands={"red": zero.copy(), "nir": zero.copy()},
                  transform=TRANSFORM, crs=CRS_UTM, meta={})
    valid, _ = sm.scene_valid_mask(scene, "ndvi")
    assert not valid.any()
    data = sm.compute_masked_index(scene.bands, "ndvi", valid)
    assert (data == sm.INDEX_NODATA).all()


def test_strict_real_bands_reject_rgb_and_no_nir_proxy(tmp_path):
    shape = (6, 6)
    rgb_bands = {name: land_bands(shape)[name] for name in ("blue", "green", "red")}
    p3 = write_scene(tmp_path / "rgb3.tif", rgb_bands)

    # 既有 load_scene 会对 3 波段影像给出无 NIR 的代理风险，这里先确认它确实没有 NIR
    plain = load_scene(p3)
    assert plain.has_nir is False

    # 本模块必须拒绝，而不是退化成 green/red 代理
    for index in ("ndvi", "savi", "ndwi"):
        with pytest.raises(ValueError) as excinfo:
            sm.load_scenes([p3], index)
        assert "缺少" in str(excinfo.value)
        assert "nir" in str(excinfo.value)

    # 真正的 green+nir 两波段可以算 ndwi
    gnir = {"green": np.full(shape, 0.3, np.float32), "nir": np.full(shape, 0.6, np.float32)}
    p2 = write_scene(tmp_path / "green_nir.tif", gnir)
    scenes = sm.load_scenes([p2], "ndwi")
    assert set(scenes[0].bands) == {"green", "nir"}
    assert sm.compute_index(scenes[0].bands, "ndwi")[0, 0] == pytest.approx(-1.0 / 3.0, rel=1e-6)


def test_scene_valid_mask_applies_scl_classes_and_ignores_other_classes():
    shape = (2, 12)
    bands = {"red": np.full(shape, 0.2, np.float32), "nir": np.full(shape, 0.6, np.float32)}
    scl = np.tile(np.arange(12, dtype=np.uint8), (2, 1))
    scene = Scene(bands=bands, transform=TRANSFORM, crs=CRS_UTM, meta={"scl": scl})

    valid, scl_available = sm.scene_valid_mask(scene, "ndvi")
    assert scl_available is True
    for cls in sm.SCL_INVALID_CLASSES:
        assert not valid[0, int(cls)], f"SCL {cls} 应为无效类别"
    for cls in (2, 4, 5, 6, 7):
        assert valid[0, cls], f"SCL {cls} 应为有效类别"
    assert set(sm.SCL_INVALID_CLASSES) == {0, 1, 3, 8, 9, 10, 11}


# --------------------------------------------------------------------------
# 2. 云 / NoData 不产生假变化
# --------------------------------------------------------------------------


def test_cloud_and_nodata_do_not_create_fake_change(tmp_path):
    shape = (20, 20)
    cloud = (slice(2, 8), slice(2, 8))
    nodata = (slice(12, 18), slice(12, 18))

    a = land_bands(shape)
    scl_a = np.full(shape, 4, np.uint8)

    b = land_bands(shape)
    for name in b:  # 亮且"光谱平坦"的云，不加 SCL 就会变成假变化
        b[name][cloud] = 0.55
    for name in b:  # 无数据像元
        b[name][nodata] = -9999.0
    scl_b = np.full(shape, 4, np.uint8)
    scl_b[cloud] = 9  # CLOUD_HIGH_PROBABILITY

    pa = write_scene(tmp_path / "a.tif", a, scl=scl_a)
    pb = write_scene(tmp_path / "b.tif", b, nodata=-9999.0, scl=scl_b)

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    scene_b = summary["scenes"][1]
    assert scene_b["scl_available"] is True
    assert scene_b["status"] == "ok"

    pair = summary["changes"][0]
    assert pair["status"] == "ok"
    assert pair["mean_change"] == pytest.approx(0.0, abs=1e-6), "云/无数据不得造成假变化"

    common = _read_tif(pair["common_valid_tif"])
    assert (common[cloud] == 0).all()
    assert (common[nodata] == 0).all()
    assert common[0, 0] == 1

    idx_b = _read_tif(scene_b["index_tif"])
    assert (idx_b[cloud] == sm.INDEX_NODATA).all()
    assert (idx_b[nodata] == sm.INDEX_NODATA).all()
    assert idx_b[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_no_scl_sidecar_warns_but_still_runs(tmp_path):
    shape = (8, 8)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape))
    pb = write_scene(tmp_path / "b.tif", land_bands(shape))
    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    assert all(scene["scl_available"] is False for scene in summary["scenes"])
    assert summary["warnings"]


# --------------------------------------------------------------------------
# 3. 错位配准 -> 对齐到首景网格
# --------------------------------------------------------------------------


def test_misregistered_scene_aligned_to_first_grid(tmp_path):
    shape = (20, 20)
    big = (30, 30)
    tf_big = from_origin(499990.0, 3000010.0, 10.0, 10.0)  # 平移半格并放大覆盖

    pa = write_scene(tmp_path / "ref.tif", land_bands(shape),
                     transform=TRANSFORM, scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "shift.tif", land_bands(big),
                     transform=tf_big, scl=np.full(big, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    scene_b = summary["scenes"][1]
    assert scene_b["reprojected_to_reference"] is True
    assert (scene_b["width"], scene_b["height"]) == shape
    assert scene_b["transform"] == summary["reference_grid"]["transform"]

    pair = summary["changes"][0]
    assert pair["status"] == "ok"
    assert pair["common_valid_pct"] == pytest.approx(100.0, abs=0.1)
    assert pair["mean_change"] == pytest.approx(0.0, abs=1e-6)

    import rasterio

    with rasterio.open(summary["scenes"][0]["index_tif"]) as d0, \
            rasterio.open(scene_b["index_tif"]) as d1:
        assert d1.transform == d0.transform
        assert (d1.height, d1.width) == shape


def test_cross_crs_scene_reprojected_to_first_grid(tmp_path):
    shape = (20, 20)
    # 用首景四角反算经纬度范围，保证地理坐标场景覆盖参考网格
    xs = [500000.0, 500200.0, 500200.0, 500000.0]
    ys = [3000000.0, 3000000.0, 2999800.0, 2999800.0]
    lons, lats = warp_transform(CRS_UTM, "EPSG:4326", xs, ys)
    res = 0.0001
    lon0 = min(lons) - 0.002
    lat0 = max(lats) + 0.002
    width = int((max(lons) + 0.002 - lon0) / res) + 2
    height = int((lat0 - (min(lats) - 0.002)) / res) + 2

    pa = write_scene(tmp_path / "utm.tif", land_bands(shape),
                     transform=TRANSFORM, crs=CRS_UTM, scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "ll.tif", land_bands((height, width)),
                     transform=from_origin(lon0, lat0, res, res), crs="EPSG:4326",
                     scl=np.full((height, width), 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    scene_b = summary["scenes"][1]
    assert scene_b["reprojected_to_reference"] is True
    assert (scene_b["width"], scene_b["height"]) == shape
    assert scene_b["crs"] == str(CRS.from_user_input(CRS_UTM))

    pair = summary["changes"][0]
    assert pair["status"] == "ok"
    assert pair["common_valid_pixels"] > 0
    assert pair["mean_change"] == pytest.approx(0.0, abs=1e-4)


def test_missing_crs_or_transform_raises(tmp_path):
    shape = (6, 6)
    no_crs = write_scene(tmp_path / "nocrs.tif", land_bands(shape), crs=None,
                         scl=np.full(shape, 4, np.uint8))
    with pytest.raises(ValueError, match="CRS"):
        sm.run_monitor([no_crs], "ndvi", tmp_path / "out")

    ok = write_scene(tmp_path / "ok.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    good = sm.load_scenes([ok], "ndvi")[0]
    bad = Scene(bands=dict(good.bands), transform=None, crs=good.crs, path="bad")
    with pytest.raises(ValueError, match="transform"):
        sm.align_scenes([bad])


# --------------------------------------------------------------------------
# 4. 缺测报告 null，绝不写 0
# --------------------------------------------------------------------------


def test_below_min_valid_pct_reports_null_not_zero(tmp_path):
    shape = (20, 20)
    a = land_bands(shape)
    pa = write_scene(tmp_path / "a.tif", a, scl=np.full(shape, 4, np.uint8))

    b = land_bands(shape)
    mostly_nodata = (slice(0, 18), slice(0, 20))  # 只剩 10% 有效
    for name in b:
        b[name][mostly_nodata] = -9999.0
    pb = write_scene(tmp_path / "b.tif", b, nodata=-9999.0, scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=50.0)
    scene_b = summary["scenes"][1]
    assert scene_b["valid_pct"] < 50.0
    assert scene_b["status"] == "missing"
    assert scene_b["mean"] is None
    assert "null" in scene_b["reason"]

    pair = summary["changes"][0]
    assert pair["status"] == "insufficient"
    assert pair["mean_change"] is None
    assert pair["observable_area_km2"] is None

    # 落盘的指数栅格也不能把缺测写成 0
    idx_b = _read_tif(scene_b["index_tif"])
    assert (idx_b[mostly_nodata] == sm.INDEX_NODATA).all()
    assert (idx_b[mostly_nodata] != 0.0).all(), "缺测不能被写成 0"
    assert np.allclose(idx_b[18:, :], 0.5, atol=1e-6)


def test_empty_common_valid_reports_null(tmp_path):
    shape = (20, 20)
    a = land_bands(shape)
    a_nodata = (slice(0, 20), slice(10, 20))
    for name in a:
        a[name][a_nodata] = -9999.0

    b = land_bands(shape)
    b_nodata = (slice(0, 20), slice(0, 10))
    for name in b:
        b[name][b_nodata] = -9999.0

    pa = write_scene(tmp_path / "a.tif", a, nodata=-9999.0, scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", b, nodata=-9999.0, scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    pair = summary["changes"][0]
    assert pair["common_valid_pixels"] == 0
    assert pair["status"] == "insufficient"
    assert pair["mean_change"] is None
    assert pair["observable_area_km2"] is None
    assert "共同有效区为空" in pair["reason"]

    common = _read_tif(pair["common_valid_tif"])
    assert not common.any()
    diff = _read_tif(pair["change_tif"])
    assert (diff == sm.INDEX_NODATA).all()


# --------------------------------------------------------------------------
# 5. 输出栅格：nodata / 网格 / 面积
# --------------------------------------------------------------------------


def test_outputs_nodata_grid_and_area(tmp_path):
    shape = (20, 20)
    change_block = (slice(4, 10), slice(4, 10))
    nodata_block = (slice(14, 18), slice(14, 18))

    a = land_bands(shape)
    b = land_bands(shape)
    b["nir"][change_block] = 0.40  # ndvi 由 0.5 降到约 0.333
    for name in b:
        b[name][nodata_block] = -9999.0

    pa = write_scene(tmp_path / "a.tif", a, scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", b, nodata=-9999.0, scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)

    import rasterio

    for scene in summary["scenes"]:
        with rasterio.open(scene["index_tif"]) as ds:
            assert ds.dtypes[0] == "float32"
            assert ds.nodata == sm.INDEX_NODATA
            assert ds.crs == CRS.from_user_input(CRS_UTM)
            assert ds.transform.a == pytest.approx(10.0)
            assert ds.transform.e == pytest.approx(-10.0)
            assert (ds.height, ds.width) == shape
            assert ds.transform == rasterio.Affine(*summary["reference_grid"]["transform"])

    pair = summary["changes"][0]
    with rasterio.open(pair["change_tif"]) as ds:
        assert ds.dtypes[0] == "float32"
        assert ds.nodata == sm.INDEX_NODATA
        assert ds.crs == CRS.from_user_input(CRS_UTM)
        assert ds.transform.a == pytest.approx(10.0)
        diff = ds.read(1)

    with rasterio.open(pair["common_valid_tif"]) as ds:
        assert ds.dtypes[0] == "uint8"
        assert ds.nodata is None, "0 是真实无效类别，不能声明为 nodata"
        assert ds.transform.a == pytest.approx(10.0)
        common = ds.read(1)

    assert set(np.unique(common).tolist()) <= {0, 1}
    assert (common[nodata_block] == 0).all()
    assert (diff[nodata_block] == sm.INDEX_NODATA).all(), "缺测不能写成 0"
    assert diff[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert diff[change_block].mean() < 0.0

    assert summary["area"]["available"] is True
    assert summary["area"]["pixel_area_m2"] == pytest.approx(100.0)
    assert pair["observable_area_km2"] == pytest.approx(int(common.sum()) * 100.0 / 1e6)
    assert pair["observable_area_basis"] == "共同有效区像元数 × 投影网格像元面积"


def test_area_null_for_geographic_crs(tmp_path):
    shape = (10, 10)
    tf_ll = from_origin(85.0, 28.0, 0.0001, 0.0001)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape), transform=tf_ll,
                     crs="EPSG:4326", scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", land_bands(shape), transform=tf_ll,
                     crs="EPSG:4326", scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    assert summary["area"]["available"] is False
    assert summary["area"]["pixel_area_m2"] is None
    assert summary["area"]["note"]

    pair = summary["changes"][0]
    assert pair["status"] == "ok"
    assert pair["observable_area_km2"] is None
    assert pair["area_note"]


def test_metric_pixel_area_helper_rejects_non_metric():
    area, note = sm.metric_pixel_area_m2(TRANSFORM, CRS_UTM)
    assert area == pytest.approx(100.0)
    assert note

    for crs in ("EPSG:2229", None, "EPSG:4326"):
        area, note = sm.metric_pixel_area_m2(TRANSFORM, crs)
        assert area is None
        assert note

    area, note = sm.metric_pixel_area_m2(None, CRS_UTM)
    assert area is None
    assert note


def test_min_valid_pct_out_of_range(tmp_path):
    shape = (6, 6)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    for bad in (-1.0, 120.0, float("nan")):
        with pytest.raises(ValueError):
            sm.run_monitor([pa], "ndvi", tmp_path / "out", min_valid_pct=bad)


# --------------------------------------------------------------------------
# 6. 日期与输入指纹
# --------------------------------------------------------------------------


def test_dates_user_provided_or_unverified_never_guessed(tmp_path):
    shape = (8, 8)
    # 文件名故意带日期：绝不能被解析
    pa = write_scene(tmp_path / "S2_20200101.tif", land_bands(shape),
                     scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "S2_20200201.tif", land_bands(shape),
                     scl=np.full(shape, 4, np.uint8))

    s1 = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out1", min_valid_pct=5.0)
    assert s1["dates_user_provided"] is False
    assert s1["dates_unverified"] is True
    assert s1["date_source"] == "unverified"
    assert s1["dates_guessed_from_filename"] is False
    assert all(sc["date"] is None and sc["date_source"] == "unverified" for sc in s1["scenes"])

    s2 = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out2", min_valid_pct=5.0,
                        dates=["2020-01-01", "2020-02-01"])
    assert s2["dates_user_provided"] is True
    assert s2["dates_unverified"] is True  # 用户声明仍未与影像元数据核验
    assert s2["dates_verified_against_source"] is False
    assert s2["date_source"] == "user_provided"
    assert [sc["date"] for sc in s2["scenes"]] == ["2020-01-01", "2020-02-01"]
    assert all(sc["date_source"] == "user_provided" for sc in s2["scenes"])

    with pytest.raises(ValueError, match="日期数量"):
        sm.run_monitor([pa, pb], "ndvi", tmp_path / "out3", dates=["2020-01-01"])


def test_summary_records_sha256_paths_means_and_grid(tmp_path):
    shape = (8, 8)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    for scene in summary["scenes"]:
        expected = hashlib.sha256(open(scene["path"], "rb").read()).hexdigest()
        assert scene["sha256"] == expected
        assert len(scene["sha256"]) == 64
        assert scene["valid_pct"] == pytest.approx(100.0)
        assert scene["mean"] == pytest.approx(0.5, abs=1e-6)
        assert scene["path"] == os.path.abspath(scene["path"])

    payload = json.loads(open(summary["summary_json"], encoding="utf-8").read())
    assert payload["index"] == "ndvi"
    assert payload["formula"] == "(nir - red) / (nir + red)"
    assert payload["required_bands"] == ["nir", "red"]
    assert payload["min_valid_pct"] == 5.0
    assert payload["reference_grid"]["width"] == shape[1]
    assert payload["changes"][0]["common_valid_pct"] == pytest.approx(100.0)
    assert payload["changes"][0]["mean_change"] == pytest.approx(0.0, abs=1e-6)


def test_format_summary_text_marks_nulls(tmp_path):
    shape = (8, 8)
    a = land_bands(shape)
    pa = write_scene(tmp_path / "a.tif", a, scl=np.full(shape, 4, np.uint8))
    b = land_bands(shape)
    for name in b:
        b[name][:] = -9999.0
    pb = write_scene(tmp_path / "b.tif", b, nodata=-9999.0, scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "ndvi", tmp_path / "out", min_valid_pct=5.0)
    text = sm.format_summary_text(summary)
    assert "null" in text
    assert "均值 null" in text


# --------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------


def test_cli_main_end_to_end(tmp_path, capsys):
    cli = importlib.import_module("scripts.run_spectral")
    shape = (10, 10)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    out = tmp_path / "cli_out"

    rc = cli.main(["--index", "ndvi", "--images", pa, pb,
                   "--out-dir", str(out), "--min-valid-pct", "5"])
    assert rc == 0
    text = capsys.readouterr().out
    assert str(out / "summary.json") in text
    assert (out / "summary.json").is_file()


def test_cli_reports_errors_without_crashing(tmp_path, capsys):
    cli = importlib.import_module("scripts.run_spectral")
    shape = (8, 8)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))

    rc = cli.main(["--index", "ndvi", "--images", pa, pb,
                   "--out-dir", str(tmp_path / "o1"), "--dates", "2020-01-01"])
    assert rc == 1
    assert "日期数量" in capsys.readouterr().err

    rc = cli.main(["--index", "ndvi", "--images", str(tmp_path / "missing.tif"),
                   "--out-dir", str(tmp_path / "o2")])
    assert rc == 1
    assert "光谱监测失败" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cli.main(["--index", "evi", "--images", pa, "--out-dir", str(tmp_path / "o3")])


def test_cli_dates_flag_comma_separated(tmp_path):
    cli = importlib.import_module("scripts.run_spectral")
    shape = (8, 8)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    out = tmp_path / "o"
    rc = cli.main(["--index", "savi", "--images", pa, pb, "--out-dir", str(out),
                   "--dates", "2020-01-01,2020-02-01", "--min-valid-pct", "5"])
    assert rc == 0
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [s["date"] for s in payload["scenes"]] == ["2020-01-01", "2020-02-01"]
    assert payload["index"] == "savi"
    assert payload["formula"] == "1.5 * (nir - red) / (nir + red + 0.5)"


def test_one_meter_grid_shift_requires_reprojection(tmp_path):
    shape = (12, 12)
    first = write_scene(tmp_path / "first.tif", land_bands(shape))
    shifted = write_scene(tmp_path / "shifted.tif", land_bands(shape),
                          transform=from_origin(500000, 3000001, 10, 10))
    result = sm.run_monitor([first, shifted], "ndvi", tmp_path / "out", min_valid_pct=5)
    assert result["scenes"][1]["reprojected_to_reference"] is True
    assert result["scenes"][1]["transform"] == result["reference_grid"]["transform"]
    with __import__("rasterio").open(result["scenes"][1]["index_tif"]) as ds:
        assert ds.transform == TRANSFORM


def test_required_band_only_nodata_is_invalid_after_loading_and_alignment(tmp_path):
    shape = (12, 12)
    first = write_scene(tmp_path / "first.tif", land_bands(shape))
    second_bands = land_bands(shape)
    second_bands["nir"][5, 5] = -9999
    second = write_scene(tmp_path / "second.tif", second_bands, nodata=-9999,
                         transform=from_origin(500005, 3000000, 10, 10))
    loaded = sm.load_scenes([first, second], "ndvi")
    second_valid, _ = sm.scene_valid_mask(loaded[1], "ndvi")
    assert not second_valid[5, 5], "一个必需波段 NoData 就应无效"
    aligned = sm.align_scenes(loaded)
    aligned_valid, _ = sm.scene_valid_mask(aligned[1], "ndvi")
    assert np.allclose(aligned[1].bands["nir"][aligned_valid], 0.6, atol=1e-5), (
        "双线性重投影不能把无效 NIR=0 混进有效邻像元"
    )


def test_scl_tiff_grid_mismatch_rejected_and_hash_recorded(tmp_path):
    shape = (8, 8)
    first = write_scene(tmp_path / "first.tif", land_bands(shape),
                        scl=np.full(shape, 4, np.uint8))
    sidecar = tmp_path / "first_scl.tif"
    _write_scl(sidecar, np.full(shape, 9, np.uint8),
               from_origin(500001, 3000000, 10, 10), CRS_UTM)
    with pytest.raises(ValueError, match="SCL GeoTIFF 网格"):
        sm.run_monitor([first], "ndvi", tmp_path / "bad")

    _write_scl(sidecar, np.full(shape, 4, np.uint8), TRANSFORM, CRS_UTM)
    good = sm.run_monitor([first], "ndvi", tmp_path / "good")
    recorded = good["scenes"][0]["quality_sidecar"]
    assert recorded["path"] == os.path.abspath(sidecar)
    assert recorded["sha256"] == sm.sha256_file(sidecar)
    assert recorded["alignment"] == "verified_grid"


def test_common_valid_fraction_has_its_own_gate(tmp_path):
    shape = (20, 20)
    a, b = land_bands(shape), land_bands(shape)
    for band in a.values():
        band[:, 12:] = -9999  # 第一景 60% 有效
    for band in b.values():
        band[:, :8] = -9999  # 第二景 60% 有效；共同仅 20%
    first = write_scene(tmp_path / "first.tif", a, nodata=-9999)
    second = write_scene(tmp_path / "second.tif", b, nodata=-9999)
    result = sm.run_monitor([first, second], "ndvi", tmp_path / "out", min_valid_pct=50)
    assert all(scene["status"] == "ok" for scene in result["scenes"])
    pair = result["changes"][0]
    assert pair["common_valid_pct"] == pytest.approx(20)
    assert pair["status"] == "insufficient"
    assert pair["mean_change"] is None
    assert pair["observable_area_km2"] is None


def test_run_directory_guard_and_dates_validation(tmp_path):
    shape = (8, 8)
    first = write_scene(tmp_path / "first.tif", land_bands(shape))
    second = write_scene(tmp_path / "second.tif", land_bands(shape))
    out = tmp_path / "out"
    result = sm.run_monitor([first, second], "ndvi", out,
                            dates=["2020-01-01", "2020-02-01"])
    previous = (out / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="输出目录"):
        sm.run_monitor([first, second], "savi", out)
    assert (out / "summary.json").read_bytes() == previous
    assert result["dates_verified_against_source"] is False
    for dates in (["2020-02-01", "2020-01-01"], ["2020-13-01", "2020-02-01"]):
        with pytest.raises(ValueError, match="日期|严格递增"):
            sm.run_monitor([first, second], "ndvi", tmp_path / "invalid", dates=dates)
