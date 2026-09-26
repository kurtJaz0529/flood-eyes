"""Synthetic regressions of scene rules, not claims of real-world accuracy."""
from dataclasses import replace
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.baseline import predict
from src.infer import FloodDetector
from src.preprocess import Scene, load_scene, reproject_scene_to, same_geo_grid
from src.pipeline import run_pipeline
from src.task_contracts import PipelineRequest

NAMES = ("blue", "green", "red", "nir", "swir1", "swir2")
PATCH = (slice(16, 48), slice(16, 48))
BASE = dict(lon=116, lat=29, pre_start="2020-01-01", pre_end="2020-01-03",
            post_start="2020-02-01", post_end="2020-02-03", size=64)


def scene(water=True):
    bands = {k: np.full((64, 64), v, np.float32) for k, v in
             zip(NAMES, (.08, .10, .12, .30, .25, .20))}
    if water:
        for k, v in zip(NAMES, (.03, .08, .02, .02, .005, .005)):
            bands[k][PATCH] = v
    return Scene(bands, from_origin(500000, 3200000, 10, 10), "EPSG:32650",
                 meta={"scl": np.full((64, 64), 4, np.uint8)})


def detector(profile="plain", **kwargs):
    return FloodDetector(mode="baseline", detection_strategy="adaptive",
                         terrain_profile=profile, **kwargs)


def write_scene(path, sc):
    with rasterio.open(path, "w", driver="GTiff", count=len(sc.bands),
                       height=sc.height, width=sc.width, dtype="float32",
                       crs=sc.crs, transform=sc.transform, nodata=-9999) as ds:
        ds.write(np.stack(list(sc.bands.values())))
        ds.descriptions = tuple(sc.bands)
    return str(path)


def dem(path, rise=0, hole=False):
    z = np.tile(np.arange(64, dtype=np.float32) * rise, (64, 1))
    if hole:
        z[PATCH] = -9999
    with rasterio.open(path, "w", driver="GTiff", count=1, height=64,
                       width=64, dtype="float32", crs="EPSG:32650",
                       transform=scene().transform, nodata=-9999) as ds:
        ds.write(z, 1)
    return str(path)


@pytest.mark.parametrize("profile,index", [("plain", "ndwi"), ("hilly", "ndwi"),
    ("mountain", "ndwi"), ("urban", "mndwi"), ("coastal", "mndwi"),
    ("wetland", "ndwi"), ("arid", "mndwi")])
def test_recipes_detect_clear_open_water(profile, index):
    result = detector(profile).detect_scene(scene())
    assert result.meta["adaptive"]["water_index"] == index
    assert result.mask[30, 30]
    assert not result.mask[0, 0]


def test_urban_swir_suppresses_ndwi_built_up_false_positive():
    sc = scene(False)
    sc.bands["green"][PATCH] = .1
    sc.bands["nir"][PATCH] = .04
    sc.bands["swir1"][PATCH] = .2
    ndwi = detector("plain", index_threshold=0).detect_scene(sc)
    urban = detector("urban", index_threshold=0).detect_scene(sc)
    assert ndwi.mask[30, 30]
    assert not urban.mask[30, 30]


def test_shadow_conflict_is_unknown_not_dry():
    sc = scene()
    for k, v in zip(NAMES, (.001, .02, .005, .001, .019, .15)):
        sc.bands[k][PATCH] = v
    result = detector("urban", index_threshold=0).detect_scene(sc)
    assert result.meta["review_mask"][30, 30] == 1
    assert not result.valid_mask[30, 30]
    assert not result.mask[30, 30]


@pytest.mark.parametrize("hole,code", [(False, 2), (True, 4)])
def test_dem_candidate_review_never_becomes_change(tmp_path, hole, code):
    path = dem(tmp_path / "dem.tif", rise=10, hole=hole)
    result = detector("mountain", dem_path=path).compare(
        before_scene=scene(False), after_scene=scene())
    assert result.meta["review_mask"][30, 30] == code
    assert not result.change["masks"]["common_valid"][30, 30]
    assert result.change["new_water_km2"] == 0
    assert result.stats["review_area_km2"] > 0


def test_narrow_wetland_channels_survive():
    sc = scene(False)
    for k, v in zip(NAMES, (.03, .08, .02, .02, .005, .005)):
        sc.bands[k][10:54, 30:32] = v
    assert detector("wetland").detect_scene(sc).mask[30, 30]
    assert not FloodDetector(mode="baseline").detect_scene(sc).mask[30, 30]


def test_pair_uses_same_available_index_and_records_degradation():
    pre = scene()
    del pre.bands["swir1"]
    del pre.bands["swir2"]
    result = detector("coastal").compare(before_scene=pre, after_scene=scene())
    assert result.meta["adaptive"]["water_index"] == "ndwi"
    assert result.meta["before_adaptive"]["water_index"] == "ndwi"
    assert any("降级" in text for text in result.meta["warnings"])
    assert result.change["new_water_km2"] == 0
    with pytest.raises(ValueError, match="swir1"):
        detector("urban", water_index="mndwi").detect_scene(pre)


def test_cloud_or_swir_nodata_not_false_flood():
    pre, post = scene(False), scene()
    pre.meta["scl"][PATCH] = 3
    result = detector("urban").compare(before_scene=pre, after_scene=post)
    assert result.change["new_water_km2"] == 0
    post.meta["spectral_band_valid"] = {"swir1": np.ones(post.shape, bool)}
    post.meta["spectral_band_valid"]["swir1"][PATCH] = False
    output = detector("urban").detect_scene(post)
    assert output.meta["review_mask"][30, 30] == 255
    assert not output.valid_mask[30, 30]


def test_metric_rectangular_pixel_area_and_nonmetric_rejected():
    sc = replace(scene(), transform=from_origin(500000, 3200000, 10, 20))
    result = detector("urban").detect_scene(sc)
    assert result.area_km2 == pytest.approx(result.mask.sum() * .0002)
    with pytest.raises(ValueError, match="米制"):
        detector().detect_scene(replace(sc, crs="EPSG:4326"))


def test_description_or_explicit_preset_required_for_swir(tmp_path):
    path = write_scene(tmp_path / "six.tif", scene())
    loaded = load_scene(path)
    assert set(loaded.bands) == set(NAMES)
    assert loaded.channel_names == list(NAMES[:4])  # original model tensor stays 4ch
    with pytest.raises(ValueError, match="不匹配"):
        from src.preprocess import resolve_band_order
        resolve_band_order(4, "s2_6band")


def test_shifted_swir_nodata_does_not_pollute_neighbours():
    src = scene()
    src.bands["swir1"][20:24, 20:24] = -9999
    src.meta["spectral_band_valid"] = {"swir1": src.bands["swir1"] != -9999}
    ref = replace(scene(), transform=from_origin(500001, 3200000, 10, 10))
    assert not same_geo_grid(src, ref)
    aligned = reproject_scene_to(src, ref)
    good = aligned.meta["spectral_band_valid"]["swir1"]
    assert aligned.bands["swir1"][good].min() >= .0049
    assert not good[21, 21]


def test_threshold_fit_ignores_invalid_values_and_constant_gate():
    values = np.tile(np.linspace(-.2, .4, 10), (10, 1)).astype(np.float32)
    invalid = np.zeros(values.shape, bool)
    invalid[:5] = True
    dirty = values.copy()
    dirty[invalid] = 9000
    _, _, first = predict(values, nodata_mask=invalid)
    _, _, second = predict(dirty, nodata_mask=invalid)
    assert first["threshold_global"] == second["threshold_global"]
    mask, _, _ = predict(np.full((10, 10), .5), nir=np.full((10, 10), .8))
    assert not mask.any()
    mask, _, _ = predict(np.full((10, 10), .5), fixed_threshold=.8)
    assert not mask.any()


def test_request_identity_covers_recipe_and_calibration():
    req = PipelineRequest(**BASE, detection_strategy="adaptive")
    for kw in (dict(water_index="mndwi"), dict(index_threshold=.2),
               dict(terrain_profile="wetland"), dict(slope_threshold_deg=5),
               dict(band_order="s2_6band")):
        assert replace(req, **kw).cache_key != req.cache_key
    with pytest.raises(ValueError):
        replace(req, index_threshold=float("nan"))


def test_pipeline_exports_review_and_preserves_grid(tmp_path):
    pre = write_scene(tmp_path / "pre.tif", scene(False))
    post = write_scene(tmp_path / "post.tif", scene())
    request = PipelineRequest(**BASE, terrain_profile="mountain", detection_strategy="adaptive")
    result = run_pipeline(request, str(tmp_path / "out"), local_pair=(pre, post),
                          dem_path=dem(tmp_path / "dem.tif", rise=10))
    assert result["meta"]["terrain"]["scope"] == "user_declared_adaptive_recipe"
    assert result["meta"]["adaptive"]["review_pixels"] > 0
    assert "review_mask" not in result["meta"]
    for key, value in (("water_mask_tif", 255), ("review_mask_tif", 2), ("change_tif", 255)):
        with rasterio.open(result["files"][key]) as ds:
            assert ds.transform == scene().transform
            assert ds.crs == rasterio.crs.CRS.from_epsg(32650)
            assert ds.read(1)[30, 30] == value
    assert "before_review_mask_tif" in result["files"]
    json.dumps(result["meta"], allow_nan=False)


def test_adaptive_rejects_misaligned_scl_and_sidecar_changes_identity(tmp_path):
    from src.pipeline import _build_identity
    path = write_scene(tmp_path / "pre.tif", scene())
    req = PipelineRequest(**BASE, detection_strategy="adaptive")
    identity = _build_identity(req, (path, path), None, False)
    scl_path = tmp_path / "pre_scl.tif"
    with rasterio.open(scl_path, "w", driver="GTiff", count=1, height=64,
                       width=64, dtype="uint8", crs="EPSG:32650",
                       transform=from_origin(500001, 3200000, 10, 10)) as ds:
        ds.write(np.full((64, 64), 4, np.uint8), 1)
    assert _build_identity(req, (path, path), None, False) != identity
    with pytest.raises(ValueError, match="SCL.*网格"):
        detector().detect(path)


def test_invalid_only_scene_and_nir_proxy_are_rejected():
    sc = scene()
    sc.meta["scl"][:] = 9
    result = detector("urban").detect_scene(sc)
    assert result.stats["quality_status"] == "data_insufficient"
    assert not result.valid_mask.any()
    sc = scene()
    del sc.bands["nir"]
    with pytest.raises(ValueError, match="真实"):
        detector().detect_scene(sc)
