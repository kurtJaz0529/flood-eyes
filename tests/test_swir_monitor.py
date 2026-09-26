"""
慧眼识灾 · SWIR 光谱指数（MNDWI / NDMI / NBR）针对性测试
=========================================================

运行：
    python -m pytest tests/test_swir_monitor.py -q

覆盖：
    · mndwi / ndmi / nbr 手算公式与分子分母
    · 缺真实 SWIR（四波段影像 / auto 解析）必须报错，不得用代理波段
    · 六波段 B2 B3 B4 B8 B11 B12 GeoTIFF 用 s2_6band 预设往返监测
    · 单个 SWIR 波段 NoData 不产生假变化（不写成 0 或假指数）
    · summary 公式、所需波段与差值符号：统一"后减前"，NBR 不是传统 dNBR

本文件复用 tests/test_spectral_monitor.py 的构造辅助，不修改既有测试。
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np
import pytest
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rasterio.crs import CRS  # noqa: E402

from src import spectral_monitor as sm  # noqa: E402
from src.preprocess import BAND_ORDERS, load_scene  # noqa: E402

try:  # 复用既有 fixtures（pytest 可能以不同导入模式加载旧测试）
    from tests.test_spectral_monitor import (  # noqa: E402
        CRS_UTM,
        TRANSFORM,
        _read_tif,
        _write_scl,
        land_bands,
        write_scene,
    )
except ImportError:  # pragma: no cover - 兼容顶层模块导入模式
    from test_spectral_monitor import (  # type: ignore[no-redef]  # noqa: E402
        CRS_UTM,
        TRANSFORM,
        _read_tif,
        _write_scl,
        land_bands,
        write_scene,
    )

#: 六波段预设 s2_6band：B2 B3 B4 B8 B11 B12 -> blue/green/red/nir/swir1/swir2
SIX_ORDER = ("blue", "green", "red", "nir", "swir1", "swir2")
SIX_DESCRIPTIONS = ("B2", "B3", "B4", "B8", "B11", "B12")
SIX_PRESET = "s2_6band"

#: 手算用基准反射率（真实 SWIR 反射率量级）
LAND6 = {
    "blue": 0.08,
    "green": 0.20,
    "red": 0.15,
    "nir": 0.40,
    "swir1": 0.30,
    "swir2": 0.10,
}

#: 指数 -> (公式文本, 所需波段, 基准值手算结果)
EXPECTED = {
    "mndwi": ("(green - swir1) / (green + swir1)", ["green", "swir1"],
              (0.20 - 0.30) / (0.20 + 0.30)),
    "ndmi": ("(nir - swir1) / (nir + swir1)", ["nir", "swir1"],
             (0.40 - 0.30) / (0.40 + 0.30)),
    "nbr": ("(nir - swir2) / (nir + swir2)", ["nir", "swir2"],
            (0.40 - 0.10) / (0.40 + 0.10)),
}


# --------------------------------------------------------------------------
# 构造辅助（六波段 GeoTIFF）
# --------------------------------------------------------------------------


def six_bands(shape):
    return {name: np.full(shape, value, np.float32) for name, value in LAND6.items()}


def write_six_band(path, bands, *, transform=TRANSFORM, crs=CRS_UTM, nodata=None,
                   scl=None, descriptions=SIX_DESCRIPTIONS):
    """写一个六波段 B2 B3 B4 B8 B11 B12 GeoTIFF（描述默认写真实 S2 波段名）。"""
    stack = np.stack([np.asarray(bands[name], dtype=np.float32) for name in SIX_ORDER], axis=0)
    profile = {
        "driver": "GTiff",
        "height": int(stack.shape[1]),
        "width": int(stack.shape[2]),
        "count": len(SIX_ORDER),
        "dtype": "float32",
        "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = float(nodata)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(stack)
        if descriptions:
            ds.descriptions = tuple(descriptions)
    if scl is not None:
        _write_scl(os.path.splitext(str(path))[0] + "_scl.tif", scl, transform, crs)
    return str(path)


# --------------------------------------------------------------------------
# 1. 公式与预设契约
# --------------------------------------------------------------------------


def test_swir_index_formulas_match_spec():
    green = np.array([[0.20]], np.float32)
    nir = np.array([[0.40]], np.float32)
    swir1 = np.array([[0.30]], np.float32)
    swir2 = np.array([[0.10]], np.float32)

    num, den = sm.index_terms({"green": green, "swir1": swir1}, "mndwi")
    assert num[0, 0] == pytest.approx(-0.10, abs=1e-6)
    assert den[0, 0] == pytest.approx(0.50, abs=1e-6)
    assert sm.compute_index({"green": green, "swir1": swir1}, "mndwi")[0, 0] == pytest.approx(
        -0.2, abs=1e-6
    )

    num, den = sm.index_terms({"nir": nir, "swir1": swir1}, "ndmi")
    assert num[0, 0] == pytest.approx(0.10, abs=1e-6)
    assert den[0, 0] == pytest.approx(0.70, abs=1e-6)
    assert sm.compute_index({"nir": nir, "swir1": swir1}, "ndmi")[0, 0] == pytest.approx(
        0.10 / 0.70, rel=1e-6
    )

    num, den = sm.index_terms({"nir": nir, "swir2": swir2}, "nbr")
    assert num[0, 0] == pytest.approx(0.30, abs=1e-6)
    assert den[0, 0] == pytest.approx(0.50, abs=1e-6)
    assert sm.compute_index({"nir": nir, "swir2": swir2}, "nbr")[0, 0] == pytest.approx(
        0.6, abs=1e-6
    )

    for index, (formula, bands, _) in EXPECTED.items():
        assert sm.INDEX_FORMULAS[index] == formula
        assert list(sm.REQUIRED_BANDS[index]) == bands
        assert index in sm.supported_indices()


def test_s2_6band_preset_contract():
    assert SIX_PRESET in BAND_ORDERS, "preprocess 需提供显式预设 s2_6band（B2 B3 B4 B8 B11 B12）"
    mapping = BAND_ORDERS[SIX_PRESET]
    assert mapping["blue"] == 0 and mapping["green"] == 1 and mapping["red"] == 2
    assert mapping["nir"] == 3 and mapping["swir1"] == 4 and mapping["swir2"] == 5


# --------------------------------------------------------------------------
# 2. 缺真实 SWIR 必须失败
# --------------------------------------------------------------------------


def test_missing_swir_bands_raise_for_four_band_scene(tmp_path):
    shape = (6, 6)
    four = write_scene(tmp_path / "four.tif", land_bands(shape))

    plain = load_scene(four)
    assert set(plain.bands) == {"blue", "green", "red", "nir"}
    assert "swir1" not in plain.bands and "swir2" not in plain.bands

    for index in ("mndwi", "ndmi", "nbr"):
        with pytest.raises(ValueError) as excinfo:
            sm.load_scenes([four], index)
        message = str(excinfo.value)
        assert "缺少" in message
        assert "swir" in message


def test_six_band_auto_without_descriptions_does_not_fake_swir(tmp_path):
    shape = (6, 6)
    # 六波段但无波段描述：auto 只能按通道数推断，绝不能凭空造出 swir1/swir2
    six = write_six_band(tmp_path / "six_nodesc.tif", six_bands(shape), descriptions=None)

    auto = load_scene(six, band_order="auto")
    assert "swir1" not in auto.bands and "swir2" not in auto.bands

    for index in ("mndwi", "ndmi", "nbr"):
        with pytest.raises(ValueError, match="swir"):
            sm.load_scenes([six], index)


def test_s2_6band_preset_exposes_real_swir(tmp_path):
    shape = (6, 6)
    six = write_six_band(tmp_path / "six.tif", six_bands(shape))

    preset = load_scene(six, band_order=SIX_PRESET)
    assert preset.bands["swir1"][0, 0] == pytest.approx(LAND6["swir1"], abs=1e-6)
    assert preset.bands["swir2"][0, 0] == pytest.approx(LAND6["swir2"], abs=1e-6)
    assert sm.compute_index(preset.bands, "ndmi")[0, 0] == pytest.approx(
        EXPECTED["ndmi"][2], rel=1e-6
    )


def test_compute_index_directly_requires_real_swir():
    green = np.full((2, 2), 0.2, np.float32)
    nir = np.full((2, 2), 0.4, np.float32)
    for index in ("mndwi", "ndmi", "nbr"):
        with pytest.raises(ValueError) as excinfo:
            sm.compute_index({"green": green, "nir": nir}, index)
        assert "swir" in str(excinfo.value)


# --------------------------------------------------------------------------
# 3. 六波段 GeoTIFF 往返监测
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index", ["mndwi", "ndmi", "nbr"])
def test_six_band_roundtrip_monitor(tmp_path, index):
    shape = (16, 16)
    formula, bands, expected_mean = EXPECTED[index]

    pa = write_six_band(tmp_path / "pre.tif", six_bands(shape),
                        scl=np.full(shape, 4, np.uint8))
    pb = write_six_band(tmp_path / "post.tif", six_bands(shape),
                        scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], index, tmp_path / "out",
                             min_valid_pct=5.0, band_order=SIX_PRESET)

    assert summary["index"] == index
    assert summary["formula"] == formula
    assert summary["required_bands"] == bands

    for scene in summary["scenes"]:
        assert scene["status"] == "ok"
        assert scene["valid_pct"] == pytest.approx(100.0)
        assert scene["mean"] == pytest.approx(expected_mean, abs=1e-5)
        with rasterio.open(scene["index_tif"]) as ds:
            assert ds.count == 1
            assert ds.dtypes[0] == "float32"
            assert ds.nodata == sm.INDEX_NODATA
            assert ds.crs == CRS.from_user_input(CRS_UTM)
            assert ds.transform == TRANSFORM
            assert (ds.height, ds.width) == shape
            assert np.allclose(ds.read(1), expected_mean, atol=1e-5)

    pair = summary["changes"][0]
    assert pair["status"] == "ok"
    assert pair["common_valid_pct"] == pytest.approx(100.0)
    assert pair["mean_change"] == pytest.approx(0.0, abs=1e-6)

    payload = json.loads(open(summary["summary_json"], encoding="utf-8").read())
    assert payload["formula"] == formula
    assert payload["required_bands"] == bands


# --------------------------------------------------------------------------
# 4. 单个 SWIR 波段 NoData 不产生假变化
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index,missing_band", [
    ("mndwi", "swir1"),
    ("ndmi", "swir1"),
    ("nbr", "swir2"),
])
def test_single_swir_band_nodata_creates_no_fake_change(tmp_path, index, missing_band):
    shape = (20, 20)
    hole = (slice(4, 10), slice(4, 10))

    a = six_bands(shape)
    b = six_bands(shape)
    b[missing_band][hole] = -9999.0  # 只有该 SWIR 波段缺测，其余波段正常

    pa = write_six_band(tmp_path / "pre.tif", a, scl=np.full(shape, 4, np.uint8))
    pb = write_six_band(tmp_path / "post.tif", b, nodata=-9999.0,
                        scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], index, tmp_path / "out",
                             min_valid_pct=5.0, band_order=SIX_PRESET)

    scene_b = summary["scenes"][1]
    assert scene_b["status"] == "ok"
    assert scene_b["valid_pct"] == pytest.approx(100.0 * (shape[0] * shape[1] - 36)
                                                 / (shape[0] * shape[1]), abs=1e-6)

    # 缺测像元不得写出任何"看似有效"的指数值
    idx_b = _read_tif(scene_b["index_tif"])
    assert (idx_b[hole] == sm.INDEX_NODATA).all()
    assert not np.isclose(idx_b[hole], 0.0).any(), "缺测不能写成 0"
    assert not np.isclose(idx_b[hole], 1.0).any(), "缺测不能写成假指数 1"

    pair = summary["changes"][0]
    assert pair["status"] == "ok"
    assert pair["mean_change"] == pytest.approx(0.0, abs=1e-6), "单波段缺测不得造成假变化"

    common = _read_tif(pair["common_valid_tif"])
    assert (common[hole] == 0).all()
    assert common[0, 0] == 1

    diff = _read_tif(pair["change_tif"])
    assert (diff[hole] == sm.INDEX_NODATA).all()
    assert not np.isclose(diff[hole], 0.0).any()


# --------------------------------------------------------------------------
# 5. summary 公式与差值符号（后减前；NBR 不是传统 dNBR）
# --------------------------------------------------------------------------


def test_summary_states_later_minus_earlier_and_nbr_is_not_dnbr(tmp_path):
    shape = (12, 12)
    pre = six_bands(shape)
    post = six_bands(shape)
    post["swir2"][:] = 0.20  # NBR 由 0.6 降到 0.2

    pa = write_six_band(tmp_path / "pre.tif", pre, scl=np.full(shape, 4, np.uint8))
    pb = write_six_band(tmp_path / "post.tif", post, scl=np.full(shape, 4, np.uint8))

    summary = sm.run_monitor([pa, pb], "nbr", tmp_path / "out",
                             min_valid_pct=5.0, band_order=SIX_PRESET)

    assert summary["formula"] == "(nir - swir2) / (nir + swir2)"
    assert summary["required_bands"] == ["nir", "swir2"]
    assert summary["change_direction"] == "later_minus_earlier"
    assert "后减前" in summary["change_direction_note"]
    assert "dNBR" in summary["change_direction_note"]
    assert "火前" in summary["change_direction_note"]
    assert summary["fire_severity_grades_provided"] is False
    assert summary["drought_area_provided"] is False

    # NBR：pre = 0.6，post = 0.2 -> 后减前 = -0.4
    nbr_pre = (0.40 - 0.10) / (0.40 + 0.10)
    nbr_post = (0.40 - 0.20) / (0.40 + 0.20)
    pair = summary["changes"][0]
    assert pair["status"] == "ok"
    assert pair["mean_change"] == pytest.approx(nbr_post - nbr_pre, abs=1e-6)
    assert pair["mean_change"] < 0.0
    # 传统 dNBR = 火前 - 火后，符号相反，不得混用
    assert pair["mean_change"] == pytest.approx(-(nbr_pre - nbr_post), abs=1e-6)
    assert summary["dnbr_note"] and "火前" in summary["dnbr_note"]

    diff = _read_tif(pair["change_tif"])
    assert np.allclose(diff, nbr_post - nbr_pre, atol=1e-6)

    text = sm.format_summary_text(summary)
    assert "后减前" in text
    assert "dNBR" in text


def test_non_nbr_summary_has_no_dnbr_note(tmp_path):
    shape = (8, 8)
    pa = write_six_band(tmp_path / "a.tif", six_bands(shape))
    pb = write_six_band(tmp_path / "b.tif", six_bands(shape))
    summary = sm.run_monitor([pa, pb], "ndmi", tmp_path / "out",
                             min_valid_pct=5.0, band_order=SIX_PRESET)
    assert "dnbr_note" not in summary
    assert summary["change_direction"] == "later_minus_earlier"
    assert summary["fire_severity_grades_provided"] is False
    assert summary["drought_area_provided"] is False


# --------------------------------------------------------------------------
# 6. CLI：SWIR 指数可直接用现有入口执行
# --------------------------------------------------------------------------


def test_cli_runs_swir_index_with_preset(tmp_path, capsys):
    cli = importlib.import_module("scripts.run_spectral")
    shape = (8, 8)
    pa = write_six_band(tmp_path / "a.tif", six_bands(shape), scl=np.full(shape, 4, np.uint8))
    pb = write_six_band(tmp_path / "b.tif", six_bands(shape), scl=np.full(shape, 4, np.uint8))
    out = tmp_path / "cli_out"

    rc = cli.main(["--index", "nbr", "--band-order", SIX_PRESET,
                   "--images", pa, pb, "--out-dir", str(out), "--min-valid-pct", "5"])
    assert rc == 0
    assert (out / "summary.json").is_file()
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["index"] == "nbr"
    assert payload["required_bands"] == ["nir", "swir2"]
    assert payload["change_direction"] == "later_minus_earlier"


def test_cli_fails_without_real_swir(tmp_path, capsys):
    cli = importlib.import_module("scripts.run_spectral")
    shape = (8, 8)
    pa = write_scene(tmp_path / "a.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))
    pb = write_scene(tmp_path / "b.tif", land_bands(shape), scl=np.full(shape, 4, np.uint8))

    rc = cli.main(["--index", "ndmi", "--images", pa, pb,
                   "--out-dir", str(tmp_path / "out"), "--min-valid-pct", "5"])
    assert rc == 1
    assert "swir" in capsys.readouterr().err
