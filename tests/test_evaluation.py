import json
import zipfile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from src.evaluation import evaluate_water


def write(path, data, transform=None):
    data = np.asarray(data, dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", count=1, width=data.shape[1],
                       height=data.shape[0], dtype="uint8", crs="EPSG:32650", nodata=255,
                       transform=transform or from_origin(500000, 3200000, 10, 10)) as ds:
        ds.write(data, 1)
    return str(path)


def test_metrics_report_coverage_and_unknown_water(tmp_path):
    ref = write(tmp_path / "ref.tif", [[1, 1, 1, 0, 0, 255]])
    pred = write(tmp_path / "pred.tif", [[1, 0, 255, 1, 0, 1]])
    m = evaluate_water(pred, ref, "urban")
    assert [m[k] for k in ("tp", "fp", "fn", "tn")] == [1, 1, 1, 1]
    assert m["f1"] == .5
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["classification_coverage"] == .8
    assert m["recall"] == .5
    assert m["recall_over_all_labelled_water"] == pytest.approx(1 / 3)
    assert m["unclassified_reference_water_pixels"] == 1
    json.dumps(m, allow_nan=False)


def test_no_label_coverage_is_missing_and_shift_rejected(tmp_path):
    ref = write(tmp_path / "ref.tif", [[1, 0]])
    pred = write(tmp_path / "pred.tif", [[255, 255]])
    m = evaluate_water(pred, ref)
    assert m["status"] == "data_insufficient"
    assert m["f1"] is None
    shifted = write(tmp_path / "shift.tif", [[1, 0]], from_origin(500001, 3200000, 10, 10))
    with pytest.raises(ValueError, match="transform"):
        evaluate_water(shifted, ref)


def test_spectral_ui_uses_same_monitor_and_packages_outputs(tmp_path):
    from app.spectral import run_spectral_ui
    from test_adaptive_flood import scene, write_scene
    paths = [write_scene(tmp_path / (tag + ".tif"), scene()) for tag in ("pre", "post")]
    text, archive = run_spectral_ui(paths, "ndmi", "2020-01-01 2020-02-01",
                                    "s2_6band", 50, tmp_path / "results")
    assert "NDMI" in text.upper()
    with zipfile.ZipFile(archive) as zf:
        assert "summary.json" in zf.namelist()
        result = json.loads(zf.read("summary.json"))
        assert result["index"] == "ndmi"
        assert result["changes"][0]["mean_change"] == 0
        assert any(name.endswith(".tif") for name in zf.namelist())


def test_ui_and_batch_accept_adaptive_inputs(tmp_path, monkeypatch):
    from app import main, automation
    from test_adaptive_flood import BASE, scene, write_scene
    paths = [write_scene(tmp_path / (tag + ".tif"), scene()) for tag in ("pre", "post")]
    payload = automation._validate_row({**BASE, "detection_strategy": "adaptive",
        "terrain_profile": "arid", "water_index": "mndwi", "index_threshold": "0.2",
        "slope_threshold_deg": "12", "local_pre": paths[0], "local_post": paths[1]})
    assert payload["request"]["index_threshold"] == .2
    demo = main.build_ui(baseline_only=True)
    components = demo.get_config_file()["components"]
    labels = {item.get("props", {}).get("label") for item in components}
    assert "识别策略" in labels
    assert "监测方向" in labels
    demo.close()
