"""标签编码 / NoData / 事件划分的回归测试。

覆盖三处易错语义：
- ``_read_mask`` 必须把 -1 保留为 ignore，而不是当成 0/1 水体；
- ``DiceBCELoss`` 对 ignore 像素不产生梯度，且全 ignore 时不报错；
- ``segmentation_metrics`` 与 ``split_samples`` 按事件划分、拒绝重叠与单事件。

真实函数签名取自 train.py 与 src/model_unet.py。
"""
import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin

from src.model_unet import DiceBCELoss, segmentation_metrics
from train import _read_mask, split_samples


# --------------------------------------------------------------------------
# _read_mask
# --------------------------------------------------------------------------


def _write_tif(path, array, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", count=1, height=array.shape[0], width=array.shape[1],
        dtype=array.dtype.name, nodata=nodata, crs="EPSG:32650",
        transform=from_origin(500000, 3200000, 10, 10),
    ) as ds:
        ds.write(array, 1)


def test_read_mask_tiff_keeps_ignore_label(tmp_path):
    """TIFF 的 -1 必须是 ignore，不随 nodata 声明与否而变成水体/陆地。"""
    arr = np.array([[-1, 0, 1]], dtype=np.int16)
    expected = [[-1, 0, 1]]

    without_nodata = tmp_path / "label_plain.tif"
    _write_tif(without_nodata, arr, nodata=None)
    out = _read_mask(str(without_nodata))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)

    with_nodata = tmp_path / "label_nodata.tif"
    _write_tif(with_nodata, arr, nodata=-1)
    np.testing.assert_array_equal(_read_mask(str(with_nodata)), expected)


def test_read_mask_png255_maps_to_binary(tmp_path):
    """PNG 的 0/255 自动识别为 png255，映射成 0/1 且不产生 ignore。"""
    path = tmp_path / "label.png"
    Image.fromarray(np.array([[0, 255, 0]], dtype=np.uint8)).save(path)
    out = _read_mask(str(path))
    np.testing.assert_array_equal(out, [[0, 1, 0]])
    assert out.min() >= 0  # 纯 0/255 的 PNG 不应有 ignore 像素


# --------------------------------------------------------------------------
# DiceBCELoss
# --------------------------------------------------------------------------


def test_dice_bce_zero_gradient_on_ignored_pixels():
    torch.manual_seed(0)
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0], [-1.0, -1.0]]]])

    loss = DiceBCELoss()(logits, targets)
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    loss.backward()

    grad = logits.grad
    assert grad is not None and torch.isfinite(grad).all()
    # targets 第二行为 -1（ignore），该行不得贡献梯度
    assert torch.count_nonzero(grad[0, 0, 1, :]) == 0


def test_dice_bce_all_ignored_is_safe():
    logits = torch.randn(2, 1, 3, 3, requires_grad=True)
    targets = torch.full((2, 1, 3, 3), -1.0)

    loss = DiceBCELoss()(logits, targets)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


# --------------------------------------------------------------------------
# segmentation_metrics
# --------------------------------------------------------------------------


def test_segmentation_metrics_ignores_ignore_label():
    pred = np.array([[1, 1, 1]])
    target = np.array([[1, -1, 0]])
    m = segmentation_metrics(pred, target)

    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1.0, 1.0, 0.0, 0.0)
    assert m["iou"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    # 与手工剔除 -1 后的结果一致
    reference = segmentation_metrics(np.array([[1, 1]]), np.array([[1, 0]]))
    assert m["iou"] == pytest.approx(reference["iou"])
    assert m["precision"] == pytest.approx(reference["precision"])


# --------------------------------------------------------------------------
# split_samples
# --------------------------------------------------------------------------


def _items(*groups):
    return [{"id": f"{g}_{i}", "group": g} for g in groups for i in range(2)]


def test_split_samples_events_stay_disjoint():
    items = _items("event_a", "event_b", "event_c")
    train, val, test = split_samples(items, val_groups="event_a", test_groups="event_c")

    def groups(part):
        return {i["group"] for i in part}

    assert groups(train) == {"event_b"}
    assert groups(val) == {"event_a"}
    assert groups(test) == {"event_c"}
    # 训练/验证/测试事件两两不重叠，且无样本丢失
    assert not (groups(train) & groups(val))
    assert not (groups(train) & groups(test))
    assert not (groups(val) & groups(test))
    assert len(train) + len(val) + len(test) == len(items)


def test_split_samples_rejects_single_event():
    with pytest.raises(ValueError, match="distinct nonempty"):
        split_samples(_items("event_a"))


def test_split_samples_rejects_overlapping_groups():
    with pytest.raises(ValueError, match="overlapping"):
        split_samples(_items("event_a", "event_b"), val_groups="event_a", test_groups="event_a")


def test_experimental_real_weights_require_explicit_unet(tmp_path):
    from src.infer import FloodDetector
    path = tmp_path / "pilot.pt"
    torch.save({"meta": {"data_note": "real labelled pilot", "auto_eligible": False}}, path)
    assert FloodDetector(mode="auto", weights=str(path)).resolved_mode == "baseline"
    assert FloodDetector(mode="unet", weights=str(path)).resolved_mode == "unet"


def test_nonfinite_prediction_is_not_silently_ignored():
    with pytest.raises(ValueError, match="Nonfinite"):
        segmentation_metrics(np.array([float("nan")]), np.array([1]))


def test_dataset_checks_geo_grid_and_excludes_nodata_from_stats(tmp_path):
    from train import FloodDataset, compute_stats
    image = tmp_path / "case_S2Hand.tif"
    label = tmp_path / "case_LabelHand.tif"
    transform = from_origin(500000, 3200000, 10, 10)
    array = np.full((4, 4, 4), .2, np.float32)
    array[:, 0, 0] = -9999
    with rasterio.open(image, "w", driver="GTiff", count=4, height=4, width=4,
                       dtype="float32", nodata=-9999, crs="EPSG:32650", transform=transform) as ds:
        ds.write(array)
        ds.descriptions = ("B02", "B03", "B04", "B08")
    _write_tif(label, np.ones((4, 4), np.int16), nodata=-1)
    items = [{"id": "case", "image": str(image), "mask": str(label)}]
    _, mask = FloodDataset(items)._load(0)
    assert mask[0, 0] == -1
    mean, _ = compute_stats(items)
    np.testing.assert_allclose(mean, [.2] * 4, atol=1e-6)
    with rasterio.open(label, "r+") as ds:
        ds.transform = from_origin(500010, 3200000, 10, 10)
    with pytest.raises(ValueError, match="grid"):
        FloodDataset(items)._load(0)
