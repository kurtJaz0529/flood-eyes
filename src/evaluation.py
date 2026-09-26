"""Evaluate classified water against independent binary labels on an exact grid."""
import numpy as np

from .spectral_monitor import sha256_file
from .terrain import terrain_context


def evaluate_water(prediction_path, reference_path, terrain_profile="unspecified"):
    import rasterio
    terrain_context(terrain_profile)
    with rasterio.open(prediction_path) as pred, rasterio.open(reference_path) as ref:
        if (pred.count != 1 or ref.count != 1 or pred.crs is None or ref.crs is None
                or pred.crs != ref.crs or pred.transform != ref.transform
                or pred.shape != ref.shape):
            raise ValueError("预测与真值必须为相同 CRS、transform、尺寸的单波段栅格；不自动移动真值")
        for ds in (pred, ref):
            if ds.nodata not in (None, 255):
                raise ValueError("分类栅格只允许 0=非水、1=水、255=未知，NoData 必须为255或未设置")
        p, r = pred.read(1), ref.read(1)
        if not np.isin(p, (0, 1, 255)).all() or not np.isin(r, (0, 1, 255)).all():
            raise ValueError("分类值必须为 0/1/255")
        labelled = (r != 255) & (ref.read_masks(1) > 0)
        classified = (p != 255) & (pred.read_masks(1) > 0)
    common = labelled & classified
    tp = int(np.sum(common & (p == 1) & (r == 1)))
    fp = int(np.sum(common & (p == 1) & (r == 0)))
    fn = int(np.sum(common & (p == 0) & (r == 1)))
    tn = int(np.sum(common & (p == 0) & (r == 0)))
    def ratio(n, d):
        return n / d if d else None
    water_labels = int(np.sum(labelled & (r == 1)))
    return {
        "terrain_profile": terrain_profile,
        "status": "ok" if common.any() else "data_insufficient",
        "prediction_sha256": sha256_file(prediction_path),
        "reference_sha256": sha256_file(reference_path),
        "labelled_pixels": int(labelled.sum()), "evaluated_pixels": int(common.sum()),
        "unclassified_labelled_pixels": int(np.sum(labelled & ~classified)),
        "classification_coverage": ratio(int(common.sum()), int(labelled.sum())),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
        "reference_water_pixels": water_labels,
        "unclassified_reference_water_pixels": int(np.sum(labelled & ~classified & (r == 1))),
        "recall_over_all_labelled_water": ratio(tp, water_labels),
        "note": "precision/recall/F1/IoU仅在共同有效区计算，必须同时报告覆盖率；未知水体仍计入全标注水体召回率的分母。需独立真值，合成样本不代表真实精度。",
    }
