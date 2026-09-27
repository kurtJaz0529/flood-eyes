"""Evaluate a checkpoint on complete labelled chips, including ignored pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model_unet import load_checkpoint, predict_tiled, segmentation_metrics
from train import FloodDataset, discover_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, meta = load_checkpoint(args.weights, device="cpu")
    groups = set(args.groups.split(","))
    items = [i for i in discover_samples(args.data) if i["group"] in groups]
    if not items or groups != {i["group"] for i in items}:
        raise ValueError("Missing requested groups")
    if set(i["id"] for i in items) & set(meta.get("train_scenes", [])):
        raise ValueError("Evaluation groups overlap training scenes")
    dataset = FloodDataset(items)
    totals = dict.fromkeys(("tp", "fp", "fn", "tn"), 0.0)
    rows = []
    for index, item in enumerate(items):
        image, labels = dataset._load(index)
        prob, _, _ = predict_tiled(model, image, tile=512, overlap=64, device="cpu",
                                  mean=meta["mean"], std=meta["std"])
        metrics = segmentation_metrics(prob, labels)
        for key in totals:
            totals[key] += metrics[key]
        rows.append({"id": item["id"], "metrics": metrics,
                     "valid_pixels": int((labels >= 0).sum()), "pixels": int(labels.size)})
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    report = {"weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              "groups": sorted(groups), "threshold": 0.5, "tile": 512, "tta": False,
              "data_note": meta.get("data_note"), "counts": totals,
              "iou": tp / max(tp + fp + fn, 1), "dice": 2*tp / max(2*tp + fp + fn, 1),
              "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
              "valid_pixels": int(tp + fp + fn + tn), "samples": rows}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("groups", "iou", "dice", "precision", "recall")}))


if __name__ == "__main__":
    main()
