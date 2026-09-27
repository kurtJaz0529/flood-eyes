"""Download a reproducible, bounded optical pilot from the official public GCS bucket.

Keep event grouping, real -1/0/1 labels, source hashes and L1C provenance.
This is a pilot dataset, not regional accuracy certification.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.io import MemoryFile

BUCKET = "https://storage.googleapis.com/sen1floods11/"
PREFIX = "v1.1/data/flood_events/HandLabeled/"


def download(key):
    if not key.startswith(PREFIX) or ".." in key.split("/"):
        raise ValueError("Unexpected dataset object")
    with urllib.request.urlopen(BUCKET + urllib.parse.quote(key, safe="/"), timeout=45) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Dataset object exceeds 32 MiB limit")
    return data


def prepare(key, out):
    base = Path(key).name.removesuffix("_S2Hand.tif")
    label_key = PREFIX + "LabelHand/" + base + "_LabelHand.tif"
    raw_image, raw_label = download(key), download(label_key)
    image_path, label_path = out / (base + "_S2Hand.tif"), out / (base + "_LabelHand.tif")
    with MemoryFile(raw_image) as imem, MemoryFile(raw_label) as lmem:
        with imem.open() as image, lmem.open() as label:
            if image.count != 13 or label.count != 1:
                raise ValueError("Unexpected official image/label bands")
            tolerance = min(abs(image.transform.a), abs(image.transform.e)) * 1e-6
            if (image.crs != label.crs or not image.transform.almost_equals(label.transform, precision=tolerance)
                    or image.shape != label.shape):
                raise ValueError("Official image/label grids differ")
            gt = label.read(1)
            if not set(np.unique(gt).tolist()) <= {-1, 0, 1}:
                raise ValueError("Unexpected official label values")
            # Explicit B2/B3/B4/B8; L1C DN scale is 10000. No BOA offset heuristic.
            arr = image.read([2, 3, 4, 8]).astype(np.float32) / 10000.0
            valid = image.read_masks([2, 3, 4, 8]).all(axis=0) & np.isfinite(arr).all(axis=0)
            arr[:, ~valid] = -9999
            profile = image.profile.copy()
            profile.update(count=4, dtype="float32", nodata=-9999, compress="deflate")
            with rasterio.open(image_path, "w", **profile) as dst:
                dst.write(arr)
                dst.descriptions = ("B02", "B03", "B04", "B08")
                dst.update_tags(product_level="L1C", reflectance_units="TOA", dataset="Sen1Floods11-v1.1")
            profile.update(count=1, dtype="int16", nodata=-1)
            with rasterio.open(label_path, "w", **profile) as dst:
                dst.write(gt.astype(np.int16), 1)
    return {"id": base, "event": base.split("_")[0], "image": image_path.name,
            "label": label_path.name, "source_image": BUCKET+key, "source_label": BUCKET+label_key,
            "image_source_sha256": hashlib.sha256(raw_image).hexdigest(),
            "label_source_sha256": hashlib.sha256(raw_label).hexdigest(),
            "prepared_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "prepared_label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "valid_pixels": int((gt >= 0).sum()), "water_pixels": int((gt == 1).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sen1floods11/pilot")
    parser.add_argument("--chips-per-event", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.chips_per_event <= 20:
        parser.error("chips-per-event must be 1..20 for a bounded pilot")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.tif")):
        parser.error("Output already contains TIFF files; use a new directory")
    url = BUCKET + "?" + urllib.parse.urlencode({"prefix": PREFIX+"S2Hand/", "max-keys": 1000})
    with urllib.request.urlopen(url, timeout=30) as response:
        root = ET.fromstring(response.read(3_000_000))
    if any(x.tag.endswith("}IsTruncated") and x.text == "true" for x in root.iter()):
        raise RuntimeError("Dataset catalog is truncated")
    keys = sorted(x.text for x in root.iter() if x.tag.endswith("}Key") and x.text.endswith("_S2Hand.tif"))
    groups = {}
    for key in keys:
        groups.setdefault(Path(key).name.split("_")[0], []).append(key)
    selected = []
    for event, candidates in sorted(groups.items()):
        selected.extend(random.Random(f"{args.seed}:{event}").sample(candidates, min(len(candidates), args.chips_per_event)))
    manifest = {"dataset": "Sen1Floods11 v1.1", "source": "https://github.com/cloudtostreet/Sen1Floods11",
                "input_product": "Sentinel-2 L1C TOA", "seed": args.seed, "pilot_only": True, "samples": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(prepare, key, out) for key in selected]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            manifest["samples"].append(item)
            manifest["samples"].sort(key=lambda x: x["id"])
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{len(manifest['samples'])}/{len(selected)} {item['id']} water={item['water_pixels']}", flush=True)
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
