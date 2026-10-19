"""Load and prepare the radar images and labels."""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 256
DB_MIN, DB_MAX = -35.0, 5.0


def load_manifest():
    path = ROOT / "data" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run python -m src.prepare_data from the floodscope folder first.")
    return json.loads(path.read_text())


def image_path(row):
    group, layer = ("HandLabeled", "S1Hand") if row["label_source"] == "manual" else ("WeaklyLabeled", "S1Weak")
    return ROOT / "data/raw/data/flood_events" / group / layer / row["image"]


def label_path(row):
    group, layer = ("HandLabeled", "LabelHand") if row["label_source"] == "manual" else ("WeaklyLabeled", "S2IndexLabelWeak")
    return ROOT / "data/raw/data/flood_events" / group / layer / row["label"]


def read_radar(path, size=IMAGE_SIZE):
    with rasterio.open(path) as source:
        if source.count != 2:
            raise ValueError("Expected two Sentinel-1 bands in VV, VH order.")
        radar = source.read(out_shape=(2, size, size), resampling=Resampling.average, masked=True)
        radar = radar.filled(np.nan).astype(np.float32)
        valid = np.isfinite(radar).all(axis=0)
        profile = source.profile.copy()
        profile.update(height=size, width=size,
                       transform=source.transform * source.transform.scale(source.width / size, source.height / size))
    radar = np.nan_to_num(radar, nan=DB_MIN, posinf=DB_MAX, neginf=DB_MIN)
    return radar, valid, profile


def normalize(radar):
    return ((np.clip(radar, DB_MIN, DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)).astype(np.float32)


def read_label(path, size=IMAGE_SIZE):
    with rasterio.open(path) as source:
        label = source.read(1, out_shape=(size, size), resampling=Resampling.nearest)
    return np.where(np.isin(label, [0, 1]), label, -1).astype(np.int8)


def load_arrays(manifest=None, rebuild=False):
    manifest = manifest or load_manifest()
    import hashlib
    signature = hashlib.sha256(json.dumps({"rows": manifest["rows"], "files": manifest["files"],
                                            "size": IMAGE_SIZE, "range": [DB_MIN, DB_MAX]}, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/cache" / f"prepared_{signature[:16]}.npz"
    if cache.exists() and not rebuild:
        with np.load(cache) as saved:
            return saved["images"].astype(np.float32), saved["labels"], manifest["rows"]
    images, labels = [], []
    for row in manifest["rows"]:
        radar, valid, _ = read_radar(image_path(row))
        label = read_label(label_path(row))
        label[~valid] = -1
        images.append(normalize(radar))
        labels.append(label)
    images, labels = np.stack(images), np.stack(labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, images=images.astype(np.float16), labels=labels)
    # Keep the same precision when rebuilding the cache.
    return images.astype(np.float16).astype(np.float32), labels, manifest["rows"]


def split_indices(rows):
    result = {name: np.array([i for i, row in enumerate(rows) if row["split"] == name], dtype=int)
              for name in ["train", "valid", "test", "bolivia", "weak_train"]}
    sets = [set(result[name]) for name in result]
    assert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i + 1:])
    assert all(rows[i]["event"] != "Bolivia" for name in ["train", "valid", "weak_train"] for i in result[name])
    return result
