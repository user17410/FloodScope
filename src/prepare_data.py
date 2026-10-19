"""Download Sen1Floods11 files and verify their public storage checksums."""

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import certifi

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "sen1floods11"
BASE = f"https://storage.googleapis.com/{BUCKET}/"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
SPLIT_NAMES = ("train", "valid", "test", "bolivia")


def read_url(url):
    for attempt in range(5):
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            with urlopen(Request(url, headers={"User-Agent": "FloodScope-research/1.0"}), timeout=90, context=context) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1 + 2 ** attempt)


def list_objects(prefix):
    objects, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": 1000}
        if token:
            query["pageToken"] = token
        page = json.loads(read_url(API + "?" + urlencode(query)))
        objects.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return {item["name"]: item for item in objects if not item["name"].endswith("/")}


def fetch_object(item):
    name = item["name"]
    relative = name.removeprefix("v1.1/")
    path = ROOT / "data" / "raw" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = item.get("md5Hash")
    payload = path.read_bytes() if path.exists() else b""
    md5 = lambda b: base64.b64encode(hashlib.md5(b).digest()).decode()
    if len(payload) != int(item["size"]) or (expected_md5 and md5(payload) != expected_md5):
        url = BASE + quote(name, safe="/") + "?" + urlencode({"generation": item["generation"]})
        payload = read_url(url)
        if len(payload) != int(item["size"]) or (expected_md5 and md5(payload) != expected_md5):
            raise ValueError(f"The downloaded file failed its integrity check: {name}")
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return {
        "object": name, "local_path": str(path.relative_to(ROOT)),
        "url": BASE + quote(name, safe="/"), "bytes": len(payload),
        "md5_base64": expected_md5, "sha256": hashlib.sha256(payload).hexdigest(),
        "generation": item["generation"],
    }


def prepare(weak_per_event=8, workers=8):
    (ROOT / "data").mkdir(exist_ok=True)
    split_objects = list_objects("v1.1/splits/flood_handlabeled/")
    files, rows = {}, []
    for split in SPLIT_NAMES:
        name = f"v1.1/splits/flood_handlabeled/flood_{split}_data.csv"
        record = fetch_object(split_objects[name])
        files[name] = record
        for image, label in csv.reader(io.StringIO((ROOT / record["local_path"]).read_text())):
            chip = image.removesuffix("_S1Hand.tif")
            rows.append({"chip": chip, "event": chip.split("_")[0], "split": split,
                         "image": image, "label": label, "label_source": "manual"})
    chips = [row["chip"] for row in rows]
    if len(chips) != len(set(chips)):
        raise ValueError("The published splits contain duplicate chips.")
    if any(row["event"] == "Bolivia" and row["split"] != "bolivia" for row in rows):
        raise ValueError("Bolivia leaked into the development splits.")

    selected = []
    for layer, key in [("S1Hand", "image"), ("LabelHand", "label"), ("JRCWaterHand", None)]:
        objects = list_objects(f"v1.1/data/flood_events/HandLabeled/{layer}/")
        for row in rows:
            filename = row[key] if key else row["chip"] + "_JRCWaterHand.tif"
            name = f"v1.1/data/flood_events/HandLabeled/{layer}/{filename}"
            selected.append(objects[name])

    weak_rows = []
    if weak_per_event:
        weak_images = list_objects("v1.1/data/flood_events/WeaklyLabeled/S1Weak/")
        weak_labels = list_objects("v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/")
        rng = random.Random(42)
        train_events = sorted({row["event"] for row in rows if row["split"] == "train"})
        for event in train_events:
            candidates = []
            for name in sorted(weak_images):
                filename = Path(name).name
                chip = filename.removesuffix("_S1Weak.tif")
                label_name = f"v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/{chip}_S2IndexLabelWeak.tif"
                if chip.startswith(event + "_") and chip not in chips and label_name in weak_labels:
                    candidates.append((chip, name, label_name))
            for chip, image_name, label_name in rng.sample(candidates, min(weak_per_event, len(candidates))):
                selected.extend([weak_images[image_name], weak_labels[label_name]])
                weak_rows.append({"chip": chip, "event": event, "split": "weak_train",
                                  "image": Path(image_name).name, "label": Path(label_name).name,
                                  "label_source": "automatic_sentinel2"})

    total_bytes = sum(int(item["size"]) for item in selected)
    print(f"Downloading/verifying {len(rows)} manual chips and {len(weak_rows)} weak chips ({total_bytes / 1e6:.1f} MB).", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(fetch_object, item) for item in selected]
        for count, task in enumerate(as_completed(tasks), 1):
            record = task.result()
            files[record["object"]] = record
            if count % 100 == 0 or count == len(tasks):
                print(f"Verified {count}/{len(tasks)} files", flush=True)

    metadata_url = "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/Sen1Floods11_Metadata.geojson"
    metadata = read_url(metadata_url)
    json.loads(metadata)
    (ROOT / "data" / "events.geojson").write_bytes(metadata)
    manifest = {
        "dataset": "Sen1Floods11", "version": "v1.1", "seed": 42,
        "source": "https://github.com/cloudtostreet/Sen1Floods11",
        "weak_per_event": weak_per_event, "rows": rows + weak_rows,
        "files": sorted(files.values(), key=lambda item: item["object"]),
        "event_metadata_url": metadata_url,
        "event_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved data/manifest.json with source URLs, object generations, and checksums.", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-per-event", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.weak_per_event, args.workers)
