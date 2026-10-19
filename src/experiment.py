"""Train the models and compare their water predictions."""

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import uniform_filter
import sklearn
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import Dataset, DataLoader

from .data import ROOT, IMAGE_SIZE, DB_MIN, DB_MAX, load_manifest, load_arrays, split_indices
from .model import SmallUNet, masked_loss

CONFIG = {
    "seed": 42, "device": "cpu", "threads": 4, "image_size": IMAGE_SIZE,
    "crop_size": 128, "crops_per_image": 2, "batch_size": 16,
    "epochs": 18, "weak_pretrain_epochs": 4, "learning_rate": 0.001,
    "weight_decay": 0.0001, "unet_width": 8,
    "rf_trees": 80, "rf_max_depth": 14, "rf_min_leaf": 8, "rf_pixels_per_image": 1536,
    "probability_thresholds": [round(float(x), 2) for x in np.arange(0.2, 0.81, 0.1)],
    "vh_thresholds_db": list(range(-30, -7)),
    "selection_metric": "validation pooled water IoU",
    "evaluation_grid": "256x256: average radar resampling, nearest-neighbor labels",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)


def counts(truth, prediction):
    valid = truth >= 0
    return np.array([np.sum(valid & (truth == 1) & prediction),
                     np.sum(valid & (truth == 0) & prediction),
                     np.sum(valid & (truth == 1) & ~prediction),
                     np.sum(valid & (truth == 0) & ~prediction)], dtype=np.int64)


def scores(c):
    tp, fp, fn, tn = (int(x) for x in c)
    ratio = lambda a, b: float(a / b) if b else None
    return {"water_iou": ratio(tp, tp + fp + fn), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "accuracy": ratio(tp + tn, tp + fp + fn + tn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "valid_pixels": tp + fp + fn + tn}


def summarize(truth, prediction):
    c = [counts(y, p) for y, p in zip(truth, prediction)]
    result = scores(np.sum(c, axis=0))
    individual = [scores(item)["water_iou"] for item in c]
    result["mean_chip_water_iou"] = float(np.mean([x for x in individual if x is not None]))
    result["chips_with_defined_water_iou"] = sum(x is not None for x in individual)
    return result


def choose_threshold(probabilities, truth):
    trials = [(threshold, summarize(truth, probabilities >= threshold)["water_iou"])
              for threshold in CONFIG["probability_thresholds"]]
    return max(trials, key=lambda pair: (pair[1], -abs(pair[0] - 0.5))), trials


def radar_features(image):
    vv, vh = image
    mean_vv = uniform_filter(vv, size=5, mode="reflect")
    mean_vh = uniform_filter(vh, size=5, mode="reflect")
    std_vh = np.sqrt(np.maximum(uniform_filter(vh * vh, size=5, mode="reflect") - mean_vh * mean_vh, 0))
    return np.stack([vv, vh, vv - vh, mean_vv, mean_vh, std_vh], axis=-1).reshape(-1, 6)


class RadarCrops(Dataset):
    def __init__(self, images, labels, indices, seed):
        self.images, self.labels, self.indices = images, labels, indices
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices) * CONFIG["crops_per_image"]

    def __getitem__(self, index):
        i = self.indices[index % len(self.indices)]
        crop = CONFIG["crop_size"]
        for _ in range(10):
            top, left = self.rng.integers(0, IMAGE_SIZE - crop + 1, size=2)
            y = self.labels[i, top:top + crop, left:left + crop]
            if np.mean(y >= 0) > 0.1:
                break
        x = self.images[i, :, top:top + crop, left:left + crop]
        turns = int(self.rng.integers(0, 4))
        x, y = np.rot90(x, turns, axes=(1, 2)), np.rot90(y, turns)
        if self.rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()).long()


@torch.inference_mode()
def unet_probabilities(model, images, indices):
    model.eval()
    result = []
    for start in range(0, len(indices), 8):
        x = torch.from_numpy(images[indices[start:start + 8]])
        result.append(model(x).sigmoid().numpy())
    return np.concatenate(result)


def fit_unet(images, labels, splits, rows, name, weak=False):
    seed_everything(CONFIG["seed"])
    model = SmallUNet(CONFIG["unet_width"])
    history = []

    def stage(indices, epochs, phase):
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
        generator = torch.Generator().manual_seed(CONFIG["seed"])
        dataset = RadarCrops(images, labels, indices, CONFIG["seed"])
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, generator=generator)
        best_iou, best_state, best_epoch, best_threshold = -1, None, None, None
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            model.train()
            total_loss, examples = 0.0, 0
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(x)
                examples += len(x)
            row = {"model": name, "phase": phase, "epoch": epoch,
                   "loss": total_loss / examples, "seconds": time.perf_counter() - started}
            if phase == "manual":
                probabilities = unet_probabilities(model, images, splits["valid"])
                (threshold, iou), _ = choose_threshold(probabilities, labels[splits["valid"]])
                row.update(validation_water_iou=iou, threshold=threshold)
                if iou > best_iou:
                    best_iou, best_state, best_epoch, best_threshold = iou, copy.deepcopy(model.state_dict()), epoch, threshold
                print(f"{name}: epoch {epoch:02d}/{epochs}, loss={row['loss']:.4f}, validation IoU={iou:.4f}", flush=True)
            else:
                print(f"{name}: weak pretraining {epoch}/{epochs}, loss={row['loss']:.4f}", flush=True)
            history.append(row)
        return best_state, best_epoch, best_threshold

    if weak:
        if not len(splits["weak_train"]):
            raise ValueError("The weak-label comparison requires downloaded weak images.")
        stage(splits["weak_train"], CONFIG["weak_pretrain_epochs"], "weak")
    state, epoch, threshold = stage(splits["train"], CONFIG["epochs"], "manual")
    model.load_state_dict(state)
    checkpoint = {
        "state_dict": state, "width": CONFIG["unet_width"], "image_size": IMAGE_SIZE,
        "db_min": DB_MIN, "db_max": DB_MAX, "threshold": threshold,
        "selected_epoch": epoch, "config": CONFIG, "model_name": name,
        "training_chips": [rows[i]["chip"] for i in splits["train"]],
        "weak_training_chips": [rows[i]["chip"] for i in splits["weak_train"]] if weak else [],
    }
    torch.save(checkpoint, ROOT / "models" / f"{name}.pth")
    return model, threshold, history, epoch


def dataset_audit(images, labels, rows, splits):
    # Only coordinates are used for the overlap check.
    from rasterio.coords import disjoint_bounds
    import rasterio
    from .data import image_path
    bounds = []
    for row in rows:
        with rasterio.open(image_path(row)) as source:
            bounds.append(source.bounds)
    weak = splits["weak_train"]
    checking = np.concatenate([splits["valid"], splits["test"], splits["bolivia"]])
    overlaps = []
    for i in weak:
        for j in checking:
            if rows[i]["event"] == rows[j]["event"] and not disjoint_bounds(bounds[i], bounds[j]):
                a, b = bounds[i], bounds[j]
                # Shared edges do not count as overlap.
                if min(a.right, b.right) > max(a.left, b.left) + 1e-8 and min(a.top, b.top) > max(a.bottom, b.bottom) + 1e-8:
                    overlaps.append((int(i), int(j)))
    if overlaps:
        excluded = {i for i, _ in overlaps}
        splits["weak_train"] = np.array([i for i in weak if i not in excluded], dtype=int)
    audit = {
        "source_grid": "512x512", "experiment_grid": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "splits": {name: len(indices) for name, indices in splits.items()},
        "events": {name: sorted({rows[i]["event"] for i in indices}) for name, indices in splits.items()},
        "overlapping_weak_chips_removed": sorted({rows[i]["chip"] for i, _ in overlaps}),
        "disjoint_chip_ids": len({row["chip"] for row in rows}) == len(rows),
        "bolivia_excluded_from_development": all(rows[i]["event"] != "Bolivia" for name in ["train", "valid", "weak_train"] for i in splits[name]),
        "finite_images": bool(np.isfinite(images).all()), "label_values": np.unique(labels).tolist(),
    }
    if not audit["finite_images"] or not audit["disjoint_chip_ids"]:
        raise ValueError("Dataset audit failed.")
    return audit


def run_experiment(force=False):
    summary_path = ROOT / "results/summary.json"
    if summary_path.exists() and not force:
        print("Using the saved completed experiment. Set force=True to retrain.")
        return json.loads(summary_path.read_text())
    for folder in ["models", "results", "results/predictions"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = load_manifest()
    images, labels, rows = load_arrays(manifest)
    splits = split_indices(rows)
    audit = dataset_audit(images, labels, rows, splits)
    (ROOT / "results/data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    source_hash = hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest()
    config_record = {**CONFIG, "manifest_sha256": source_hash, "splits": audit["splits"],
                     "test_access_policy": "Freeze model checkpoints and thresholds using train/valid before scoring test or Bolivia."}
    (ROOT / "results/config.json").write_text(json.dumps(config_record, indent=2) + "\n")
    print("Fixed split sizes:", audit["splits"], flush=True)
    seed_everything(CONFIG["seed"])
    models, histories, validation_records = {}, [], []

    vh_valid = images[splits["valid"], 1] * (DB_MAX - DB_MIN) + DB_MIN
    threshold_trials = []
    for threshold in CONFIG["vh_thresholds_db"]:
        iou = summarize(labels[splits["valid"]], vh_valid < threshold)["water_iou"]
        threshold_trials.append({"model": "vh_threshold", "threshold": threshold, "water_iou": iou})
    best = max(threshold_trials, key=lambda item: item["water_iou"])
    models["vh_threshold"] = {"threshold": best["threshold"]}
    validation_records.extend(threshold_trials)
    (ROOT / "models/vh_threshold.json").write_text(json.dumps(models["vh_threshold"], indent=2) + "\n")

    rng = np.random.default_rng(CONFIG["seed"])
    features, targets = [], []
    for i in splits["train"]:
        valid = np.flatnonzero(labels[i].ravel() >= 0)
        if not len(valid):
            continue
        chosen = rng.choice(valid, min(CONFIG["rf_pixels_per_image"], len(valid)), replace=False)
        features.append(radar_features(images[i])[chosen])
        targets.append(labels[i].ravel()[chosen])
    forest = RandomForestClassifier(n_estimators=CONFIG["rf_trees"], max_depth=CONFIG["rf_max_depth"],
                                    min_samples_leaf=CONFIG["rf_min_leaf"], class_weight="balanced_subsample",
                                    n_jobs=CONFIG["threads"], random_state=CONFIG["seed"])
    forest.fit(np.concatenate(features), np.concatenate(targets))
    forest_prob = lambda indices: np.stack([forest.predict_proba(radar_features(images[i]))[:, 1].reshape(IMAGE_SIZE, IMAGE_SIZE) for i in indices])
    (threshold, iou), trials = choose_threshold(forest_prob(splits["valid"]), labels[splits["valid"]])
    models["random_forest"] = {"object": forest, "threshold": threshold}
    validation_records.extend({"model": "random_forest", "threshold": t, "water_iou": score} for t, score in trials)
    joblib.dump({"model": forest, "threshold": threshold, "config": CONFIG,
                 "feature_order": ["VV", "VH", "VV-VH", "mean_VV_5x5", "mean_VH_5x5", "std_VH_5x5"]}, ROOT / "models/random_forest.joblib", compress=3)
    print(f"Random forest: validation IoU={iou:.4f}", flush=True)

    for name, weak in [("unet_manual", False), ("unet_weak_then_manual", True)]:
        model, threshold, history, epoch = fit_unet(images, labels, splits, rows, name, weak)
        histories.extend(history)
        probabilities = unet_probabilities(model, images, splits["valid"])
        metrics = summarize(labels[splits["valid"]], probabilities >= threshold)
        models[name] = {"object": model, "threshold": threshold, "epoch": epoch}
        validation_records.append({"model": name, "threshold": threshold, "water_iou": metrics["water_iou"]})

    pd.DataFrame(histories).to_csv(ROOT / "results/training_history.csv", index=False)
    pd.DataFrame(validation_records).to_csv(ROOT / "results/validation_search.csv", index=False)
    best_by_name = {name: max(row["water_iou"] for row in validation_records if row["model"] == name) for name in models}
    selected_name = max(best_by_name, key=best_by_name.get)
    selection = {"selected_model": selected_name, "criterion": CONFIG["selection_metric"],
                 "validation_water_iou": best_by_name, "thresholds": {name: item["threshold"] for name, item in models.items()}}
    (ROOT / "models/selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Model selection frozen before test scoring:", selected_name, flush=True)

    results, chip_results = [], []
    for split in ["valid", "test", "bolivia"]:
        indices = splits[split]
        for name, item in models.items():
            if name == "vh_threshold":
                probabilities = images[indices, 1] * (DB_MAX - DB_MIN) + DB_MIN
                prediction = probabilities < item["threshold"]
            else:
                probabilities = forest_prob(indices) if name == "random_forest" else unet_probabilities(item["object"], images, indices)
                prediction = probabilities >= item["threshold"]
            result = {"split": split, "model": name, "threshold": item["threshold"], "chips": len(indices),
                      **summarize(labels[indices], prediction)}
            results.append(result)
            np.savez_compressed(ROOT / f"results/predictions/{name}_{split}.npz", chips=np.array([rows[i]["chip"] for i in indices]),
                                predictions=prediction.astype(np.uint8), probabilities=probabilities.astype(np.float16))
            for i, pred in zip(indices, prediction):
                chip_results.append({"split": split, "model": name, "chip": rows[i]["chip"], "event": rows[i]["event"],
                                     **scores(counts(labels[i], pred))})
            print(f"{split:7s} {name:24s} water IoU={result['water_iou']:.4f}", flush=True)
    pd.DataFrame(results).to_csv(ROOT / "results/metrics.csv", index=False)
    pd.DataFrame(chip_results).to_csv(ROOT / "results/per_chip_metrics.csv", index=False)
    versions = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    summary = {"config": config_record, "data_audit": audit, "selection": selection, "metrics": results,
               "versions": versions, "elapsed_seconds": time.perf_counter() - started,
               "interpretation": "Scores measure water segmentation on the prepared 256x256 grid. Flood-only accuracy needs separate flood labels."}
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_experiment(args.force)
