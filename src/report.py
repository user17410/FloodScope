"""Plot the experiment results."""

import json
import os

from .data import ROOT, DB_MIN, DB_MAX, load_arrays, read_label

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

WATER = ListedColormap(["#edece7", "#2185c5"])
WATER.set_bad("#9ba3aa")
ERRORS = ListedColormap(["#edece7", "#d89a24", "#b33951", "#2185c5"])
ERRORS.set_bad("#9ba3aa")


def setup_style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "figure.facecolor": "white"})


def save_figure(fig, name):
    path = ROOT / "results/figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return fig


def training_figure():
    setup_style()
    history = pd.read_csv(ROOT / "results/training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    for name, color in [("unet_manual", "#225f9b"), ("unet_weak_then_manual", "#c17424")]:
        rows = history[(history.model == name) & (history.phase == "manual")]
        label = "Manual labels" if name == "unet_manual" else "Weak pretraining + manual labels"
        axes[0].plot(rows.epoch, rows.loss, color=color, label=label)
        axes[1].plot(rows.epoch, rows.validation_water_iou, color=color, label=label)
    for axis in axes:
        axis.set_xlabel("Manual training epoch")
        axis.grid(alpha=0.2)
    axes[0].set(title="Training loss", ylabel="Masked BCE + Dice loss")
    axes[1].set(title="Validation water overlap", ylabel="Pooled water IoU", ylim=(0, 1))
    axes[1].legend(fontsize=8, loc="lower right")
    return save_figure(fig, "training_curves.png")


def comparison_figure():
    setup_style()
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    models = ["vh_threshold", "random_forest", "unet_manual", "unet_weak_then_manual"]
    labels = ["VH threshold", "Random forest", "U-Net: manual", "U-Net: weak + manual"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for offset, split, color, title in [(-0.2, "test", "#225f9b", "Published test split"),
                                      (0.2, "bolivia", "#c17424", "Unseen Bolivia event")]:
        values = metrics[metrics.split == split].set_index("model").loc[models, "water_iou"].to_numpy()
        bars = ax.bar(np.arange(4) + offset, values, width=0.38, color=color, label=title)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set(xticks=np.arange(4), xticklabels=labels, ylabel="Pooled water IoU", ylim=(0, 1.05),
           title="Performance after freezing validation choices")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    return save_figure(fig, "model_comparison.png")


def examples_figure(model_name=None, split="bolivia"):
    setup_style()
    if model_name is None:
        model_name = json.loads((ROOT / "models/selection.json").read_text())["selected_model"]
    images, labels, rows = load_arrays()
    index = {row["chip"]: i for i, row in enumerate(rows)}
    with np.load(ROOT / f"results/predictions/{model_name}_{split}.npz") as saved:
        predictions = {str(chip): pred.astype(bool) for chip, pred in zip(saved["chips"], saved["predictions"])}
    metrics = pd.read_csv(ROOT / "results/per_chip_metrics.csv")
    ordered = metrics[(metrics.model == model_name) & (metrics.split == split)].dropna(subset=["water_iou"]).sort_values(["water_iou", "chip"])
    positions = [0, len(ordered) // 2, len(ordered) - 1]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5), constrained_layout=True)
    for line, position, description in zip(axes, positions, ["Lowest IoU", "Middle IoU", "Highest IoU"]):
        record = ordered.iloc[position]
        i = index[record.chip]
        truth, pred = labels[i], predictions[record.chip]
        invalid = truth < 0
        vh = images[i, 1] * (DB_MAX - DB_MIN) + DB_MIN
        error = np.zeros_like(truth)
        error[(truth == 0) & pred] = 1
        error[(truth == 1) & ~pred] = 2
        error[(truth == 1) & pred] = 3
        panels = [(vh, "gray", None, "Sentinel-1 VH radar"),
                  (truth, WATER, invalid, "Reference water"),
                  (pred, WATER, invalid, "Predicted water"),
                  (error, ERRORS, invalid, "Errors and correct water")]
        for axis, (array, cmap, mask, title) in zip(line, panels):
            data = np.ma.masked_where(mask, array) if mask is not None else array
            axis.imshow(data, cmap=cmap, vmin=DB_MIN if cmap == "gray" else 0,
                        vmax=DB_MAX if cmap == "gray" else (3 if cmap is ERRORS else 1), interpolation="nearest")
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        line[0].set_ylabel(f"{description}\n{record.chip}\nIoU = {record.water_iou:.3f}", fontsize=8)
    fig.suptitle(f"{model_name}: {split} diagnostics\nBlue: correct water · amber: false alarm · red: missed water · gray: ignored reference", fontsize=12)
    return save_figure(fig, "prediction_examples.png")


def permanent_water_figure(chip, prediction):
    setup_style()
    path = ROOT / "data/raw/data/flood_events/HandLabeled/JRCWaterHand" / f"{chip}_JRCWaterHand.tif"
    reference = read_label(path)
    valid = reference >= 0
    candidate = prediction & (reference == 0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    for axis, array, title in zip(axes, [reference == 1, prediction, candidate],
                                  ["JRC permanent-water reference", "Current predicted water", "Water outside permanent reference"]):
        axis.imshow(np.ma.masked_where(~valid, array), cmap=WATER, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
    fig.suptitle(f"{chip}: an illustrative candidate flood mask", fontsize=12)
    return save_figure(fig, "candidate_flood_mask.png")
