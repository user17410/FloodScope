"""Check saved predictions and GeoTIFF exports."""

import json
import tempfile

import numpy as np
import rasterio
import torch

from .data import ROOT, IMAGE_SIZE, load_manifest, load_arrays, split_indices, image_path, read_radar
from .experiment import counts, scores
from .model import masked_loss
from .predict import Predictor


def verify():
    summary = json.loads((ROOT / "results/summary.json").read_text())
    torch.set_num_threads(4)
    images, labels, rows = load_arrays()
    splits = split_indices(rows)
    assert [len(splits[name]) for name in ["train", "valid", "test", "bolivia"]] == [252, 89, 90, 15]
    assert summary["data_audit"]["bolivia_excluded_from_development"]

    # Ignored labels must not affect the loss or its gradient.
    logits = torch.tensor([[[0.2, -0.3], [0.7, -0.8]]], requires_grad=True)
    targets = torch.tensor([[[1, -1], [0, 1]]])
    loss = masked_loss(logits, targets)
    loss.backward()
    assert logits.grad[0, 0, 1] == 0
    altered = logits.detach().clone()
    altered[0, 0, 1] = 100
    assert torch.equal(masked_loss(altered, targets), loss.detach())
    known = np.array([[1, 0], [-1, 1]])
    predicted = np.array([[True, True], [True, False]])
    assert counts(known, predicted).tolist() == [1, 1, 1, 0]
    assert abs(scores(counts(known, predicted))["water_iou"] - 1 / 3) < 1e-12

    reloads = []
    indices = splits["bolivia"][:8]
    for name in summary["selection"]["validation_water_iou"]:
        predictor = Predictor(name)
        if name.startswith("unet_"):
            assert all(torch.isfinite(parameter).all() for parameter in predictor.model.parameters())
        result = predictor.predict(images[indices])
        with np.load(ROOT / f"results/predictions/{name}_bolivia.npz") as saved:
            expected = saved["predictions"][:len(indices)].astype(bool)
        assert np.array_equal(result, expected), f"Reloaded predictions changed: {name}"
        assert result.shape == (8, IMAGE_SIZE, IMAGE_SIZE)
        reloads.append({"model": name, "chips": 8, "identical_predictions": True})

    predictor = Predictor()
    source = image_path(rows[int(indices[0])])
    _, valid, expected_profile = read_radar(source)
    with tempfile.TemporaryDirectory(prefix="floodscope-verify-") as temporary:
        output = predictor.predict_file(source, temporary + "/prediction.tif")
        with rasterio.open(output) as saved:
            array = saved.read(1)
            assert saved.shape == (IMAGE_SIZE, IMAGE_SIZE)
            assert saved.count == 1 and saved.nodata == 255
            assert saved.crs == expected_profile["crs"]
            assert saved.transform.almost_equals(expected_profile["transform"])
            assert np.all(array[~valid] == 255)
            expected = predictor.predict(images[int(indices[0])])
            assert np.array_equal(array[valid], expected[valid].astype(np.uint8))

    import pandas as pd
    details = pd.read_csv(ROOT / "results/per_chip_metrics.csv")
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    for split in ["test", "bolivia"]:
        group = details[(details.model == predictor.name) & (details.split == split)]
        pooled = scores(group[["tp", "fp", "fn", "tn"]].sum().to_numpy())
        reported = metrics[(metrics.model == predictor.name) & (metrics.split == split)].iloc[0]
        assert abs(pooled["water_iou"] - reported.water_iou) < 1e-12

    report = {"passed": True, "split_counts": {name: len(indices) for name, indices in splits.items()},
              "ignored_label_loss_and_metrics": True, "model_reloads": reloads,
              "geotiff_shape_crs_transform_and_nodata": True, "aggregate_scores_recomputed": True}
    (ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify()
