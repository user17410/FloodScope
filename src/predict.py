"""Predict water masks from the saved models."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio
import torch

from .data import ROOT, IMAGE_SIZE, DB_MIN, DB_MAX, normalize, read_radar
from .experiment import radar_features
from .model import SmallUNet


class Predictor:
    def __init__(self, model_name=None, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else ROOT / "models"
        self.name = model_name or json.loads((self.model_dir / "selection.json").read_text())["selected_model"]
        if self.name.startswith("unet_"):
            saved = torch.load(self.model_dir / f"{self.name}.pth", map_location="cpu", weights_only=True)
            self.model = SmallUNet(saved["width"])
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()
            self.threshold = saved["threshold"]
        elif self.name == "random_forest":
            saved = joblib.load(self.model_dir / "random_forest.joblib")
            self.model, self.threshold = saved["model"], saved["threshold"]
        elif self.name == "vh_threshold":
            self.model = None
            self.threshold = json.loads((self.model_dir / "vh_threshold.json").read_text())["threshold"]
        else:
            raise ValueError(f"Unknown model: {self.name}")

    @torch.inference_mode()
    def probabilities(self, images):
        images = np.asarray(images, dtype=np.float32)
        single = images.ndim == 3
        if single:
            images = images[None]
        if images.shape[1:] != (2, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected normalized input with shape (N, 2, {IMAGE_SIZE}, {IMAGE_SIZE}).")
        if self.name.startswith("unet_"):
            result = self.model(torch.from_numpy(images)).sigmoid().numpy()
        elif self.name == "random_forest":
            result = np.stack([self.model.predict_proba(radar_features(x))[:, 1].reshape(IMAGE_SIZE, IMAGE_SIZE) for x in images])
        else:
            # The threshold rule returns a mask rather than a probability.
            result = (images[:, 1] * (DB_MAX - DB_MIN) + DB_MIN < self.threshold).astype(np.float32)
        return result[0] if single else result

    def predict(self, images):
        return self.probabilities(images) >= (0.5 if self.name == "vh_threshold" else self.threshold)

    def predict_file(self, image, output):
        radar, valid, profile = read_radar(image)
        normalized = normalize(radar).astype(np.float16).astype(np.float32)
        mask = self.predict(normalized).astype(np.uint8)
        mask[~valid] = 255
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as destination:
            destination.write(mask, 1)
            destination.update_tags(model=self.name, description="Water mask: 0=not water, 1=water, 255=invalid radar")
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="A two-band VV/VH Sentinel-1 GeoTIFF in dB, matching dataset preprocessing")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    predictor = Predictor(args.model)
    print(predictor.predict_file(args.image, args.output))
