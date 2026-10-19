"""Write the README and model card from the saved results."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / "results/summary.json").read_text())
metrics = pd.read_csv(ROOT / "results/metrics.csv")
selected = summary["selection"]["selected_model"]
checking = metrics[metrics.split.isin(["test", "bolivia"])]


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


metric_table = table(checking, ["split", "model", "water_iou", "precision", "recall", "f1"])
weak = checking[checking.model.str.startswith("unet")].pivot(index="split", columns="model", values="water_iou")
delta_lines = []
for split, row in weak.iterrows():
    delta = 100 * (row["unet_weak_then_manual"] - row["unet_manual"])
    delta_lines.append(f"- {split}: weak pretraining changed water IoU by **{delta:+.2f} percentage points**.")

readme = f"""# FloodScope

For this project, I use real Sentinel-1 satellite radar measurements to map water during flood events and compare a VH threshold with a random forest and a small U-Net so I can see which method performs best on the reserved images. I also train the same U-Net on extra automatically labeled images before using the manual labels to check whether that extra training improves its predictions.

The main walkthrough is in **[floodscope.ipynb](floodscope.ipynb)** where I explain how I prepare the data and train the models before comparing their results through tables, training curves and examples of correct and incorrect predictions. I also reload the selected model and export a GeoTIFF to check that the saved files can produce the same predictions.

![Model comparison](results/figures/model_comparison.png)

## What I used

- Dataset: [Sen1Floods11 v1.1](https://github.com/cloudtostreet/Sen1Floods11) with all 446 manually labeled patches and 80 selected automatic-label patches of which {summary['data_audit']['splits']['weak_train']} passed the spatial overlap check.
- Published splits: 252 training images, 89 validation images, 90 ordinary test images and 15 Bolivia images with Bolivia kept outside training, validation and weak pretraining.
- Inputs: VV and VH radar measurements in dB using the dataset's existing preprocessing.
- Image size: 256 × 256 for all methods after reducing the original 512 × 512 grid.
- Training: my CPU with four threads and seed 42 using a U-Net with 122,393 parameters.

The labels describe **water** which includes permanent rivers and lakes so the scores measure water segmentation during flood events. I also compare the predictions with the JRC permanent-water reference to illustrate a candidate flood mask although I have not measured flood-only accuracy separately.

## Results

{metric_table}

I selected **`{selected}`** using pooled validation water IoU before scoring either checking split so the test answers did not determine which method I chose. The ordinary test split checks new patches from events already represented in development while Bolivia checks an event that stayed outside training and validation.

I ignore reference pixels without reliable answers in both the loss and the metrics and calculate every score on the prepared 256 × 256 grid which means these results are not directly comparable with published scores at the original resolution.

I trained each U-Net for 18 manual-label epochs with four extra pretraining epochs for the weak-label model and selected the checkpoints and probability thresholds using validation only. The complete experiment took **{summary['elapsed_seconds'] / 60:.2f} minutes** on my machine although the runtime will depend on the hardware used.

### Did the extra labels help?

{chr(10).join(delta_lines)}

This comparison uses one seed and a limited weak-label sample with extra training time for the pretrained model so I use it to describe what happened under these settings without claiming that either method will always perform better.

## Run it locally

Use Python 3.12 and run these commands from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.prepare_data
python -m src.experiment
```

The data command downloads approximately 865 MB through public HTTPS links and verifies the file sizes and MD5 checksums before saving the source generations and SHA-256 checksums in `data/manifest.json`. I exclude the environment, raw data and prepared cache through `.gitignore` so they can be downloaded separately when sharing the repository while the source data remain subject to their provider's terms and attribution requirements.

Open `floodscope.ipynb` in VS Code or Jupyter and select this folder's `.venv/bin/python` interpreter to use the saved results without training again. Set `RETRAIN = True` or run `python -m src.experiment --force` to repeat training. I reused the existing system PyTorch and pandas packages for the original run although a fresh environment can install all the dependencies through `requirements.txt`.

To predict a matching two-band VV/VH GeoTIFF:

```bash
python -m src.predict path/to/sentinel1.tif results/my_water_prediction.tif
```

The input should match Sen1Floods11's calibrated and georeferenced dB preprocessing with the same VV/VH band order and an approximately 5 km patch footprint. The command prepares an individual patch on a 256 × 256 grid and preserves its geographic extent through the updated transform with `0` for not water, `1` for water and `255` for invalid radar input.

## Files for checking the work

| File | Purpose |
| --- | --- |
| `floodscope.ipynb` | Notebook with the explanations and executed results |
| `src/prepare_data.py` | Public data download and integrity checks |
| `src/data.py` | Shared radar and label preprocessing |
| `src/model.py` | U-Net architecture and masked loss |
| `src/experiment.py` | Training and validation selection followed by checking |
| `src/predict.py` | Reloaded-model inference and GeoTIFF output |
| `models/selection.json` | Method selected using validation |
| `models/*.pth` | Both trained U-Net checkpoints |
| `models/random_forest.joblib` | Trained forest and its threshold |
| `results/metrics.csv` | Aggregate measurements |
| `results/per_chip_metrics.csv` | Per-image scores and confusion counts |
| `results/config.json` | Fixed settings and dataset-manifest fingerprint |
| `results/data_audit.json` | Split separation and spatial-overlap checks |
| `results/verification.json` | Checks for model reloads, metrics, ignored labels and exports |
| `results/notebook_execution.json` | Full notebook execution record |

Run `python -m src.verify` to repeat the saved-model and export checks or use `python execute_notebook.py` to execute every notebook cell with the current interpreter and save the outputs. The `python build_notebook.py` command rebuilds the notebook without outputs so run the execution script afterwards if you use it.

## Limits and next steps

I have checked one completely unseen event and one ordinary test split so these results do not establish operational reliability or performance on Philippine flood imagery. The manual labels contain ignored and uncertain regions while the automatic labels can contain classification mistakes and reducing the image size removes smaller details. I would need to repeat the comparison with more seeds and unseen events and check larger weak-label samples and independently labeled local cases before making broader claims about flood mapping.

## Sources

Bonafilia, D., Tellman, B., Anderson, T., and Issenberg, E. (2020). **Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1.** IEEE/CVF CVPR Workshops. [Dataset](https://github.com/cloudtostreet/Sen1Floods11), [paper](https://www.grss-ieee.org/wp-content/uploads/earthvision2020/july_stuff/webpage/papers/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf).

Ronneberger, O., Fischer, P., and Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** [Paper](https://arxiv.org/abs/1505.04597). This project uses a compact adaptation with two radar bands and group normalization.

[ESA Sentinel-1 mission description](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Introducing_the_Sentinel-1_mission).

"""
(ROOT / "README.md").write_text(readme)
card = {
    "project": "FloodScope", "task": "Binary water segmentation during observed flood events",
    "selected_model": selected, "selection": summary["selection"],
    "data": summary["data_audit"], "evaluation": summary["metrics"],
    "input": {"bands": ["VV", "VH"], "unit": "dB", "size": [256, 256], "fixed_clip_db": [-35, 5]},
    "output": {"mask": {"0": "not water", "1": "water", "255": "invalid radar"}},
    "scope": "Research demonstration on Sen1Floods11; water scores do not establish flood-only accuracy.",
    "versions": summary["versions"],
}
(ROOT / "models/model_card.json").write_text(json.dumps(card, indent=2) + "\n")
print("Wrote README.md and models/model_card.json from measured results.")
