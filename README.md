---
title: GI Disease Image Classification with XAI
emoji: "🩺"
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---
---
---
# GI Disease Image Classification with XAI & MLOps

![Grad-CAM Example](https://example.com/gradcam-placeholder.png)

## Overview
This project delivers an end-to-end pipeline for gastrointestinal disease image classification on the Kvasir / Kvasir-v2 dataset. It combines PyTorch training, MLflow experiment tracking, Grad-CAM and LIME explainability, plus a Gradio UI ready for Hugging Face Spaces with GPU acceleration.

## Highlights
- EfficientNet-B0 or ResNet-50 backbones with configurable hyperparameters.
- Albumentations-based augmentations and deterministic data splits.
- MLflow logs metrics, parameters, checkpoints, and label metadata.
- Grad-CAM overlays via OpenCV and optional LIME superpixel explanations.
- Gradio app deployable on Hugging Face Spaces (T4-small, autolaunch).
- Git LFS configuration for large artifacts (`models/best.pt`, `models/labels.json`).

## Dataset Preparation
1. Download Kvasir or Kvasir-v2 (one folder per class) from https://datasets.simula.no/kvasir/.
2. Extract into `data/kvasir/<class_name>/*.jpg`.
3. Adjust `configs/default.yaml` if you use a different root or subset.

```bash
mkdir -p data/kvasir
# Example download command; replace with the release you need
wget -O kvasir.zip https://datasets.simula.no/downloads/kvasir/Kvasir-SEG.zip
unzip kvasir.zip -d data/kvasir
```

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
make install
cp .env.example .env       # optional: override MLflow settings
```

## Training & Experiment Tracking
```bash
make train
```

- Metrics for train/val/test and artifacts (`models/best.pt`, `models/labels.json`) go to MLflow.
- Launch the MLflow UI with `make mlflow-ui` and open http://localhost:5000.
- Modify hyperparameters in `configs/default.yaml` or point `--config` at a different file.

## Explainability
- **Grad-CAM**: Produced during inference and in the Gradio app with OpenCV overlays.
- **LIME**: Enable the checkbox in the app to generate superpixel explanations (tunable under `inference.lime`).

## Running the Gradio App Locally
```bash
make app
# or
APP_CONFIG_PATH=configs/default.yaml python app.py
```
The interface returns top-k predictions, Grad-CAM overlays, an optional LIME overlay, plus a textual summary of the explanation. GPU inference is used automatically when `torch.cuda.is_available()` is true.

## Using the Gradio API (Free Option)
Running the Gradio app locally or on Hugging Face Spaces automatically exposes a REST endpoint at `/api/predict/`, providing a zero-cost API.

- **Local**: Start the app (`make app`) and POST to `http://127.0.0.1:7860/api/predict/`.
- **Hugging Face Spaces**: After deployment, POST to `https://<your-space>.hf.space/api/predict/`.

Example `curl` request (replace `sample.jpg` and the URL as needed):

```bash
curl -X POST "http://127.0.0.1:7860/api/predict/" \
  -H "Content-Type: application/json" \
  -d '{
        "data": [
          {"name": "image", "data": "'"$(base64 -w0 sample.jpg)"'", "is_file": true},
          false
        ]
      }'
```

The response JSON includes the prediction table, Grad-CAM overlay (base64), optional LIME overlay, and the textual explanation. For more client examples, see https://www.gradio.app/guides/querying-gradio-apps-with-curl.

## Deploying to Hugging Face Spaces
1. Install Git LFS: `git lfs install`.
2. Commit the project; ensure `models/best.pt` and `models/labels.json` use LFS.
3. Create a Gradio Space (GPU) and push this repository, including `.huggingface.yml`.
4. Upload or train to populate `models/` before launching.

## Project Structure
```
gi-xai-mlops/
|-- app.py
|-- requirements.txt
|-- README.md
|-- Makefile
|-- .env.example
|-- .gitattributes
|-- .huggingface.yml
|-- configs/
|   `-- default.yaml
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- model.py
|   |-- train.py
|   |-- infer.py
|   |-- utils.py
|   `-- xai/
|       |-- __init__.py
|       |-- gradcam.py
|       `-- lime_image.py
|-- models/
|   `-- (best.pt and labels.json generated after training)
`-- notebooks/
    `-- (optional exploration)
```

## Makefile Cheatsheet
- `make install` â€“ install dependencies.
- `make train` â€“ train and log to MLflow.
- `make app` â€“ run the Gradio interface locally.
- `make mlflow-ui` â€“ launch the MLflow dashboard.

## Hugging Face GPU Notes
- `.huggingface.yml` requests the free `T4-small` GPU tier and enables autolaunch.
- Ensure `models/` contains the trained weights and labels before deployment.

## Troubleshooting
- **Dataset errors**: Check that `data/kvasir` exists with one folder per class.
- **Missing artifacts**: Run `make train` to generate `models/best.pt` and `models/labels.json`.
- **MLflow write issues**: Update `MLFLOW_TRACKING_URI` in `.env` or the config file.
- **Slow LIME runs**: Adjust `inference.lime.num_samples` to balance fidelity and speed.

---

Adapt configs, track experiments, and deploy interpretable GI classifiers with confidence.
