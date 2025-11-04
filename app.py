"""Gradio app for GI disease classification with Grad-CAM and optional LIME."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from infer import (  # type: ignore  # noqa: E402
    build_inference_transform,
    format_topk,
    preprocess_image,
    predict_proba,
)
from model import build_model, get_default_target_layer  # type: ignore  # noqa: E402
from utils import get_device, load_config, load_json  # type: ignore  # noqa: E402
from xai.gradcam import GradCAM, overlay_heatmap, resolve_target_layer  # type: ignore  # noqa: E402
from xai.lime_image import LimeImageExplainerWrapper  # type: ignore  # noqa: E402

CONFIG_PATH = Path(os.getenv("APP_CONFIG_PATH", BASE_DIR / "configs/default.yaml"))
CONFIG = load_config(CONFIG_PATH)

DATA_CFG = CONFIG.get("data", {})
MODEL_CFG = CONFIG.get("model", {})
ARTIFACTS_CFG = CONFIG.get("artifacts", {})
INFER_CFG = CONFIG.get("inference", {})

DEVICE = get_device(prefer_gpu=True)

MODELS_DIR = Path(ARTIFACTS_CFG.get("model_dir", "models"))
if not MODELS_DIR.is_absolute():
    MODELS_DIR = (BASE_DIR / MODELS_DIR).resolve()
MODEL_PATH = MODELS_DIR / ARTIFACTS_CFG.get("best_model_filename", "best.pt")
LABELS_PATH = MODELS_DIR / ARTIFACTS_CFG.get("labels_filename", "labels.json")

IMAGE_SIZE = int(DATA_CFG.get("image_size", 224))
MEAN = DATA_CFG.get("mean", [0.485, 0.456, 0.406])
STD = DATA_CFG.get("std", [0.229, 0.224, 0.225])

TRANSFORM = build_inference_transform(IMAGE_SIZE, MEAN, STD)
TOP_K = int(INFER_CFG.get("top_k", 3))

MODEL_LOCK = threading.Lock()
MODEL: Optional[torch.nn.Module] = None
GRADCAM_EXPLAINER: Optional[GradCAM] = None
LIME_EXPLAINER: Optional[LimeImageExplainerWrapper] = None
IDX_TO_CLASS: List[str] = []


def _load_labels() -> List[str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing labels file at {LABELS_PATH}. "
            "Train the model or upload labels.json."
        )
    data = load_json(LABELS_PATH)
    if "idx_to_class" in data:
        classes = list(data["idx_to_class"])
    elif "class_to_idx" in data:
        classes = [cls for cls, _ in sorted(data["class_to_idx"].items(), key=lambda x: x[1])]
    else:
        raise ValueError("labels.json must include idx_to_class or class_to_idx.")
    if not classes:
        raise ValueError("labels.json does not contain any classes.")
    return classes


def _initialize_model() -> None:
    global MODEL, GRADCAM_EXPLAINER, IDX_TO_CLASS
    if MODEL is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Upload best.pt from training or run training locally first."
        )
    with MODEL_LOCK:
        if MODEL is not None:
            return
        IDX_TO_CLASS = _load_labels()
        num_classes = len(IDX_TO_CLASS)
        model, default_gradcam_layer = build_model(
            name=MODEL_CFG.get("name", "efficientnet_b0"),
            num_classes=num_classes,
            pretrained=False,
            dropout=float(MODEL_CFG.get("dropout", 0.0)),
            freeze_backbone=False,
        )
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()

        target_layer_name = MODEL_CFG.get("gradcam_layer") or default_gradcam_layer
        if not target_layer_name:
            target_layer_name = get_default_target_layer(MODEL_CFG.get("name", "efficientnet_b0"))
        target_layer = resolve_target_layer(model, target_layer_name)

        MODEL = model
        GRADCAM_EXPLAINER = GradCAM(model, target_layer)
        print(
            f"Loaded model from {MODEL_PATH} with Grad-CAM layer '{target_layer_name}' "
            f"on device {DEVICE}."
        )


def _get_lime_explainer() -> Optional[LimeImageExplainerWrapper]:
    global LIME_EXPLAINER
    if not INFER_CFG.get("enable_lime", True):
        return None
    if LIME_EXPLAINER is None:
        lime_cfg = INFER_CFG.get("lime", {})
        LIME_EXPLAINER = LimeImageExplainerWrapper(
            model=MODEL,
            preprocess_fn=lambda img: preprocess_image(img, TRANSFORM),
            device=DEVICE,
            class_names=IDX_TO_CLASS,
            num_samples=int(lime_cfg.get("num_samples", 1000)),
            num_features=int(lime_cfg.get("num_features", 8)),
            hide_rest=bool(lime_cfg.get("hide_rest", False)),
        )
    return LIME_EXPLAINER


def _describe_focus_region(mask: np.ndarray) -> Tuple[str, float]:
    """Summarize where Grad-CAM focuses within the image."""
    coverage = float(mask.mean())
    if not mask.any():
        return "no concentrated focus", coverage
    ys, xs = np.where(mask)
    height, width = mask.shape
    cx = xs.mean() / max(width, 1)
    cy = ys.mean() / max(height, 1)

    if cx < 0.33:
        horizontal = "left"
    elif cx > 0.66:
        horizontal = "right"
    else:
        horizontal = "central"

    if cy < 0.33:
        vertical = "upper"
    elif cy > 0.66:
        vertical = "lower"
    else:
        vertical = "central"

    if vertical == "central":
        focus_phrase = f"{horizontal} region"
    elif horizontal == "central":
        focus_phrase = f"{vertical} region"
    else:
        focus_phrase = f"{vertical}-{horizontal} region"
    return focus_phrase, coverage


def predict_and_explain(image: np.ndarray, with_lime: bool) -> tuple:
    if image is None:
        raise gr.Error("Please upload an endoscopy image before running inference.")
    try:
        _initialize_model()
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(str(exc)) from exc

    try:
        image_uint8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
        input_tensor = preprocess_image(image_uint8, TRANSFORM).unsqueeze(0)
        probs = predict_proba(MODEL, input_tensor, DEVICE, apply_softmax=True)[0]
        top_predictions = format_topk(probs, IDX_TO_CLASS, TOP_K)
        top_index = int(np.argmax(probs))

        heatmap = GRADCAM_EXPLAINER.generate(input_tensor.clone(), class_idx=top_index)
        heatmap = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))
        gradcam_overlay = overlay_heatmap(image_uint8, heatmap)

        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        focus_mask = heatmap_norm >= float(INFER_CFG.get("gradcam_focus_threshold", 0.5))
        focus_phrase, focus_coverage = _describe_focus_region(focus_mask)
        primary_pred = top_predictions[0] if top_predictions else {"class": "N/A", "score": 0.0}

        lime_overlay = None
        if with_lime and INFER_CFG.get("enable_lime", True):
            explainer = _get_lime_explainer()
            if explainer is not None:
                lime_overlay = explainer.explain(image_uint8, top_index)

        predictions_lines = [
            "| Rank | Class | Probability |",
            "| --- | --- | --- |",
        ]
        for rank, item in enumerate(top_predictions, start=1):
            predictions_lines.append(
                f"| {rank} | {item['class']} | {item['score']:.4f} |"
            )
        predictions_markdown = "\n".join(predictions_lines)

        explanation_lines = [
            f"**Predicted:** `{primary_pred['class']}` with probability {primary_pred['score']:.1%}.",
            f"Grad-CAM highlights the {focus_phrase}, covering about {focus_coverage:.1%} of the image pixels.",
        ]
        if lime_overlay is not None:
            explanation_lines.append(
                "LIME superpixels (in the third panel) indicate complementary regions supporting the prediction."
            )
        elif with_lime and INFER_CFG.get("enable_lime", True):
            explanation_lines.append(
                "LIME explanation could not be generated for this image."
            )
        else:
            explanation_lines.append(
                "Enable the LIME checkbox to generate a superpixel-based textual explanation."
            )
        explanation_text = "\n\n".join(explanation_lines)

        return predictions_markdown, gradcam_overlay, lime_overlay, explanation_text
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Inference failed: {exc}") from exc


with gr.Blocks(
    title="GI Disease Image Classification with XAI",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
# GI Disease Image Classification with Explainability

Upload a gastrointestinal endoscopy image to view model predictions with Grad-CAM heatmaps.
Optionally enable LIME for a complementary superpixel-based explanation.
        """.strip()
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Endoscopy Image",
                type="numpy",
                image_mode="RGB",
                height=256,
            )
            lime_checkbox = gr.Checkbox(
                label="Generate LIME explanation (slower)",
                value=False,
                visible=INFER_CFG.get("enable_lime", True),
            )
            run_button = gr.Button("Run Inference", variant="primary")
        with gr.Column(scale=1):
            predictions_table = gr.Markdown(label="Top Predictions")
            gradcam_output = gr.Image(label="Grad-CAM Overlay", height=256)
            lime_output = gr.Image(
                label="LIME Overlay",
                height=256,
                visible=INFER_CFG.get("enable_lime", True),
            )

        with gr.Column(scale=1):
            explanation_output = gr.Markdown(label="Explanation")

    run_button.click(
        fn=predict_and_explain,
        inputs=[image_input, lime_checkbox],
        outputs=[predictions_table, gradcam_output, lime_output, explanation_output],
    )

    gr.Markdown(
        """
**Tips**

- Grad-CAM highlights discriminative regions using convolutional activations.
- LIME explanations approximate the model locally via interpretable superpixels.
- For best results, ensure images are pre-cleaned and cropped around the lesion.
        """.strip()
    )


if __name__ == "__main__":
    queue_max = int(os.getenv("QUEUE_MAX_SIZE", "32"))
    demo.queue(max_size=queue_max).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
