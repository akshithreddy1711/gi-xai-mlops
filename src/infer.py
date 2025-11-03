"""Inference helpers for preprocessing and formatting model outputs."""
from __future__ import annotations

from typing import Dict, List, Sequence

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def build_inference_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> A.Compose:
    """Construct an evaluation-time albumentations pipeline."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def preprocess_image(image: np.ndarray, transform: A.Compose) -> torch.Tensor:
    """Apply the albumentations transform and return a CHW tensor."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    transformed = transform(image=image)
    tensor = transformed["image"].float()
    return tensor


def predict_proba(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    apply_softmax: bool = True,
) -> np.ndarray:
    """Run a forward pass and return probabilities."""
    model.eval()
    with torch.no_grad():
        outputs = model(batch.to(device, non_blocking=True))
        if apply_softmax:
            outputs = torch.softmax(outputs, dim=1)
    return outputs.detach().cpu().numpy()


def format_topk(
    probs: np.ndarray,
    classes: Sequence[str],
    topk: int,
) -> List[Dict[str, float | str]]:
    """Format top-k predictions as class/score mappings."""
    topk = min(topk, len(classes))
    indices = np.argsort(probs)[::-1][:topk]
    return [
        {"class": classes[idx], "score": float(probs[idx])}
        for idx in indices
    ]
