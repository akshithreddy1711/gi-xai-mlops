"""LIME image explanation helper."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries


class LimeImageExplainerWrapper:
    """Thin wrapper around lime_image for RGB classifier explanations."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess_fn: Callable[[np.ndarray], torch.Tensor],
        device: torch.device,
        class_names: Sequence[str],
        num_samples: int = 1000,
        num_features: int = 8,
        hide_rest: bool = False,
    ) -> None:
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.class_names = list(class_names)
        self.num_samples = num_samples
        self.num_features = num_features
        self.hide_rest = hide_rest
        self.explainer = lime_image.LimeImageExplainer()

    def _classifier_fn(self, images: Sequence[np.ndarray]) -> np.ndarray:
        batch = []
        for image in images:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            tensor = self.preprocess_fn(image).to(self.device, non_blocking=True)
            batch.append(tensor)
        batch_tensor = torch.stack(batch, dim=0)
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.detach().cpu().numpy()

    def explain(self, image: np.ndarray, top_label: int) -> np.ndarray:
        """Generate a LIME explanation overlay for the specified class index."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        explanation = self.explainer.explain_instance(
            image,
            classifier_fn=self._classifier_fn,
            top_labels=1,
            hide_color=None,
            num_samples=self.num_samples,
        )
        temp, mask = explanation.get_image_and_mask(
            label=top_label,
            positive_only=True,
            num_features=self.num_features,
            hide_rest=self.hide_rest,
        )
        overlay = mark_boundaries(temp.astype(np.float32) / 255.0, mask)
        return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
