"""Grad-CAM utilities for model explainability."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM heatmaps for convolutional neural networks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:  # PyTorch < 1.8 fallback
            self.handles.append(self.target_layer.register_backward_hook(backward_hook))

    def close(self) -> None:
        """Remove registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __del__(self) -> None:
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for the provided input and target class."""
        self.model.eval()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
        target = output[:, class_idx]
        target.sum().backward(retain_graph=retain_graph)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured for Grad-CAM.")

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * activations, dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam.squeeze().detach().cpu().numpy()
        heatmap -= heatmap.min()
        denominator = heatmap.max() + 1e-8
        heatmap /= denominator
        return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap onto an RGB image using OpenCV color maps."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(colored, alpha, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def resolve_target_layer(model: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    """Resolve a dotted path (with optional indices) to a module on the model."""
    tokens = []
    for token in layer_path.replace("[", ".").replace("]", "").split("."):
        token = token.strip()
        if token:
            tokens.append(token)

    current: torch.nn.Module = model
    for token in tokens:
        if token.lstrip("-").isdigit():
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current
