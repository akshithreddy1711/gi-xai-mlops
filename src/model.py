"""Model factory for EfficientNet-B0 and ResNet-50 classifiers."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "efficientnet_b0": {
        "builder": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "head_attr": "classifier",
        "target_layer": "features.7",
    },
    "resnet50": {
        "builder": resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "head_attr": "fc",
        "target_layer": "layer4.2",
    },
}


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
) -> Tuple[nn.Module, str]:
    """Construct a classifier model and return it with the default Grad-CAM layer."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{name}'. Choose from {list(MODEL_REGISTRY)}.")

    entry = MODEL_REGISTRY[name]
    weights = entry["weights"] if pretrained else None
    model: nn.Module = entry["builder"](weights=weights)
    head_attr = entry["head_attr"]

    if name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        classifier_layers = []
        if dropout and dropout > 0:
            classifier_layers.append(nn.Dropout(p=dropout, inplace=True))
        classifier_layers.append(nn.Linear(in_features, num_classes))
        model.classifier = nn.Sequential(*classifier_layers)
    elif name.startswith("resnet"):
        in_features = model.fc.in_features
        if dropout and dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Head configuration missing for model '{name}'.")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        head_module = resolve_target_layer(model, head_attr)
        for param in head_module.parameters():
            param.requires_grad = True

    return model, entry["target_layer"]


def get_default_target_layer(model_name: str) -> str:
    """Return the default Grad-CAM layer name for a given model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'.")
    return MODEL_REGISTRY[model_name]["target_layer"]


def resolve_target_layer(model: nn.Module, layer_path: str) -> nn.Module:
    """Resolve a dot/index notation layer path into an actual module reference."""
    tokens = []
    for token in layer_path.replace("[", ".").replace("]", "").split("."):
        token = token.strip()
        if token:
            tokens.append(token)

    current: nn.Module = model
    for token in tokens:
        if token.lstrip("-").isdigit():
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current
