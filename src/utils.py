"""Utility helpers for reproducibility, configuration management, and metrics."""
from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

__all__ = [
    "AverageMeter",
    "count_parameters",
    "ensure_dir",
    "flatten_config",
    "get_device",
    "load_config",
    "load_json",
    "macro_f1_from_confusion",
    "save_json",
    "seed_worker",
    "set_seed",
    "topk_accuracy",
    "update_confusion_matrix",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """Seed dataloader workers deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the appropriate torch.device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path | str) -> Path:
    """Create the directory if it does not exist and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(obj: Any, path: Path | str) -> None:
    """Persist a JSON-serializable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def load_json(path: Path | str) -> Any:
    """Load JSON data from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_config(config: Mapping[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flatten a nested configuration dict for MLflow logging."""
    items: Dict[str, Any] = {}
    for key, value in config.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, Mapping):
            items.update(flatten_config(value, new_key))
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            items[new_key] = ",".join(str(v) for v in value)
        else:
            items[new_key] = value
    return items


class AverageMeter:
    """Track running averages for losses and metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> Tuple[float, ...]:
    """Compute the top-k accuracy for the specified values of k."""
    if output.ndim != 2:
        raise ValueError("Expected output tensor with shape (batch_size, num_classes).")
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    batch_size = target.size(0)
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append((correct_k.mul_(100.0 / batch_size)).item())
    return tuple(results)


def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    """Accumulate predictions into a confusion matrix."""
    if confusion.shape != (num_classes, num_classes):
        raise ValueError("Confusion matrix has unexpected shape.")
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)
    with torch.no_grad():
        indices = targets * num_classes + preds
        cm = torch.bincount(indices, minlength=num_classes * num_classes)
        cm = cm.reshape(num_classes, num_classes)
        confusion += cm


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    """Compute macro F1 score from a confusion matrix."""
    num_classes = confusion.size(0)
    f1_scores = []
    for cls_idx in range(num_classes):
        tp = confusion[cls_idx, cls_idx].item()
        fp = confusion[:, cls_idx].sum().item() - tp
        fn = confusion[cls_idx, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        f1_scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
