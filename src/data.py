"""Data loading utilities for the Kvasir gastrointestinal image dataset."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .utils import load_json, save_json, seed_worker

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class KvasirDataset(Dataset):
    """PyTorch dataset for GI images with albumentations transforms."""

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        transforms: Optional[A.BasicTransform] = None,
    ) -> None:
        if not samples:
            raise ValueError("Received empty sample list for dataset split.")
        self.samples = [
            {
                "path": Path(sample["path"]),
                "label_index": int(sample["label_index"]),
            }
            for sample in samples
        ]
        self.transforms = transforms

    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        path: Path = sample["path"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            tensor = augmented["image"].float()
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor, sample["label_index"]


def build_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    augment: bool,
) -> A.Compose:
    """Create albumentations transform pipeline for training or evaluation."""
    if augment:
        def _random_resized_crop() -> A.BasicTransform:
            kwargs = {
                "scale": (0.85, 1.0),
                "ratio": (0.9, 1.1),
                "p": 1.0,
            }
            try:
                return A.RandomResizedCrop(
                    height=image_size,
                    width=image_size,
                    **kwargs,
                )
            except (TypeError, ValueError):
                return A.RandomResizedCrop(
                    size=(image_size, image_size),
                    **kwargs,
                )

        transforms = [
            _random_resized_crop(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(
                max_holes=1,
                max_height=max(1, int(0.1 * image_size)),
                max_width=max(1, int(0.1 * image_size)),
                fill_value=0,
                p=0.2,
            ),
        ]
    else:
        transforms = [A.Resize(height=image_size, width=image_size)]
    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def _discover_classes(root: Path, explicit_classes: Optional[Sequence[str]] = None) -> List[str]:
    if explicit_classes:
        class_names = list(explicit_classes)
    else:
        class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not class_names:
        raise FileNotFoundError(
            f"No class folders found under {root}. Expected one folder per class."
        )
    return class_names


def _collect_samples(root: Path, class_to_idx: Dict[str, int]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for class_name, class_index in class_to_idx.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class directory {class_dir} to exist.")
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append(
                    {
                        "path": str(path.resolve()),
                        "label": class_name,
                        "label_index": class_index,
                    }
                )
    if not samples:
        raise RuntimeError(f"No image files discovered in {root}.")
    return samples


def _load_cached_splits(path: Path) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Malformed split cache at {path}.")
    return {split: list(records) for split, records in data.items()}


def _create_splits(
    samples: List[Dict[str, Any]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)

    if val_ratio > 0 and val_count == 0:
        val_count = 1
    if test_ratio > 0 and test_count == 0:
        test_count = 1

    train_count = total - val_count - test_count
    while train_count <= 0 and (val_count > 0 or test_count > 0):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        train_count = total - val_count - test_count

    if train_count <= 0:
        raise ValueError("Not enough samples to satisfy train/val/test split ratios.")

    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count : train_count + val_count]
    test_samples = shuffled[train_count + val_count :]
    return {"train": train_samples, "val": val_samples, "test": test_samples}


def create_dataloaders(
    data_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    seed: int,
) -> tuple[Dict[str, DataLoader], Dict[str, Any]]:
    root = Path(data_cfg["root"]).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root {root} does not exist. Please download Kvasir images first."
        )

    class_names = _discover_classes(root, data_cfg.get("classes"))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples = _collect_samples(root, class_to_idx)

    cache_enabled = bool(data_cfg.get("cache_splits", True))
    splits_file = data_cfg.get("splits_file")
    splits_path = Path(splits_file).expanduser() if splits_file else root / "splits.json"
    if not splits_path.is_absolute():
        splits_path = (root / splits_path).resolve()

    splits = _load_cached_splits(splits_path) if cache_enabled else None
    if splits is None:
        splits = _create_splits(
            samples,
            seed=seed,
            val_ratio=float(data_cfg.get("val_ratio", 0.1)),
            test_ratio=float(data_cfg.get("test_ratio", 0.1)),
        )
        if cache_enabled:
            save_json(splits, splits_path)
    else:
        for split_samples in splits.values():
            for record in split_samples:
                record["path"] = str(Path(record["path"]).resolve())

    image_size = int(data_cfg.get("image_size", 224))
    mean = data_cfg.get("mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("std", [0.229, 0.224, 0.225])

    train_samples = splits.get("train", [])
    if not train_samples:
        raise ValueError("Training split is empty; verify dataset structure and split ratios.")
    val_samples = splits.get("val", [])
    test_samples = splits.get("test", [])

    train_transform = build_transforms(image_size, mean, std, augment=True)
    eval_transform = build_transforms(image_size, mean, std, augment=False)

    train_dataset = KvasirDataset(train_samples, transforms=train_transform)

    dataloaders: Dict[str, DataLoader] = {}
    num_workers = int(data_cfg.get("num_workers", 4))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _build_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
            "worker_init_fn": seed_worker,
            "generator": generator,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = persistent_workers
        return DataLoader(**kwargs)

    train_loader = _build_loader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 32)),
        shuffle=True,
    )
    dataloaders["train"] = train_loader

    eval_batch_size = int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 32)))

    if val_samples:
        val_dataset = KvasirDataset(val_samples, transforms=eval_transform)
        dataloaders["val"] = _build_loader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    if test_samples:
        test_dataset = KvasirDataset(test_samples, transforms=eval_transform)
        dataloaders["test"] = _build_loader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    metadata = {
        "class_to_idx": class_to_idx,
        "idx_to_class": class_names,
        "split_counts": {split: len(records) for split, records in splits.items()},
        "image_size": image_size,
        "mean": mean,
        "std": std,
    }

    return dataloaders, metadata
