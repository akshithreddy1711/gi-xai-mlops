"""Training entrypoint with MLflow logging for GI disease classification."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import create_dataloaders  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    AverageMeter,
    count_parameters,
    ensure_dir,
    flatten_config,
    get_device,
    load_config,
    macro_f1_from_confusion,
    save_json,
    set_seed,
    topk_accuracy,
    update_confusion_matrix,
)


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience


def configure_scheduler(
    optimizer: optim.Optimizer,
    scheduler_cfg: Dict[str, Any],
    total_epochs: int,
) -> Tuple[Optional[object], bool, str]:
    """Instantiate a learning-rate scheduler based on config."""
    if not scheduler_cfg or not scheduler_cfg.get("name"):
        return None, False, ""
    name = scheduler_cfg["name"].lower()
    if name == "cosineannealinglr":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", total_epochs)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )
        return scheduler, False, ""
    if name == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
        return scheduler, False, ""
    if name == "reducelronplateau":
        monitor = scheduler_cfg.get("monitor", "val_loss")
        mode = "max" if monitor == "val_f1" else "min"
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 3)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
        )
        return scheduler, True, monitor
    raise ValueError(f"Unsupported scheduler '{scheduler_cfg['name']}'")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    max_grad_norm: Optional[float],
    num_classes: int,
    use_amp: bool,
) -> Dict[str, float]:
    """Run one epoch of training."""
    model.train()
    losses = AverageMeter()
    top1_meter = AverageMeter()
    top3_meter = AverageMeter()
    confusion = torch.zeros(num_classes, num_classes, device="cpu")

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler and use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        probs = torch.softmax(outputs, dim=1)
        top1, top3 = topk_accuracy(probs, targets, topk=(1, 3))
        top1_meter.update(top1, images.size(0))
        top3_meter.update(top3, images.size(0))
        preds = torch.argmax(probs, dim=1)
        update_confusion_matrix(confusion, preds.cpu(), targets.cpu(), num_classes)

    metrics = {
        "loss": losses.avg,
        "top1": top1_meter.avg,
        "top3": top3_meter.avg,
        "f1": macro_f1_from_confusion(confusion),
    }
    return metrics


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Evaluate the model on a validation or test split."""
    model.eval()
    losses = AverageMeter()
    top1_meter = AverageMeter()
    top3_meter = AverageMeter()
    confusion = torch.zeros(num_classes, num_classes, device="cpu")

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))
            probs = torch.softmax(outputs, dim=1)
            top1, top3 = topk_accuracy(probs, targets, topk=(1, 3))
            top1_meter.update(top1, images.size(0))
            top3_meter.update(top3, images.size(0))
            preds = torch.argmax(probs, dim=1)
            update_confusion_matrix(confusion, preds.cpu(), targets.cpu(), num_classes)

    metrics = {
        "loss": losses.avg,
        "top1": top1_meter.avg,
        "top3": top3_meter.avg,
        "f1": macro_f1_from_confusion(confusion),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GI disease classifier with MLflow tracking."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional MLflow experiment name override.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = get_device(prefer_gpu=True)
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    data_cfg = dict(config.get("data", {}))
    if "root" not in data_cfg:
        raise KeyError("Config must include data.root")
    data_root = Path(data_cfg["root"]).expanduser()
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    data_cfg["root"] = str(data_root)
    if data_cfg.get("splits_file"):
        splits_path = Path(data_cfg["splits_file"]).expanduser()
        if not splits_path.is_absolute():
            splits_path = (PROJECT_ROOT / splits_path).resolve()
        data_cfg["splits_file"] = str(splits_path)

    training_cfg = dict(config.get("training", {}))
    dataloaders, metadata = create_dataloaders(data_cfg, training_cfg, seed=seed)
    if "val" not in dataloaders:
        raise RuntimeError("Validation split is required for model selection.")

    model_cfg = dict(config.get("model", {}))
    model_name = model_cfg.get("name", "efficientnet_b0")
    dropout = float(model_cfg.get("dropout", 0.0))
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))

    num_classes = len(metadata["idx_to_class"])
    model, _ = build_model(
        name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )

    mixed_precision = bool(training_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=mixed_precision)

    scheduler_cfg = dict(training_cfg.get("lr_scheduler", {}))
    epochs = int(training_cfg.get("epochs", 30))
    scheduler, scheduler_requires_metric, scheduler_monitor = configure_scheduler(
        optimizer, scheduler_cfg, total_epochs=epochs
    )

    early_cfg = training_cfg.get("early_stopping")
    early_stopper = None
    if early_cfg:
        early_stopper = EarlyStopping(
            patience=int(early_cfg.get("patience", 5)),
            min_delta=float(early_cfg.get("min_delta", 0.0)),
        )

    artifacts_cfg = dict(config.get("artifacts", {}))
    model_dir = Path(artifacts_cfg.get("model_dir", "models"))
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()
    ensure_dir(model_dir)
    best_model_path = model_dir / artifacts_cfg.get("best_model_filename", "best.pt")
    labels_path = model_dir / artifacts_cfg.get("labels_filename", "labels.json")

    mlflow_cfg = dict(config.get("mlflow", {}))
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg.get("tracking_uri", "file:./mlruns"),
    )
    if tracking_uri.startswith("file:"):
        uri_path = tracking_uri[len("file:") :]
        uri_path_obj = Path(uri_path)
        if not uri_path_obj.is_absolute():
            tracking_uri = f"file:{(PROJECT_ROOT / uri_path_obj).resolve()}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = args.experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        mlflow_cfg.get("experiment_name", "gi-xai-classifier"),
    )
    mlflow.set_experiment(experiment_name)
    run_name = args.run_name or os.getenv("MLFLOW_RUN_NAME", mlflow_cfg.get("run_name"))

    print(f"Using device: {device}")
    print(f"Discovered {num_classes} classes: {metadata['idx_to_class']}")
    print(f"Split counts: {metadata.get('split_counts')}")

    max_grad_norm_cfg = training_cfg.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm_cfg) if max_grad_norm_cfg is not None else None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(config))
        mlflow.log_param("model/name", model_name)
        mlflow.log_param("model/num_parameters", count_parameters(model))
        mlflow.log_param("data/num_classes", num_classes)
        mlflow.log_param("system/device", str(device))
        mlflow.log_param("training/epochs", epochs)
        for split_name, count in metadata.get("split_counts", {}).items():
            mlflow.log_param(f"data/{split_name}_samples", count)
        if config_path.exists():
            mlflow.log_artifact(str(config_path))

        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        test_loader = dataloaders.get("test")

        best_metric = -float("inf")
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                max_grad_norm=max_grad_norm,
                num_classes=num_classes,
                use_amp=mixed_precision,
            )
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
            )
            epoch_time = time.perf_counter() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{epochs:03d} "
                f"| train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"| train_f1={train_metrics['f1']:.2f} val_f1={val_metrics['f1']:.2f} "
                f"| top1={val_metrics['top1']:.2f} top3={val_metrics['top3']:.2f} "
                f"| lr={current_lr:.3e} time={epoch_time:.1f}s"
            )

            mlflow.log_metric("train/loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric("train/f1", train_metrics["f1"], step=epoch)
            mlflow.log_metric("train/top1", train_metrics["top1"], step=epoch)
            mlflow.log_metric("train/top3", train_metrics["top3"], step=epoch)
            mlflow.log_metric("val/loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val/f1", val_metrics["f1"], step=epoch)
            mlflow.log_metric("val/top1", val_metrics["top1"], step=epoch)
            mlflow.log_metric("val/top3", val_metrics["top3"], step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)
            mlflow.log_metric("epoch/time_sec", epoch_time, step=epoch)

            if scheduler:
                if scheduler_requires_metric:
                    monitor_value = (
                        val_metrics["f1"] if scheduler_monitor == "val_f1" else val_metrics["loss"]
                    )
                    scheduler.step(monitor_value)
                else:
                    scheduler.step()

            if val_metrics["f1"] > best_metric:
                best_metric = val_metrics["f1"]
                best_epoch = epoch
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1": val_metrics["f1"],
                }
                torch.save(checkpoint, best_model_path)
                label_payload = {
                    "idx_to_class": metadata["idx_to_class"],
                    "class_to_idx": metadata["class_to_idx"],
                }
                save_json(label_payload, labels_path)

            if early_stopper and early_stopper.step(val_metrics["f1"]):
                print("Early stopping triggered.")
                break

        mlflow.log_metric("best/val_f1", best_metric)
        mlflow.log_param("training/best_epoch", best_epoch)
        mlflow.log_artifact(str(best_model_path), artifact_path="models")
        mlflow.log_artifact(str(labels_path), artifact_path="models")

        if test_loader:
            checkpoint = torch.load(best_model_path, map_location=device)
            state_dict = checkpoint.get("model_state", checkpoint)
            model.load_state_dict(state_dict)
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
            )
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test/{key}", value)
            print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
