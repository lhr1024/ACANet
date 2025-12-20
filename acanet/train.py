"""简化的 ACANet 训练脚本，便于对照论文实验流程。

运行方式：
    python -m acanet.train --features data/X.npy --labels data/Y.npy
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import NpyDatasetConfig, load_datasets
from .metrics import dice_coefficient, dice_loss, iou_score
from .model import ACANet, ACANetConfig, count_parameters


@dataclass
class TrainingConfig:
    dataset: NpyDatasetConfig
    model: ACANetConfig
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-3
    num_workers: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def create_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    train_set, val_set = load_datasets(config.dataset)
    if len(train_set) == 0:
        raise ValueError("Training set is empty")
    if len(val_set) == 0:
        raise ValueError("Validation set is empty")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader, device: torch.device, num_classes: int
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = dice_loss(logits, labels, num_classes=num_classes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return {"loss": total_loss / max(total_batches, 1)}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_dice += dice_coefficient(logits, labels, num_classes=num_classes).item()
            total_iou += iou_score(logits, labels, num_classes=num_classes).item()
            total_batches += 1
    denom = max(total_batches, 1)
    return {"dice": total_dice / denom, "iou": total_iou / denom}


def fit(config: TrainingConfig) -> Dict[str, float]:
    config.validate()
    device = torch.device(config.device)
    train_loader, val_loader = create_dataloaders(config)
    model = ACANet(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {}
    for epoch in range(config.num_epochs):
        train_stats = train_one_epoch(model, optimizer, train_loader, device, num_classes=config.model.num_classes)
        val_stats = evaluate(model, val_loader, device, num_classes=config.model.num_classes)
        history[epoch] = {"train_loss": train_stats["loss"], **val_stats}
        print(
            f"Epoch {epoch+1}/{config.num_epochs} | train_loss={train_stats['loss']:.4f} "
            f"| val_dice={val_stats['dice']:.4f} | val_iou={val_stats['iou']:.4f}")
    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACANet on .npy datasets")
    parser.add_argument("--features", required=True, help="Path to features .npy")
    parser.add_argument("--labels", required=True, help="Path to labels .npy")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI 入口
    args = parse_args()
    dataset_cfg = NpyDatasetConfig(features_path=args.features, labels_path=args.labels)
    model_cfg = ACANetConfig(in_channels=1, num_classes=args.num_classes, base_channels=args.base_channels)
    train_cfg = TrainingConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )
    fit(train_cfg)


if __name__ == "__main__":
    main()
