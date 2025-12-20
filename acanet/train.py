"""简化的 ACANet 训练脚本，便于对照论文实验流程。

运行方式：
    直接运行 python -m acanet.train
    所有必填参数在下方 CONFIG_BLOCK 中集中配置（无需命令行必填参数）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import NpyDatasetConfig, load_datasets
from .metrics import dice_coefficient, dice_loss, iou_score
from .model import ACANet, ACANetConfig, count_parameters


DEFAULT_FEATURES_PATH = "data/features"
DEFAULT_LABELS_PATH = "data/labels"


CONFIG_BLOCK = {
    # 数据路径（可指向目录或单个 .npy）
    "features": DEFAULT_FEATURES_PATH,
    "labels": DEFAULT_LABELS_PATH,
    # 训练超参
    "batch_size": 4,
    "epochs": 50,
    "learning_rate": 1e-3,
    "num_workers": 0,
    # 模型配置
    "num_classes": 4,
    "base_channels": 32,
    # 设备（GPU 环境下设置为 "cuda"，若无 GPU 可改为 "cpu"）
    "device": "cuda",
}


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
        outputs = model(images)
        pa, pd, pf = outputs["pa"], outputs["pd"], outputs["pf"]
        # 混合损失：对 Pa、Pd、Pf 分别计算 Dice+CE，累加
        loss = (
            dice_loss(pa, labels, num_classes=num_classes)
            + dice_loss(pd, labels, num_classes=num_classes)
            + dice_loss(pf, labels, num_classes=num_classes)
        )
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
            outputs = model(images)
            pf = outputs["pf"]
            total_dice += dice_coefficient(pf, labels, num_classes=num_classes).item()
            total_iou += iou_score(pf, labels, num_classes=num_classes).item()
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


def build_dataset_config(features: str | None, labels: str | None) -> NpyDatasetConfig:
    """根据传入路径或默认路径构建数据配置，便于显式设置特征/标签目录。"""

    feats = features if features is not None else DEFAULT_FEATURES_PATH
    labs = labels if labels is not None else DEFAULT_LABELS_PATH
    return NpyDatasetConfig(features_path=feats, labels_path=labs)


def main() -> None:  # pragma: no cover - CLI 入口
    cfg = CONFIG_BLOCK
    dataset_cfg = build_dataset_config(cfg["features"], cfg["labels"])
    model_cfg = ACANetConfig(in_channels=4, num_classes=cfg["num_classes"], base_channels=cfg["base_channels"])
    train_cfg = TrainingConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        batch_size=cfg["batch_size"],
        num_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        num_workers=cfg["num_workers"],
        device=cfg["device"],
    )
    fit(train_cfg)


if __name__ == "__main__":
    main()
