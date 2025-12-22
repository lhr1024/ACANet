"""简化的 ACANet 训练脚本，便于对照论文实验流程。

运行方式：
    直接运行 python -m acanet.train
    所有必填参数在下方 CONFIG_BLOCK 中集中配置（无需命令行必填参数）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .data import NpyDatasetConfig, load_datasets
from .metrics import brats_dice_iou, dice_loss
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
    # 指标曲线保存路径
    "plot_path": "training_metrics.png",
    # 每隔多少轮保存一次权重（None 则不保存）
    "checkpoint_interval": 5,
    "checkpoint_dir": "checkpoints",
    # 是否在训练前做快速有效性检查（数据量、batch、标签分布、loss 是否下降）
    "sanity_check": True,
    # 是否在训练过程中打印详细诊断信息（每 N 个 batch）
    "diag_interval": 10,
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
    checkpoint_interval: int | None = None
    checkpoint_dir: str | None = None

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when set")


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
    progress = tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(progress):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        pa, pd, pf = outputs["pa"], outputs["pd"], outputs["pf"]
        # 若预测与标签空间尺寸不一致，先插值对齐再计算损失
        if pf.shape[-2:] != labels.shape[-2:]:
            pa = nn.functional.interpolate(pa, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pd = nn.functional.interpolate(pd, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pf = nn.functional.interpolate(pf, size=labels.shape[-2:], mode="bilinear", align_corners=False)
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
        progress.set_postfix(loss=loss.item())
        # 诊断输出：每 diag_interval 个 batch 打印一次标签范围与 loss
        if CONFIG_BLOCK.get("diag_interval", 0) > 0 and (step + 1) % CONFIG_BLOCK["diag_interval"] == 0:
            with torch.no_grad():
                lbl_min = labels.min().item()
                lbl_max = labels.max().item()
                unique_vals = torch.unique(labels).cpu().tolist()
            print(f"[Diag] step {step+1}: loss={loss.item():.4f}, label min/max={lbl_min}/{lbl_max}, unique={unique_vals}")
    return {"loss": total_loss / max(total_batches, 1)}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "dice_wt": 0.0,
        "dice_tc": 0.0,
        "dice_et": 0.0,
        "iou_wt": 0.0,
        "iou_tc": 0.0,
        "iou_et": 0.0,
    }
    total_batches = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pa, pd, pf = outputs["pa"], outputs["pd"], outputs["pf"]
            if pf.shape[-2:] != labels.shape[-2:]:
                pa = nn.functional.interpolate(pa, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                pd = nn.functional.interpolate(pd, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                pf = nn.functional.interpolate(pf, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = (
                dice_loss(pa, labels, num_classes=num_classes)
                + dice_loss(pd, labels, num_classes=num_classes)
                + dice_loss(pf, labels, num_classes=num_classes)
            )
            metrics = brats_dice_iou(pf, labels, num_classes=num_classes)
            totals["loss"] += loss.item()
            totals["dice"] += metrics["dice_mean"]
            totals["iou"] += metrics["iou_mean"]
            totals["dice_wt"] += metrics["dice_wt"]
            totals["dice_tc"] += metrics["dice_tc"]
            totals["dice_et"] += metrics["dice_et"]
            totals["iou_wt"] += metrics["iou_wt"]
            totals["iou_tc"] += metrics["iou_tc"]
            totals["iou_et"] += metrics["iou_et"]
            total_batches += 1
    denom = max(total_batches, 1)
    return {k: v / denom for k, v in totals.items()}


def fit(config: TrainingConfig) -> Dict[str, float]:
    config.validate()
    device = torch.device(config.device)
    train_loader, val_loader = create_dataloaders(config)
    model = ACANet(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    history = {}
    for epoch in range(config.num_epochs):
        train_stats = train_one_epoch(model, optimizer, train_loader, device, num_classes=config.model.num_classes)
        val_stats = evaluate(model, val_loader, device, num_classes=config.model.num_classes)
        history[epoch] = {"train_loss": train_stats["loss"], **val_stats}
        metrics_msg = (
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"train_loss={train_stats['loss']:.4f} | val_loss={val_stats['loss']:.4f} | "
            f"dice_wt={val_stats['dice_wt']:.4f} | dice_tc={val_stats['dice_tc']:.4f} | "
            f"dice_et={val_stats['dice_et']:.4f} | dice={val_stats['dice']:.4f} | "
            f"iou_wt={val_stats['iou_wt']:.4f} | iou_tc={val_stats['iou_tc']:.4f} | "
            f"iou_et={val_stats['iou_et']:.4f} | iou={val_stats['iou']:.4f}"
        )
        print(metrics_msg)
        if config.checkpoint_interval and (epoch + 1) % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir or "", f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")
    return history


def save_history_plots(history: Dict[int, Dict[str, float]], plot_path: str) -> None:
    """将训练/验证指标可视化为折线图并保存。"""

    if not history:
        print("No history to plot.")
        return
    epochs = sorted(history.keys())

    def _collect(key: str) -> list[float]:
        return [history[e][key] for e in epochs if key in history[e]]

    base, ext = os.path.splitext(plot_path)
    ext = ext if ext else ".png"

    # 1) Dice 曲线：WT/TC/ET/Mean
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, _collect("dice_wt"), label="Dice WT")
    plt.plot(epochs, _collect("dice_tc"), label="Dice TC")
    plt.plot(epochs, _collect("dice_et"), label="Dice ET")
    plt.plot(epochs, _collect("dice"), label="Dice Mean")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice Curves (WT/TC/ET/Mean)")
    plt.legend()
    plt.tight_layout()
    dice_path = f"{base}_dice{ext}"
    plt.savefig(dice_path)
    plt.close()

    # 2) IoU 曲线：WT/TC/ET/Mean
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, _collect("iou_wt"), label="IoU WT")
    plt.plot(epochs, _collect("iou_tc"), label="IoU TC")
    plt.plot(epochs, _collect("iou_et"), label="IoU ET")
    plt.plot(epochs, _collect("iou"), label="IoU Mean")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU Curves (WT/TC/ET/Mean)")
    plt.legend()
    plt.tight_layout()
    iou_path = f"{base}_iou{ext}"
    plt.savefig(iou_path)
    plt.close()

    # 3) Loss 曲线：Train / Val
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, _collect("train_loss"), label="Train Loss")
    plt.plot(epochs, _collect("loss"), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (Train vs Val)")
    plt.legend()
    plt.tight_layout()
    loss_path = f"{base}_loss{ext}"
    plt.savefig(loss_path)
    plt.close()

    print(f"Saved dice curves to {dice_path}")
    print(f"Saved IoU curves to {iou_path}")
    print(f"Saved loss curves to {loss_path}")


def run_sanity_checks(
    train_loader: DataLoader, model: nn.Module, device: torch.device, num_classes: int, learning_rate: float
) -> None:
    """快速检查：数据量/批大小/标签分布/一次前向反向是否正常，loss 是否下降趋势。"""

    if len(train_loader) == 0:
        raise ValueError("Sanity check failed: train_loader is empty")
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"[Sanity] batch size: {images.shape[0]}, image shape: {tuple(images.shape)}, label shape: {tuple(labels.shape)}")
    labels_np = labels.detach().cpu().numpy()
    unique_vals = set(labels_np.reshape(-1).tolist())
    print(f"[Sanity] label unique values: {sorted(list(unique_vals))}")
    if max(unique_vals) >= num_classes or min(unique_vals) < 0:
        raise ValueError(f"Sanity check failed: label values out of range [0,{num_classes-1}]")

    # 简单前向+反向一步，观察 loss 是否可计算
    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    outputs = model(images)
    pa, pd, pf = outputs["pa"], outputs["pd"], outputs["pf"]
    # 若空间尺寸不一致，先对齐
    if pf.shape[-2:] != labels.shape[-2:]:
        pa = nn.functional.interpolate(pa, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pd = nn.functional.interpolate(pd, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pf = nn.functional.interpolate(pf, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    loss1 = (
        dice_loss(pa, labels, num_classes=num_classes)
        + dice_loss(pd, labels, num_classes=num_classes)
        + dice_loss(pf, labels, num_classes=num_classes)
    )
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    print(f"[Sanity] initial mixed loss: {loss1.item():.4f}")
    # 再跑一次前向，不反向，观察 loss 是否有变化趋势
    with torch.no_grad():
        outputs2 = model(images)
        pa2, pd2, pf2 = outputs2["pa"], outputs2["pd"], outputs2["pf"]
        if pf2.shape[-2:] != labels.shape[-2:]:
            pa2 = nn.functional.interpolate(pa2, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pd2 = nn.functional.interpolate(pd2, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pf2 = nn.functional.interpolate(pf2, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss2 = (
            dice_loss(pa2, labels, num_classes=num_classes)
            + dice_loss(pd2, labels, num_classes=num_classes)
            + dice_loss(pf2, labels, num_classes=num_classes)
        )
    print(f"[Sanity] post-step mixed loss (no backward): {loss2.item():.4f}")


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
        checkpoint_interval=cfg.get("checkpoint_interval"),
        checkpoint_dir=cfg.get("checkpoint_dir"),
    )
    # 可选的快速有效性检查
    history = {}
    if cfg.get("sanity_check", False):
        tmp_model = ACANet(model_cfg)
        tmp_train_loader, _ = create_dataloaders(train_cfg)
        run_sanity_checks(tmp_train_loader, tmp_model, torch.device(cfg["device"]), cfg["num_classes"], cfg["learning_rate"])
        del tmp_model, tmp_train_loader
    history = fit(train_cfg)
    save_history_plots(history, cfg["plot_path"])


if __name__ == "__main__":
    main()
