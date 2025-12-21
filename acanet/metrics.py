"""评价指标：Dice 与 IoU。"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _one_hot(labels: Tensor, num_classes: int) -> Tensor:
    if labels.dtype not in (torch.long, torch.int):
        labels = labels.long()
    # 防止越界导致 CUDA device-side assert，先在 CPU 校验 + 编码，再移回设备
    with torch.no_grad():
        labels_cpu = labels.detach().cpu()
        min_val = labels_cpu.min().item()
        max_val = labels_cpu.max().item()
        if min_val < 0 or max_val >= num_classes:
            raise ValueError(f"Label values must be in [0, {num_classes-1}], got min={min_val}, max={max_val}")
        if labels.dim() != 3:
            raise ValueError(f"Labels must have shape (N, H, W), got {labels.shape}")
        one_hot_cpu = torch.nn.functional.one_hot(labels_cpu, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # 与 labels.device 对齐，避免作用域问题导致 None 返回
    return one_hot_cpu.to(labels.device)


def dice_coefficient(pred: Tensor, target: Tensor, num_classes: int, epsilon: float = 1e-6) -> Tensor:
    """计算平均 Dice 系数。

    Args:
        pred: logits 或概率，形状 (N, C, H, W)
        target: 标签，形状 (N, H, W)
    """
    if pred.numel() == 0:
        raise ValueError("pred is empty")
    if target.numel() == 0:
        raise ValueError("target is empty")
    pred_classes = torch.argmax(pred, dim=1)
    pred_one_hot = _one_hot(pred_classes, num_classes)
    target_one_hot = _one_hot(target, num_classes)
    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=dims)
    cardinality = torch.sum(pred_one_hot + target_one_hot, dim=dims)
    dice = (2 * intersection + epsilon) / (cardinality + epsilon)
    return dice.mean()


def iou_score(pred: Tensor, target: Tensor, num_classes: int, epsilon: float = 1e-6) -> Tensor:
    """计算平均 IoU。"""
    pred_classes = torch.argmax(pred, dim=1)
    pred_one_hot = _one_hot(pred_classes, num_classes)
    target_one_hot = _one_hot(target, num_classes)
    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=dims)
    union = torch.sum(pred_one_hot + target_one_hot, dim=dims) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean()


def dice_loss(pred: Tensor, target: Tensor, num_classes: int, weight_ce: Optional[float] = 0.5) -> Tensor:
    """Dice + 交叉熵混合损失，对应论文的目标函数。"""
    ce = torch.nn.functional.cross_entropy(pred, target)
    dice = 1 - dice_coefficient(pred, target, num_classes)
    if weight_ce is None:
        return dice
    return weight_ce * ce + (1 - weight_ce) * dice


__all__ = ["dice_coefficient", "dice_loss", "iou_score"]
