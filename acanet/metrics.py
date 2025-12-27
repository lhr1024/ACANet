"""评价指标：Dice 与 IoU。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

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


def per_class_dice_iou(pred: Tensor, target: Tensor, num_classes: int, epsilon: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """返回每个类别的 Dice、IoU（形状: num_classes）。"""
    if pred.numel() == 0 or target.numel() == 0:
        raise ValueError("pred or target is empty")
    pred_classes = torch.argmax(pred, dim=1)
    pred_one_hot = _one_hot(pred_classes, num_classes)
    target_one_hot = _one_hot(target, num_classes)
    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=dims)
    cardinality = torch.sum(pred_one_hot + target_one_hot, dim=dims)
    dice = (2 * intersection + epsilon) / (cardinality + epsilon)
    union = torch.sum(pred_one_hot + target_one_hot, dim=dims) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return dice, iou


def brats_region_masks(one_hot_labels: Tensor) -> Dict[str, Tensor]:
    """根据 BraTS 定义组合区域：WT(1/2/3), TC(1/3), ET(3)。输入 one_hot_labels 形状 (N,C,H,W)。"""
    # 假设通道含背景+ 1,2,3 三类
    wt = one_hot_labels[:, 1:4].sum(dim=1, keepdim=True)  # WT = 1|2|3
    tc = one_hot_labels[:, (1, 3)].sum(dim=1, keepdim=True)  # TC = 1|3
    et = one_hot_labels[:, 3:4]  # ET = 3
    return {"wt": wt, "tc": tc, "et": et}


def brats_dice_iou(pred: Tensor, target: Tensor, num_classes: int, epsilon: float = 1e-6) -> Dict[str, float]:
    """计算 WT/TC/ET 以及平均 Dice/IoU（使用 softmax 概率进行软指标计算）。"""
    pred_probs = torch.softmax(pred, dim=1)
    tgt_oh = _one_hot(target, num_classes)
    pred_masks = brats_region_masks(pred_probs)
    tgt_masks = brats_region_masks(tgt_oh)

    def _binary_dice(pred_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        inter = (pred_mask * tgt_mask).sum(dim=(0, 2, 3))
        card = pred_mask.sum(dim=(0, 2, 3)) + tgt_mask.sum(dim=(0, 2, 3))
        return (2 * inter + epsilon) / (card + epsilon)

    def _binary_iou(pred_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        inter = (pred_mask * tgt_mask).sum(dim=(0, 2, 3))
        union = pred_mask.sum(dim=(0, 2, 3)) + tgt_mask.sum(dim=(0, 2, 3)) - inter
        return (inter + epsilon) / (union + epsilon)

    dice_wt = _binary_dice(pred_masks["wt"], tgt_masks["wt"])
    dice_tc = _binary_dice(pred_masks["tc"], tgt_masks["tc"])
    dice_et = _binary_dice(pred_masks["et"], tgt_masks["et"])
    iou_wt = _binary_iou(pred_masks["wt"], tgt_masks["wt"])
    iou_tc = _binary_iou(pred_masks["tc"], tgt_masks["tc"])
    iou_et = _binary_iou(pred_masks["et"], tgt_masks["et"])

    return {
        "dice_wt": dice_wt.item(),
        "dice_tc": dice_tc.item(),
        "dice_et": dice_et.item(),
        "dice_mean": float(torch.stack([dice_wt, dice_tc, dice_et]).mean().item()),
        "iou_wt": iou_wt.item(),
        "iou_tc": iou_tc.item(),
        "iou_et": iou_et.item(),
        "iou_mean": float(torch.stack([iou_wt, iou_tc, iou_et]).mean().item()),
    }


def dice_loss(
    pred: Tensor,
    target: Tensor,
    num_classes: int,
    weight_ce: Optional[float] = 0.5,
    class_weights: Optional[Tensor] = None,
) -> Tensor:
    """Dice + 交叉熵混合损失，对应论文的目标函数。"""
    ce = torch.nn.functional.cross_entropy(pred, target, weight=class_weights)
    dice = 1 - dice_coefficient(pred, target, num_classes)
    if weight_ce is None:
        return dice
    return weight_ce * ce + (1 - weight_ce) * dice


__all__ = ["dice_coefficient", "dice_loss", "iou_score"]
