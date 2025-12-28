import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import pytest
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# 若核心依赖缺失，则跳过整份测试，方便在最小依赖环境下导入
_missing = [name for name, mod in {"numpy": np, "torch": torch}.items() if mod is None]
if _missing:
    pytest.skip(
        "Skipping ACANet NPY inference test because missing dependencies: " + ", ".join(_missing),
        allow_module_level=True,
    )

try:
    from acanet.metrics import dice_coefficient, iou_score
    from acanet.model import ACANet, ACANetConfig
except ImportError:  # pragma: no cover - package not installed in minimal envs
    pytest.skip(
        "Skipping ACANet NPY inference test because project package is not importable (install in editable mode).",
        allow_module_level=True,
    )

# 默认配置：在这里填好路径后可直接运行 python tests/test_acanet.py
CONFIG = {
    "images": "data/test_images_npy",   # 测试集 .npy 目录（特征）
    "labels": "data/test_labels_npy",   # 测试集标签目录
    "model": "last_model.pth",          # 训练好的模型权重
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 运行设备
    "num_classes": 4,                   # 类别数（含背景）
}

DEFAULT_TEST_IMAGES_PATH = "/LHRP/ACANet/tests/test_images_npy"
DEFAULT_TEST_LABELS_PATH = "/LHRP/ACANet/tests/test_labels_npy"
DEFAULT_GT_NPY_DIR = "/LHRP/ACANet/tests/gt_npy"
DEFAULT_PRED_NPY_DIR = "/LHRP/ACANet/tests/result"


def load_npy_slice(path: Path) -> np.ndarray:
    """读取单个 .npy 切片，返回形状 (C,H,W)。"""
    arr = np.load(path)
    if arr.size == 0:
        raise ValueError(f"Empty feature slice in {path}")
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = np.transpose(arr, (2, 0, 1))  # (H,W,4) -> (4,H,W)
    if arr.ndim != 3:
        raise ValueError(f"Expected (C,H,W) slice in {path}, got {arr.shape}")
    return arr.astype(np.float32)


def run_inference(model_path: str, feature_paths: List[Path], device: str, num_classes: int) -> List[torch.Tensor]:
    """对切片进行预测，返回 logits 列表（每个切片一个 Tensor，形状 CxHxW）。"""
    device_obj = torch.device(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    cfg = ACANetConfig(in_channels=4, num_classes=num_classes, base_channels=32)
    model = ACANet(cfg)
    state = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(state)
    model.to(device_obj)
    model.eval()
    preds = []
    with torch.no_grad():
        for p in feature_paths:
            arr = load_npy_slice(p)  # (C,H,W)
            x = torch.from_numpy(arr).unsqueeze(0).to(device_obj)  # (1,C,H,W)
            out = model(x)["pf"].squeeze(0).cpu()  # (C,H,W)
            preds.append(out)
    return preds


def save_prediction_masks(
    preds: List[torch.Tensor], out_dir: Path, base_name: str
) -> None:
    """保存预测掩码为 .npy，形状 (H,W,3) 对应 WT/TC/ET。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, pred in enumerate(preds):
        # pred: (C,H,W)
        if pred.numel() == 0:
            print(f"[WARN] Skip empty prediction at index {idx}; no mask saved.")
            continue
        pred_classes = torch.argmax(pred, dim=0).cpu().numpy()
        wt = (pred_classes > 0).astype(np.uint8)
        tc = np.logical_or(pred_classes == 1, pred_classes == 3).astype(np.uint8)
        et = (pred_classes == 3).astype(np.uint8)
        mask = np.stack([wt, tc, et], axis=-1)  # (H,W,3)
        save_path = out_dir / f"{base_name}_slice{idx:03d}.npy"
        np.save(save_path, mask)


def dice_coefficient_np(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    return float((2.0 * intersection + smooth) / (np.sum(pred_flat) + np.sum(gt_flat) + smooth))


def iou_score_np(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    return float((intersection + smooth) / (union + smooth))


def convert_brats_gt_label(gt_label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wt_mask = np.zeros_like(gt_label, dtype=np.float32)
    wt_mask[(gt_label == 1) | (gt_label == 2) | (gt_label == 4)] = 1.0

    tc_mask = np.zeros_like(gt_label, dtype=np.float32)
    tc_mask[(gt_label == 1) | (gt_label == 4)] = 1.0

    et_mask = np.zeros_like(gt_label, dtype=np.float32)
    et_mask[gt_label == 4] = 1.0

    return wt_mask, tc_mask, et_mask


def parse_brats_pred(pred_npy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_wt = pred_npy[:, :, 0].astype(np.float32)
    pred_tc = pred_npy[:, :, 1].astype(np.float32)
    pred_et = pred_npy[:, :, 2].astype(np.float32)

    pred_wt = (pred_wt > 0.5).astype(np.float32)
    pred_tc = (pred_tc > 0.5).astype(np.float32)
    pred_et = (pred_et > 0.5).astype(np.float32)

    return pred_wt, pred_tc, pred_et


def calculate_brats_metrics(gt_dir: Path, pred_dir: Path) -> None:
    gt_paths = sorted(glob.glob(str(gt_dir / "*.npy")))
    pred_paths = sorted(glob.glob(str(pred_dir / "*.npy")))

    if len(pred_paths) == 0:
        print(f"[WARN] No prediction files found in {pred_dir}; skipping BraTS metrics.")
        return

    if len(gt_paths) != len(pred_paths):
        raise ValueError(
            f"The number of GT files ({len(gt_paths)}) does not match the number of prediction files ({len(pred_paths)})!"
        )
    print(f"Found {len(gt_paths)} sets of data for evaluation")

    dice_wt_sum, dice_tc_sum, dice_et_sum = 0.0, 0.0, 0.0
    iou_wt_sum, iou_tc_sum, iou_et_sum = 0.0, 0.0, 0.0

    for gt_path, pred_path in tqdm(zip(gt_paths, pred_paths), total=len(gt_paths), desc="Calculating metrics"):
        gt_label = np.load(gt_path)
        pred_npy = np.load(pred_path)

        if gt_label.shape != (224, 224):
            raise ValueError(
                f"GT file {os.path.basename(gt_path)} has an abnormal shape. Expected (224,224), but got {gt_label.shape}"
            )
        if pred_npy.shape != (224, 224, 3):
            raise ValueError(
                f"Prediction file {os.path.basename(pred_path)} has an abnormal shape. Expected (224,224,3), but got {pred_npy.shape}"
            )

        gt_wt, gt_tc, gt_et = convert_brats_gt_label(gt_label)
        pred_wt, pred_tc, pred_et = parse_brats_pred(pred_npy)

        dice_wt_sum += dice_coefficient_np(pred_wt, gt_wt)
        dice_tc_sum += dice_coefficient_np(pred_tc, gt_tc)
        dice_et_sum += dice_coefficient_np(pred_et, gt_et)

        iou_wt_sum += iou_score_np(pred_wt, gt_wt)
        iou_tc_sum += iou_score_np(pred_tc, gt_tc)
        iou_et_sum += iou_score_np(pred_et, gt_et)

    num_samples = len(gt_paths)
    dice_wt_avg = dice_wt_sum / num_samples
    dice_tc_avg = dice_tc_sum / num_samples
    dice_et_avg = dice_et_sum / num_samples
    dice_total_avg = (dice_wt_avg + dice_tc_avg + dice_et_avg) / 3.0

    iou_wt_avg = iou_wt_sum / num_samples
    iou_tc_avg = iou_tc_sum / num_samples
    iou_et_avg = iou_et_sum / num_samples
    iou_total_avg = (iou_wt_avg + iou_tc_avg + iou_et_avg) / 3.0

    dice_iou_avg = (dice_total_avg + iou_total_avg) / 2.0
    print("\n" + "=" * 50)
    print("BraTS Segmentation Metrics Evaluation Results")
    print("=" * 50)
    print("Dice Coefficient")
    print(f"  WT: {dice_wt_avg:.4f}")
    print(f"  TC: {dice_tc_avg:.4f}")
    print(f"  ET: {dice_et_avg:.4f}")
    print(f"  Average Dice: {dice_total_avg:.4f}")
    print("-" * 30)
    print("IoU")
    print(f"  WT: {iou_wt_avg:.4f}")
    print(f"  TC: {iou_tc_avg:.4f}")
    print(f"  ET: {iou_et_avg:.4f}")
    print(f"  Average IoU: {iou_total_avg:.4f}")
    print("=" * 50)
    print(f"  Average dice and IoU: {dice_iou_avg:.4f}")


def compute_metrics(preds: List[torch.Tensor], labels: List[torch.Tensor], num_classes: int) -> Tuple[float, float]:
    """逐切片计算 Dice/IoU，返回均值。"""
    if not preds or not labels:
        raise ValueError("No predictions or labels to evaluate; check that .npy inputs are non-empty.")
    dices, ious = [], []
    for idx, (pred, target) in enumerate(zip(preds, labels)):
        if pred.numel() == 0 or target.numel() == 0:
            print(f"[WARN] Skip empty prediction/label at index {idx}; check input slice sizes.")
            continue
        pred = pred.unsqueeze(0)  # (1,C,H,W)
        target = target.unsqueeze(0)  # (1,H,W)
        dices.append(dice_coefficient(pred, target, num_classes=num_classes).item())
        ious.append(iou_score(pred, target, num_classes=num_classes).item())
    if not dices:
        print("[WARN] All prediction/label slices were empty; returning 0.0 for metrics.")
        return 0.0, 0.0
    return float(np.mean(dices)), float(np.mean(ious))


def load_label_npy_slices(label_paths: List[Path]) -> List[torch.Tensor]:
    """读取 .npy 标签切片，每个形状 (H,W)。"""
    slices = []
    for path in label_paths:
        arr = np.load(path)
        if arr.size == 0:
            raise ValueError(f"Empty label slice in {path}")
        if arr.ndim != 2:
            raise ValueError(f"Label slice expected (H,W) in {path}, got {arr.shape}")
        slices.append(torch.from_numpy(arr.astype(np.int64)))
    return slices


def main() -> None:
    parser = argparse.ArgumentParser(description="ACANet NPY inference")
    parser.add_argument("--images", help="测试集 .npy 目录（特征）")
    parser.add_argument("--labels", help="测试集标签 .npy 目录")
    parser.add_argument("--model", help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--device", help="运行设备")
    parser.add_argument("--num-classes", type=int, help="类别数（含背景）")
    parser.add_argument("--gt-npy-dir", help="GT .npy 目录（224x224）")
    parser.add_argument("--pred-npy-dir", help="预测 .npy 目录（224x224x3）")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if not args.images:
        cfg["images"] = DEFAULT_TEST_IMAGES_PATH
    if args.images:
        cfg["images"] = args.images
    if not args.labels:
        cfg["labels"] = DEFAULT_TEST_LABELS_PATH
    if args.labels:
        cfg["labels"] = args.labels
    if args.model:
        cfg["model"] = args.model
    if args.device:
        cfg["device"] = args.device
    if args.num_classes:
        cfg["num_classes"] = args.num_classes

    img_dir = Path(cfg["images"])
    lbl_dir = Path(cfg["labels"])
    gt_dir = Path(args.gt_npy_dir) if args.gt_npy_dir else Path(DEFAULT_GT_NPY_DIR)
    pred_dir = Path(args.pred_npy_dir) if args.pred_npy_dir else Path(DEFAULT_PRED_NPY_DIR)
    result_dir = Path("/LHRP/ACANet/tests/result")

    img_files_npy = sorted(glob.glob(str(img_dir / "*.npy")))
    lbl_files_npy = sorted(glob.glob(str(lbl_dir / "*.npy")))

    if len(img_files_npy) == 0 or len(img_files_npy) != len(lbl_files_npy):
        raise FileNotFoundError("Npy 图像与标签数量不匹配或为空")

    all_dices, all_ious = [], []
    preds = run_inference(cfg["model"], [Path(p) for p in img_files_npy], cfg["device"], cfg["num_classes"])
    label_slices = load_label_npy_slices([Path(p) for p in lbl_files_npy])
    n = min(len(preds), len(label_slices))
    dice, iou = compute_metrics(preds[:n], label_slices[:n], cfg["num_classes"])
    all_dices.append(dice)
    all_ious.append(iou)
    print(f"Npy slices: Dice={dice:.4f}, IoU={iou:.4f}")
    save_prediction_masks(preds[:n], result_dir, "npy")

    print(f"Test Mean Dice={np.mean(all_dices):.4f}, Mean IoU={np.mean(all_ious):.4f}")
    calculate_brats_metrics(gt_dir, pred_dir)


if __name__ == "__main__":
    main()
