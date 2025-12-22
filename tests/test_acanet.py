import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import pytest

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - handled lazily to keep tests importable without nibabel
    nib = None
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# 若核心依赖缺失，则跳过整份测试，方便在最小依赖环境下导入
_missing = [name for name, mod in {"nibabel": nib, "numpy": np, "torch": torch}.items() if mod is None]
if _missing:
    pytest.skip(
        "Skipping ACANet NIfTI inference test because missing dependencies: " + ", ".join(_missing),
        allow_module_level=True,
    )

try:
    from acanet.metrics import dice_coefficient, iou_score
    from acanet.model import ACANet, ACANetConfig
except ImportError:  # pragma: no cover - package not installed in minimal envs
    pytest.skip(
        "Skipping ACANet NIfTI inference test because project package is not importable (install in editable mode).",
        allow_module_level=True,
    )

# 默认配置：在这里填好路径后可直接运行 python tests/test_acanet.py
CONFIG = {
    "images": "data/test_images_nii",   # 测试集 .nii/.nii.gz 目录（特征）
    "labels": "data/test_labels_nii",   # 测试集标签目录
    "model": "last_model.pth",          # 训练好的模型权重
    "out_dir": "test_slices",           # 切片输出目录
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 运行设备
    "num_classes": 4,                   # 类别数（含背景）
}

def load_nii_slices(img_path: str) -> np.ndarray:
    """读取 .nii/.nii.gz，返回形状 (C,H,W,D)。若为单通道，则 C=1。"""
    _ensure_nib()
    nii = nib.load(img_path)
    data = nii.get_fdata()
    # 常见 BraTS 存储为 (H,W,D) 单通道或 (H,W,D,C) 多通道
    if data.ndim == 3:
        data = data[..., None]  # (H,W,D,1)
    # 重排为 (C,H,W,D)
    data = np.transpose(data, (3, 0, 1, 2))
    return data.astype(np.float32)


def volume_to_npy_slices(volume: np.ndarray, out_dir: Path, base_name: str) -> List[Path]:
    """将 (C,H,W,D) 体数据按深度 D 切片保存为 .npy，返回保存路径列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    c, h, w, d = volume.shape
    paths = []
    for idx in range(d):
        slice_arr = volume[:, :, :, idx]  # (C,H,W)
        save_path = out_dir / f"{base_name}_slice{idx:03d}.npy"
        np.save(save_path, slice_arr)
        paths.append(save_path)
    return paths


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
            arr = np.load(p)  # (C,H,W)
            if arr.ndim != 3:
                raise ValueError(f"Expected (C,H,W) slice in {p}, got {arr.shape}")
            x = torch.from_numpy(arr).unsqueeze(0).to(device_obj)  # (1,C,H,W)
            out = model(x)["pf"].squeeze(0).cpu()  # (C,H,W)
            preds.append(out)
    return preds


def compute_metrics(preds: List[torch.Tensor], labels: List[torch.Tensor], num_classes: int) -> Tuple[float, float]:
    """逐切片计算 Dice/IoU，返回均值。"""
    dices, ious = [], []
    for pred, target in zip(preds, labels):
        pred = pred.unsqueeze(0)  # (1,C,H,W)
        target = target.unsqueeze(0)  # (1,H,W)
        dices.append(dice_coefficient(pred, target, num_classes=num_classes).item())
        ious.append(iou_score(pred, target, num_classes=num_classes).item())
    return float(np.mean(dices)), float(np.mean(ious))


def load_label_slices(label_path: str) -> List[torch.Tensor]:
    """读取标签体，切片为 (H,W)，输出列表。假设标签为单通道。"""
    _ensure_nib()
    nii = nib.load(label_path)
    data = nii.get_fdata()
    if data.ndim != 3:
        raise ValueError(f"Label volume expected 3D, got {data.shape}")
    data = data.astype(np.int64)
    slices = [torch.from_numpy(data[:, :, i]) for i in range(data.shape[2])]
    return slices


def _ensure_nib() -> None:
    """确保 nibabel 可用，在缺失时给出清晰错误。"""
    if nib is None:
        raise ImportError(
            "nibabel is required for NIfTI I/O. Install it with `pip install nibabel` "
            "or add it to your environment before running this script."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ACANet NIfTI inference")
    parser.add_argument("--images", help="测试集 .nii/.nii.gz 目录（特征）")
    parser.add_argument("--labels", help="测试集标签 .nii/.nii.gz 目录")
    parser.add_argument("--model", help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--out-dir", help="临时保存切片的目录")
    parser.add_argument("--device", help="运行设备")
    parser.add_argument("--num-classes", type=int, help="类别数（含背景）")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.images:
        cfg["images"] = args.images
    if args.labels:
        cfg["labels"] = args.labels
    if args.model:
        cfg["model"] = args.model
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.device:
        cfg["device"] = args.device
    if args.num_classes:
        cfg["num_classes"] = args.num_classes

    img_dir = Path(cfg["images"])
    lbl_dir = Path(cfg["labels"])
    out_dir = Path(cfg["out_dir"])

    img_files = sorted(glob.glob(str(img_dir / "*.nii*")))
    lbl_files = sorted(glob.glob(str(lbl_dir / "*.nii*")))
    if len(img_files) == 0 or len(img_files) != len(lbl_files):
        raise FileNotFoundError("图像与标签数量不匹配或为空")

    all_dices, all_ious = [], []
    for img_path, lbl_path in zip(img_files, lbl_files):
        base = Path(img_path).stem.replace(".nii", "")
        vol = load_nii_slices(img_path)  # (C,H,W,D)
        slice_paths = volume_to_npy_slices(vol, out_dir, base)
        preds = run_inference(cfg["model"], slice_paths, cfg["device"], cfg["num_classes"])
        label_slices = load_label_slices(lbl_path)
        # 对齐长度：只评估最短的切片数，避免越界
        n = min(len(preds), len(label_slices))
        dice, iou = compute_metrics(preds[:n], label_slices[:n], cfg["num_classes"])
        all_dices.append(dice); all_ious.append(iou)
        print(f"{base}: Dice={dice:.4f}, IoU={iou:.4f}")

    print(f"Test Mean Dice={np.mean(all_dices):.4f}, Mean IoU={np.mean(all_ious):.4f}")


if __name__ == "__main__":
    main()
