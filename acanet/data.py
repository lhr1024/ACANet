"""数据加载与划分工具。

读取 .npy 特征与标签文件：
- features.npy: 形状 (N, C, H, W)
- labels.npy:   形状 (N, H, W)，值为 [0, num_classes-1]
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class NpyDatasetConfig:
    """数据路径配置。

    - 若为文件：单个 features.npy / labels.npy。
    - 若为目录：目录下的 .npy 文件按文件名匹配（如 BraTS20_Training_001_30.npy），
      并根据病人编号（例如 001）进行患者级划分，防止数据泄露。
    """

    features_path: str
    labels_path: str
    test_size: float = 0.2
    random_state: int = 42

    def validate(self) -> None:
        for p, name in [(self.features_path, "features_path"), (self.labels_path, "labels_path")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{name} not found: {p}")
            if os.path.isdir(p):
                npy_files = glob(os.path.join(p, "*.npy"))
                if len(npy_files) == 0:
                    raise FileNotFoundError(f"{name} dir has no .npy files: {p}")
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be in (0,1)")


class NpySegmentationDataset(Dataset):
    """基于 .npy 文件的分割数据集。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        if images.ndim != 4:
            raise ValueError(f"Expected images shape (N, C, H, W) or (N, H, W, C), got {images.shape}")
        if labels.ndim != 3:
            raise ValueError(f"Expected labels shape (N, H, W), got {labels.shape}")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("Images and labels must share batch dimension")
        # 如果通道在最后一维，则转为 NCHW
        if images.shape[1] not in (1, 3, 4) and images.shape[-1] in (1, 3, 4):
            images = np.transpose(images, (0, 3, 1, 2))
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds for dataset")
        x = torch.from_numpy(self.images[idx])
        y = torch.from_numpy(self.labels[idx])
        # 将标签值规范到 0..K-1，适配 BraTS 标签 {0,1,2,4} -> {0,1,2,3}
        if y.max() >= 4:
            y = y.clone()
            y[y == 4] = 3
        return x, y


def load_datasets(config: NpyDatasetConfig) -> Tuple[NpySegmentationDataset, NpySegmentationDataset]:
    """读取 .npy 文件并切分训练/验证集（患者级划分，防止泄露）。

    支持：
    - 单文件：features.npy / labels.npy （旧方式），随机切分。
    - 目录：匹配同名 .npy 文件，按文件名中的患者编号分组（如 BraTS20_Training_001_30.npy => 患者 001）。
    """

    config.validate()

    def _load_from_file(path: str) -> np.ndarray:
        return np.load(path)

    def _extract_patient_id(filename: str) -> str:
        stem = os.path.splitext(os.path.basename(filename))[0]
        # 典型格式：BraTS20_Training_001_30 -> tokens = [BraTS20, Training, 001, 30]
        tokens = stem.split("_")
        for tok in tokens:
            if tok.isdigit():
                return tok
        # 兜底：若未找到纯数字，使用去掉尾部索引后的前半部分
        return "_".join(tokens[:-1]) if len(tokens) > 1 else stem

    def _load_from_dir(feat_dir: str, label_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        feat_files = sorted(glob(os.path.join(feat_dir, "*.npy")))
        label_files = sorted(glob(os.path.join(label_dir, "*.npy")))
        feat_map = {os.path.basename(f): f for f in feat_files}
        label_map = {os.path.basename(f): f for f in label_files}
        common_names = sorted(set(feat_map) & set(label_map))
        if len(common_names) == 0:
            raise FileNotFoundError("No paired .npy files between features and labels directories")

        images: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        patient_ids: List[str] = []
        for name in common_names:
            f_path = feat_map[name]
            l_path = label_map[name]
            img = np.load(f_path)
            lab = np.load(l_path)
            if img.shape[0] != lab.shape[0] and img.ndim == 3 and lab.ndim == 2:
                # 单张切片 (C,H,W) vs (H,W) 是允许的
                pass
            elif img.shape[0] == lab.shape[0] and img.ndim == 4 and lab.ndim == 3:
                # 批量切片，逐张展开
                pass
            patient_ids.append(_extract_patient_id(name))
            images.append(img)
            labels.append(lab)

        images_arr = np.stack(images, axis=0)
        labels_arr = np.stack(labels, axis=0)
        return images_arr, labels_arr, patient_ids

    # 读取数据
    if os.path.isdir(config.features_path) and os.path.isdir(config.labels_path):
        images, labels, patient_ids = _load_from_dir(config.features_path, config.labels_path)
        unique_patients = sorted(set(patient_ids))
        if len(unique_patients) < 2:
            raise ValueError("Need at least 2 patients for train/val split to avoid leakage")
        # 按患者划分，防止同患者切片泄露到不同集合
        train_patients, val_patients = train_test_split(
            unique_patients, test_size=config.test_size, random_state=config.random_state
        )
        print(f"[Data] Validation patient IDs: {sorted(val_patients)}")
        train_mask = np.array([pid in train_patients for pid in patient_ids])
        val_mask = ~train_mask
        train_x, val_x = images[train_mask], images[val_mask]
        train_y, val_y = labels[train_mask], labels[val_mask]
    else:
        images = _load_from_file(config.features_path)
        labels = _load_from_file(config.labels_path)
        train_x, val_x, train_y, val_y = train_test_split(
            images,
            labels,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=labels.reshape(len(labels), -1).max(axis=1),
        )

    return NpySegmentationDataset(train_x, train_y), NpySegmentationDataset(val_x, val_y)


__all__ = ["NpyDatasetConfig", "NpySegmentationDataset", "load_datasets"]
