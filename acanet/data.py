"""数据加载与划分工具。

读取 .npy 特征与标签文件：
- features.npy: 形状 (N, C, H, W)
- labels.npy:   形状 (N, H, W)，值为 [0, num_classes-1]
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class NpyDatasetConfig:
    features_path: str
    labels_path: str
    test_size: float = 0.2
    random_state: int = 42

    def validate(self) -> None:
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"features_path not found: {self.features_path}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"labels_path not found: {self.labels_path}")
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be in (0,1)")


class NpySegmentationDataset(Dataset):
    """基于 .npy 文件的分割数据集。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        if images.ndim != 4:
            raise ValueError(f"Expected images shape (N, C, H, W), got {images.shape}")
        if labels.ndim != 3:
            raise ValueError(f"Expected labels shape (N, H, W), got {labels.shape}")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("Images and labels must share batch dimension")
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds for dataset")
        x = torch.from_numpy(self.images[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y


def load_datasets(config: NpyDatasetConfig) -> Tuple[NpySegmentationDataset, NpySegmentationDataset]:
    """读取 .npy 文件并切分训练/验证集。"""

    config.validate()
    images = np.load(config.features_path)
    labels = np.load(config.labels_path)

    train_x, val_x, train_y, val_y = train_test_split(
        images, labels, test_size=config.test_size, random_state=config.random_state, stratify=labels.reshape(len(labels), -1).max(axis=1)
    )
    return NpySegmentationDataset(train_x, train_y), NpySegmentationDataset(val_x, val_y)


__all__ = ["NpyDatasetConfig", "NpySegmentationDataset", "load_datasets"]
