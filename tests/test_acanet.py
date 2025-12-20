import os
import numpy as np
import torch

from acanet.data import NpyDatasetConfig, load_datasets
from acanet.metrics import dice_coefficient, iou_score
from acanet.model import ACANet, ACANetConfig


def test_model_forward_shapes():
    config = ACANetConfig(in_channels=4, num_classes=3, base_channels=8)
    model = ACANet(config)
    x = torch.randn(2, 4, 64, 64)
    outputs = model(x)
    assert outputs["pf"].shape == (2, 3, 64, 64)
    assert outputs["pa"].shape == (2, 3, 64, 64)
    assert outputs["pd"].shape == (2, 3, 64, 64)


def test_dataset_loading_and_split(tmp_path):
    # 构造两名患者（001, 002），每名各 2 张切片，确保按患者划分不会泄露
    feat_dir = tmp_path / "features"
    lab_dir = tmp_path / "labels"
    feat_dir.mkdir()
    lab_dir.mkdir()

    def save_case(pid: str, idx: int):
        name = f"BraTS20_Training_{pid}_{idx:02d}.npy"
        img = np.random.rand(4, 32, 32).astype(np.float32)
        lab = np.random.randint(0, 3, size=(32, 32), dtype=np.int64)
        np.save(feat_dir / name, img)
        np.save(lab_dir / name, lab)

    for i in range(2):
        save_case("001", i)
        save_case("002", i)

    cfg = NpyDatasetConfig(features_path=str(feat_dir), labels_path=str(lab_dir), test_size=0.5, random_state=0)
    train_ds, val_ds = load_datasets(cfg)
    # 按患者划分，故每侧应含 2 张切片（1 个患者）
    assert len(train_ds) == 2
    assert len(val_ds) == 2
    x, y = train_ds[0]
    assert x.shape == (4, 32, 32)
    assert y.shape == (32, 32)


def test_metrics_range():
    logits = torch.tensor(
        [
            [[[0.1, 0.9], [0.8, 0.2]], [[0.9, 0.1], [0.2, 0.8]], [[0.0, 0.0], [0.0, 0.0]]]],
            dtype=torch.float32,
    ).unsqueeze(0)
    target = torch.tensor([[1, 0], [0, 1]])
    dice = dice_coefficient(logits, target, num_classes=3)
    iou = iou_score(logits, target, num_classes=3)
    assert 0 <= dice <= 1
    assert 0 <= iou <= 1
