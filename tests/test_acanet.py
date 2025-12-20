import numpy as np
import torch

from acanet.data import NpyDatasetConfig, load_datasets
from acanet.metrics import dice_coefficient, iou_score
from acanet.model import ACANet, ACANetConfig


def test_model_forward_shapes():
    config = ACANetConfig(in_channels=1, num_classes=3, base_channels=8)
    model = ACANet(config)
    x = torch.randn(2, 1, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 3, 64, 64)


def test_dataset_loading_and_split(tmp_path):
    images = np.random.rand(10, 1, 32, 32).astype(np.float32)
    labels = np.random.randint(0, 3, size=(10, 32, 32), dtype=np.int64)
    features_path = tmp_path / "features.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(features_path, images)
    np.save(labels_path, labels)

    cfg = NpyDatasetConfig(features_path=str(features_path), labels_path=str(labels_path), test_size=0.2)
    train_ds, val_ds = load_datasets(cfg)
    assert len(train_ds) == 8
    assert len(val_ds) == 2
    x, y = train_ds[0]
    assert x.shape == (1, 32, 32)
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
