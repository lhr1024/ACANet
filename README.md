# ACANet 复现（基于论文截图）

本仓库依据论文截图《Adaptive Context Aggregation Network With Prediction-Aware Decoding for Multimodal Brain Tumor Segmentation》（IEEE TIM 2024）梳理并复现 ACANet 的核心思想：双分支编码、预测感知区域探索（PRE）、自适应上下文聚合（ACA）和预测引导解码（PDS/PD）。为便于快速运行，编码器采用轻量 CNN 近似原文的 PVTv2-B2，但模块和公式对应关系保持一致。

## 1. 核心结构与论文公式对应
- **双分支特征提取**：输入 4 个模态 (T1, T1CE, T2, FLAIR) 分为上下两路编码器，得到多尺度特征，并由部分解码器（PPD）输出暂态预测 `P_a`、`P_d`。
- **Prediction-aware Region Exploration（PRE，公式 5/6）**：`P_H = (σ(P_a) + σ(P_d)) / 2` 提供候选肿瘤区域；`P_L = |σ(P_a) - σ(P_d)|` 提供边界/不确定区域。
- **Adaptive Context Aggregation（ACA，公式 1-4）**：通道权重 `w1/w2` 融合两路特征，多尺度空洞卷积提取上下文，并用 `P_H` 引导，生成融合特征 `F_i`。
- **Prediction-Guided Decoding（PD，公式 7/8）**：利用 `P_L` 逐层引导解码融合特征，输出最终预测 `P_f`；训练时对 `P_a`、`P_d`、`P_f` 同时监督（Dice+CE）。
- **评估指标**：代码内置 Dice、IoU，可扩展 HD95、Sensitivity（论文使用）。

对应实现：
- `acanet/model.py`：双分支编码、PPD、PRE、ACA、PD 以及多路输出。
- `acanet/metrics.py`：Dice/IoU 与混合损失。
- `acanet/train.py`：训练/验证循环，多路损失求和。

## 2. 依赖安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
主要依赖：PyTorch、NumPy、SciPy、scikit-learn、rich（可选日志美化）。

## 3. 数据格式与划分
- 特征：`features.npy`，形状 `(N, 4, H, W)`，通道顺序为 T1、T1CE、T2、FLAIR；通道不足将报错。
- 标签：`labels.npy`，形状 `(N, H, W)`，像素值范围 `[0, num_classes-1]`。
- 划分：若提供目录（如 BraTS20_Training_001_30.npy 这类文件），按文件名中的患者编号（例如 `001`）做患者级 8:2 划分，防止同一患者切片出现在不同集合；若提供单一 .npy 文件则为随机划分，可通过 `test_size`、`random_state` 调整。

## 4. 核心代码示例
```python
from acanet.model import ACANet, ACANetConfig
from acanet.data import NpyDatasetConfig, load_datasets
from acanet.metrics import dice_loss, dice_coefficient

model_cfg = ACANetConfig(in_channels=4, num_classes=4)
model = ACANet(model_cfg)

dataset_cfg = NpyDatasetConfig(features_path="data/features.npy", labels_path="data/labels.npy")
train_ds, val_ds = load_datasets(dataset_cfg)
```

## 5. 训练与验证
```bash
python -m acanet.train --features data/features --labels data/labels \
    --batch-size 4 --epochs 50 --lr 1e-3 --num-classes 4 --base-channels 32 --device cuda
```
输出包含每个 epoch 的训练损失（含 `P_a/P_d/P_f` 三路）与验证 Dice/IoU，以及参数量，可对照论文表格。若追求论文设定，可将编码器替换为 PVTv2-B2，batch size=12，训练 100 轮，poly 学习率策略，BraTS 2D 切片。

## 6. 测试
使用合成数据的快速单元测试：
```bash
pytest -q
```

## 7. 与论文对比/消融
- **模块对齐**：PRE (5/6)、ACA (1-4)、PD (7/8) 均已实现；可禁用 ACA（直接 concat）、或将 `P_H/P_L` 置为常数 1 做消融。
- **指标扩展**：若需 HD95、Sensitivity，可在 `metrics.py` 增加；保持与论文一致的数据预处理、切片策略与训练轮次。
- **编码器替换**：为匹配论文，可将 `SimpleEncoder` 换成 PVTv2-B2 双分支，保持下游模块不变。

## 8. 边界处理
- 输入维度、通道不足或空数据时会抛出明确错误。
- 数据划分为空批次会报错防止静默失败。
- 前向、损失均在多路输出上做维度/类型检查。 
