# ACANet 复现与说明

本仓库给出论文中 ACANet（用于脑肿瘤分割）的核心结构梳理、可运行的 PyTorch 复现代码、依赖安装、测试用例与实验复现步骤。

## 1. ACANet 核心算法结构
- **编码器-解码器骨架**：整体沿用 U-Net 框架。编码端三层下采样提取多尺度特征；解码端逐层上采样并与跳跃特征融合。
- **多尺度空洞上下文聚合（ACA）**：在瓶颈处使用多种扩张率的空洞卷积（默认膨胀率 1/3/5）提取不同感受野的特征后拼接，通过 1×1 卷积融合，再叠加通道注意力与空间注意力实现上下文自适应聚合。
- **双重注意力跳跃融合**：解码阶段的跳跃连接先经过通道注意力和空间注意力门控，抑制无关区域，再与上采样特征拼接。
- **损失与指标**：使用 Dice + 交叉熵混合损失，对照论文的分割目标函数；评估采用 Dice 与 IoU。

对应实现位置：
- `acanet/model.py`：`AtrousContextAggregation`（多尺度空洞+CA+SA），`DecoderBlock`（跳跃注意力），`ACANet`（整体骨架）。
- `acanet/metrics.py`：Dice/IoU 与混合损失。
- `acanet/train.py`：训练与验证流程。

## 2. 依赖安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
主要依赖：PyTorch、NumPy、SciPy、scikit-learn、rich（可选日志美化）。

## 3. 数据格式与划分
- 特征：`features.npy`，形状 `(N, C, H, W)`，C 通常为 1（单模态）或多模态堆叠。
- 标签：`labels.npy`，形状 `(N, H, W)`，像素值为 `[0, num_classes-1]`。
- 划分：`acanet/data.py` 使用 `train_test_split` 默认 8:2 划分，可通过 `test_size` 与 `random_state` 控制。

## 4. 核心代码示例
```python
from acanet.model import ACANet, ACANetConfig
from acanet.data import NpyDatasetConfig, load_datasets
from acanet.metrics import dice_loss, dice_coefficient

model_cfg = ACANetConfig(in_channels=1, num_classes=4)
model = ACANet(model_cfg)

# 读取数据
dataset_cfg = NpyDatasetConfig(features_path="data/features.npy", labels_path="data/labels.npy")
train_ds, val_ds = load_datasets(dataset_cfg)
```

## 5. 运行训练与验证步骤
1. 准备 `features.npy` 与 `labels.npy`（可用论文中的预处理方式生成）。
2. 安装依赖并启动虚拟环境。
3. 运行训练脚本：
   ```bash
   python -m acanet.train --features data/features.npy --labels data/labels.npy \
       --batch-size 4 --epochs 50 --lr 1e-3 --num-classes 4 --base-channels 32 --device cuda
   ```
   输出包含每个 epoch 的训练损失、验证 Dice/IoU 以及参数量，便于和论文表格对比。
4. 若需修改网络宽度/扩张率，可调整 `ACANetConfig` 中的 `base_channels` 与 `dilations`。

## 6. 测试用例
使用内置的合成数据测试前向、数据加载与指标范围：
```bash
pytest -q
```

## 7. 与论文实验的对比方法
- **参数量**：训练结束会打印 `Trainable parameters`，可与论文报告的模型规模核对。
- **指标对齐**：脚本输出的验证 Dice/IoU 即论文常用指标。如需精确复现，确保：
  - 使用与论文一致的预处理、类别划分与数据分割方式；
  - 训练轮数、学习率策略、损失权重与论文保持一致；
  - 若论文采用 3D 体数据，将卷积/池化/反卷积替换为 3D 版本即可。
- **消融验证**：可在 `AtrousContextAggregation` 中调整 `dilations`，或在 `DecoderBlock` 去掉注意力以复现实验对照组。

## 8. 边界情况处理
- 数据文件不存在或形状异常时会抛出清晰错误。
- 空批次或标签数不足会触发验证逻辑，避免静默失败。
- 输入维度不符（非 NCHW）时前向会提示具体形状错误。
