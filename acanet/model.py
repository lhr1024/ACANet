"""ACANet 模型实现。

本文件根据论文中描述的 ACANet 框架进行 Python 复现：
- 采用编码器-解码器结构（对应论文的整体公式 F = D(E(X))）。
- 在编码阶段使用多尺度空洞卷积聚合（对应论文的上下文聚合公式）。
- 在跳跃连接处加入通道/空间注意力（对应论文的自适应注意力权重计算公式）。

实现假设输入为二维分割（N, C, H, W），便于快速复现和测试。如果需要 3D，
仅需将 Conv2d/BatchNorm2d/MaxPool2d 改为 3D 版本并调整卷积核尺寸。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn


class ConvBNReLU(nn.Sequential):
    """基础卷积模块，对应论文中最底层的特征提取公式。

    该模块实现 (Conv -> BN -> ReLU)，可在多个位置复用。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2 * dilation
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ChannelAttention(nn.Module):
    """通道注意力，对应论文的通道权重归一化公式（soft attention）。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.mlp(self.pool(x))  # 通道权重 α_c
        return x * weights


class SpatialAttention(nn.Module):
    """空间注意力，对应论文的空间权重分布公式。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        # 聚合通道信息后生成空间掩码
        self.compress = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True).values
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        pooled = torch.cat([max_pool, mean_pool], dim=1)
        mask = self.activation(self.compress(pooled))  # 空间权重 β_(h,w)
        return x * mask


class AtrousContextAggregation(nn.Module):
    """多尺度空洞卷积上下文聚合模块（ACA）。

    该模块对应论文中的上下文编码公式：使用多种扩张率提取局部与全局特征，
    然后通过逐像素加权的注意力进行融合。
    """

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 3, 5)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNReLU(channels, channels, kernel_size=3, dilation=d, padding=d) for d in dilations]
        )
        self.fuse = nn.Conv2d(len(dilations) * channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        if x.numel() == 0:
            raise ValueError("Input to AtrousContextAggregation is empty")
        features = [branch(x) for branch in self.branches]
        concat = torch.cat(features, dim=1)
        fused = self.fuse(concat)
        fused = self.act(self.norm(fused))
        # 自适应注意力融合，对应论文的 γ = σ(CA + SA)
        fused = self.channel_attn(fused)
        fused = self.spatial_attn(fused)
        return fused


class EncoderBlock(nn.Module):
    """编码器块：两层卷积 + 下采样，模拟论文中自下而上的特征抽取。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(in_channels, out_channels), ConvBNReLU(out_channels, out_channels))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class DecoderBlock(nn.Module):
    """解码器块：上采样 + 跳跃注意力融合 + 卷积，还原分割分辨率。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_attn = nn.Sequential(ChannelAttention(skip_channels), SpatialAttention())
        self.conv = nn.Sequential(
            ConvBNReLU(out_channels + skip_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # 对齐尺寸，防止奇数尺寸导致的边界错位
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))
        skip = self.skip_attn(skip)
        fused = torch.cat([x, skip], dim=1)
        return self.conv(fused)


@dataclass
class ACANetConfig:
    """可配置参数，便于测试不同宽度与类别数。"""

    in_channels: int = 1
    num_classes: int = 4
    base_channels: int = 32
    dilations: Sequence[int] = (1, 3, 5)

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >= 2 for segmentation")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if len(self.dilations) == 0:
            raise ValueError("dilations must be non-empty")


class ACANet(nn.Module):
    """ACANet 主体网络。

    流程：
    1) 编码器获取多尺度特征。
    2) 最底层通过 AtrousContextAggregation 聚合上下文。
    3) 解码器利用注意力筛选的跳跃连接恢复分辨率。
    4) 最终 1x1 卷积生成分割 logits，对应论文的预测公式 P = softmax(W * F).
    """

    def __init__(self, config: ACANetConfig) -> None:
        super().__init__()
        config.validate()
        ch = config.base_channels
        self.enc1 = EncoderBlock(config.in_channels, ch)
        self.enc2 = EncoderBlock(ch, ch * 2)
        self.enc3 = EncoderBlock(ch * 2, ch * 4)

        self.bottleneck = nn.Sequential(
            ConvBNReLU(ch * 4, ch * 8),
            AtrousContextAggregation(ch * 8, dilations=config.dilations),
        )

        self.dec3 = DecoderBlock(ch * 8, ch * 4, ch * 4)
        self.dec2 = DecoderBlock(ch * 4, ch * 2, ch * 2)
        self.dec1 = DecoderBlock(ch * 2, ch, ch)

        self.classifier = nn.Conv2d(ch, config.num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (N, C, H, W), got {x.shape}")
        x1, down1 = self.enc1(x)
        x2, down2 = self.enc2(down1)
        x3, down3 = self.enc3(down2)

        bottleneck = self.bottleneck(down3)
        d3 = self.dec3(bottleneck, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        logits = self.classifier(d1)
        return logits


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数数量，方便与论文报告对比。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo_forward() -> Tensor:
    """用于快速单元测试的前向示例。"""

    config = ACANetConfig()
    model = ACANet(config)
    dummy = torch.randn(2, config.in_channels, 128, 128)
    return model(dummy)


__all__ = [
    "ACANet",
    "ACANetConfig",
    "AtrousContextAggregation",
    "ChannelAttention",
    "SpatialAttention",
    "count_parameters",
    "demo_forward",
]
