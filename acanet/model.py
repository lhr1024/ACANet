"""ACANet 模型复现（基于论文截图梳理的核心结构）。

核心组件（对应论文图 1、图 2、图 3、公式 (1)-(8)）：
1) 双分支特征提取：上支路(T1/T1CE)与下支路(T2/FLAIR)分别编码，得到多层特征与各自的暂态预测 P_a、P_d（通过部分解码器 PPD）。
2) 预测感知区域探索 PRE：由 P_a、P_d 得到全局候选区域 P_H（公式 (5)）和边界/不确定区域 P_L（公式 (6)）。
3) 自适应上下文聚合 ACA：利用 P_H 引导对齐后的多模态特征融合，使用通道权重 w_1, w_2（公式 (1)、(2)）和多尺度卷积（公式 (3)、(4)）生成融合特征 F_i。
4) 预测引导解码 PDS（含 PD）：使用 P_L 逐层引导解码融合特征 F_i，生成最终预测 P_f（公式 (7)、(8)）。

为便于复现与测试，这里使用轻量卷积编码器替代原文的 PVTv2-B2，但保留双分支、ACA、PRE、PD 的设计思想与公式对应关系。
输入假设为 (N, 4, H, W)，按论文分成两对模态：上支路 (T1, T1CE)，下支路 (T2, FLAIR)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ----------------------------- 基础模块 ----------------------------- #
class ConvBNReLU(nn.Sequential):
    """3x3 卷积 + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    """两层卷积 + 下采样，用于逐层提取特征。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(in_channels, out_channels), ConvBNReLU(out_channels, out_channels))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class SimpleEncoder(nn.Module):
    """轻量级编码器（替代原文 PVTv2-B2，用于示意复现）。

    返回三个尺度的特征列表 [F1, F2, F3]，从浅到深。
    """

    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, base)
        self.enc1 = EncoderBlock(base, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        f1, x = self.enc1(x)  # H/2
        f2, x = self.enc2(x)  # H/4
        f3, _ = self.enc3(x)  # H/8
        return [f1, f2, f3]


# ----------------------------- 预测感知模块 ----------------------------- #
class PartialPredictionDecoder(nn.Module):
    """部分解码器（PPD），从单分支顶层特征生成暂态预测。

    这里用浅层上采样路径近似论文中的分支解码器，输出大小与输入原图一致。
    """

    def __init__(self, channels: Sequence[int], num_classes: int):
        super().__init__()
        c1, c2, c3 = channels  # 对应 F1, F2, F3
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.conv2 = ConvBNReLU(c2 * 2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(c1 * 2, c1)
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, features: List[Tensor]) -> Tensor:
        f1, f2, f3 = features  # 低 -> 高
        x = self.up3(f3)
        if x.shape[-2:] != f2.shape[-2:]:
            x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv2(torch.cat([x, f2], dim=1))
        x = self.up2(x)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv1(torch.cat([x, f1], dim=1))
        return self.classifier(x)


class PREModule(nn.Module):
    """Prediction-aware Region Exploration（公式 (5)、(6)）。

    P_H = σ(P_a) + σ(P_d) / 2   （候选肿瘤区域，覆盖 WT）
    P_L = |σ(P_a) - σ(P_d)|     （边界/不确定区域）
    """

    def forward(self, pa: Tensor, pd: Tensor) -> tuple[Tensor, Tensor]:
        pa_sig = torch.sigmoid(pa)
        pd_sig = torch.sigmoid(pd)
        ph = (pa_sig + pd_sig) / 2
        pl = torch.abs(pa_sig - pd_sig)
        return ph, pl


# ----------------------------- ACA 模块 ----------------------------- #
class ACAModule(nn.Module):
    """自适应上下文聚合（公式 (1)-(4)）。

    1) w1 = σ(conv3(F_u))，w2 = σ(conv3(F_d))            —— 公式 (1)、(2)
    2) F̃ = w1 * F_u + w2 * F_d                          —— 融合权重
    3) 多尺度空洞卷积抽取 F̃_u, F̃_d                     —— 公式 (3)、(4) 的 multi-scale conv
    4) 拼接 [F̃_u * P_H, F̃_d * P_H, F̃] 后 3x3 卷积得到 F_i
    """

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 3, 5, 7)):
        super().__init__()
        self.wu = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.Sigmoid())
        self.wd = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.Sigmoid())
        self.ms_u = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False) for d in dilations]
        )
        self.ms_d = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False) for d in dilations]
        )
        self.fuse = ConvBNReLU(channels * 3, channels)

    def _multi_scale(self, x: Tensor, convs: nn.ModuleList) -> Tensor:
        feats = [conv(x) for conv in convs]
        return sum(feats) / len(feats)

    def forward(self, fu: Tensor, fd: Tensor, ph: Tensor) -> Tensor:
        if fu.numel() == 0 or fd.numel() == 0:
            raise ValueError("ACA 输入特征为空")
        # 权重 (1)(2)
        wu = self.wu(fu)
        wd = self.wd(fd)
        f_tilde = wu * fu + wd * fd

        # 多尺度卷积 (3)(4)
        fu_ms = self._multi_scale(fu, self.ms_u)
        fd_ms = self._multi_scale(fd, self.ms_d)

        # 结合 P_H 引导，强调候选肿瘤区域
        fu_guided = fu_ms * ph
        fd_guided = fd_ms * ph

        fused = torch.cat([fu_guided, fd_guided, f_tilde], dim=1)
        return self.fuse(fused)


# ----------------------------- 预测引导解码 PDS ----------------------------- #
class PredictionGuidedDecoder(nn.Module):
    """使用 P_L 逐层解码融合特征（对应公式 (7)、(8) 思路简化实现）。"""

    def __init__(self, channels: Sequence[int], num_classes: int):
        super().__init__()
        c1, c2, c3 = channels
        self.conv3 = ConvBNReLU(c3, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.conv2 = ConvBNReLU(c2 + c2, c2)  # concat PL 引导
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.conv1 = ConvBNReLU(c1 + c1, c1)
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, feats: List[Tensor], pl: Tensor) -> Tensor:
        f1, f2, f3 = feats
        x = self.conv3(f3)
        x = self.up3(x)
        if x.shape[-2:] != f2.shape[-2:]:
            x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        pl2 = F.interpolate(pl, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv2(torch.cat([x, pl2], dim=1))

        x = self.up2(x)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        pl1 = F.interpolate(pl, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv1(torch.cat([x, pl1], dim=1))
        return self.classifier(x)


# ----------------------------- 配置 ----------------------------- #
@dataclass
class ACANetConfig:
    in_channels: int = 4  # 按论文 4 个模态
    num_classes: int = 4  # WT/TC/ET/背景
    base_channels: int = 32
    dilations: Sequence[int] = (1, 3, 5, 7)

    def validate(self) -> None:
        if self.in_channels < 4:
            raise ValueError("需要至少 4 个通道（T1, T1CE, T2, FLAIR）")
        if self.num_classes < 2:
            raise ValueError("num_classes 必须 >=2")
        if self.base_channels <= 0:
            raise ValueError("base_channels 必须为正")
        if len(self.dilations) == 0:
            raise ValueError("dilations 不能为空")


# ----------------------------- 主模型 ----------------------------- #
def _pad_to_multiple(x: Tensor, factor: int = 8) -> tuple[Tensor, tuple[int, int, int, int]]:
    """将输入在 H/W 维度补齐到 factor 的倍数，并确保最小尺寸为 factor，避免多次下采样后尺寸为 0。"""

    _, _, h, w = x.shape
    target_h = max(factor, ((h + factor - 1) // factor) * factor)
    target_w = max(factor, ((w + factor - 1) // factor) * factor)
    pad_h = target_h - h
    pad_w = target_w - w
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    if pad_h or pad_w:
        x = F.pad(x, pad, mode="replicate")
    return x, pad


def _crop_to_original(x: Tensor, pad: tuple[int, int, int, int]) -> Tensor:
    """去除补齐边界，恢复到原始尺寸。"""

    left, right, top, bottom = pad
    if top == bottom == left == right == 0:
        return x
    h, w = x.shape[-2:]
    return x[..., top : h - bottom if bottom > 0 else h, left : w - right if right > 0 else w]


class ACANet(nn.Module):
    """ACANet 主体：双分支编码 -> PPD -> PRE -> ACA -> PD."""

    def __init__(self, config: ACANetConfig):
        super().__init__()
        config.validate()
        ch = config.base_channels

        # 双分支编码
        self.encoder_up = SimpleEncoder(in_channels=2, base=ch)
        self.encoder_down = SimpleEncoder(in_channels=2, base=ch)

        # 部分解码器（生成 Pa, Pd）
        channels = [ch, ch * 2, ch * 4]
        self.ppd_up = PartialPredictionDecoder(channels, num_classes=config.num_classes)
        self.ppd_down = PartialPredictionDecoder(channels, num_classes=config.num_classes)

        # ACA 多尺度融合（逐层）
        self.aca1 = ACAModule(ch, config.dilations)
        self.aca2 = ACAModule(ch * 2, config.dilations)
        self.aca3 = ACAModule(ch * 4, config.dilations)

        # 预测引导解码
        self.pd = PredictionGuidedDecoder(channels, num_classes=config.num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.dim() != 4:
            raise ValueError(f"期望输入维度 (N, C, H, W)，实际 {x.shape}")
        if x.shape[1] < 4:
            raise ValueError("输入通道不足 4，无法按模态对分支")

        # 记录原始尺寸并补齐到 8 的倍数，避免多次下采样后尺寸变为 0
        x, pad = _pad_to_multiple(x, factor=8)

        # 划分模态：上 (T1, T1CE)，下 (T2, FLAIR)
        x_up = x[:, :2]
        x_down = x[:, 2:4]

        feats_up = self.encoder_up(x_up)
        feats_down = self.encoder_down(x_down)

        # 暂态预测（Pa, Pd）
        pa = self.ppd_up(feats_up)
        pd = self.ppd_down(feats_down)

        # PRE 生成 P_H, P_L
        ph, pl = PREModule()(pa, pd)

        # ACA 融合多模态特征（逐层对应公式 (1)-(4)）
        f1 = self.aca1(feats_up[0], feats_down[0], ph)
        f2 = self.aca2(feats_up[1], feats_down[1], ph)
        f3 = self.aca3(feats_up[2], feats_down[2], ph)
        fused_feats = [f1, f2, f3]

        # 预测引导解码（公式 (7)-(8) 简化）
        pf = self.pd(fused_feats, pl)

        # 去掉补齐，恢复到原始尺寸
        pa = _crop_to_original(pa, pad)
        pd = _crop_to_original(pd, pad)
        pf = _crop_to_original(pf, pad)
        ph = _crop_to_original(ph, pad)
        pl = _crop_to_original(pl, pad)

        return {"pa": pa, "pd": pd, "pf": pf, "ph": ph, "pl": pl}


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数数量。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo_forward() -> Dict[str, Tensor]:
    """快速前向示例。"""

    cfg = ACANetConfig()
    model = ACANet(cfg)
    dummy = torch.randn(2, cfg.in_channels, 128, 128)
    return model(dummy)


__all__ = ["ACANet", "ACANetConfig", "count_parameters", "demo_forward"]
