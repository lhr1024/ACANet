"""ACANet 模型复现（基于论文截图梳理的核心结构）。

核心组件（对应论文图 1、图 2、图 3、公式 (1)-(8)）：
1) 双分支特征提取：上支路(T1/T1CE)与下支路(T2/FLAIR)分别编码，得到多层特征与各自的暂态预测 P_a、P_d（通过部分解码器 PPD）。
2) 预测感知区域探索 PRE：由 P_a、P_d 得到全局候选区域 P_H（公式 (5)）和边界/不确定区域 P_L（公式 (6)）。
3) 自适应上下文聚合 ACA：利用 P_H 引导对齐后的多模态特征融合，使用通道权重 w_1, w_2（公式 (1)、(2)）和多尺度卷积（公式 (3)、(4)）生成融合特征 F_i。
4) 预测引导解码 PDS（含 PD）：使用 P_L 逐层引导解码融合特征 F_i，生成最终预测 P_f（公式 (7)、(8)）。

默认使用 PVTv2-B2（ImageNet 预训练，依赖 timm），若未安装 timm 可切换为轻量卷积编码器作为回退，但仍保留双分支、ACA、PRE、PD 的设计思想与公式对应关系。
输入假设为 (N, 4, H, W)，按论文分成两对模态：上支路 (T1, T1CE)，下支路 (T2, FLAIR)。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import timm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    timm = None


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
    """轻量级卷积编码器（当 timm 不可用时的回退），输出四个尺度的特征。"""

    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, base)
        self.enc1 = EncoderBlock(base, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)
        self.enc4 = EncoderBlock(base * 4, base * 8)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        f1, x = self.enc1(x)  # H/2
        f2, x = self.enc2(x)  # H/4
        f3, x = self.enc3(x)  # H/8
        f4, _ = self.enc4(x)  # H/16
        return [f1, f2, f3, f4]


class PVTv2B2Encoder(nn.Module):
    """PVTv2-B2 特征提取（需要 timm，支持本地权重文件或不加载预训练）。"""

    def __init__(self, in_channels: int = 2, pretrained: bool = False, weights_path: str | None = None):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for PVTv2-B2 encoder. Install timm>=0.9")
        # features_only 提取四个 stage
        self.backbone = timm.create_model(
            "pvt_v2_b2",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_channels,
        )
        self.out_channels = self.backbone.feature_info.channels()
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"[PVTv2-B2] Missing keys when loading local weights: {missing}")
            if unexpected:
                print(f"[PVTv2-B2] Unexpected keys when loading local weights: {unexpected}")

    def forward(self, x: Tensor) -> List[Tensor]:
        feats = self.backbone(x)
        if len(feats) != 4:
            raise ValueError(f"PVTv2-B2 expected 4 feature maps, got {len(feats)}")
        return feats


# ----------------------------- 预测感知模块 ----------------------------- #
class PartialPredictionDecoder(nn.Module):
    """部分解码器（PPD），从单分支多尺度特征生成暂态预测，输出对齐原图。"""

    def __init__(self, channels: Sequence[int], num_classes: int):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"PartialPredictionDecoder expects 4 scales, got {len(channels)}")
        c1, c2, c3, c4 = channels  # 对应 F1, F2, F3, F4
        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.conv3 = ConvBNReLU(c3 * 2, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.conv2 = ConvBNReLU(c2 * 2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(c1 * 2, c1)
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, features: List[Tensor]) -> Tensor:
        f1, f2, f3, f4 = features  # 低 -> 高
        x = self.up4(f4)
        if x.shape[-2:] != f3.shape[-2:]:
            x = F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv3(torch.cat([x, f3], dim=1))

        x = self.up3(x)
        if x.shape[-2:] != f2.shape[-2:]:
            x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv2(torch.cat([x, f2], dim=1))

        x = self.up2(x)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv1(torch.cat([x, f1], dim=1))
        return self.classifier(x)


class PREModule(nn.Module):
    """Prediction-aware Region Exploration（公式 (5)、(6)），仅使用 WT/前景概率。"""

    def forward(self, pa: Tensor, pd: Tensor) -> tuple[Tensor, Tensor]:
        # 使用 softmax 取背景通道 (0)，其余视为 WT 前景
        pa_prob = torch.softmax(pa, dim=1)
        pd_prob = torch.softmax(pd, dim=1)
        pa_wt = 1 - pa_prob[:, :1]  # 前景概率
        pd_wt = 1 - pd_prob[:, :1]
        ph = (pa_wt + pd_wt) / 2
        pl = torch.abs(pa_wt - pd_wt)
        return ph, pl


# ----------------------------- ACA 模块 ----------------------------- #
class ACAModule(nn.Module):
    """自适应上下文聚合（更贴近论文图 2 的多尺度加权实现）。"""

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 3, 5, 7)):
        super().__init__()
        self.wu = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.Sigmoid())
        self.wd = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.Sigmoid())
        self.ms = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False), nn.Sigmoid()) for d in dilations]
        )
        self.conv_ms_fuse = ConvBNReLU(channels * len(dilations), channels)
        # 最终融合 Fu/Fd 与多尺度特征
        self.conv_out = ConvBNReLU(channels * 3, channels)

    def forward(self, fu: Tensor, fd: Tensor, ph: Tensor) -> Tensor:
        if fu.numel() == 0 or fd.numel() == 0:
            raise ValueError("ACA 输入特征为空")

        # 权重 (1)(2)
        wu = self.wu(fu)
        wd = self.wd(fd)
        f_prime = wu * fu + wd * fd  # F'

        # 多尺度卷积 + Sigmoid
        ms_feats = [m(f_prime) for m in self.ms]
        ms_concat = torch.cat(ms_feats, dim=1)
        f_ms = self.conv_ms_fuse(ms_concat)  # F''

        # PH 引导到与特征同尺寸的单通道 mask
        if ph.shape[-2:] != f_ms.shape[-2:]:
            ph = F.interpolate(ph, size=f_ms.shape[-2:], mode="bilinear", align_corners=False)
        ph_mask = ph.mean(dim=1, keepdim=True)

        guided_u = f_ms * fu * ph_mask
        guided_d = f_ms * fd * ph_mask
        fused = torch.cat([guided_u, guided_d, f_ms], dim=1)
        return self.conv_out(fused)


# ----------------------------- 预测引导解码 PDS ----------------------------- #
class PredictionGuidedDecoder(nn.Module):
    """使用 P_L 逐层解码融合特征（更贴近图 5 的逐级组件形式）。"""

    def __init__(self, channels: Sequence[int], num_classes: int):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"PredictionGuidedDecoder expects 4 scales, got {len(channels)}")
        c1, c2, c3, c4 = channels
        self.conv3 = ConvBNReLU(c4, c3)
        self.conv2 = ConvBNReLU(c3, c2)
        self.conv1 = ConvBNReLU(c2, c1)
        self.proj_pl3 = ConvBNReLU(1, c4, kernel_size=1)
        self.proj_pl2 = ConvBNReLU(1, c3, kernel_size=1)
        self.proj_pl1 = ConvBNReLU(1, c2, kernel_size=1)
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def _resize_pl(self, pl: Tensor, size: Sequence[int]) -> Tensor:
        return F.interpolate(pl, size=size, mode="bilinear", align_corners=False)

    def forward(self, feats: List[Tensor], pl: Tensor) -> Tensor:
        f1, f2, f3, f4 = feats  # f1:最高分辨率

        # C3: 使用 F4 作为 D4
        pl3 = self.proj_pl3(self._resize_pl(pl, f4.shape[-2:]).mean(dim=1, keepdim=True))
        d3_input = (f4 * pl3 + f4)
        d3 = self.conv3(d3_input)
        d3 = F.interpolate(d3, size=f3.shape[-2:], mode="bilinear", align_corners=False) * f3

        # C2
        pl2 = self.proj_pl2(self._resize_pl(pl, f3.shape[-2:]).mean(dim=1, keepdim=True))
        d2_input = (d3 * pl2 + d3)
        d2 = self.conv2(d2_input)
        d2 = F.interpolate(d2, size=f2.shape[-2:], mode="bilinear", align_corners=False) * f2

        # C1
        pl1 = self.proj_pl1(self._resize_pl(pl, f2.shape[-2:]).mean(dim=1, keepdim=True))
        d1_input = (d2 * pl1 + d2)
        d1 = self.conv1(d1_input)
        d1 = F.interpolate(d1, size=f1.shape[-2:], mode="bilinear", align_corners=False) * f1

        return self.classifier(d1)


# ----------------------------- 配置 ----------------------------- #
@dataclass
class ACANetConfig:
    in_channels: int = 4  # 按论文 4 个模态
    num_classes: int = 4  # WT/TC/ET/背景
    base_channels: int = 32
    dilations: Sequence[int] = (1, 3, 5, 7)
    use_pvt: bool = True
    pretrained_backbone: bool = False
    pvt_weights_path: str | None = None

    def validate(self) -> None:
        if self.in_channels < 4:
            raise ValueError("需要至少 4 个通道（T1, T1CE, T2, FLAIR）")
        if self.num_classes < 2:
            raise ValueError("num_classes 必须 >=2")
        if self.base_channels <= 0:
            raise ValueError("base_channels 必须为正")
        if len(self.dilations) == 0:
            raise ValueError("dilations 不能为空")
        if self.pvt_weights_path is not None and not os.path.exists(self.pvt_weights_path):
            raise FileNotFoundError(f"pvt_weights_path not found: {self.pvt_weights_path}")


# ----------------------------- 主模型 ----------------------------- #
def _pad_to_multiple(x: Tensor, factor: int = 16) -> tuple[Tensor, tuple[int, int, int, int]]:
    """将输入在 H/W 维度补齐到 factor 的倍数，并确保最小尺寸为 factor，避免多次下采样后尺寸为 0。

    采用 16 是因为网络包含 3 次 2x 下采样，16 / 2 / 2 / 2 = 2，仍大于 0，给出冗余安全边际。
    """

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
        if config.use_pvt:
            if timm is None:
                raise ImportError("timm is required when use_pvt=True. Install timm>=0.9 or set use_pvt=False.")
            self.encoder_up = PVTv2B2Encoder(
                in_channels=2, pretrained=config.pretrained_backbone, weights_path=config.pvt_weights_path
            )
            self.encoder_down = PVTv2B2Encoder(
                in_channels=2, pretrained=config.pretrained_backbone, weights_path=config.pvt_weights_path
            )
            channels = self.encoder_up.out_channels
        else:
            self.encoder_up = SimpleEncoder(in_channels=2, base=ch)
            self.encoder_down = SimpleEncoder(in_channels=2, base=ch)
            channels = [ch, ch * 2, ch * 4, ch * 8]

        # 部分解码器（生成 Pa, Pd）
        self.ppd_up = PartialPredictionDecoder(channels, num_classes=config.num_classes)
        self.ppd_down = PartialPredictionDecoder(channels, num_classes=config.num_classes)

        # ACA 多尺度融合（逐层）
        self.aca1 = ACAModule(channels[0], config.dilations)
        self.aca2 = ACAModule(channels[1], config.dilations)
        self.aca3 = ACAModule(channels[2], config.dilations)
        self.aca4 = ACAModule(channels[3], config.dilations)

        # PRE & 预测引导解码
        self.pre = PREModule()
        self.pd = PredictionGuidedDecoder(channels, num_classes=config.num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.dim() != 4:
            raise ValueError(f"期望输入维度 (N, C, H, W)，实际 {x.shape}")
        if x.shape[1] < 4:
            raise ValueError("输入通道不足 4，无法按模态对分支")

        # 记录原始尺寸并补齐到 16 的倍数，避免多次下采样后尺寸变为 0
        x, pad = _pad_to_multiple(x, factor=16)

        # 划分模态：上 (T1, T1CE)，下 (T2, FLAIR)
        x_up = x[:, :2]
        x_down = x[:, 2:4]

        feats_up = self.encoder_up(x_up)
        feats_down = self.encoder_down(x_down)

        # 暂态预测（Pa, Pd）
        pa = self.ppd_up(feats_up)
        pd = self.ppd_down(feats_down)

        # PRE 生成 P_H, P_L（仅使用前景/WT 概率）
        ph, pl = self.pre(pa, pd)

        # ACA 融合多模态特征（逐层对应公式 (1)-(4)）
        f1 = self.aca1(feats_up[0], feats_down[0], ph)
        f2 = self.aca2(feats_up[1], feats_down[1], ph)
        f3 = self.aca3(feats_up[2], feats_down[2], ph)
        f4 = self.aca4(feats_up[3], feats_down[3], ph)
        fused_feats = [f1, f2, f3, f4]

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
