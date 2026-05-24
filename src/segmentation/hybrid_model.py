"""
Hybrid Segmentation Model: ConvNeXt-Tiny encoder + UNet++ decoder.

ConvNeXt-Tiny skip channels (torchvision features[0..7]):
    enc1  features[0]+[1]  stem + stage 1  →   96 ch  H/4
    enc2  features[2]+[3]  down + stage 2  →  192 ch  H/8
    enc3  features[4]+[5]  down + stage 3  →  384 ch  H/16
    enc4  features[6]+[7]  down + stage 4  →  768 ch  H/32
    (no H/2 skip — stem uses 4×4 stride-4 patch embedding)

UNet++ node layout  (final output at H/4 → ×4 upsample to H):
    Scale H/16 :  x31
    Scale H/8  :  x21  x22
    Scale H/4  :  x11  x12  x13  ← decoder output

Competes with: Cell_ACDC (https://github.com/SchmollerLab/Cell_ACDC)
Dataset: LIVECell — MCF7, SkBr3, SKOV3 (EAT-relevant cancer cell lines)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


# ── Primitives ────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpProject(nn.Module):
    """Bilinear upsample ×2 then project channels with 1×1 conv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class DenseNode(nn.Module):
    """One UNet++ dense node: concatenate inputs → ConvBnRelu."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBnRelu(in_ch, out_ch)

    def forward(self, tensors: list) -> torch.Tensor:
        return self.conv(torch.cat(tensors, dim=1))


# ── Main Model ────────────────────────────────────────────────────────────────

class HybridSegmentationModel(nn.Module):
    """
    Input:  (B, 1, H, W)  phase-contrast grayscale. H, W divisible by 32.

    Output dict:
        mask         (B, 1, H, W)  sigmoid segmentation probability ∈ [0, 1]
        distance     (B, 1, H, W)  distance transform regression (watershed)

    Args:
        pretrained:      load ConvNeXt-Tiny ImageNet weights
        use_checkpoint:  gradient checkpointing on ConvNeXt encoder stages
                         (saves ~30% VRAM at +20% compute cost)
    """
    def __init__(self, pretrained: bool = True, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # ── ConvNeXt-Tiny Encoder ─────────────────────────────────────────────
        weights  = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)

        # Adapt stem Conv2d(3, 96, 4, 4) → Conv2d(1, 96, 4, 4) for grayscale
        stem_conv = backbone.features[0][0]
        grey_stem = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0, bias=False)
        if pretrained:
            grey_stem.weight.data.copy_(
                stem_conv.weight.data.mean(dim=1, keepdim=True)
            )
        backbone.features[0][0] = grey_stem

        # Group into 4 encoder stages; each includes its leading downsample
        self.enc1 = nn.Sequential(backbone.features[0], backbone.features[1])  #  96ch H/4
        self.enc2 = nn.Sequential(backbone.features[2], backbone.features[3])  # 192ch H/8
        self.enc3 = nn.Sequential(backbone.features[4], backbone.features[5])  # 384ch H/16
        self.enc4 = nn.Sequential(backbone.features[6], backbone.features[7])  # 768ch H/32

        # ── UNet++ upsampling projections ─────────────────────────────────────
        self.up_e4  = UpProject(768, 256)
        self.up_e3  = UpProject(384, 128)
        self.up_x31 = UpProject(256, 128)
        self.up_e2  = UpProject(192,  64)
        self.up_x21 = UpProject(128,  64)
        self.up_x22 = UpProject(128,  64)

        # ── UNet++ dense nodes ────────────────────────────────────────────────
        self.x31 = DenseNode(384 + 256,       256)
        self.x21 = DenseNode(192 + 128,       128)
        self.x22 = DenseNode(192 + 128 + 128, 128)
        self.x11 = DenseNode( 96 + 64,              64)
        self.x12 = DenseNode( 96 + 64 + 64,         64)
        self.x13 = DenseNode( 96 + 64 + 64 + 64,    64)

        # ── Output heads ──────────────────────────────────────────────────────
        self.coarse_head   = nn.Conv2d(64, 1, 1)
        self.distance_head = nn.Conv2d(64, 1, 1)

    def _ckpt(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x: torch.Tensor, return_feats: bool = False) -> dict:
        e1 = self._ckpt(self.enc1, x)    # (B,  96, H/4,  W/4)
        e2 = self._ckpt(self.enc2, e1)   # (B, 192, H/8,  W/8)
        e3 = self._ckpt(self.enc3, e2)   # (B, 384, H/16, W/16)
        e4 = self._ckpt(self.enc4, e3)   # (B, 768, H/32, W/32)

        x31 = self.x31([e3,      self.up_e4(e4,   e3.shape[2:])])
        x21 = self.x21([e2,      self.up_e3(e3,   e2.shape[2:])])
        x22 = self.x22([e2, x21, self.up_x31(x31, e2.shape[2:])])
        x11 = self.x11([e1,           self.up_e2(e2,   e1.shape[2:])])
        x12 = self.x12([e1, x11,      self.up_x21(x21, e1.shape[2:])])
        x13 = self.x13([e1, x11, x12, self.up_x22(x22, e1.shape[2:])])

        d = F.interpolate(x13, size=x.shape[2:], mode="bilinear", align_corners=False)

        out = {
            "mask":     torch.sigmoid(self.coarse_head(d)),
            "distance": self.distance_head(d),
        }
        if return_feats:
            return out, d
        return out
