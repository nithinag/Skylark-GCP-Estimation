"""Two-stage Coarse-to-Fine PyTorch models, Soft-Argmax decoding, losses, and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

from src.dataset import CLASS_NAMES


class DecoderBlock(nn.Module):
    """Bilinear upsample + skip concatenation + two Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNetBackbone(nn.Module):
    """Shared ResNet-34 encoder + UNet decoder outputting stride-2 heatmap features and bottleneck features."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64ch
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # /4, 64ch
        self.layer2 = backbone.layer2  # /8, 128ch
        self.layer3 = backbone.layer3  # /16, 256ch
        self.layer4 = backbone.layer4  # /32, 512ch

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # CenterNet-style prior bias (sigmoid(-4.6) ~= 0.01) prevents background focal-loss explosion at step 0.
        nn.init.constant_(self.heatmap_head[-1].bias, -4.6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s2 = self.stem(x)
        s4 = self.layer1(self.maxpool(s2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)

        d = self.dec4(s32, s16)
        d = self.dec3(d, s8)
        d = self.dec2(d, s4)
        d = self.dec1(d, s2)
        heatmap_logits = self.heatmap_head(d)
        return heatmap_logits, s32


class Stage1CoarseModel(nn.Module):
    """Stage 1: Full-image aspect-preserved coarse heatmap localizer."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.unet = ResNet34UNetBackbone(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heatmap_logits, _ = self.unet(x)
        return heatmap_logits


class Stage2FineModel(nn.Module):
    """Stage 2: Multi-task native-crop refiner (localization heatmap + 3-class shape head)."""

    def __init__(self, num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> None:
        super().__init__()
        self.unet = ResNet34UNetBackbone(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heatmap_logits, bottleneck = self.unet(x)
        avg_pool = bottleneck.mean(dim=(-2, -1))
        max_pool = bottleneck.amax(dim=(-2, -1))
        feat = torch.cat([avg_pool, max_pool], dim=1)
        class_logits = self.classifier(feat)
        return heatmap_logits, class_logits


def soft_argmax_2d(
    heatmap_logits: torch.Tensor,
    stride: int = 2,
) -> torch.Tensor:
    """Differentiable global 2D soft-argmax returning (B, 2) coordinates in input-image space."""
    logits_f32 = heatmap_logits.float()
    b, _, h, w = logits_f32.shape
    probs = F.softmax(logits_f32.view(b, -1), dim=-1).view(b, h, w)

    device = logits_f32.device
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)
    ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1)

    cx_hm = (probs * xs).sum(dim=(-2, -1))
    cy_hm = (probs * ys).sum(dim=(-2, -1))
    return torch.stack([cx_hm * stride, cy_hm * stride], dim=-1)


@torch.no_grad()
def decode_heatmap_peak(
    heatmap_logits: torch.Tensor,
    stride: int = 2,
    window_radius: int = 4,
) -> torch.Tensor:
    """Inference decoder: finds global argmax peak and refines sub-pixel center in a local window."""
    b, _, h, w = heatmap_logits.shape
    probs = torch.sigmoid(heatmap_logits[:, 0])
    flat_idx = probs.view(b, -1).argmax(dim=-1)
    peak_y = flat_idx // w
    peak_x = flat_idx % w

    coords = []
    for i in range(b):
        px = int(peak_x[i].item())
        py = int(peak_y[i].item())
        x1 = max(0, px - window_radius)
        x2 = min(w, px + window_radius + 1)
        y1 = max(0, py - window_radius)
        y2 = min(h, py + window_radius + 1)

        patch = probs[i, y1:y2, x1:x2]
        patch_sum = patch.sum().clamp(min=1e-8)
        xs = torch.arange(x1, x2, device=patch.device, dtype=patch.dtype).view(1, -1)
        ys = torch.arange(y1, y2, device=patch.device, dtype=patch.dtype).view(-1, 1)
        cx = (patch * xs).sum() / patch_sum
        cy = (patch * ys).sum() / patch_sum
        coords.append(torch.stack([cx * stride, cy * stride]))
    return torch.stack(coords, dim=0)


class GaussianFocalLoss(nn.Module):
    """CenterNet-style penalty-reduced focal loss for 2D Gaussian heatmaps."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute focal log-probs in FP32 so FP16 autocast never rounds (1 - 1e-6) to 1.0 -> log(0)=NaN.
        logits_f32 = logits.float()
        targets_f32 = targets.float()
        probs = torch.sigmoid(logits_f32).clamp(1e-6, 1.0 - 1e-6)
        peak_vals = targets_f32.amax(dim=(-2, -1), keepdim=True)
        pos_mask = targets_f32.ge(peak_vals - 1e-5).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = -((1.0 - probs) ** self.alpha) * torch.log(probs) * pos_mask
        neg_weights = (1.0 - targets_f32).clamp(min=0.0) ** self.beta
        neg_loss = -((probs) ** self.alpha) * torch.log(1.0 - probs) * neg_weights * neg_mask

        num_pos = pos_mask.sum().clamp(min=1.0)
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


def compute_localization_loss(
    heatmap_logits: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_xy: torch.Tensor,
    stride: int,
    focal_loss_fn: GaussianFocalLoss,
    coord_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine Gaussian heatmap focal loss with SmoothL1 loss on Soft-Argmax coordinates."""
    with torch.amp.autocast("cuda", enabled=False):
        hm_loss = focal_loss_fn(heatmap_logits, target_heatmap)
        pred_xy = soft_argmax_2d(heatmap_logits, stride=stride)
        coord_loss = F.smooth_l1_loss(pred_xy, target_xy.float(), beta=2.0)
        total = hm_loss + coord_weight * coord_loss
    return total, hm_loss, coord_loss


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint cleanly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
        "config": config or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load model weights (and optionally optimizer state) from a checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
