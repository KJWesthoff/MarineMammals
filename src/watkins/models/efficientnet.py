"""EfficientNet-B0 on log-mel spectrograms, optionally ImageNet-pretrained.

EfficientNet's headline idea (Tan & Le 2019) is compound scaling depth,
width and input resolution together under a fixed compute budget, using
MBConv (inverted residual + squeeze-excite) blocks instead of ResNet's
plain residual blocks. It's a useful second CNN family to compare against
ResNet precisely because it's a different set of architectural choices at
a similar (B0: ~5M params) parameter budget -- differences you see
between the two here are more about inductive bias than raw capacity.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision

from ._torchvision_backbone import SpectrogramBackbone


def build_efficientnet_b0(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = torchvision.models.efficientnet_b0(weights=weights)
    feature_dim = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()
    return SpectrogramBackbone(backbone, feature_dim, num_classes, freeze_backbone=freeze_backbone)
