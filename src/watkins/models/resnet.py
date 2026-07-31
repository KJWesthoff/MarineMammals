"""ResNet-18 on log-mel spectrograms, optionally initialized from
ImageNet-pretrained weights (transfer learning).

Why ResNet at all for spectrograms: it's a translation-equivariant conv
stack with residual connections, originally built for natural images.
Spectrograms aren't natural images (the two axes -- frequency and time --
mean very different things, unlike an image's x/y), but local
time-frequency patterns (a whistle's slope, a click train's spacing) are
still something 2D convolution picks up well, and this is exactly the
architecture the exercise brief asks you to compare against a plain CNN
and against AST's global self-attention.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision

from ._torchvision_backbone import SpectrogramBackbone


def build_resnet18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = torchvision.models.resnet18(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return SpectrogramBackbone(backbone, feature_dim, num_classes, freeze_backbone=freeze_backbone)
