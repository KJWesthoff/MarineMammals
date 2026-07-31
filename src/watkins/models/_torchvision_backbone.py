"""Shared wrapper for adapting torchvision ImageNet backbones (ResNet,
EfficientNet, ...) to single-channel spectrogram input.

torchvision backbones expect 3-channel RGB input and their first
convolution's pretrained weights were learned for that. Rather than
surgically resizing the first conv layer (which throws away 2/3 of the
pretrained filters), we repeat the single spectrogram channel three times
before the backbone -- a standard, simple trick for reusing ImageNet
weights on single-channel inputs. It costs a bit of redundant compute for
a tensor this small; that's a fine trade on this hardware.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectrogramBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.backbone(x)
        return self.classifier(feats)
