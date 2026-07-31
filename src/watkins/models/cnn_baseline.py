"""A small CNN trained from scratch on log-mel spectrograms.

This is the "sanity check" model of the project: no pretraining, no
transfer learning, just four conv blocks and a linear head. It exists so
you have a fast, honest baseline to compare the transfer-learning models
(ResNet, EfficientNet) and the transformer (AST) against. If a fancy
pretrained model can't beat this by a meaningful margin, that tells you
something about whether the extra complexity is earning its keep on a
54-species problem with severe long-tail imbalance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """Input: [B, 1, n_mels, frames] log-mel spectrogram.
    Output: [B, num_classes] logits.
    """

    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            _conv_block(1, c),
            _conv_block(c, c * 2),
            _conv_block(c * 2, c * 4),
            _conv_block(c * 4, c * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_baseline_cnn(num_classes: int, **kwargs) -> BaselineCNN:
    return BaselineCNN(num_classes=num_classes, **kwargs)
