"""Model registry: build any model in this project by name.

    from watkins.models import build_model
    from watkins.data import NUM_CLASSES
    model, input_kind = build_model("resnet18", num_classes=NUM_CLASSES)

`input_kind` tells you which feature pipeline the model expects:
  - "logmel"  -> watkins.features.LogMelSpectrogram
  - "ast"     -> watkins.models.ast_model.ASTPreprocessor
"""
from __future__ import annotations

from .ast_model import build_ast_classifier
from .cnn_baseline import build_baseline_cnn
from .efficientnet import build_efficientnet_b0
from .resnet import build_resnet18

_REGISTRY = {
    "baseline_cnn": (build_baseline_cnn, "logmel"),
    "resnet18": (build_resnet18, "logmel"),
    "efficientnet_b0": (build_efficientnet_b0, "logmel"),
    "ast": (build_ast_classifier, "ast"),
}


def build_model(name: str, num_classes: int, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Choices: {sorted(_REGISTRY)}")
    builder, input_kind = _REGISTRY[name]
    return builder(num_classes=num_classes, **kwargs), input_kind


__all__ = ["build_model"]
