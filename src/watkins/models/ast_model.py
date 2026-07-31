"""Audio Spectrogram Transformer (AST, Gong et al. 2021) fine-tuning.

AST treats a spectrogram as a grid of patches (like ViT treats an image)
and lets self-attention mix information across *all* time-frequency
patches at every layer, instead of a CNN's local receptive field that
only grows gradually with depth. That's the architectural contrast this
project is built to let you observe directly: does global attention help
or hurt on short, single-source marine mammal recordings where the CNNs
above already capture most of the useful local time-frequency structure?

Practical constraints baked into this module, given this machine has no
usable GPU (see utils.get_device) and AST-base is ~86M parameters:

* We default to **linear probing** (freeze the pretrained transformer,
  train only a fresh linear classifier head on top of its pooled output).
  This is dramatically cheaper on CPU and is a legitimate, widely-used
  transfer-learning baseline in its own right.
* `extract_embeddings()` lets you run the frozen backbone over a split
  exactly once and cache the resulting 768-d vectors, so repeated epochs
  of "training" the linear head cost almost nothing -- you're just
  training logistic regression on fixed features.
* Full fine-tuning (`freeze_backbone=False`) is supported and wired
  through configs/ast_finetune.yaml, but expect it to be slow on CPU;
  use a small subset/epoch count if you try it, or see README for notes
  on renting a GPU for that one run.

Note on input length: the pretrained checkpoint
(MIT/ast-finetuned-audioset-10-10-0.4593) expects up to 1024 time frames
(~10.24s). Our clips are 3.0s (~300 frames), so the feature extractor
zero-pads roughly 70% of every input. That wastes most of the model's
positional embedding range -- a good "next steps" exercise is trimming
`max_length` and interpolating positional embeddings for shorter clips.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTForAudioClassification

DEFAULT_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTPreprocessor:
    """waveform [1, T] float32 @ 16kHz -> AST input_values [1024, 128].

    Drop-in replacement for LogMelSpectrogram in the Dataset transform
    pipeline, except AST needs its *own* fbank recipe (mean/std, window,
    mel count) to match what it was pretrained with -- reusing our
    LogMelSpectrogram here would silently distribution-shift every input.
    """

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT):
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(checkpoint)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        array = wav.squeeze(0).numpy()
        out = self.feature_extractor(array, sampling_rate=16_000, return_tensors="pt")
        return out["input_values"].squeeze(0)  # [1024, 128]


class ASTClassifier(nn.Module):
    """Wraps ASTForAudioClassification with a head resized to num_classes."""

    def __init__(self, num_classes: int, checkpoint: str = DEFAULT_CHECKPOINT,
                 freeze_backbone: bool = True):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        if freeze_backbone:
            for name, p in self.model.named_parameters():
                if not name.startswith("classifier"):
                    p.requires_grad = False

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values).logits

    @property
    def backbone(self) -> nn.Module:
        return self.model.audio_spectrogram_transformer


def build_ast_classifier(num_classes: int, freeze_backbone: bool = True,
                          checkpoint: str = DEFAULT_CHECKPOINT) -> ASTClassifier:
    return ASTClassifier(num_classes=num_classes, checkpoint=checkpoint, freeze_backbone=freeze_backbone)


@torch.no_grad()
def extract_embeddings(model: ASTClassifier, dataloader, device: torch.device,
                        amp_dtype: torch.dtype | None = None):
    """Run the (frozen) backbone once over a dataloader of AST input
    features and return (embeddings [N, hidden], labels [N]).

    Use this to turn a slow "re-run the transformer every epoch" linear
    probe into a fast "train logistic regression on cached features"
    problem -- the whole point of freezing the backbone in the first
    place.

    `amp_dtype` (see utils.autocast_dtype) runs the backbone in reduced
    precision, which is where nearly all of this path's wall-clock time
    goes on a GPU. Embeddings are always cast back to float32 before
    caching so the downstream head trains in full precision regardless.
    """
    model.eval()
    all_feats, all_labels = [], []
    for batch in dataloader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pooled = model.backbone(x).pooler_output  # [B, hidden]
        all_feats.append(pooled.float().cpu())
        all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


class LinearProbeHead(nn.Module):
    """A tiny classifier trained on cached AST embeddings.

    Deliberately mirrors ASTForAudioClassification's own `ASTMLPHead`
    (LayerNorm -> Linear) module-for-module -- same submodule names, same
    shapes -- so its trained state_dict can be `load_state_dict`'d
    straight into `model.model.classifier` afterwards, producing a
    checkpoint whose full forward pass exactly reproduces what the cached
    fast training loop optimized.
    """

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.layernorm(x))
