"""Glue between a Split (data.py), a feature pipeline, and DataLoaders.

Centralized here so train.py, evaluate.py and robustness.py all build
their transforms and loaders the exact same way -- a common source of
subtle train/eval bugs is small mismatches between how features were
computed at training time vs. evaluation time.
"""
from __future__ import annotations

from torch.utils.data import DataLoader

from .augment import Compose, RandomNoiseInjection, SpecAugment, add_synthetic_noise
from .data import Split, WatkinsDataset, collate_with_clips
from .features import LogMelSpectrogram
from .models.ast_model import ASTPreprocessor


def build_transforms(input_kind: str, train_augment: bool = True,
                      noise_augment_p: float = 0.3, spec_augment: bool = True):
    """Returns (train_transform, eval_transform) for the given input_kind
    ("logmel" or "ast"). Augmentation (noise injection, SpecAugment) is
    only ever applied to the train transform."""
    if input_kind == "logmel":
        feat = LogMelSpectrogram()
        eval_t = feat
        if train_augment:
            steps = [RandomNoiseInjection(p=noise_augment_p), feat]
            if spec_augment:
                steps.append(SpecAugment())
            train_t = Compose(*steps)
        else:
            train_t = feat
        return train_t, eval_t

    if input_kind == "ast":
        pre = ASTPreprocessor()
        eval_t = pre
        if train_augment:
            train_t = Compose(RandomNoiseInjection(p=noise_augment_p), pre)
        else:
            train_t = pre
        return train_t, eval_t

    raise ValueError(f"Unknown input_kind: {input_kind!r}")


def build_dataloaders(split: Split, input_kind: str, batch_size: int = 32,
                       num_workers: int = 0, train_augment: bool = True,
                       noise_augment_p: float = 0.3, spec_augment: bool = True,
                       drop_last_train: bool = True, pin_memory: bool = False):
    """`drop_last_train=True` (the default) is the right choice for normal
    multi-epoch training: dropping a partial final batch each epoch is
    harmless since shuffling means a different few samples are dropped
    every epoch, averaging out over training. Set it to False for any
    *single-pass* use of the train loader (e.g. one-time frozen-backbone
    embedding extraction in ast_model.extract_embeddings) -- there, a
    dropped partial batch means those samples are never used at all.

    `pin_memory=True` (callers pass this when training on CUDA) allocates
    batches in page-locked memory so the host->device copy can overlap
    compute. It costs a little host RAM and does nothing on CPU, so it
    defaults off. Worker processes are kept alive between epochs whenever
    `num_workers > 0`: feature extraction here is per-sample STFT work, and
    respawning workers every epoch is pure overhead once the dataset is
    large enough to matter.
    """
    train_t, eval_t = build_transforms(
        input_kind, train_augment=train_augment,
        noise_augment_p=noise_augment_p, spec_augment=spec_augment,
    )
    train_ds = WatkinsDataset(split.train, transform=train_t, random_crop=True)
    val_ds = WatkinsDataset(split.val, transform=eval_t, random_crop=False)
    test_ds = WatkinsDataset(split.test, transform=eval_t, random_crop=False)

    common = dict(num_workers=num_workers, collate_fn=collate_with_clips,
                   pin_memory=pin_memory, persistent_workers=num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last_train, **common)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_dl, val_dl, test_dl


def build_noisy_eval_transform(input_kind: str, snr_db: float | None, noise_kind: str = "white"):
    """Eval-time feature transform that injects synthetic noise into the
    raw waveform *before* feature extraction (matches how real acoustic
    degradation would enter the signal chain, e.g. a more distant animal
    or rougher sea state). `snr_db=None` returns the clean transform.

    Used by robustness.py to sweep accuracy vs. SNR for a trained model.
    """
    _, eval_feat = build_transforms(input_kind, train_augment=False)
    if snr_db is None:
        return eval_feat

    def _noisy(wav):
        noisy_wav = add_synthetic_noise(wav, snr_db=snr_db, kind=noise_kind)
        return eval_feat(noisy_wav)

    return _noisy


def build_eval_dataloader(clips, input_kind: str, transform, batch_size: int = 32, num_workers: int = 0):
    ds = WatkinsDataset(clips, transform=transform, random_crop=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       collate_fn=collate_with_clips)
