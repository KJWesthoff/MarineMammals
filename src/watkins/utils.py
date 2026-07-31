"""Small cross-cutting helpers: seeding, device selection, parameter counting."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch RNGs for reproducibility.

    Note: full bit-for-bit reproducibility across CPU thread counts is not
    guaranteed by PyTorch, but this removes the main sources of run-to-run
    variance for a teaching project.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Pick a usable compute device.

    This machine's GPU (Quadro P520) is Pascal-generation (compute
    capability 6.1). Current PyTorch CUDA wheels are built for CC >= 7.5,
    so `torch.cuda.is_available()` can return True while every actual CUDA
    kernel launch fails. We defend against that by running a trivial op on
    the device and falling back to CPU if it raises.
    """
    if prefer_cuda and torch.cuda.is_available():
        try:
            x = torch.zeros(1, device="cuda")
            _ = x + 1
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class AverageMeter:
    """Tracks a running mean (loss, accuracy, ...) over an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Timer:
    """Context manager: `with Timer() as t: ...; print(t.elapsed)`."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        return False


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Assumes higher-is-better (e.g. validation macro-F1). Negate the metric
    yourself if you want to track a loss instead.
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop


def repo_root() -> Path:
    """Path to the project root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Where the materialized Watkins dataset lives.

    Defaults to `<repo_root>/Watkins/watkins_16k`, but honors
    `WATKINS_DATA_ROOT` if set -- useful when code and data shouldn't
    share a filesystem root, e.g. on Colab: code mounted from Google
    Drive (slow, quota-limited), data downloaded fresh each session to
    fast local/ephemeral disk.
    """
    override = os.environ.get("WATKINS_DATA_ROOT")
    if override:
        return Path(override)
    return repo_root() / "Watkins" / "watkins_16k"


def results_root() -> Path:
    """Where checkpoints/logs/metrics/figures get written.

    Defaults to `<repo_root>/results`, but honors `WATKINS_RESULTS_ROOT`
    if set -- e.g. on Colab, pointing this at a Google Drive path keeps
    trained checkpoints and metrics after the runtime disconnects, even
    though the (re-derivable) dataset itself lives on ephemeral local
    disk via `data_root()`.
    """
    override = os.environ.get("WATKINS_RESULTS_ROOT")
    if override:
        return Path(override)
    return repo_root() / "results"
