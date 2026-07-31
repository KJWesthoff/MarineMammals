"""Synthetic noise injection and SpecAugment-style masking.

Two uses in this project:

1. **Training-time augmentation** -- mix in noise/SpecAugment masks while
   training so models don't overfit to the exact recording conditions of
   any one species' original tapes (this matters a lot for the 14 species
   with only a single source recording, see data.py's split docstring).
2. **Robustness evaluation** (robustness.py) -- take a *trained* model and
   re-run it on test clips with progressively more noise added, to plot
   accuracy vs. SNR. This mimics degraded real-world SNR conditions
   (distant animals, sea state, hydrophone self-noise, historical tape
   hiss) that a real bioacoustic monitoring system has to cope with.
"""
from __future__ import annotations

import torch


def white_noise(num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(1, num_samples, generator=generator)


def pink_noise(num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """1/f ("pink") noise via spectral shaping of white noise.

    Pink noise is a closer match to real ambient ocean noise than white
    noise (ocean ambient spectra roughly follow a 1/f-ish decay above the
    shipping-noise-dominated band -- see the Wenz curves) than pure white
    noise.
    """
    white = torch.randn(num_samples, generator=generator)
    spectrum = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(num_samples)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    scale = 1.0 / torch.sqrt(freqs)
    shaped = torch.fft.irfft(spectrum * scale, n=num_samples)
    return (shaped / shaped.std().clamp_min(1e-8)).unsqueeze(0)


NOISE_GENERATORS = {"white": white_noise, "pink": pink_noise}


def mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Scale `noise` so the mix (signal + noise) has the target SNR in dB,
    relative to `signal`'s own power, then return signal + scaled noise.

    signal, noise: [1, T] tensors of equal length.
    """
    sig_power = signal.pow(2).mean().clamp_min(1e-12)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / noise_power)
    return signal + scale * noise


def add_synthetic_noise(wav: torch.Tensor, snr_db: float, kind: str = "white",
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """Convenience wrapper: generate `kind` noise matching wav's length and
    mix it in at `snr_db`."""
    gen_fn = NOISE_GENERATORS[kind]
    noise = gen_fn(wav.shape[-1], generator=generator)
    return mix_at_snr(wav, noise, snr_db)


class RandomNoiseInjection:
    """Waveform-domain training augmentation: with probability `p`, mix in
    white or pink noise at a uniformly sampled SNR in `snr_range_db`."""

    def __init__(self, p: float = 0.5, snr_range_db: tuple[float, float] = (0.0, 20.0),
                 kinds: tuple[str, ...] = ("white", "pink")):
        self.p = p
        self.snr_range_db = snr_range_db
        self.kinds = kinds

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return wav
        kind = self.kinds[torch.randint(len(self.kinds), (1,)).item()]
        snr = torch.empty(1).uniform_(*self.snr_range_db).item()
        return add_synthetic_noise(wav, snr_db=snr, kind=kind)


class SpecAugment:
    """Time/frequency masking (Park et al. 2019) applied to a spectrogram
    tensor [1, n_mels, frames]. A cheap, effective regularizer for
    spectrogram-based classifiers with limited training data."""

    def __init__(self, freq_mask_width: int = 8, time_mask_width: int = 16,
                 num_freq_masks: int = 2, num_time_masks: int = 2):
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()
        _, n_mels, frames = spec.shape
        fill = spec.mean()
        for _ in range(self.num_freq_masks):
            w = torch.randint(0, self.freq_mask_width + 1, (1,)).item()
            if w == 0 or w >= n_mels:
                continue
            start = torch.randint(0, n_mels - w, (1,)).item()
            spec[:, start:start + w, :] = fill
        for _ in range(self.num_time_masks):
            w = torch.randint(0, self.time_mask_width + 1, (1,)).item()
            if w == 0 or w >= frames:
                continue
            start = torch.randint(0, frames - w, (1,)).item()
            spec[:, :, start:start + w] = fill
        return spec


class Compose:
    """Chain callables: waveform -> ... -> features."""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
