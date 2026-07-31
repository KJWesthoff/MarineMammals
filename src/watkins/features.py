"""Waveform -> time-frequency feature transforms.

The core exercise in this project is: turn a 1D pressure waveform into a
2D image-like representation (a spectrogram) so image architectures
(CNNs, ViT-style transformers) can be applied to it. This module builds
that representation with `torchaudio`, entirely on CPU/GPU tensors (no
librosa/numba dependency, which matters since this machine runs a very
new Python where numba support lags).

Two representations are provided:

``LogMelSpectrogram``
    Standard log-mel spectrogram: STFT -> mel filterbank -> log
    compression. This is what feeds the baseline CNN, ResNet and
    EfficientNet in this project.

``lofar_gram``
    A plain linear-frequency log-power spectrogram (no mel warping).
    Useful for visualization/discussion in the signal-processing
    notebook -- a linear axis keeps narrowband tonals at a fixed pixel
    row across recordings, which the mel scale's compression distorts.

``demon_spectrum``
    Classic passive-sonar DEMON analysis (DEModulation Of Envelope on
    Noise): high-pass, rectify, low-pass, then FFT the envelope itself to
    expose a periodic pulse-repetition rate hidden inside broadband
    energy. Originally used to read propeller shaft/blade rate off
    cavitation noise; here it reads the click-repetition rate of an
    echolocating species' pulse train (sperm whale codas, dolphin
    echolocation trains) -- a periodicity a plain spectrogram doesn't
    surface as a single clean peak.

``matched_filter``
    Normalized cross-correlation of a waveform against a reference
    template (e.g. an averaged click or call). The classic active/
    passive sonar detection primitive: slide a known target shape across
    the signal and look for where it matches, rather than feeding the
    whole clip to a classifier.

``matched_filter_bank``
    `matched_filter` run against several templates at once (e.g. a
    handful of clips from the same species), taking the best match at
    each position. A single template is a narrow, unrepresentative
    stand-in for a species' natural call-shape variation; a small bank
    is a cheap step towards something closer to a real detector -- see
    `docs/theory-deck` for how much further deployment-grade systems go
    (FFT/GCC-PHAT correlation, CFAR thresholding, whitening, and more).

A caveat worth keeping in mind throughout: this project resamples every
clip to 16kHz (see `data.py`), so the Nyquist ceiling is 8kHz. Baleen
whale calls (fin, blue, humpback, right whale) mostly sit well under that.
Many odontocete (toothed whale/dolphin/porpoise) whistles and especially
echolocation clicks extend well past 8kHz -- often into the tens or
hundreds of kHz -- so some of what would otherwise be the most
diagnostic content for those species is not visible to any model trained
on this pipeline's spectrograms. See `docs/next_steps.md` for the
higher-sample-rate variant this motivates as a follow-up.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    """waveform [1, T] -> log-mel spectrogram [1, n_mels, frames].

    Defaults (25ms window / 10ms hop / 64 mel bins) are conventional
    speech/audio-classification defaults, not tuned for marine mammal
    calls -- see notebook 01 for an exercise comparing window sizes
    against the frequency content of a few example species.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,        # 25ms @ 16kHz
        hop_length: int = 160,   # 10ms @ 16kHz
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float | None = None,
        top_db: float = 80.0,
        per_instance_normalize: bool = True,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate / 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.per_instance_normalize = per_instance_normalize

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.to_db(self.mel(wav))  # [1, n_mels, frames]
        if self.per_instance_normalize:
            mean = spec.mean()
            std = spec.std().clamp_min(1e-6)
            spec = (spec - mean) / std
        return spec

    @property
    def output_shape(self) -> tuple[int, int]:
        """(n_mels, frames) for one clip, given this config."""
        from .data import CLIP_SAMPLES
        hop = self.mel.hop_length
        frames = 1 + CLIP_SAMPLES // hop
        return self.mel.n_mels, frames


def lofar_gram(
    wav: torch.Tensor,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    top_db: float = 80.0,
) -> torch.Tensor:
    """Linear-frequency log-power spectrogram (a LOFARgram-style view).

    Returns [freq_bins, frames] in dB. Intended for plotting/inspection in
    notebook 01, not as a direct model input tensor.
    """
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
    return to_db(spec_transform(wav)).squeeze(0)


def demon_spectrum(
    wav: torch.Tensor,
    sample_rate: int = 16_000,
    hp_cutoff: float = 2000.0,
    envelope_lp_cutoff: float = 500.0,
    n_fft: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DEMON spectrum: reveals a periodic pulse-repetition rate hidden inside
    broadband energy.

    Pipeline: high-pass `wav` to isolate broadband click/cavitation-like
    energy -> full-wave rectify (envelope detector) -> low-pass the envelope
    -> FFT the envelope itself. A sharp peak in the *envelope's* spectrum at
    frequency f means the underlying signal contains a pulse train repeating
    f times per second -- e.g. an echolocating species' click-repetition
    rate, which a plain spectrogram of the raw waveform doesn't surface as
    a single clean peak (it shows up smeared across many broadband clicks
    instead).

    Returns (freqs [Hz], magnitude), DC bin included at index 0.
    """
    hp = torchaudio.functional.highpass_biquad(wav, sample_rate, cutoff_freq=hp_cutoff)
    envelope = hp.abs()
    envelope = torchaudio.functional.lowpass_biquad(envelope, sample_rate, cutoff_freq=envelope_lp_cutoff)
    envelope = (envelope - envelope.mean(dim=-1, keepdim=True)).squeeze(0)
    magnitude = torch.fft.rfft(envelope, n=n_fft).abs()
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, magnitude


def matched_filter(wav: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Normalized cross-correlation of `wav` against `template`.

    At each sample position, computes the cosine similarity between
    `template` and the equal-length window of `wav` centered there --
    the classic matched-filter/template-matching detector. A response
    near 1.0 marks where (and how strongly) the template's shape appears
    in `wav`; near 0 means no resemblance.

    `template` is typically a short averaged click/call waveform (fewer
    samples than `wav`). Returns a 1D response the same length as `wav`.
    """
    signal = wav.squeeze(0)
    tmpl = template.squeeze(0) if template.ndim > 1 else template
    tmpl = tmpl - tmpl.mean()
    tmpl_norm = tmpl.norm().clamp_min(1e-8)

    n = tmpl.shape[-1]
    pad_left = n // 2
    pad_right = n - 1 - pad_left
    padded = torch.nn.functional.pad(signal, (pad_left, pad_right))
    windows = padded.unfold(0, n, 1)  # [len(signal), n]
    windows = windows - windows.mean(dim=1, keepdim=True)
    window_norms = windows.norm(dim=1).clamp_min(1e-8)

    return (windows @ tmpl) / (window_norms * tmpl_norm)


def matched_filter_bank(wav: torch.Tensor, templates: list[torch.Tensor]) -> torch.Tensor:
    """Score `wav` against a *bank* of templates rather than a single one.

    Runs `matched_filter` against every template in `templates` and takes
    the elementwise maximum response across the bank at each sample
    position -- the standard way a real matched-filter bank combines
    multiple templates: a strong match against *any one* of them counts.

    A single template (e.g. one 50ms click snippet) is a poor stand-in
    for a species' natural call-shape variation -- amplitude, exact
    duration, and fine structure all vary clip to clip. A bank built from
    a handful of different clips is a cheap, meaningful step up, and is
    still nowhere near what a deployment-grade detector does (see
    `docs/theory-deck` for what's missing beyond this).

    Returns a 1D response the same length as `wav`.
    """
    if not templates:
        raise ValueError("templates must be a non-empty list")
    responses = torch.stack([matched_filter(wav, t) for t in templates])
    return responses.max(dim=0).values
