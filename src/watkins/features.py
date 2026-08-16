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

``demon_envelope`` / ``demon_spectrum``
    Classic passive-sonar DEMON analysis (DEModulation Of Envelope on
    Noise): high-pass, rectify, low-pass, decimate, then FFT the envelope
    itself to expose a periodic pulse-repetition rate hidden inside
    broadband energy. Originally used to read propeller shaft/blade rate
    off cavitation noise; here it reads the click-repetition rate of an
    echolocating species' pulse train (sperm whale codas, dolphin
    echolocation trains) -- a periodicity a plain spectrogram doesn't
    surface as a single clean peak. ``demon_envelope`` returns the
    intermediate pulse train on its own, for plotting next to the
    spectrum.

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


def demon_envelope(
    wav: torch.Tensor,
    sample_rate: int = 16_000,
    hp_cutoff: float = 2000.0,
    envelope_lp_cutoff: float = 250.0,
    envelope_rate: float = 1000.0,
) -> tuple[torch.Tensor, float]:
    """The envelope-detector front half of DEMON, exposed on its own.

    High-pass -> full-wave rectify -> low-pass -> decimate. Returns
    (envelope [samples], envelope_sample_rate). Useful for plotting the
    pulse train that `demon_spectrum` then measures the rate of: the
    peaks you can count by eye in this signal are the same peaks the
    DEMON peak reports as a frequency.

    Decimation is what makes the FFT downstream usable. Click-repetition
    rates of interest are tens of Hz, so carrying the envelope at the
    original 16kHz wastes almost every FFT bin on frequencies that cannot
    contain a repetition rate.
    """
    hp = torchaudio.functional.highpass_biquad(wav, sample_rate, cutoff_freq=hp_cutoff)
    envelope = torchaudio.functional.lowpass_biquad(hp.abs(), sample_rate, cutoff_freq=envelope_lp_cutoff)
    decimation = max(1, int(sample_rate // envelope_rate))
    return envelope[..., ::decimation].squeeze(0), sample_rate / decimation


def demon_spectrum(
    wav: torch.Tensor,
    sample_rate: int = 16_000,
    hp_cutoff: float = 2000.0,
    envelope_lp_cutoff: float = 250.0,
    envelope_rate: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DEMON spectrum: reveals a periodic pulse-repetition rate hidden inside
    broadband energy.

    Pipeline: high-pass `wav` to isolate broadband click/cavitation-like
    energy -> full-wave rectify (envelope detector) -> low-pass the envelope
    -> decimate -> FFT the envelope itself. A sharp peak in the *envelope's*
    spectrum at frequency f means the underlying signal contains a pulse
    train repeating f times per second -- e.g. an echolocating species'
    click-repetition rate, which a plain spectrogram of the raw waveform
    doesn't surface as a single clean peak (it shows up smeared across many
    broadband clicks instead).

    The FFT runs over the **whole** decimated envelope, so the frequency
    resolution is 1/duration -- about 0.3Hz for a 3-4 second clip, fine
    enough to separate click rates that sit only a few Hz apart. (An
    earlier version passed a fixed `n_fft` at the audio sample rate, which
    both truncated the envelope to its first 0.26s and quantized the
    spectrum to 3.9Hz bins -- roughly a dozen usable points below 50Hz,
    far too coarse to see a click rate at all.)

    A Hann window plus mean removal keeps the DC component from leaking a
    broad skirt over the low-frequency bins where the rates of interest
    live. Magnitudes are scaled to envelope-amplitude units, so the peak
    height reads as the depth of the modulation.

    Returns (freqs [Hz], magnitude), DC bin included at index 0. The
    highest frequency returned is `envelope_rate`/2.
    """
    envelope, rate = demon_envelope(
        wav,
        sample_rate=sample_rate,
        hp_cutoff=hp_cutoff,
        envelope_lp_cutoff=envelope_lp_cutoff,
        envelope_rate=envelope_rate,
    )
    n = envelope.shape[-1]
    if n < 8:
        raise ValueError(f"envelope too short for a DEMON spectrum ({n} samples)")

    window = torch.hann_window(n, dtype=envelope.dtype, device=envelope.device)
    windowed = (envelope - envelope.mean()) * window

    n_fft = 1 << ((n - 1).bit_length() + 1)  # next power of two, zero-padded 2x
    magnitude = torch.fft.rfft(windowed, n=n_fft).abs() * (2.0 / window.sum())
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / rate)
    return freqs, magnitude


def matched_filter(wav: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Normalized cross-correlation of `wav` against `template`.

    At each sample position, computes the cosine similarity between
    `template` and the equal-length window of `wav` centered there --
    the classic matched-filter/template-matching detector. A response
    near 1.0 marks where (and how strongly) the template's shape appears
    in `wav`; near 0 means no resemblance.

    `template` is typically a short averaged click/call shape (fewer
    samples than `wav`). Returns a 1D response the same length as `wav`.

    Note what "shape" means here. This correlates whatever signal you
    hand it, and for click trains in this dataset that should be the
    *envelope* (`demon_envelope`), not the raw waveform -- see
    `average_pulse_template` for why the raw-waveform version fails.
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


def detect_peaks(
    response: torch.Tensor,
    sample_rate: float,
    threshold: float,
    refractory_s: float = 0.015,
) -> torch.Tensor:
    """Positions of local maxima above `threshold`, one per refractory window.

    The thresholding stage every detector needs after a matched filter:
    the filter gives a continuous response, and turning that into a list
    of *detections* means picking local maxima over a threshold and then
    enforcing a minimum spacing so that one broad peak is not counted
    several times. `refractory_s` is that spacing -- it also caps the
    highest pulse rate that can be reported (1/`refractory_s` Hz), so
    keep it well under the interval you expect between pulses.

    Suppression is greedy left-to-right (first qualifying peak wins, the
    rest of its window is discarded), which is the simple textbook
    version; a deployment detector would keep the strongest peak in each
    window instead.

    Returns a 1D LongTensor of sample indices into `response`.
    """
    signal = response.squeeze()
    if signal.ndim != 1:
        raise ValueError(f"expected a 1D response, got shape {tuple(response.shape)}")

    interior = signal[1:-1]
    is_local_max = (interior >= signal[:-2]) & (interior > signal[2:]) & (interior > threshold)
    candidates = torch.nonzero(is_local_max).squeeze(-1) + 1

    gap = max(1, int(refractory_s * sample_rate))
    kept: list[int] = []
    last = -gap - 1
    for index in candidates.tolist():
        if index - last > gap:
            kept.append(index)
            last = index
    return torch.tensor(kept, dtype=torch.long)


def average_pulse_template(
    envelope: torch.Tensor,
    sample_rate: float,
    duration_s: float = 0.025,
    n_sigma: float = 3.0,
    max_pulses: int = 40,
) -> torch.Tensor:
    """Build a click/pulse template by averaging the envelope around the
    strongest pulses in `envelope` (as returned by `demon_envelope`).

    Two decisions are baked in here, both of which matter more than they
    look.

    **Work on the envelope, not the raw waveform.** The obvious way to
    template a click -- slice 50ms of raw waveform and cross-correlate --
    does not work on this dataset. A click's fine waveform structure is
    the part destroyed by the 16kHz resampling (`data.py`), by whatever
    tape and hydrophone the recording came off, and by propagation; what
    survives is the *shape of the energy burst* and its timing. Measured
    on notebook 01's example clips, a raw-waveform template scores a
    different-species clip (0.44) marginally higher than a same-species
    one (0.40) -- i.e. no discrimination at all. The envelope keeps the
    repeatable part.

    **Average many pulses, don't take one.** A single pulse carries the
    noise realization that happened to sit under it. Averaging the
    strongest `max_pulses` detections suppresses that noise while leaving
    the common shape, the same reason a coda/click "exemplar" in the
    bioacoustics literature is an average rather than a hand-picked
    example.

    Pulses are found with `detect_peaks` at a `mean + n_sigma * std`
    threshold. Raises `ValueError` if the envelope contains no pulse far
    enough from its edges to cut a whole template out of -- which is the
    right outcome: a clip with no detectable pulse train has no click
    template to give.

    Returns a 1D tensor of `duration_s * sample_rate` samples (rounded to
    an even length).
    """
    env = envelope.squeeze()
    if env.ndim != 1:
        raise ValueError(f"expected a 1D envelope, got shape {tuple(envelope.shape)}")

    half = max(1, int(duration_s * sample_rate / 2))
    threshold = (env.mean() + n_sigma * env.std()).item()
    # Refractory shorter than the template, so two pulses inside one
    # template width still register separately.
    found = detect_peaks(env, sample_rate, threshold, refractory_s=duration_s * 0.6)
    usable = found[(found >= half) & (found < env.shape[-1] - half)]
    if usable.numel() == 0:
        raise ValueError(
            "no pulse found far enough from the clip edges to build a template -- "
            "this clip probably does not contain a click train"
        )

    strongest = usable[torch.argsort(env[usable], descending=True)[:max_pulses]]
    return torch.stack([env[i - half:i + half] for i in strongest.tolist()]).mean(dim=0)


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
