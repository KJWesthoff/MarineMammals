"""Render the theory deck's figure assets from the real dataset.

Every image in `assets/` comes from here, and every one is produced by the
same `watkins.features` calls notebook 01 runs, on the same clips notebook
01 names in its `EXAMPLE_RECORDS`. That is deliberate: the deck and the
notebook should be showing the audience the same signals, so a slide can
be trusted as a picture of what the code actually does rather than an
artist's impression of it.

What differs is only the styling -- deck palette, no axes, transparent
background -- so the figures sit inside the slide panels instead of
fighting them. Labels and numbers live in the HTML, not burned into the
PNGs, so they stay selectable and restyle with the CSS.

Run after changing notebook 01's example clips or any feature transform:

    python docs/theory-deck/make_assets.py

Requires the materialized dataset (`python -m watkins.prepare_data`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from watkins.data import load_manifest  # noqa: E402
from watkins.features import (  # noqa: E402
    LogMelSpectrogram,
    lofar_gram,
    demon_envelope,
    demon_spectrum,
    matched_filter,
    detect_peaks,
    average_pulse_template,
    pulse_train_stats,
)
import torchaudio  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"

# Deck palette (styles.css)
BG = "#060f18"
PANEL = "#0d1c29"
TEAL = "#2dd4bf"
AMBER = "#f4a261"
ROSE = "#f43f5e"
MUTED = "#7a97a3"
LINE = "#1c3644"

# Waterfall colormap: sonar-console navy through phosphor teal, with the
# hottest cells pushed to amber so peaks read as "detection" in the same
# colour the deck uses for detections everywhere else.
WATERFALL = LinearSegmentedColormap.from_list(
    "sonar",
    [
        (0.00, "#040a11"),
        (0.30, "#0b2b38"),
        (0.55, "#12666a"),
        (0.78, TEAL),
        (0.92, "#a9efe1"),
        (1.00, AMBER),
    ],
)

# The clips notebook 01 uses -- keep in sync with its EXAMPLE_RECORDS.
TONAL = "58018011"      # HumpbackWhale, clear harmonic stack
CLICKS = "7501101V"     # SpermWhale, ~20.5Hz click train
MF_SAME_TAPE = "7501101R"
MF_OTHER_TAPE = "61027004"
BANK_RECORDS = ["93570011", "72009005", "9357300H", "9358400V"]

DETECTION_THRESHOLD = 0.6


def load(record: str):
    clip = BY_RECORD[record]
    y, sr = sf.read(clip.path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def as_wav(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(y).unsqueeze(0)


# Panels are at most ~600 CSS px wide on a 1280x720 slide, so ~2x that is
# already retina-sharp. Higher just inflates the repo -- spectrogram noise
# compresses badly in PNG and these files are committed.
DPI = 120


def save(fig, name: str):
    path = ASSETS / name
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  {name}")


def bare_axes(figsize):
    """A figure that is nothing but the data -- no frame, ticks or margins."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    return fig, ax


def image(spec, name, figsize=(5.0, 4.0), **kwargs):
    fig, ax = bare_axes(figsize)
    ax.imshow(spec, aspect="auto", origin="lower", cmap=WATERFALL, **kwargs)
    save(fig, name)


def trace(name, series, figsize=(6.0, 2.2), baseline=True, ylim=None):
    """One or more (values, colour, width, alpha) line series, no axes."""
    fig, ax = bare_axes(figsize)
    for x, y, colour, width, alpha in series:
        ax.plot(x, y, color=colour, linewidth=width, alpha=alpha,
                solid_capstyle="round")
    if baseline:
        ax.axhline(0, color=LINE, linewidth=0.8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.margins(x=0.005)
    save(fig, name)


# --------------------------------------------------------------------------
# Part 2 -- spectrogram basics, on the notebook's humpback clip
# --------------------------------------------------------------------------

def spectrogram_assets():
    print("spectrograms (HumpbackWhale %s):" % TONAL)
    y, sr = load(TONAL)
    wav = as_wav(y)

    mel64 = LogMelSpectrogram(n_mels=64, per_instance_normalize=False)(wav).squeeze(0).numpy()
    image(mel64, "hero_spectrogram.png", figsize=(7.0, 3.2))
    image(mel64, "mel_spectrogram.png", figsize=(5.0, 4.0))

    for name, n_fft, hop in [
        ("window_fine_time.png", 160, 80),
        ("window_balanced.png", 400, 160),
        ("window_fine_freq.png", 1024, 512),
    ]:
        feat = LogMelSpectrogram(n_fft=n_fft, hop_length=hop, n_mels=32,
                                 per_instance_normalize=False)
        image(feat(wav).squeeze(0).numpy(), name)

    # LOFARgram, zoomed to the band the harmonics live in (matches the
    # notebook's 0-4kHz ylim).
    lofar = lofar_gram(wav, n_fft=1024, hop_length=256).numpy()
    cutoff = int(4000 / (sr / 2) * lofar.shape[0])
    image(lofar[:cutoff], "lofar_spectrogram.png")

    # The same view of a click train, for the tonal-vs-click comparison:
    # vertical striations instead of horizontal lines. Clamped to
    # percentiles -- a broadband click train fills most of the 80dB range,
    # so the default scaling washes the whole panel out to one flat colour.
    yc, src = load(CLICKS)
    lofar_c = lofar_gram(as_wav(yc), n_fft=1024, hop_length=256).numpy()
    image(lofar_c, "lofar_clicks.png",
          vmin=np.percentile(lofar_c, 40), vmax=np.percentile(lofar_c, 99.8))


# --------------------------------------------------------------------------
# Per-instance normalization -- the notebook's section 4, shared colour scale
# --------------------------------------------------------------------------

def normalization_assets():
    print("per-instance normalization:")
    rng = np.random.default_rng(0)
    sample = rng.choice(MANIFEST, size=200, replace=False)
    rms = np.array([np.sqrt(np.mean(sf.read(c.path)[0].astype(np.float64) ** 2))
                    for c in sample])
    loud, quiet = sample[rms.argmax()], sample[rms.argmin()]

    raw_feat = LogMelSpectrogram(per_instance_normalize=False)
    norm_feat = LogMelSpectrogram(per_instance_normalize=True)

    specs = {}
    for role, clip in [("loud", loud), ("quiet", quiet)]:
        y, sr = sf.read(clip.path)
        w = as_wav(y.astype(np.float32))
        specs[role] = (raw_feat(w).squeeze(0).numpy(), norm_feat(w).squeeze(0).numpy())

    # The whole point: one scale per column, shared by both clips. Autoscaling
    # each panel is what makes raw and normalized look identical.
    raw_lim = (min(s[0].min() for s in specs.values()), max(s[0].max() for s in specs.values()))
    norm_lim = (min(s[1].min() for s in specs.values()), max(s[1].max() for s in specs.values()))

    for role, (raw, norm) in specs.items():
        image(raw, f"norm_raw_{role}.png", figsize=(4.4, 3.0),
              vmin=raw_lim[0], vmax=raw_lim[1])
        image(norm, f"norm_z_{role}.png", figsize=(4.4, 3.0),
              vmin=norm_lim[0], vmax=norm_lim[1])

    print(f"    RMS ratio {rms.max() / rms.min():.0f}:1, "
          f"raw dB means {specs['loud'][0].mean():.1f} vs {specs['quiet'][0].mean():.1f}")


# --------------------------------------------------------------------------
# Filters and envelopes -- the new signal-processing section
# --------------------------------------------------------------------------

def filter_assets():
    print("filters + envelope (SpermWhale %s):" % CLICKS)
    y, sr = load(CLICKS)
    wav = as_wav(y)
    t = np.arange(len(y)) / sr

    # A 0.4s window around the middle of the click train: at full clip width
    # individual clicks merge into a solid block and the point is lost.
    lo, hi = int(1.4 * sr), int(1.8 * sr)
    seg_t, seg = t[lo:hi], y[lo:hi]

    # All three panels share the raw signal's y-scale. Letting each autoscale
    # would hide the only thing a filter does -- remove energy -- and make the
    # low-passed residual look bigger than the signal it came from.
    span = float(np.abs(seg).max()) * 1.1
    ylim = (-span, span)

    trace("filter_raw.png", [(seg_t, seg, TEAL, 0.7, 0.95)], ylim=ylim)

    hp = torchaudio.functional.highpass_biquad(wav, sr, cutoff_freq=2000.0).squeeze(0).numpy()
    trace("filter_highpassed.png", [(seg_t, hp[lo:hi], TEAL, 0.7, 0.95)], ylim=ylim)

    # Low-passed instead: the complementary half, which for a click train is
    # mostly the rumble the high-pass was there to remove.
    lp = torchaudio.functional.lowpass_biquad(wav, sr, cutoff_freq=500.0).squeeze(0).numpy()
    trace("filter_lowpassed.png", [(seg_t, lp[lo:hi], MUTED, 0.7, 0.95)], ylim=ylim)

    # Rectified, and then the envelope over the top of it.
    rect = np.abs(hp[lo:hi])
    trace("filter_rectified.png", [(seg_t, rect, TEAL, 0.7, 0.9)], baseline=False)

    env, env_rate = demon_envelope(wav, sample_rate=sr)
    env_np = env.numpy()
    env_t = np.arange(len(env_np)) / env_rate
    env_mask = (env_t >= 1.4) & (env_t <= 1.8)
    fig, ax = bare_axes((6.0, 2.2))
    ax.plot(seg_t, rect, color=TEAL, linewidth=0.6, alpha=0.45)
    ax.plot(env_t[env_mask], env_np[env_mask], color=AMBER, linewidth=2.2)
    ax.margins(x=0.005)
    save(fig, "envelope_overlay.png")

    # The full-clip envelope: the pulse train DEMON measures the rate of.
    trace("envelope_full.png", [(env_t, env_np, AMBER, 0.9, 0.95)],
          figsize=(7.0, 2.0), baseline=False)

    # Same treatment on the tonal clip -- no pulse train to find.
    yt, srt = load(TONAL)
    env_t2, rate2 = demon_envelope(as_wav(yt), sample_rate=srt)
    trace("envelope_tonal.png",
          [(np.arange(len(env_t2)) / rate2, env_t2.numpy(), MUTED, 0.9, 0.95)],
          figsize=(7.0, 2.0), baseline=False)


# --------------------------------------------------------------------------
# DEMON -- the real spectra, replacing the schematic
# --------------------------------------------------------------------------

def demon_assets():
    print("DEMON spectra:")
    for record, name, colour in [(CLICKS, "demon_spectrum_clicks.png", TEAL),
                                 (TONAL, "demon_spectrum_tonal.png", MUTED)]:
        y, sr = load(record)
        freqs, mag = demon_spectrum(as_wav(y), sample_rate=sr)
        keep = (freqs >= 2) & (freqs <= 60)
        f, m = freqs[keep].numpy(), mag[keep].numpy()
        peak = f[m.argmax()]

        fig, ax = bare_axes((6.0, 2.6))
        ax.plot(f, m, color=colour, linewidth=1.6)
        ax.fill_between(f, 0, m, color=colour, alpha=0.18)
        if colour == TEAL:
            ax.axvline(peak, color=AMBER, linestyle="--", linewidth=1.4)
            ax.axvline(2 * peak, color=AMBER, linestyle=":", linewidth=1.1, alpha=0.7)
        ax.set_xlim(f.min(), f.max())
        ax.set_ylim(0, m.max() * 1.12)
        save(fig, name)
        print(f"    {record}: peak {peak:.1f}Hz, {m.max() / np.median(m):.0f}x median")


# --------------------------------------------------------------------------
# Matched filtering in the envelope domain
# --------------------------------------------------------------------------

def matched_filter_assets():
    print("matched filter:")
    y, sr = load(CLICKS)
    env_t, env_rate = demon_envelope(as_wav(y), sample_rate=sr)
    template = average_pulse_template(env_t, env_rate)

    # The template itself -- an averaged click, not one hand-picked pulse.
    trace("pulse_template.png",
          [(np.arange(len(template)) / env_rate * 1000, template.numpy(), AMBER, 3.0, 1.0)],
          figsize=(3.0, 2.0), baseline=False)

    cases = [
        (MF_SAME_TAPE, "same_tape", TEAL),
        (MF_OTHER_TAPE, "other_tape", TEAL),
        (TONAL, "humpback", MUTED),
    ]
    for record, slug, colour in cases:
        yc, src = load(record)
        env, rate = demon_envelope(as_wav(yc), sample_rate=src)
        response = matched_filter(env.unsqueeze(0), template)
        stats = pulse_train_stats(env, response, rate, DETECTION_THRESHOLD)
        det = stats.detections
        e = env.numpy()
        t = np.arange(len(e)) / rate

        # Envelope with detections marked, full clip.
        fig, ax = bare_axes((6.0, 2.0))
        ax.plot(t, e, color=colour, linewidth=0.7, alpha=0.9)
        ax.plot(det.numpy() / rate, e[det.numpy()], ".", color=AMBER, markersize=6)
        ax.margins(x=0.005)
        ax.set_ylim(0, e.max() * 1.12)
        save(fig, f"mf_detections_{slug}.png")

        # Response against the threshold, first second only.
        zoom = t <= 1.0
        r = response.numpy()
        fig, ax = bare_axes((6.0, 2.0))
        ax.plot(t[zoom], r[zoom], color=colour, linewidth=1.0)
        ax.axhline(DETECTION_THRESHOLD, color=AMBER, linestyle="--", linewidth=1.2)
        ax.set_ylim(-1, 1)
        ax.margins(x=0.005)
        save(fig, f"mf_response_{slug}.png")

        if stats.note is None:
            print(f"    {record} ({slug}): {stats.n} det, {stats.rate_hz:.1f}/s, "
                  f"CV {stats.cv:.2f}, {stats.on_pulse:.0%} on pulse")
        else:
            # The slide quotes these numbers, so never print one the library
            # declined to estimate -- see pulse_train_stats' docstring.
            print(f"    {record} ({slug}): {stats.note}")


if __name__ == "__main__":
    MANIFEST = load_manifest()
    BY_RECORD = {clip.record_number: clip for clip in MANIFEST}
    ASSETS.mkdir(exist_ok=True)

    spectrogram_assets()
    normalization_assets()
    filter_assets()
    demon_assets()
    matched_filter_assets()
    print("\nwrote assets to", ASSETS)
