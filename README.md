# Learning passive sonar signal analysis and classification with the Watkins Marine Mammal Sound Database

A hands-on curriculum for learning passive-sonar signal analysis and
classification: detecting, characterizing, and identifying acoustic
targets from underwater recordings, using real marine-mammal recordings
in the [Watkins Marine Mammal Sound
Database](https://cis.whoi.edu/science/B/whalesounds/) (WMMSD), a
~70-year archive collected and curated by the Woods Hole Oceanographic
Institution, as the target signals. Several of these species are
themselves biological sonar systems -- dolphins, porpoises, and sperm
whales navigate and hunt by echolocation, producing literal biosonar
click trains -- which makes classic passive-sonar analysis techniques
(LOFARgrams, DEMON, matched filtering; see notebook 01) a direct fit for
this data, not just a borrowed metaphor.

WHOI's original Watkins download page is currently down, so this project
runs on a community-maintained Hugging Face re-release of Watkins instead
(`ivangtorre/watkins-marine-mammal-full-cuts`, which recovered the audio
from Internet Archive snapshots -- see Citations below). The curriculum
covers waveform -> spectrogram preprocessing and sonar-analysis
techniques, a from-scratch CNN baseline, transfer learning from ImageNet
(ResNet-18, EfficientNet-B0), a fine-tuned Audio Spectrogram Transformer
(AST), and robustness testing against synthetic noise, applied to a
realistic classification problem: 54 species with real, severe long-tail
class imbalance (2,647 clips for killer whale down to a single clip for
harbour seal).

## Start here

Work through the notebooks in order -- each one is a self-contained
lesson with theory, runnable code, and exercises:

| # | Notebook | What you'll learn |
|---|----------|--------------------|
| 00 | [`dataset_exploration`](notebooks/00_dataset_exploration.ipynb) | The data: 54 species, 15,248 clips, severe long-tail imbalance, listening to tonal calls vs. biosonar click trains, and why splitting by original recording (`tape_id`), not by clip, is essential to an honest test set. |
| 01 | [`signal_processing_fundamentals`](notebooks/01_signal_processing_fundamentals.ipynb) | Waveform -> spectrogram: STFT, window/hop trade-offs, mel vs. linear frequency axis (LOFARgrams); then two classic passive-sonar analysis techniques -- DEMON (reading a click-repetition rate off an echolocating species) and matched filtering (template-matching a known call shape). |
| 02 | [`baseline_cnn`](notebooks/02_baseline_cnn.ipynb) | A from-scratch CNN and class-weighted loss for a 54-way, heavily imbalanced problem. |
| 03 | [`transfer_learning_cnns`](notebooks/03_transfer_learning_cnns.ipynb) | ResNet-18 and EfficientNet-B0 fine-tuned from ImageNet weights; frozen-backbone vs. fine-tuned comparisons. |
| 04 | [`audio_spectrogram_transformer`](notebooks/04_audio_spectrogram_transformer.ipynb) | AST (ViT-style self-attention over spectrogram patches), pretrained on AudioSet; linear probing vs. full fine-tuning, and why this project defaults to the former on this hardware. |
| 05 | [`model_comparison`](notebooks/05_model_comparison.ipynb) | All four architectures compared head-to-head: accuracy, macro-F1, per-class performance on the best-represented species, and parameter efficiency. |
| 06 | [`robustness_to_noise`](notebooks/06_robustness_to_noise.ipynb) | Accuracy vs. SNR curves under synthetic white/pink noise injection -- a proxy for range/sea-state degradation a real passive acoustic monitoring system has to handle. |

See [`docs/next_steps.md`](docs/next_steps.md) for extension ideas once
you've been through all seven (raw-waveform models, self-supervised
pretraining, few-shot learning for the long tail, and more).

An illustrated slide deck covering the theory behind the exercises --
STFT/LOFAR, DEMON, matched filtering, the three architectures, and the
imbalance/robustness pitfalls -- lives at
[`docs/theory-deck/presentation.html`](docs/theory-deck/presentation.html)
(open directly in a browser, no build step).

## Why this dataset needs more care than it looks like

Two things about the Watkins database are easy to miss and change how you
should interpret any result on it -- see
[`docs/class_reference.md`](docs/class_reference.md) for the full detail:

1. **Severe, real class imbalance.** Species representation spans three
   orders of magnitude (2,647 clips for killer whale down to 1 clip for
   harbour seal), and it's a genuine property of the historical archive
   (some species, and even some individual animals, were simply recorded
   far more often), not a preprocessing artifact. 14 of the 54 species
   have only a single original recording to their name, so there's no
   such thing as an unseen-recording test set for them -- they end up
   train-only by construction (`watkins.data.build_split`).
2. **Heterogeneous recording conditions.** 47 distinct native sample
   rates (320Hz-192kHz) and clip lengths from 16ms to over 24 minutes,
   reflecting seven decades of different recording equipment.
   `watkins.prepare_data` resamples everything to a common 16kHz, which
   is a deliberate, documented information loss: many dolphin/porpoise
   whistles and echolocation clicks extend well past the resulting 8kHz
   Nyquist ceiling. See `features.py`'s module docstring and
   `docs/next_steps.md` for the higher-sample-rate variant this
   motivates as a follow-up.

Clips are split into train/val/test by grouping on original recording
(`tape_id`, not individual clip) so that no split boundary ever cuts
through a single recording -- see `watkins/data.py` for the
implementation.

## Hardware notes: why everything here runs on CPU

This machine has an NVIDIA Quadro P520 (2GB VRAM), but it's a
Pascal-generation GPU (compute capability 6.1) and current PyTorch CUDA
wheels are built for compute capability >= 7.5 -- `torch.cuda.is_available()`
returns `True`, but any actual kernel launch fails. Combined with the
2GB VRAM ceiling (too little for comfortable AST fine-tuning regardless),
this project targets **CPU** throughout, on an 8-core / 30GB RAM machine.

- The baseline CNN and both CNN transfer-learning models are trainable
  in a few hours combined at this dataset's scale (~15,000 clips).
- AST is used primarily via **linear probing**: the pretrained 86M-
  parameter backbone is frozen and run once to extract 768-dim
  embeddings (the slow part, dominated by the one-time forward pass
  cost), after which the tiny classifier head trains in seconds. Full
  fine-tuning is wired up (`configs/ast_finetune.yaml`) but is genuinely
  slow on CPU -- see notebook 04 for when it's worth doing on a real GPU
  instead (a free Colab/Kaggle T4 is plenty).

If you're running this on different hardware with a supported GPU,
nothing in `watkins.utils.get_device()` needs to change -- it already
tries CUDA first and only falls back to CPU if a real kernel launch
fails. Use `requirements-gpu.txt` and the scaled-up `configs/gpu/`
variants there; see [Running on a rented GPU](#running-on-a-rented-gpu).

## Project layout

```
Watkins/watkins_16k/         the materialized dataset (built by prepare_data.py, not checked in)
    manifest.csv                record_number, class_id, path, orig_sample_rate, orig_duration_s
    audio/<class_id>/<record_number>.wav
src/watkins/                 the library everything else is built on
    prepare_data.py             downloads + materializes the Watkins dataset from Hugging Face
    data.py                     dataset loading, species metadata, tape-grouped train/val/test splits
                                 (plus a deliberately leaky clip_random mode for comparison)
    features.py                 waveform -> log-mel spectrogram, plus sonar-analysis views
                                 (LOFARgram, DEMON spectrum, matched filter)
    augment.py                  synthetic noise injection (white/pink, SNR-controlled), SpecAugment
    pipeline.py                 glue: splits -> transforms -> DataLoaders
    train.py                    config-driven training loop (CLI + notebook-importable)
    evaluate.py                 per-class precision/recall/F1, confusion matrices
    robustness.py                accuracy-vs-SNR sweeps against a trained checkpoint
    models/
        cnn_baseline.py           small from-scratch CNN
        resnet.py                 ResNet-18 (ImageNet transfer learning)
        efficientnet.py           EfficientNet-B0 (ImageNet transfer learning)
        ast_model.py              Audio Spectrogram Transformer (AudioSet transfer learning)
configs/                     one YAML per experiment (model + training hyperparameters)
    gpu/                        the same experiments scaled up for a rented GPU
notebooks/                   the 7-lesson curriculum described above
results/
    checkpoints/                trained model weights (+ config used to produce them)
    logs/                       per-epoch training curves (CSV)
    metrics/                    evaluation reports and robustness sweeps (JSON/CSV)
    figures/                    confusion matrices, robustness plots (PNG)
docs/
    class_reference.md          full species list, imbalance table, and metadata notes
    next_steps.md                extension ideas beyond this curriculum
```

## Setup

A virtual environment is already set up at `.venv/` with everything
needed (CPU-only PyTorch/torchaudio/torchvision + transformers/timm +
the usual data science stack, plus `datasets`/`huggingface_hub`/`pyarrow`
for downloading Watkins). To recreate it elsewhere:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio torchvision
pip install -r requirements.txt
pip install -e . --no-deps   # makes `import watkins` / `python -m watkins.train` work anywhere
```

(`requirements-lock.txt` has the exact pinned versions this project was
built and tested against, if you want bit-for-bit reproducibility.)

Then download and materialize the dataset (~10.6GB download, resampled
down to a few GB on disk; safe to interrupt and re-run):

```bash
python -m watkins.prepare_data
```

Launch the notebooks with:

```bash
source .venv/bin/activate
jupyter lab notebooks/
```

Or train/evaluate a model straight from the CLI, without touching a
notebook at all:

```bash
python -m watkins.train --config configs/baseline_cnn.yaml
python -m watkins.evaluate --checkpoint results/checkpoints/baseline_cnn_best.pt
python -m watkins.robustness --checkpoint results/checkpoints/baseline_cnn_best.pt
```

Every config in `configs/` accepts `--epochs`, `--subset-frac`,
`--batch-size`, and `--run-name` overrides for quick experiments without
editing the YAML (e.g. `--subset-frac 0.2 --epochs 3` for a fast sanity
check before committing to a full run).

## Running on a rented GPU

For a full worked example -- what gets stored where on the pod, where the
trained checkpoint ends up, how to retrieve it, measured throughput, and the
gotchas of RunPod's SSH proxy -- see [`docs/runpod_run.md`](docs/runpod_run.md).

The configs in `configs/` are sized to complete on this project's CPU-only
development machine. `configs/gpu/` holds the same five experiments scaled
up for a real GPU -- larger batches, more workers, the full dataset, and
mixed precision -- with `configs/gpu/ast_finetune.yaml` in particular
being the run the CPU default can't realistically do (86M parameters
unfrozen over all 15k clips; roughly an hour on an RTX 4090).

The dataset never needs to be uploaded: it's ~2GB materialized but fully
re-derivable from the public Hugging Face source, so rebuilding it on the
remote box is faster than copying it. On a fresh pod:

```bash
git clone <your-fork> && cd MarineMammals
pip install -r requirements-gpu.txt        # NOT requirements.txt -- see below
pip install -e . --no-deps
python -m watkins.prepare_data             # ~10-15 min
python -m watkins.train --config configs/gpu/ast_finetune.yaml
```

**Install `requirements-gpu.txt`, not `requirements.txt`.** The latter
documents the CPU-only wheel index (this machine's Quadro P520 is
unsupported by current CUDA wheels, as above); installing it on a GPU box
gives you a CPU-only torch and every run silently falls back to CPU with
no error. On Colab/Kaggle, install neither -- their images ship a
preinstalled, driver-matched torch, and the notebooks' bootstrap cell
already installs only the missing extras.

Mixed precision is controlled by the `amp` config key, default `"auto"`:
bfloat16 on Ampere-and-newer cards, float16 (with gradient scaling) on
older ones like a free-Colab T4, and off on CPU, so CPU runs are
bit-for-bit unchanged. Set `amp: false` to disable it, or `amp: true` to
force it on.

Point `WATKINS_DATA_ROOT` and `WATKINS_RESULTS_ROOT` (see
`watkins.utils`) at, respectively, fast ephemeral disk and whatever
storage outlives the instance -- a persistent volume, a mounted bucket,
or Google Drive on Colab. Training reads thousands of small files per
epoch, so the dataset in particular wants local disk, not a network
filesystem.

## Running on Google Colab

Every notebook auto-detects Colab and adapts -- the exact same `.ipynb`
files work locally (as above) and on Colab with no separate "Colab
version" to keep in sync. One-time setup:

1. Upload `src/`, `configs/`, `pyproject.toml`, and `requirements.txt` to
   `My Drive/MarineMammals/` (skip the multi-GB `Watkins/` and `results/`
   folders -- notebooks handle both automatically, see below).
2. Open a notebook in Colab (**Runtime > Change runtime type > GPU**
   first) and run the first cell.

That cell mounts Drive, installs whatever Colab's base image is missing
(`transformers`, `timm`, `soundfile`, `datasets`, `huggingface_hub`,
`pyarrow` -- deliberately *not* `torch`/`torchaudio`/`torchvision`, since
overwriting Colab's preinstalled, GPU-matched build with this project's
CPU-only wheels would silently disable the GPU), and materializes the
Watkins dataset fresh from Hugging Face if it isn't already present.

Data and results are deliberately split across two roots, both
overridable via environment variables (`watkins.utils.data_root()` /
`results_root()`, honoring `WATKINS_DATA_ROOT` / `WATKINS_RESULTS_ROOT`
if set, falling back to `<repo_root>/Watkins/watkins_16k` and
`<repo_root>/results` otherwise):

- **Data** lives on Colab's local/ephemeral disk (`/content/watkins_data`).
  It's fully re-derivable from the public Hugging Face source, and
  training reads thousands of small files per epoch -- exactly the kind
  of access pattern Drive's network filesystem is slow at. Expect a
  one-time ~10-15 minute materialization per fresh Colab runtime.
- **Results** (checkpoints, logs, metrics, figures) get written to
  `My Drive/MarineMammals/results/`, so a multi-hour training run's output
  survives a runtime disconnect -- infrequent, small writes, exactly what
  Drive is fine at.

A GPU turns the CPU-bound parts of this project (everything, on the
hardware described below, but especially `ast_finetune.yaml` and the
ResNet/EfficientNet configs) into a very different, much faster
experience -- see `watkins.utils.get_device()`, which already tries CUDA
first and only falls back to CPU if a real kernel launch fails, so no
code changes are needed either way.

## Results

All five `configs/gpu/` models, trained on the **full dataset** with the
honest tape-grouped split on a rented RTX 4090. Notebook 05 generates the
live version of this comparison from `results/metrics/`.

| Model | Params | Test accuracy | Test macro-F1 |
|---|---|---|---|
| `baseline_cnn` (from scratch) | 402,678 | 0.447 | 0.293 |
| `resnet18` (ImageNet) | 11,204,214 | 0.485 | 0.296 |
| `efficientnet_b0` (ImageNet) | 4,076,722 | 0.504 | **0.357** |
| `ast_linear_probe` (AudioSet, frozen) | 86,230,326 (43,062 trained) | 0.507 | 0.308 |
| `ast_finetune` (AudioSet, all unfrozen) | 86,230,326 | **0.576** | **0.358** |

Macro-F1 is averaged over the **40 species present in the test set** (of 54
total -- 14 have a single source recording and are train-only by
construction). These are the numbers in `results/metrics/*_eval.json` and in
notebook 05. Note that `train.py`'s per-epoch printout reports a *lower*
macro-F1 for the same predictions; see [the caveat
below](#a-caveat-on-trainpys-macro-f1).

Read macro-F1, not accuracy. With 54 species spanning three orders of
magnitude in representation, accuracy is largely a measure of how well a
model does on killer whale and sperm whale; macro-F1 weights every species
equally and is the harder, more honest number.

The headline result is that **the two metrics disagree about the winner**.
The fine-tuned AST wins accuracy outright (0.576, a clear +0.072 over the
next model) -- but on macro-F1 it is in a statistical tie with
EfficientNet-B0 (0.358 vs 0.357), a model with **1/21 the parameters**. So
AST's extra capacity is buying performance on the well-represented species,
not on the long tail, which is the part of the problem that is actually hard.
On a per-parameter basis EfficientNet-B0 is the standout model here, and it
also beats ResNet-18 on macro-F1 by a wide margin (0.357 vs 0.296) despite
being a third the size.

For the AST pair that notebook 04 is built around: the **linear probe**
trains only a 43k-parameter head on a frozen backbone and already reaches
0.507 accuracy -- AudioSet pretraining transferring to underwater
bioacoustics with no adaptation at all, for the cost of a single forward pass
over the dataset. **Full fine-tuning** then adds +0.069 accuracy and +0.050
macro-F1, paid for with backprop through all 86M parameters.

### A caveat on `train.py`'s macro-F1

The macro-F1 that `train.py` prints per epoch is computed with
`f1_score(..., average="macro")` and no explicit `labels=`, so scikit-learn
averages over the *union of true and predicted* classes. Any species the
model predicts but which never appears in the test set enters the average as
a zero.

That denominator therefore **varies by model** -- 44 classes for the AST
fine-tune, 48 for the baseline CNN -- which makes those printed numbers not
directly comparable across models, and systematically harsher on models that
make more varied predictions. It is why `train.py` reports 0.326 for the AST
fine-tune where `evaluate.py` reports 0.358 on identical predictions.

Use the `evaluate.py` / `results/metrics/` numbers (fixed 40-class
denominator) for any comparison. The training-time value is still perfectly
serviceable for its actual job -- early stopping and best-checkpoint
selection within a single run -- since the denominator is stable there.

None of these are tuned results. The fine-tune early-stops after ~9 epochs
with train macro-F1 above 0.95 while validation plateaus near 0.35: 86M
parameters against ~9,400 training clips overfits quickly, and
`docs/next_steps.md` lists the obvious levers. Treat the table as a
reproducible baseline, not a ceiling.

See [`docs/runpod_run.md`](docs/runpod_run.md) for how these were produced
(~2 hours on one 4090, well under $1).

## Citations

If you use this project's data pipeline, please cite the underlying
sources it's built on:

```bibtex
@inproceedings{sayigh2016watkins,
  author = {Sayigh, Laela S. and Daher, Mary Ann and Allen, Julie and Gordon, Helen and
            Joyce, Kathleen and Stuhlmann, Christian and Tyack, Peter L.},
  title = {The Watkins Marine Mammal Sound Database: An Online, Freely Accessible Resource},
  booktitle = {Proceedings of Meetings on Acoustics},
  volume = {27}, number = {1}, pages = {040013}, year = {2016},
  doi = {10.1121/2.0000358}
}
@techreport{watkins1992sound,
  author = {Watkins, William A. and Fristrup, Kurt M. and Daher, Mary Ann},
  title = {SOUND Database of Marine Animal Vocalizations: Structure and Operations},
  institution = {Woods Hole Oceanographic Institution}, number = {WHOI-92-31}, year = {1992}
}
```

Plus the Hugging Face re-release this project downloads from:
`ivangtorre/watkins-marine-mammal-full-cuts` (Iván G. Torre, 2026), which
also credits the SoundWave project's metadata conversion.
