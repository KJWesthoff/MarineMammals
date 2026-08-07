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
| 04 | [`audio_spectrogram_transformer`](notebooks/04_audio_spectrogram_transformer.ipynb) | AST (ViT-style self-attention over spectrogram patches), pretrained on AudioSet; linear probing vs. full fine-tuning -- why the CPU default is the probe, and what the full fine-tune actually bought once it was run on a rented GPU. |
| 05 | [`model_comparison`](notebooks/05_model_comparison.ipynb) | All four architectures compared head-to-head: accuracy, macro-F1, per-class performance on the best-represented species, and parameter efficiency. |
| 06 | [`robustness_to_noise`](notebooks/06_robustness_to_noise.ipynb) | Accuracy vs. SNR curves under synthetic white/pink noise injection -- a proxy for range/sea-state degradation a real passive acoustic monitoring system has to handle. |

Notebooks 05 and 06 read the committed GPU metrics, so they render the
real five-model comparison and SNR curves on a fresh clone before you've
trained anything -- see [Working in the
notebooks](#working-in-the-notebooks) for what each lesson runs under the
hood and how that relates to the published results.

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

## Hardware notes: CPU for development, a rented GPU for the results

Two hardware stories run side by side in this repo, and it's worth being
explicit about which is which:

- **The [results](#results) below were produced entirely on a GPU.** All
  five `configs/gpu/` runs were trained on a single rented RTX 4090
  (~2 hours total, well under $1) -- including the full AST fine-tune,
  which is the best model in the table. Those are the numbers this
  project reports. See [`docs/runpod_run.md`](docs/runpod_run.md) for
  exactly how they were produced.
- **Development happens on CPU**, which is the only reason the default
  configs in `configs/` are sized the way they are: small batches, capped
  epochs, and an AST default that avoids backprop through the backbone.
  Read those configs as "runnable while you're learning", not as the
  training budget the results came from.

The development machine has an NVIDIA Quadro P520 (2GB VRAM), but it's a
Pascal-generation GPU (compute capability 6.1) and current PyTorch CUDA
wheels are built for compute capability >= 7.5 -- `torch.cuda.is_available()`
returns `True`, but any actual kernel launch fails. Combined with the
2GB VRAM ceiling (too little for AST fine-tuning regardless), the
local-default path targets **CPU** throughout, on an 8-core / 30GB RAM
machine.

- The baseline CNN and both CNN transfer-learning models are trainable on
  CPU in a few hours combined at this dataset's scale (~15,000 clips).
- AST's CPU default is **linear probing**: the pretrained 86M-parameter
  backbone is frozen and run once to extract 768-dim embeddings (the slow
  part, dominated by the one-time forward pass cost), after which the tiny
  classifier head trains in seconds. That is a real result in its own
  right -- 0.507 accuracy in the table below -- not just a stand-in.
- Full fine-tuning (`configs/ast_finetune.yaml`) is wired up but is
  genuinely slow on CPU, so the CPU config is scoped down to a 25% subset
  and 3 epochs as a demo. The published fine-tune is
  `configs/gpu/ast_finetune.yaml` on the 4090: ~1 hour, and worth
  +0.069 accuracy over the probe.

If you're running this on different hardware with a supported GPU,
nothing in `watkins.utils.get_device()` needs to change -- it already
tries CUDA first and only falls back to CPU if a real kernel launch
fails. Use `requirements-gpu.txt` and the scaled-up `configs/gpu/`
variants there; see [Running on a rented GPU](#running-on-a-rented-gpu).

All three kinds of run coexist in `results/` under distinct names: the
scaled-up GPU runs are `*_gpu`, the notebooks' quick subset demos are
`*_demo`, and a full CPU-config run keeps the bare config name. Nothing
overwrites anything else, and notebook 05's comparison table lists the
full-scale runs (`*_gpu` alongside any bare-named CPU runs you've done)
while deliberately leaving the `*_demo` runs out, so a 5-epoch/30%-subset
demo can never be mistaken for a real result.

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
    local/                      optional private playground copies (gitignored)
results/
    checkpoints/                trained model weights (+ config used to produce them), not checked in
    logs/                       per-epoch training curves (CSV)
    metrics/                    evaluation reports and robustness sweeps (JSON/CSV)
    figures/                    confusion matrices, robustness plots (PNG)
results-local/               where the playground copies write instead (gitignored)
scripts/
    runpod_setup.sh             staged setup for a rented GPU pod (check/install/data/smoke/train)
    sync_local_notebooks.py     manages notebooks/local/ + results-local/
docs/
    class_reference.md          full species list, imbalance table, and metadata notes
    next_steps.md                extension ideas beyond this curriculum
```

## How the code works

Everything in `notebooks/` is a thin layer over the `watkins` package in
`src/`: the notebooks import the same functions the CLI entry points call,
so a lesson and a `python -m watkins.train` run execute identical code.
There is no separate "notebook version" of the pipeline -- the optional
playground copies in `notebooks/local/` are generated copies of these
same files, not a fork of them.

### Frameworks used

| Framework | What it does here |
|---|---|
| **PyTorch** (`torch`) | Everything model-side: tensors, `nn.Module` definitions, `DataLoader`, the training loop, AdamW/SGD, `torch.autocast` + `GradScaler` for mixed precision, checkpoint serialization. |
| **torchaudio** | All signal processing: `MelSpectrogram` / `Spectrogram` / `AmplitudeToDB` for the feature transforms, `highpass_biquad` / `lowpass_biquad` for DEMON's envelope detector, `functional.resample` in `prepare_data`. |
| **torchvision** | The two ImageNet CNN backbones (`resnet18`, `efficientnet_b0`) and their pretrained weight enums. |
| **Hugging Face `transformers`** | AST: `ASTFeatureExtractor` (the pretrained fbank recipe) and `ASTForAudioClassification` (the 86M-parameter AudioSet-pretrained backbone). |
| **`huggingface_hub` + `pyarrow`** | Dataset acquisition -- fetching the 9 parquet shards and iterating them in batches without loading 10.6GB into RAM. |
| **`soundfile`** | WAV encode/decode (writing the materialized dataset, reading clips at training time). |
| **scikit-learn** | Metrics only: `f1_score`, `classification_report`, `confusion_matrix`. No sklearn models. |
| **NumPy / pandas** | Class-weight computation, robustness-sweep tables and CSV output. |
| **matplotlib** | Confusion-matrix heatmaps and accuracy-vs-SNR plots. |
| **PyYAML / tqdm / argparse** | Config loading, progress bars, CLI parsing. |

Deliberately *not* used: no Keras/TensorFlow, no PyTorch Lightning or
other training-loop framework (the loop in `train.py` is ~80 lines of
plain PyTorch and is meant to be read), and no `librosa` -- `features.py`
does its STFT work in torchaudio instead, which keeps feature extraction
on the same tensors as the model and avoids a numba dependency.

### The pipeline, end to end

```
Hugging Face parquet shards
  |  prepare_data.py  -- decode, downmix to mono, resample to 16kHz, cap at 60s
  v
Watkins/watkins_16k/{manifest.csv, audio/<class_id>/<record>.wav}
  |  data.py:load_manifest    -> list[Clip]  (path, label, record_number, tape_id)
  |  data.py:build_split      -> Split(train, val, test), grouped by tape_id
  v
WatkinsDataset  -- reads a wav, crops/pads to a fixed 3.0s (48,000 samples):
                   random offset when training, center when evaluating
  |  transform (built in pipeline.py)
  v
either  augment.RandomNoiseInjection -> features.LogMelSpectrogram -> augment.SpecAugment
    or  augment.RandomNoiseInjection -> ast_model.ASTPreprocessor
  |  DataLoader (collate_with_clips keeps per-sample Clip metadata alongside the batch)
  v
model (models/build_model)  ->  logits [B, 54]
```

**1. Materialization** (`prepare_data.py`). The source dataset ships audio
as bytes embedded in parquet at 47 different native sample rates. This
script decodes each clip, downmixes to mono, resamples to a single 16kHz,
caps stored length at 60s, and writes one WAV per clip plus a
`manifest.csv`. It's idempotent -- already-written clips are skipped, so
an interrupted download is resumed by re-running it.

**2. Splitting** (`data.py`). `build_split` groups each species' clips by
`tape_id` (the first 6 characters of the record number, identifying the
original recording) and assigns whole tapes to train/val/test, so no
recording ever straddles a split boundary. `_assign_tape_groups` handles
the degenerate cases explicitly -- one tape means train-only, two means
train+test, three or more reserves the smallest tape for each split and
then greedily fills whichever split is furthest below its target share. A
`clip_random` mode implements the naive (leaky) alternative for the
leakage-quantification exercise.

**3. Features** (`features.py`, `augment.py`, `models/ast_model.py`). Two
input representations exist, and each model declares which one it needs:
`LogMelSpectrogram` (64 mel bins, 25ms window / 10ms hop, per-instance
normalized) for the CNNs, and `ASTPreprocessor` for AST -- which delegates
to the pretrained `ASTFeatureExtractor` rather than reusing the project's
own mel transform, since AST's normalization statistics and fbank recipe
are part of its pretrained weights. Augmentation is waveform-domain
(`RandomNoiseInjection`, mixing white/pink noise at a sampled SNR) plus
spectrogram-domain (`SpecAugment` time/frequency masking), chained with a
small `Compose`.

**4. The `input_kind` contract** (`models/__init__.py`). `build_model(name,
num_classes)` returns `(model, input_kind)` where `input_kind` is
`"logmel"` or `"ast"`. That string is what binds a model to its feature
pipeline: `pipeline.build_transforms` switches on it, and it's stored in
every checkpoint so `evaluate.py` and `robustness.py` rebuild the exact
transform the model was trained with. `pipeline.py` exists specifically so
that training, evaluation and robustness sweeps construct their transforms
through one code path -- a train/eval feature mismatch is otherwise a very
easy bug to introduce and a very hard one to notice.

The four models are: a from-scratch `BaselineCNN` (four conv/BN/ReLU/pool
blocks, global average pool, linear head); ResNet-18 and EfficientNet-B0,
both wrapped in `SpectrogramBackbone`, which repeats the single
spectrogram channel three times so ImageNet's 3-channel first conv can be
reused intact rather than surgically resized; and `ASTClassifier`, which
wraps `ASTForAudioClassification` and optionally freezes everything except
the head.

**5. Training** (`train.py`). Config-driven: a YAML file is merged over
`DEFAULT_CONFIG`, then CLI flags override that. The loop is standard --
class-weighted `CrossEntropyLoss` (inverse-frequency weights, which matters
a lot here), AdamW, per-epoch validation, `EarlyStopping` on validation
macro-F1, best-state snapshot restored before the test pass, per-epoch CSV
log. Mixed precision is decided by `utils.autocast_dtype`: bfloat16 on
Ampere+, float16 plus a `GradScaler` on older cards, off on CPU.

There are two training paths. `_train_generic` is the normal one. The AST
linear probe takes `_train_ast_linear_probe` instead: because the backbone
is frozen, its output never changes across epochs, so the backbone is run
over the data *once* (`extract_embeddings`) and the 768-dim vectors are
cached in memory; "training" is then a few hundred full-batch gradient
steps on a 43k-parameter head, which takes seconds. The trained head's
`state_dict` is spliced back into `model.model.classifier` before saving --
`LinearProbeHead` mirrors the HF head module-for-module precisely so this
works -- so the checkpoint is structurally identical to a normally trained
one and downstream code needs no special case.

**6. Checkpoints and downstream analysis.** `train.py` saves a dict of
`{model_name, model_kwargs, input_kind, state_dict, config}` -- enough for
`evaluate.load_checkpoint` to reconstruct the model and its data pipeline
from the file alone. `evaluate.py` produces the per-class report, confusion
matrix and `results/metrics/<run>_eval.json`; `robustness.py` re-runs the
same checkpoint over the test set at a sweep of injected-noise SNRs
(noise added to the *waveform*, before feature extraction, matching how
real degradation enters the signal chain) and writes the accuracy-vs-SNR
curve; `summarize.py` collates everything in `results/metrics/` into a
markdown table.

**7. Path and device handling** (`utils.py`). `data_root()` and
`results_root()` resolve to repo-relative defaults but honor
`WATKINS_DATA_ROOT` / `WATKINS_RESULTS_ROOT`, which is what lets the same
code put data on fast ephemeral disk and results on durable storage
(Colab/RunPod). `get_device()` doesn't trust `torch.cuda.is_available()` --
it actually launches a trivial kernel and falls back to CPU if that
raises, because on this project's development GPU the availability check
returns `True` and every real kernel then fails.

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

## Working in the notebooks

### What a notebook run actually does

Every training cell calls `watkins.train.train_run` -- the same function
`python -m watkins.train --config ...` calls, on the same configs in
`configs/`. Nothing about the pipeline changes because you're in a
notebook. What changes is the **budget**: the in-notebook runs override
`epochs` and `subset_frac` so a cell returns in minutes on CPU rather
than hours.

| Notebook | Runs it trains | Scale vs. the config |
|---|---|---|
| 02 | `baseline_cnn_demo`, `baseline_cnn_leak_check` | 5 epochs, 30% subset |
| 03 | `resnet18_demo`, `efficientnet_b0_demo`, `resnet18_frozen_demo` | 5 epochs, 30% subset |
| 04 | `ast_linear_probe_demo` | 30% subset |
| 05 | nothing -- reads `results/metrics/` only | -- |
| 06 | robustness sweeps, only for runs with no cached sweep | -- |

Read every `*_demo` number as a smoke test that the code path works, not
as a result. A 5-epoch run on 30% of the data is not a smaller version of
the published number; it is a different experiment.

### How the notebooks relate to the GPU results

The [results](#results) were produced on a rented RTX 4090 (see
[`docs/runpod_run.md`](docs/runpod_run.md)), and **the evidence for them
is committed to this repo** -- about 1.7MB of it:

```
results/metrics/*_gpu_eval.json         per-class precision/recall/F1
results/metrics/*_gpu_robustness.csv    accuracy-vs-SNR sweeps
results/logs/*_gpu.csv                  per-epoch training curves
results/figures/*_gpu_*.png             confusion matrices, robustness plots
```

That is what makes notebooks 05 and 06 useful on a fresh clone: they
render the real five-model comparison and the real SNR curves **before
you have trained anything at all**. You are not looking at placeholder
data waiting to be filled in.

What is *not* committed is `results/checkpoints/*.pt` -- ~820MB, gitignored,
of which the two AST checkpoints are 330MB each. This costs you less than
it sounds like, because notebook 06 only needs a checkpoint when a sweep
isn't already cached, and all five GPU sweeps are cached. It reads the
CSV and moves on.

Three things keep a CPU notebook session from damaging those numbers:

1. **Names never collide.** Demos write `*_demo`, the GPU runs are
   `*_gpu`, a full CPU-config run keeps the bare config name. Notebook
   05's comparison table lists the full-scale runs and deliberately
   excludes `*_demo`, so a 5-epoch demo can't be mistaken for a result.
2. **Notebook 06 is cache-first.** If `{run}_robustness.csv` exists it
   loads it and continues -- it never recomputes over a sweep that's
   already on disk.
3. **Sweeps are CPU-guarded.** A sweep is 13 full test-set passes over
   3,072 clips. `ALLOW_CPU_SWEEP = False` at the top of notebook 06 stops
   one starting implicitly just because a checkpoint happens to be
   sitting there.

So you can run all seven notebooks end to end on CPU without overwriting,
invalidating, or quietly degrading the published GPU results.

### Keeping your own runs out of git

The one thing a notebook session *does* disturb is `git status`. Six demo
artifacts are checked in -- `results/metrics/{baseline_cnn,resnet18,efficientnet_b0}_demo_eval.json`
and the matching `results/figures/*_demo_confusion.png` -- so re-running
lesson 02 or 03 shows up as modified tracked files. (Demo training logs
are untracked and just appear as new `??` entries.)

If you'd rather experiment without that, `scripts/sync_local_notebooks.py`
sets up a private playground:

```bash
python scripts/sync_local_notebooks.py     # populate notebooks/local/
jupyter lab notebooks/local/
```

Both `notebooks/local/` and `results-local/` are gitignored, so runs
launched from a playground copy never touch `git status`. Under the hood:

- The copies are the tracked notebooks with two depth-dependent lines
  rewritten (they sit one directory deeper) and a marked block injected
  that points `WATKINS_RESULTS_ROOT` at `results-local/`. Everything else
  resolves through `repo_root()` in `watkins.utils`, which keys off the
  package location rather than the notebook's.
- `results-local/` is **seeded with a copy of the tracked results**, which
  is the whole trick: notebooks 05 and 06 still read the real GPU metrics
  and sweeps, while anything you train lands in the copy instead of the
  checked-in tree.
- `results-local/checkpoints` is a symlink to `results/checkpoints`
  rather than a copy -- those files are already gitignored, so they were
  never a source of git noise, and symlinking avoids duplicating 820MB
  while letting notebook 06 load the GPU checkpoints.

Useful flags: `--status` (what has diverged), `--promote 03` (copy a
playground edit back onto the tracked notebook, reversing the rewrites
and stripping the injected block), `--retrofit` (re-apply the transforms
to existing copies in place, keeping your edits), `--seed-results`
(refresh `results-local/` from `results/`). A copy you've edited is never
overwritten without `--force`.

This is entirely optional. Working directly in `notebooks/` is fine --
you'll just want `git checkout -- results/` occasionally.

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
development hardware described [above](#hardware-notes-cpu-for-development-a-rented-gpu-for-the-results),
but especially `ast_finetune.yaml` and the ResNet/EfficientNet configs)
into a very different, much faster experience -- see
`watkins.utils.get_device()`, which already tries CUDA first and only
falls back to CPU if a real kernel launch fails, so no code changes are
needed either way. Colab is a reasonable place to reproduce the
`configs/gpu/` runs, with one caveat: a free-tier T4 has 16GB, so
`configs/gpu/ast_finetune.yaml` needs `batch_size` dropped to 4 (the
published run used a 24GB RTX 4090 at batch 16).

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
notebook 05, and `train.py` now prints the same value for the same
predictions -- see [the note on the
denominator](#why-the-macro-f1-denominator-is-pinned).

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

### Why the macro-F1 denominator is pinned

Worth knowing if you compare against other bioacoustics results, because it
is an easy mistake to make and it silently changes the ranking.

scikit-learn's `f1_score(..., average="macro")` with no explicit `labels=`
averages over the *union of true and predicted* classes. On a long-tailed
problem like this one, any species the model predicts that doesn't occur in
the split enters the average as a zero -- so the denominator depends on the
model's behaviour rather than on the data. Across these five models it varied
from 44 to 48 classes on one fixed test set, depressing every score and
penalizing most the models that predicted a wider spread of labels.

That is not a stable basis for comparison. `watkins.train._macro_f1` and
`watkins.evaluate` both pin the denominator to the classes actually present
in the split, so training-time and evaluation-time macro-F1 now agree exactly
for identical predictions.

The effect is not merely cosmetic: under the unpinned denominator the AST
fine-tune appeared to beat EfficientNet-B0 by 0.022 macro-F1, where in fact
the two are tied. The per-epoch logs in `results/logs/` predate this fix, so
their `train_f1`/`val_f1` columns use the old denominator; the test metrics
in `results/metrics/` were always computed correctly and are unaffected.

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
