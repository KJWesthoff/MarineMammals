# Running the AST fine-tune on a rented RunPod GPU

A record of the first real-GPU run of this project: what was set up, where
every artifact lives while the run is in flight, where the outputs land when
it finishes, and how to get them home. Written against a run on an RTX 4090
(24GB, Ada, compute capability 8.9).

The short version: **code comes from GitHub, the dataset is rebuilt on the
pod's local disk, and results are written to the pod's persistent volume and
pulled back with `runpodctl` (not `scp` -- see [Getting results
off](#getting-results-off-the-pod)).**

## What was set up

The project was CPU-only by design (the development machine's Quadro P520 is
Pascal, unsupported by current CUDA wheels). Making it run on rented hardware
took four changes, all already on `main`:

1. **`requirements-gpu.txt`** -- `requirements.txt` documents the CPU-only
   wheel index, and installing it on a GPU box silently yields a CPU-only
   torch with no error. The GPU file leaves the image's CUDA-matched build
   alone.
2. **Mixed precision** -- `utils.autocast_dtype()` plus an `amp` config key
   (default `"auto"`): bfloat16 on Ampere and newer, float16 with a
   `GradScaler` on older cards, off on CPU so CPU runs are unchanged.
3. **`configs/gpu/`** -- the five experiments rescaled for a real GPU (full
   dataset, larger batches, 8 workers). The CPU-sized `configs/` are
   untouched so the curriculum still runs on a laptop.
4. **`scripts/runpod_setup.sh`** -- the step-by-step bootstrap described below.

## Where everything lives

### Locally

| What | Path |
|---|---|
| Working copy | `/home/kj/Documents/CodeProjects/MarineMammals` |
| Virtualenv (CPU torch) | `.venv/` |
| Local dataset copy | `Watkins/watkins_16k/` (~1.9GB, gitignored) |
| Local results | `results/` |

The local `.venv` is CPU-only and stays that way -- it is not used for GPU
work and nothing about the pod changes it.

### GitHub

`https://github.com/KJWesthoff/MarineMammals` (public, so the pod clones it
with no credentials -- no SSH key ever goes on a rented box).

The repo deliberately excludes `Watkins/` (~1.9GB) and `node_modules/`. The
dataset is fully re-derivable from the public Hugging Face source, so
rebuilding it on the pod is faster than copying it.

### On the pod

Two filesystems with very different characteristics, and the split between
them is deliberate:

| What | Path | Filesystem | Why there |
|---|---|---|---|
| Code | `/root/MarineMammals` | container disk (30GB, local) | Cloned fresh; disposable |
| Dataset | `/root/watkins_data` (1.9GB) | container disk (local) | **Training reads thousands of small wavs per epoch.** `/workspace` is a network filesystem (MooseFS) and is poor at exactly that access pattern |
| HF parquet cache | `/workspace/.cache/huggingface` (9.9GB) | network volume | Read once during `prepare_data`, so network latency doesn't matter; keeps 9.9GB off the 30GB container disk |
| Results | `/workspace/results` | network volume | Small, infrequent writes that must **outlive the pod** |

> The HF cache location was not chosen by `runpod_setup.sh` -- the pod image
> presets `HF_HOME`, and the script's `${HF_HOME:-...}` default correctly
> defers to it. The outcome happens to be the right one. If you ever need it
> on local disk, set `HF_HOME` explicitly before running.

**Anything on the container disk dies with the pod. Only `/workspace`
survives**, and only if it is a network volume rather than ephemeral pod
storage -- check which you have before relying on it.

## Where training output ends up

`configs/gpu/ast_finetune.yaml` sets `run_name: ast_finetune_gpu`, and
`WATKINS_RESULTS_ROOT=/workspace/results`, so:

| Artifact | Path | Size |
|---|---|---|
| Live stdout (tqdm + per-epoch lines) | `/workspace/results/ast_finetune.log` | small |
| Per-epoch training curve (CSV) | `/workspace/results/logs/ast_finetune_gpu.csv` | ~2KB |
| Best checkpoint by val macro-F1 | `/workspace/results/checkpoints/ast_finetune_gpu_best.pt` | ~345MB |

The checkpoint is written **once, at the very end** of the run -- it holds
the best-validation-F1 weights, restored before the final test pass, not the
last epoch's. So an interrupted run leaves the CSV and log but no checkpoint.

The checkpoint is a dict with `model_name`, `model_kwargs`, `input_kind`,
`state_dict`, and the full `config` used to produce it, so
`watkins.evaluate` and `watkins.robustness` can reconstruct the model without
being told anything else.

Smoke-test artifacts from validation runs (`ast_smoke_*`) also sit in those
directories and can be deleted.

## Getting results off the pod

**`scp` and `sftp` do not work** over `ssh.runpod.io` -- the proxy has no
SFTP subsystem and fails with `subsystem request failed on channel 0`. It
also refuses non-PTY sessions and ignores remote commands, so it is only
usable as an interactive shell.

Use `runpodctl`, which is preinstalled on the pod:

```bash
# on the pod
runpodctl send /workspace/results/checkpoints/ast_finetune_gpu_best.pt
# prints a one-time code, e.g. 1234-word-word-word

# locally (install first: https://github.com/runpod/runpodctl releases)
runpodctl receive 1234-word-word-word
```

Small text artifacts (the CSV, the log tail) are easier to just read over the
shell and paste, since they are a couple of KB.

Alternatives if you prefer: push the checkpoint to a Hugging Face repo from
the pod, or keep the network volume and attach it to a future pod.

## Reproducing this on a fresh pod

Pod settings: **RTX 4090 (24GB)**, official RunPod PyTorch 2.x template,
**60GB container disk** (30GB works but leaves little slack), SSH exposed.

```bash
cd /root && git clone https://github.com/KJWesthoff/MarineMammals.git
bash MarineMammals/scripts/runpod_setup.sh check     # GPU + kernel launch + bf16
bash MarineMammals/scripts/runpod_setup.sh install   # deps; asserts device==cuda
bash MarineMammals/scripts/runpod_setup.sh data      # ~3-15 min
bash MarineMammals/scripts/runpod_setup.sh smoke     # ~2 min -- do not skip
bash MarineMammals/scripts/runpod_setup.sh train     # nohup'd; survives disconnect
tail -f /workspace/results/ast_finetune.log
```

Run `smoke` every time. It is a 2-minute, full-fidelity exercise of mixed
precision and the batch-16 VRAM ceiling, and an OOM caught there costs cents
instead of most of an hour of paid GPU time.

## Measured performance

Throughput is **flat in batch size** -- the GPU is already compute-saturated
at batch 16, so the spare VRAM is not worth spending:

| batch | peak VRAM | clips/s |
|---|---|---|
| 16 | 8.3 GiB | 112.2 |
| 32 | 15.1 GiB | 115.9 |
| 48 | 22.0 GiB | 117.6 |

5% throughput for 3x the memory, and batch 48 leaves no margin on a 23.5GiB
card. The config stays at **batch 16**. Real training confirms the probe:
585 steps/epoch at ~6.95 it/s = ~111 clips/s, ~93s/epoch, so 20 epochs is
roughly 30-35 minutes.

That the measured rate matches the synthetic probe means the 8 dataloader
workers keep up and the run is GPU-bound, not bottlenecked on CPU-side
spectrogram extraction.

## Gotchas worth remembering

- **The RunPod SSH proxy silently corrupts long command lines.** It wraps at
  80 columns and duplicates characters at the wrap point -- observed as
  `/rroot/.bashrc` and `KJWesthoff//MarineMammals` in echoed commands. Send
  `stty cols 4000` first, keep commands to a single line, and never send a
  heredoc (it strands the shell in continuation mode until timeout). This is
  why the setup lives in a committed script rather than being typed over the
  wire.
- **Do not install `requirements.txt` on a GPU box.** It is CPU-wheel pinned.
  `runpod_setup.sh install` asserts `get_device().type == "cuda"` afterwards
  precisely to catch this.
- **Colab/Kaggle need neither requirements file** -- their images ship a
  driver-matched torch, and the notebooks' bootstrap installs only the
  missing extras.
- **The pod bills for wall-clock uptime, not GPU work.** Training finishing
  does not stop the meter. Terminate it explicitly.

## Cost

~$0.34/hr on Community Cloud. Setup and data prep ~20 min, training ~35 min,
so a complete run is roughly **$0.30-0.50**. The network volume is billed
separately per GB-month; the 9.9GB HF cache can be deleted once the dataset
is materialized, or once you are done with the volume entirely.
