#!/usr/bin/env bash
# Set up a rented RunPod pod (or any bare CUDA box) for this project and run
# the AST fine-tune. See README, "Running on a rented GPU".
#
#   git clone https://github.com/KJWesthoff/MarineMammals.git
#   bash MarineMammals/scripts/runpod_setup.sh check
#   bash MarineMammals/scripts/runpod_setup.sh install
#   bash MarineMammals/scripts/runpod_setup.sh data      # ~10-15 min
#   bash MarineMammals/scripts/runpod_setup.sh smoke     # ~2 min, do NOT skip
#   bash MarineMammals/scripts/runpod_setup.sh train     # the real run
#
# Run the steps in order and read the output. `smoke` in particular earns its
# two minutes: it is the first thing that exercises mixed precision and the
# batch-16 VRAM ceiling, and an OOM discovered there costs cents rather than
# most of an hour of paid GPU time.
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Dataset and HF cache go on the *local container disk*: training reads
# thousands of small wav files per epoch, and RunPod's /workspace is a network
# filesystem (MooseFS) that is poor at exactly that access pattern. Results go
# on /workspace precisely because they are small, infrequent writes that must
# outlive the pod.
export HF_HOME=${HF_HOME:-/root/hf}
export WATKINS_DATA_ROOT=${WATKINS_DATA_ROOT:-/root/watkins_data}
export WATKINS_RESULTS_ROOT=${WATKINS_RESULTS_ROOT:-/workspace/results}

CONFIG=${CONFIG:-configs/gpu/ast_finetune.yaml}

check() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    # A kernel launch, not just is_available(): a driver/wheel mismatch reports
    # True and then fails on the first real op.
    python - <<'PY'
import torch
x = torch.zeros(1, device="cuda"); _ = x + 1
p = torch.cuda.get_device_properties(0)
print("torch        :", torch.__version__, "| cuda", torch.version.cuda)
print("gpu          :", p.name, f"{p.total_memory/2**30:.1f} GiB")
print("capability   :", torch.cuda.get_device_capability(0))
print("bf16         :", torch.cuda.is_bf16_supported())
PY
    df -h / "$WATKINS_RESULTS_ROOT" 2>/dev/null || true
}

install() {
    mkdir -p "$HF_HOME" "$WATKINS_DATA_ROOT" "$WATKINS_RESULTS_ROOT"
    cd "$REPO_DIR"
    # requirements-gpu.txt pins torch>=2.4, already satisfied by the pod image's
    # CUDA-matched build, so pip leaves it alone. Installing requirements.txt
    # here would pull CPU-only wheels and every run would silently use the CPU.
    python -m pip install -q -r requirements-gpu.txt
    python -m pip install -q -e . --no-deps
    python - <<'PY'
import torch
from watkins.utils import get_device, autocast_dtype, data_root, results_root
d = get_device()
assert d.type == "cuda", "fell back to CPU -- wrong requirements file installed?"
print("device       :", d)
print("amp dtype    :", autocast_dtype(d, "auto"))
print("data_root    :", data_root())
print("results_root :", results_root())
PY
}

data() {
    cd "$REPO_DIR"
    python -m watkins.prepare_data
    du -sh "$WATKINS_DATA_ROOT" "$HF_HOME"
    df -h /
}

smoke() {
    cd "$REPO_DIR"
    python -m watkins.train --config "$CONFIG" \
        --run-name ast_smoke --epochs 1 --subset-frac 0.02
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
}

train() {
    cd "$REPO_DIR"
    mkdir -p "$WATKINS_RESULTS_ROOT"
    local log="$WATKINS_RESULTS_ROOT/ast_finetune.log"
    # nohup so the run survives the SSH session dropping -- RunPod's ssh.runpod.io
    # proxy is an interactive shell and will not keep a foreground job alive.
    nohup python -m watkins.train --config "$CONFIG" > "$log" 2>&1 &
    echo "started pid $! -> $log"
}

GPU_RUNS="baseline_cnn resnet18 efficientnet_b0 ast_linear_probe ast_finetune"

# --- Train + evaluate + robustness-sweep all five configs, end to end -------
# Fills in everything notebooks 05 and 06 read. Deliberately does NOT abort the
# whole batch when one model fails: a late OOM or a bad config should cost you
# that model, not the four that already succeeded.
run_all() {
    cd "$REPO_DIR"
    local ck="$WATKINS_RESULTS_ROOT/checkpoints"
    local failed=""

    for name in $GPU_RUNS; do
        echo "=== TRAIN $name ==="
        python -m watkins.train --config "configs/gpu/$name.yaml" \
            || { echo "!!! TRAIN FAILED: $name"; failed="$failed train:$name"; }
    done

    for name in $GPU_RUNS; do
        local run="${name}_gpu"
        [ -f "$ck/${run}_best.pt" ] || { echo "--- skip $run (no checkpoint)"; continue; }
        echo "=== EVAL $run ==="
        python -m watkins.evaluate --checkpoint "$ck/${run}_best.pt" \
            || { echo "!!! EVAL FAILED: $run"; failed="$failed eval:$run"; }
        echo "=== ROBUSTNESS $run ==="
        python -m watkins.robustness --checkpoint "$ck/${run}_best.pt" \
            || { echo "!!! ROBUSTNESS FAILED: $run"; failed="$failed robust:$run"; }
    done

    echo "=== SUMMARY ==="
    ls -la "$WATKINS_RESULTS_ROOT/metrics" "$WATKINS_RESULTS_ROOT/figures" 2>/dev/null || true
    [ -n "$failed" ] && echo "FAILURES:$failed" || echo "all five configs completed"
    echo "=== RUN_ALL_DONE ==="
}

# Same, detached, so it survives the SSH session dropping.
run_all_bg() {
    mkdir -p "$WATKINS_RESULTS_ROOT"
    local log="$WATKINS_RESULTS_ROOT/run_all.log"
    setsid nohup bash "${BASH_SOURCE[0]}" run_all > "$log" 2>&1 &
    echo "started pid $! -> $log"
}

"${1:-check}"
