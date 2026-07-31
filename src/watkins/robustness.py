"""Robustness-to-noise evaluation: take a trained checkpoint and measure
how accuracy degrades as synthetic noise is added to the test waveforms
at progressively worse signal-to-noise ratios.

This is the closest thing in this project to a realistic passive
bioacoustic monitoring stress test: real hydrophone recordings degrade
with range (spreading + absorption loss), sea state, platform/flow
self-noise, and (for this historical archive specifically) old-tape
hiss/hum, all of which lower the effective SNR of an animal's call
against the ambient background. A model that only performs well at the
generally-favorable SNR of the Watkins archive's closest recordings may
not be telling you much about deployed performance on distant animals.

    from watkins.robustness import robustness_sweep
    df = robustness_sweep("results/checkpoints/resnet18_best.pt")

CLI:
    python -m watkins.robustness --checkpoint results/checkpoints/resnet18_best.pt
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .data import CLASS_IDS, CLASS_INFO, get_split
from .evaluate import load_checkpoint, predict
from .pipeline import build_eval_dataloader, build_noisy_eval_transform
from .utils import get_device, results_root

DEFAULT_SNR_LEVELS_DB = [None, 20, 10, 5, 0, -5, -10]  # None = clean (no added noise)


def robustness_sweep(checkpoint_path, snr_levels_db=None, noise_kinds=("white", "pink"),
                      save: bool = True) -> pd.DataFrame:
    snr_levels_db = DEFAULT_SNR_LEVELS_DB if snr_levels_db is None else snr_levels_db
    model, ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    run_name = cfg["run_name"]
    device = get_device()

    split = get_split(mode=cfg.get("split_mode", "tape_grouped"), seed=cfg["seed"])

    def _eval_one(snr, noise_kind_label):
        transform = build_noisy_eval_transform(ckpt["input_kind"], snr_db=snr, noise_kind=noise_kind_label)
        dl = build_eval_dataloader(split.test, ckpt["input_kind"], transform, batch_size=cfg["batch_size"])
        preds, labels, _clips = predict(model, dl, device)
        acc = float((preds == labels).mean())
        per_class_acc = {}
        for cid in CLASS_IDS:
            mask = labels == cid
            if mask.sum() > 0:
                per_class_acc[CLASS_INFO[cid]["display_name"]] = float((preds[mask] == labels[mask]).mean())
        label = "clean" if snr is None else f"{snr}dB"
        print(f"[{run_name}] noise={noise_kind_label:5s} snr={label:>6s}  acc={acc:.3f}  "
              f"(n_species_in_test={len(per_class_acc)})")
        row = dict(run_name=run_name, noise_kind=noise_kind_label, snr_db=snr if snr is not None else float("inf"),
                   accuracy=acc)
        row.update({f"acc_{k}": v for k, v in per_class_acc.items()})
        return row

    rows = []
    # "clean" (no injected noise) doesn't depend on noise_kind -- compute it once.
    rows.append(_eval_one(None, noise_kinds[0]))
    finite_levels = [s for s in snr_levels_db if s is not None]
    for noise_kind in noise_kinds:
        for snr in finite_levels:
            rows.append(_eval_one(snr, noise_kind))

    df = pd.DataFrame(rows)

    if save:
        out_dir = results_root() / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{run_name}_robustness.csv", index=False)
        _plot(df, run_name, results_root() / "figures" / f"{run_name}_robustness.png")

    return df


def _plot(df: pd.DataFrame, run_name: str, out_path):
    clean_acc = df[df["snr_db"] == float("inf")]["accuracy"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for noise_kind, group in df[df["snr_db"] != float("inf")].groupby("noise_kind"):
        group = group.sort_values("snr_db")
        ax.plot(group["snr_db"], group["accuracy"], marker="o", label=noise_kind)
    ax.axhline(clean_acc, color="gray", linestyle="--", linewidth=1, label="clean")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"{run_name} -- accuracy vs. injected noise SNR")
    ax.set_ylim(0, 1.02)
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    robustness_sweep(args.checkpoint)


if __name__ == "__main__":
    _cli()
