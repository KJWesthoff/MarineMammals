"""Load a trained checkpoint and report the metrics that actually matter
for a severely imbalanced 54-species problem: per-class precision/recall
/F1 and a confusion matrix, not just overall accuracy (which a model can
win on this dataset largely by nailing killer whale and sperm whale, the
two biggest classes by a wide margin).

    from watkins.evaluate import evaluate_checkpoint
    result = evaluate_checkpoint("results/checkpoints/baseline_cnn_best.pt")

CLI:
    python -m watkins.evaluate --checkpoint results/checkpoints/baseline_cnn_best.pt
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .data import CLASS_IDS, CLASS_INFO, get_split
from .models import build_model
from .pipeline import build_eval_dataloader, build_transforms
from .utils import get_device, results_root


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, input_kind = build_model(ckpt["model_name"], num_classes=len(CLASS_IDS), **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    assert input_kind == ckpt["input_kind"]
    return model, ckpt


@torch.no_grad()
def predict(model, dataloader, device) -> tuple[np.ndarray, np.ndarray, list]:
    model.to(device).eval()
    all_preds, all_labels, all_clips = [], [], []
    for x, y, clips in dataloader:
        logits = model(x.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
        all_clips.extend(clips)
    return np.concatenate(all_preds), np.concatenate(all_labels), all_clips


def plot_confusion_matrix(cm: np.ndarray, run_name: str, out_path):
    """54x54 is too dense for a per-cell-text layout, so this is a single
    row-normalized heatmap without per-cell text, sized to stay readable."""
    names = [CLASS_INFO[c]["display_name"] for c in CLASS_IDS]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.22), max(7, n * 0.22)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), names, rotation=90, fontsize=5)
    ax.set_yticks(range(n), names, fontsize=5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{run_name} -- confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate_checkpoint(checkpoint_path, save: bool = True) -> dict:
    model, ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    run_name = cfg["run_name"]
    device = get_device()

    split = get_split(mode=cfg.get("split_mode", "tape_grouped"), seed=cfg["seed"])
    _, eval_transform = build_transforms(ckpt["input_kind"], train_augment=False)
    test_dl = build_eval_dataloader(split.test, ckpt["input_kind"], eval_transform, batch_size=cfg["batch_size"])

    preds, labels, clips = predict(model, test_dl, device)

    names = [CLASS_INFO[c]["display_name"] for c in CLASS_IDS]
    present = sorted(set(labels.tolist()))  # species missing from test (train-only) would error target_names otherwise
    report = classification_report(
        labels, preds, labels=present, target_names=[CLASS_INFO[c]["display_name"] for c in present],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=CLASS_IDS)
    acc = float((preds == labels).mean())

    print(f"\n=== {run_name} ===")
    print(classification_report(labels, preds, labels=present,
                                 target_names=[CLASS_INFO[c]["display_name"] for c in present], zero_division=0))

    result = dict(run_name=run_name, accuracy=acc, macro_f1=report["macro avg"]["f1-score"],
                  weighted_f1=report["weighted avg"]["f1-score"], report=report,
                  confusion_matrix=cm.tolist(), class_names=names,
                  species_in_test=[CLASS_INFO[c]["display_name"] for c in present])

    if save:
        metrics_dir = results_root() / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / f"{run_name}_eval.json", "w") as f:
            json.dump(result, f, indent=2)
        plot_confusion_matrix(cm, run_name, results_root() / "figures" / f"{run_name}_confusion.png")

    return result


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint)


if __name__ == "__main__":
    _cli()
