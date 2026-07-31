"""Config-driven training loop, usable from the CLI or imported directly
from a notebook.

    from watkins.train import train_run, load_config
    result = train_run(load_config("configs/resnet18.yaml"))

CLI:
    python -m watkins.train --config configs/resnet18.yaml
    python -m watkins.train --config configs/baseline_cnn.yaml --epochs 3 --subset-frac 0.15
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

from .data import CLASS_IDS, NUM_CLASSES, get_split
from .models import build_model
from .models.ast_model import LinearProbeHead, extract_embeddings
from .pipeline import build_dataloaders
from .utils import (AverageMeter, EarlyStopping, autocast_dtype, count_parameters, get_device,
                     results_root, set_seed)

DEFAULT_CONFIG = dict(
    run_name="unnamed_run",
    model="baseline_cnn",
    model_kwargs={},
    batch_size=32,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    optimizer="adamw",
    train_augment=True,
    spec_augment=True,
    noise_augment_p=0.3,
    class_weighted_loss=True,
    seed=42,
    patience=8,
    num_workers=0,
    amp="auto",  # mixed precision: "auto" (GPU only), or True/False to force
    subset_frac=1.0,  # for quick smoke tests; 1.0 = full split
    split_mode="tape_grouped",  # "clip_random" for the deliberately leaky comparison
)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = {**DEFAULT_CONFIG, **user_cfg}
    return cfg


def _subset(clips: list, frac: float, seed: int) -> list:
    if frac >= 1.0:
        return clips
    rng = random.Random(seed)
    n = max(1, int(round(len(clips) * frac)))
    return rng.sample(clips, n)


def _class_weights(train_clips, device) -> torch.Tensor:
    counts = np.array([sum(1 for c in train_clips if c.label == cid) for cid in CLASS_IDS], dtype=np.float64)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(CLASS_IDS) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_optimizer(cfg: dict, params):
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    raise ValueError(f"Unknown optimizer: {cfg['optimizer']!r}")


def _run_epoch(model, loader, criterion, device, optimizer=None, amp_dtype=None, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    torch.set_grad_enabled(train_mode)
    iterator = tqdm(loader, leave=False, desc="train" if train_mode else "eval") if train_mode else loader
    for x, y, _clips in iterator:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                # fp16 only: gradients can underflow to zero without scaling.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        loss_meter.update(loss.item(), n=x.size(0))
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_labels.append(y.detach().cpu())
    torch.set_grad_enabled(True)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return loss_meter.avg, acc, macro_f1


def _train_generic(cfg: dict, model, input_kind: str, split, device) -> dict:
    train_dl, val_dl, test_dl = build_dataloaders(
        split, input_kind,
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        train_augment=cfg["train_augment"], noise_augment_p=cfg["noise_augment_p"],
        spec_augment=cfg["spec_augment"],
        pin_memory=device.type == "cuda",
    )
    weight = _class_weights(split.train, device) if cfg["class_weighted_loss"] else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = _build_optimizer(cfg, (p for p in model.parameters() if p.requires_grad))
    stopper = EarlyStopping(patience=cfg["patience"])

    amp_dtype = autocast_dtype(device, cfg["amp"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    if amp_dtype is not None:
        print(f"[{cfg['run_name']}] mixed precision: {amp_dtype} "
              f"(grad scaler {'on' if scaler.is_enabled() else 'off'})")

    log_rows = []
    best_f1 = -1.0
    best_state = None
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        train_loss, train_acc, train_f1 = _run_epoch(
            model, train_dl, criterion, device, optimizer, amp_dtype=amp_dtype, scaler=scaler)
        val_loss, val_acc, val_f1 = _run_epoch(
            model, val_dl, criterion, device, optimizer=None, amp_dtype=amp_dtype)
        elapsed = time.perf_counter() - t0
        log_rows.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, train_f1=train_f1,
                              val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, seconds=elapsed))
        print(f"[{cfg['run_name']}] epoch {epoch:02d}/{cfg['epochs']} "
              f"train_loss={train_loss:.3f} train_f1={train_f1:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}  ({elapsed:.1f}s)")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_f1):
            print(f"[{cfg['run_name']}] early stopping at epoch {epoch} (best val_f1={best_f1:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, test_f1 = _run_epoch(
        model, test_dl, criterion, device, optimizer=None, amp_dtype=amp_dtype)
    print(f"[{cfg['run_name']}] TEST acc={test_acc:.3f} macro_f1={test_f1:.3f}")
    return dict(log_rows=log_rows, best_val_f1=best_f1, test_acc=test_acc, test_f1=test_f1)


def _train_ast_linear_probe(cfg: dict, model, split, device) -> dict:
    """Fast path: cache frozen-backbone embeddings once, train the tiny
    head on cached tensors, then splice the trained head back into the
    full model so the saved checkpoint behaves identically to a normally
    trained one at inference time."""
    train_dl, val_dl, test_dl = build_dataloaders(
        split, "ast", batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        train_augment=False,  # caching requires deterministic inputs
        noise_augment_p=0.0, spec_augment=False,
        drop_last_train=False,  # single-pass embedding extraction: keep every sample
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    # The one-time backbone forward pass dominates this path's cost, so it's
    # where mixed precision pays off; the head then trains in fp32 on the
    # cached (float32-cast) embeddings.
    amp_dtype = autocast_dtype(device, cfg["amp"])
    print(f"[{cfg['run_name']}] extracting frozen AST embeddings (train/val/test)"
          f"{f' [{amp_dtype}]' if amp_dtype is not None else ''}...")
    train_feats, train_labels = extract_embeddings(model, train_dl, device, amp_dtype=amp_dtype)
    val_feats, val_labels = extract_embeddings(model, val_dl, device, amp_dtype=amp_dtype)
    test_feats, test_labels = extract_embeddings(model, test_dl, device, amp_dtype=amp_dtype)

    head = LinearProbeHead(hidden_size=train_feats.shape[1], num_classes=NUM_CLASSES).to(device)
    weight = None
    if cfg["class_weighted_loss"]:
        counts = np.array([(train_labels == cid).sum().item() for cid in CLASS_IDS], dtype=np.float64)
        counts = np.clip(counts, 1, None)
        weight = torch.tensor(counts.sum() / (len(CLASS_IDS) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = _build_optimizer(cfg, head.parameters())
    stopper = EarlyStopping(patience=cfg["patience"])

    train_feats, train_labels = train_feats.to(device), train_labels.to(device)
    val_feats, val_labels = val_feats.to(device), val_labels.to(device)
    test_feats, test_labels = test_feats.to(device), test_labels.to(device)

    log_rows = []
    best_f1, best_state = -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(train_feats)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        train_acc = (logits.argmax(1) == train_labels).float().mean().item()
        train_f1 = f1_score(train_labels.cpu(), logits.argmax(1).cpu(), average="macro", zero_division=0)

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats)
            val_loss = criterion(val_logits, val_labels).item()
            val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()
            val_f1 = f1_score(val_labels.cpu(), val_logits.argmax(1).cpu(), average="macro", zero_division=0)

        log_rows.append(dict(epoch=epoch, train_loss=loss.item(), train_acc=train_acc, train_f1=train_f1,
                              val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, seconds=0.0))
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{cfg['run_name']}] epoch {epoch:03d}/{cfg['epochs']} "
                  f"train_f1={train_f1:.3f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if stopper.step(val_f1):
            print(f"[{cfg['run_name']}] early stopping at epoch {epoch} (best val_f1={best_f1:.3f})")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    with torch.no_grad():
        test_logits = head(test_feats)
        test_acc = (test_logits.argmax(1) == test_labels).float().mean().item()
        test_f1 = f1_score(test_labels.cpu(), test_logits.argmax(1).cpu(), average="macro", zero_division=0)
    print(f"[{cfg['run_name']}] TEST acc={test_acc:.3f} macro_f1={test_f1:.3f}")

    # splice trained head back into the full model for a uniform checkpoint format
    model.model.classifier.load_state_dict(head.state_dict())
    return dict(log_rows=log_rows, best_val_f1=best_f1, test_acc=test_acc, test_f1=test_f1)


def train_run(cfg: dict) -> dict:
    cfg = {**DEFAULT_CONFIG, **cfg}
    set_seed(cfg["seed"])
    device = get_device()
    print(f"[{cfg['run_name']}] device={device}  config={cfg}")

    split = get_split(mode=cfg["split_mode"], seed=cfg["seed"])
    if cfg["subset_frac"] < 1.0:
        split.train = _subset(split.train, cfg["subset_frac"], cfg["seed"])
        split.val = _subset(split.val, cfg["subset_frac"], cfg["seed"])
        split.test = _subset(split.test, cfg["subset_frac"], cfg["seed"])
    print(f"[{cfg['run_name']}] split:\n{split.summary()}")

    model, input_kind = build_model(cfg["model"], num_classes=NUM_CLASSES, **cfg["model_kwargs"])
    model.to(device)
    print(f"[{cfg['run_name']}] model={cfg['model']} input_kind={input_kind} "
          f"params={count_parameters(model):,} trainable={count_parameters(model, True):,}")

    is_ast_linear_probe = (
        cfg["model"] == "ast" and cfg["model_kwargs"].get("freeze_backbone", True)
    )
    if is_ast_linear_probe:
        result = _train_ast_linear_probe(cfg, model, split, device)
    else:
        result = _train_generic(cfg, model, input_kind, split, device)

    ckpt_dir = results_root() / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['run_name']}_best.pt"
    torch.save(dict(model_name=cfg["model"], model_kwargs=cfg["model_kwargs"],
                     input_kind=input_kind, state_dict=model.state_dict(), config=cfg), ckpt_path)
    print(f"[{cfg['run_name']}] saved checkpoint -> {ckpt_path}")

    log_dir = results_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cfg['run_name']}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["log_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["log_rows"])
    print(f"[{cfg['run_name']}] saved training log -> {log_path}")

    return dict(**result, checkpoint_path=str(ckpt_path), log_path=str(log_path))


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset-frac", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split-mode", choices=["tape_grouped", "clip_random"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, val in [("run_name", args.run_name), ("epochs", args.epochs),
                      ("subset_frac", args.subset_frac), ("batch_size", args.batch_size),
                      ("split_mode", args.split_mode)]:
        if val is not None:
            cfg[key] = val
    train_run(cfg)


if __name__ == "__main__":
    _cli()
