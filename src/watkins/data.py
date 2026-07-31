"""Watkins Marine Mammal Sound Database: dataset loading, species metadata,
and train/val/test splitting.

Source
------
This project runs on the **Watkins Marine Mammal Sound Database**
(WMMSD), via a community-maintained re-release on Hugging Face
(``ivangtorre/watkins-marine-mammal-full-cuts``) that recovered the audio
from Internet Archive snapshots after WHOI's own site went offline. See
the README for citations.

15,248 clips, 54 marine mammal species, recorded 1934-2001 on wildly
heterogeneous equipment: 47 distinct native sample rates (320 Hz-192 kHz)
and clip lengths from 16ms to over 24 minutes. `prepare_data.py`
materializes this once into a fixed-sample-rate, fixed-length-friendly
form on disk; this module reads that materialized form.

Layout on disk after materialization (``Watkins/watkins_16k/``)::

    watkins_16k/
        manifest.csv            # record_number, class_id, path, orig_sample_rate, orig_duration_s
        audio/
            <class_id>/
                <record_number>.wav   # mono, 16kHz, capped at MAX_STORED_SECONDS

``class_id`` (0-53) is WHOI's own species numbering, preserved as-is from
the source dataset (not reassigned or inferred) -- see ``CLASS_INFO``
below for the id -> species mapping. Every clip's ``record_number`` is an
8-character code whose first 6 characters identify the original
recording/tape it was cut from (e.g. ``600120`` for ``6001200A``,
``60012001``, ...); we call that the clip's ``tape_id``.

Real, severe class imbalance
-----------------------------
Species representation ranges from 2,647 clips (killer whale) down to a
single clip (harbour seal). This is a genuine property of the historical
archive (some species were recorded far more often than others over the
database's ~70-year span), not a preprocessing artifact -- see notebook
00 for the full distribution.

Fourteen species have only a *single* original recording (``tape_id``) to
their name. Since clips are split by tape, not by individual clip, so
that clips from one recording never cross between train/val/test, those
14 species have nowhere else to draw a val/test set from -- they end up
train-only. That's a real, honestly-reported limitation of the source
data, not something this module tries to paper over: see
``Split.train_only_classes``.

Variable-length clips
----------------------
Watkins clips vary enormously in length (median 1.9s, but a long tail out
past a minute). `WatkinsDataset`
below crops/pads every waveform to a fixed ``CLIP_SAMPLES`` length at load
time -- randomly during training (a cheap, free augmentation), from the
center during evaluation (deterministic).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .utils import data_root

DATA_ROOT = data_root()
SAMPLE_RATE = 16_000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Species ground truth, taken directly from the source dataset's own
# `class_id` / `display_name` / `scientific_name` fields (no inference
# needed). Two small corrections to the upstream metadata, made from
# published species facts:
# HarbourSeal's `scientific_name` field upstream is wrongly copied from
# BottlenoseDolphin's row (`Tursiops truncatus`); the correct binomial is
# `Phoca vitulina`. WestIndianManatee's `scientific_name` field is blank
# upstream; filled in as `Trichechus manatus`.
CLASS_INFO = {
    0: dict(display_name="Fin_FinbackWhale", scientific_name="Balaenoptera physalus"),
    1: dict(display_name="StripedDolphin", scientific_name="Stenella coeruleoalba"),
    2: dict(display_name="SeaOtter", scientific_name="Enhydra lutris"),
    3: dict(display_name="Narwhal", scientific_name="Monodon monoceros"),
    4: dict(display_name="WestIndianManatee", scientific_name="Trichechus manatus"),
    5: dict(display_name="White_sidedDolphin", scientific_name="Lagenorhynchus acutus"),
    6: dict(display_name="Grampus_Risso'sDolphin", scientific_name="Grampus griseus"),
    7: dict(display_name="TucuxiDolphin", scientific_name="Sotalia fluviatilis"),
    8: dict(display_name="LongBeaked(Pacific)CommonDolphin", scientific_name="Delphinus bairdii"),
    9: dict(display_name="Dall'sPorpoise", scientific_name="Phocoenoides dalli"),
    10: dict(display_name="MelonHeadedWhale", scientific_name="Peponocephala electra"),
    11: dict(display_name="WeddellSeal", scientific_name="Leptonychotes weddelli"),
    12: dict(display_name="HumpbackWhale", scientific_name="Megaptera novaeangliae"),
    13: dict(display_name="RossSeal", scientific_name="Ommatophoca rossi"),
    14: dict(display_name="BottlenoseDolphin", scientific_name="Tursiops truncatus"),
    15: dict(display_name="StellerSeaLion", scientific_name="Eumetopias jubatus"),
    16: dict(display_name="White_beakedDolphin", scientific_name="Lagenorhynchus albirostris"),
    17: dict(display_name="Rough_ToothedDolphin", scientific_name="Steno bredanensis"),
    18: dict(display_name="HarborPorpoise", scientific_name="Phocoena phocoena"),
    19: dict(display_name="NorthernRightWhale", scientific_name="Eubalaena glacialis"),
    20: dict(display_name="AtlanticSpottedDolphin", scientific_name="Stenella frontalis"),
    21: dict(display_name="Walrus", scientific_name="Odobenus rosmarus"),
    22: dict(display_name="KillerWhale", scientific_name="Orcinus orca"),
    23: dict(display_name="Short_Finned(Pacific)PilotWhale", scientific_name="Globicephala macrorhynchus"),
    24: dict(display_name="Heaviside'sDolphin", scientific_name="Cephalorhynchus heavisidii"),
    25: dict(display_name="RibbonSeal", scientific_name="Phoca fasciata"),
    26: dict(display_name="FalseKillerWhale", scientific_name="Pseudorca crassidens"),
    27: dict(display_name="RingedSeal", scientific_name="Phoca hispida"),
    28: dict(display_name="SouthernRightWhale", scientific_name="Eubalaena australis"),
    29: dict(display_name="Boutu_AmazonRiverDolphin", scientific_name="Inia geoffrensis"),
    30: dict(display_name="HarpSeal", scientific_name="Phoca groenlandica"),
    31: dict(display_name="JuanFernandezFurSeal", scientific_name="Arctocephalus philippii"),
    32: dict(display_name="BeardedSeal", scientific_name="Erignathus barbatus"),
    33: dict(display_name="IrawaddyDolphin", scientific_name="Sotalia borneensis"),
    34: dict(display_name="MinkeWhale", scientific_name="Balaenoptera acutorostrata"),
    35: dict(display_name="SpottedSeal", scientific_name="Phoca largha"),
    36: dict(display_name="CommonDolphin", scientific_name="Delphinus delphis"),
    37: dict(display_name="GrayWhale", scientific_name="Eschrichtius robustus"),
    38: dict(display_name="FinlessPorpoise", scientific_name="Neophocaena phocaenoides"),
    39: dict(display_name="Long_FinnedPilotWhale", scientific_name="Globicephala melaena"),
    40: dict(display_name="ClymeneDolphin", scientific_name="Stenella clymene"),
    41: dict(display_name="Fraser'sDolphin", scientific_name="Lagenodelphis hosei"),
    42: dict(display_name="LeopardSeal", scientific_name="Hydrurga leptonyx"),
    43: dict(display_name="BowheadWhale", scientific_name="Balaena mysticetus"),
    44: dict(display_name="HoodedSeal", scientific_name="Cystophora cristata"),
    45: dict(display_name="HarbourSeal", scientific_name="Phoca vitulina"),
    46: dict(display_name="SpinnerDolphin", scientific_name="Stenella longirostris"),
    47: dict(display_name="Beluga_WhiteWhale", scientific_name="Delphinapterus leucas"),
    48: dict(display_name="GraySeal", scientific_name="Halichoerus grypus"),
    49: dict(display_name="NewZealandFurSeal", scientific_name="Arctocephalus forsteri"),
    50: dict(display_name="Commerson'sDolphin", scientific_name="Cephalorhynchus commersonii"),
    51: dict(display_name="SpermWhale", scientific_name="Physeter catodon"),
    52: dict(display_name="DuskyDolphin", scientific_name="Lagenorhynchus obscurus"),
    53: dict(display_name="PantropicalSpottedDolphin", scientific_name="Stenella attenuata"),
}
CLASS_IDS = sorted(CLASS_INFO.keys())
NUM_CLASSES = len(CLASS_IDS)
CLASS_NAMES = [CLASS_INFO[i]["display_name"] for i in CLASS_IDS]


@dataclass
class Clip:
    path: Path
    label: int
    record_number: str
    tape_id: str  # first 6 chars of record_number: groups clips from one original recording


def _tape_id(record_number: str) -> str:
    return record_number[:6]


def load_manifest(data_root: Path = DATA_ROOT) -> list[Clip]:
    """Read `manifest.csv` (written by `prepare_data.py`) and return every
    clip with its metadata."""
    import csv

    manifest_path = data_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run `python -m watkins.prepare_data` first "
            "to download and materialize the Watkins dataset."
        )
    clips = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            record_number = row["record_number"]
            clips.append(Clip(
                path=data_root / row["path"],
                label=int(row["class_id"]),
                record_number=record_number,
                tape_id=_tape_id(record_number),
            ))
    if not clips:
        raise FileNotFoundError(f"manifest at {manifest_path} has no rows.")
    return clips


@dataclass
class Split:
    train: list[Clip]
    val: list[Clip]
    test: list[Clip]
    train_only_classes: list[int] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"train={len(self.train)}  val={len(self.val)}  test={len(self.test)}"]
        if self.train_only_classes:
            names = [CLASS_INFO[c]["display_name"] for c in self.train_only_classes]
            lines.append(
                f"  {len(names)} species have only one source recording and are "
                f"train-only (excluded from val/test): {', '.join(names)}"
            )
        return "\n".join(lines)


def _assign_tape_groups(
    groups: dict[str, list[Clip]], train_frac: float, val_frac: float, rng: random.Random,
) -> tuple[list[Clip], list[Clip], list[Clip]]:
    """Split one species' clips, grouped by original recording (`tape_id`),
    into train/val/test so that no recording ever crosses a split boundary.

    - 1 tape: everything goes to train (nowhere else to draw val/test from).
    - 2 tapes: the larger goes to train, the smaller to test (val is empty
      for this species -- there's no third tape to give it).
    - >=3 tapes: the 3 smallest tapes are reserved one each for val, test,
      and train (protects val/test from being empty), then every remaining
      tape is greedily assigned to whichever split is furthest below its
      target share of this species' total clips.
    """
    tape_ids = list(groups.keys())
    if len(tape_ids) == 1:
        return list(groups[tape_ids[0]]), [], []

    if len(tape_ids) == 2:
        a, b = tape_ids
        if len(groups[a]) < len(groups[b]):
            a, b = b, a
        return list(groups[a]), [], list(groups[b])

    total = sum(len(v) for v in groups.values())
    targets = {"train": total * train_frac, "val": total * val_frac,
               "test": total * (1 - train_frac - val_frac)}
    current = {"train": 0.0, "val": 0.0, "test": 0.0}
    buckets: dict[str, list[Clip]] = {"train": [], "val": [], "test": []}

    by_size = sorted(tape_ids, key=lambda t: len(groups[t]))
    reserved, rest = by_size[:3], by_size[3:]
    rng.shuffle(reserved)
    for split_name, tid in zip(["val", "test", "train"], reserved):
        buckets[split_name].extend(groups[tid])
        current[split_name] += len(groups[tid])

    rng.shuffle(rest)
    for tid in rest:
        split_name = min(current, key=lambda s: current[s] / max(targets[s], 1e-9))
        buckets[split_name].extend(groups[tid])
        current[split_name] += len(groups[tid])

    return buckets["train"], buckets["val"], buckets["test"]


def _split_clip_random(
    manifest: list[Clip], train_frac: float, val_frac: float, rng: random.Random,
) -> tuple[list[Clip], list[Clip], list[Clip], list[int]]:
    """Naive per-class split at the individual-clip level, ignoring which
    original recording (`tape_id`) each clip came from. Deliberately
    leaky: clips cut from the same recording can land in both train and
    test, letting a model partly "recognize the recording" (tape hiss
    level, background hum, distance-to-source) instead of the species.
    Exists only for the leakage-quantification exercise in notebooks
    00/02/05 -- no model in this project trains/evaluates on this split
    by default.
    """
    by_class: dict[int, list[Clip]] = {c: [] for c in CLASS_IDS}
    for clip in manifest:
        by_class[clip.label].append(clip)

    train, val, test, train_only_classes = [], [], [], []
    for label in CLASS_IDS:
        clips = list(by_class[label])
        if not clips:
            continue
        rng.shuffle(clips)
        n = len(clips)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        c_train, c_val, c_test = clips[:n_train], clips[n_train:n_train + n_val], clips[n_train + n_val:]
        train.extend(c_train)
        val.extend(c_val)
        test.extend(c_test)
        if not c_val and not c_test:
            train_only_classes.append(label)
    return train, val, test, train_only_classes


def _split_tape_grouped(
    manifest: list[Clip], train_frac: float, val_frac: float, rng: random.Random,
) -> tuple[list[Clip], list[Clip], list[Clip], list[int]]:
    """Group-aware, class-stratified train/val/test split.

    Clips are grouped by `tape_id` (the original recording they were cut
    from) before splitting, so no split boundary ever cuts through a
    single recording -- the same recording's clips never appear in both
    train and test. This is done independently per species so that rare
    species aren't crowded out by common ones.
    """
    by_class: dict[int, dict[str, list[Clip]]] = {c: {} for c in CLASS_IDS}
    for clip in manifest:
        by_class[clip.label].setdefault(clip.tape_id, []).append(clip)

    train, val, test, train_only_classes = [], [], [], []
    for label in CLASS_IDS:
        groups = by_class[label]
        if not groups:
            continue
        c_train, c_val, c_test = _assign_tape_groups(groups, train_frac, val_frac, rng)
        train.extend(c_train)
        val.extend(c_val)
        test.extend(c_test)
        if not c_val and not c_test:
            train_only_classes.append(label)
    return train, val, test, train_only_classes


SplitMode = Literal["tape_grouped", "clip_random"]

_SPLITTERS = {"tape_grouped": _split_tape_grouped, "clip_random": _split_clip_random}


def build_split(
    data_root: Path = DATA_ROOT, mode: SplitMode = "tape_grouped",
    train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42,
) -> Split:
    """Build a train/val/test split in the given `mode` (see
    `_split_tape_grouped` for the honest default and `_split_clip_random`
    for the deliberately leaky comparison mode)."""
    manifest = load_manifest(data_root)
    rng = random.Random(seed)
    train, val, test, train_only_classes = _SPLITTERS[mode](manifest, train_frac, val_frac, rng)
    return Split(train=train, val=val, test=test, train_only_classes=train_only_classes,
                 notes={"mode": mode})


def get_split(mode: SplitMode = "tape_grouped", data_root: Path = DATA_ROOT, seed: int = 42) -> Split:
    """Kept as a thin wrapper (rather than inlining `build_split`) so
    train.py/evaluate.py/robustness.py share one entry point."""
    return build_split(data_root=data_root, mode=mode, seed=seed)


class WatkinsDataset:
    """torch.utils.data.Dataset over a list of Clips.

    Every waveform is cropped/padded to a fixed `CLIP_SAMPLES` length
    before `transform` is applied, since Watkins clips vary enormously in
    native length and every downstream feature transform needs a
    consistent shape to batch.
    `random_crop=True` (used for the training split) crops from a random
    offset and pads asymmetrically -- a free, standard time-shift
    augmentation; `random_crop=False` (val/test) always crops/pads from
    the center, so evaluation is deterministic.

    Returns (waveform[1, CLIP_SAMPLES] float32, label int, Clip).
    """

    def __init__(self, clips: list[Clip], transform=None, random_crop: bool = False):
        self.clips = clips
        self.transform = transform
        self.random_crop = random_crop

    def __len__(self):
        return len(self.clips)

    def _fit_length(self, wav):
        import torch

        n = wav.shape[-1]
        if n == CLIP_SAMPLES:
            return wav
        if n > CLIP_SAMPLES:
            if self.random_crop:
                start = torch.randint(0, n - CLIP_SAMPLES + 1, (1,)).item()
            else:
                start = (n - CLIP_SAMPLES) // 2
            return wav[..., start:start + CLIP_SAMPLES]
        pad = CLIP_SAMPLES - n
        if self.random_crop:
            left = torch.randint(0, pad + 1, (1,)).item()
        else:
            left = pad // 2
        right = pad - left
        return torch.nn.functional.pad(wav, (left, right))

    def __getitem__(self, idx: int):
        import soundfile as sf
        import torch

        clip = self.clips[idx]
        wav, sr = sf.read(str(clip.path), dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{clip.path} has sample rate {sr}, expected {SAMPLE_RATE}")
        wav = torch.from_numpy(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        wav = self._fit_length(wav)
        if self.transform is not None:
            wav = self.transform(wav)
        return wav, clip.label, clip


def collate_with_clips(batch):
    """DataLoader collate_fn that batches (features, label) normally and
    keeps the per-sample `Clip` metadata as a plain list (torch's default
    collate can't stack dataclasses). Use whenever you need to trace a
    prediction back to its source file/recording, e.g. in evaluate.py."""
    import torch

    feats = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    clips = [b[2] for b in batch]
    return feats, labels, clips
