"""Download the Watkins Marine Mammal Sound Database and materialize it to
local disk in the layout `data.py` expects.

The source (`ivangtorre/watkins-marine-mammal-full-cuts` on Hugging Face)
ships 15,248 clips as embedded audio bytes inside 9 parquet shards
(~10.6GB total), at 47 different native sample rates and lengths from
16ms to 24+ minutes. This script decodes every clip, downmixes to mono,
resamples to a single `SAMPLE_RATE`, caps stored duration at
`MAX_STORED_SECONDS` (clips get randomly/center-cropped further down to
`CLIP_SECONDS` at *load* time -- see `WatkinsDataset`; this cap just keeps
on-disk size and per-file I/O sane for the handful of extreme outliers),
and writes one wav file per clip plus a `manifest.csv`.

    python -m watkins.prepare_data
    python -m watkins.prepare_data --force   # redo even if manifest exists

Safe to interrupt and re-run: already-written wav files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path

from .data import DATA_ROOT, SAMPLE_RATE

HF_REPO = "ivangtorre/watkins-marine-mammal-full-cuts"
NUM_SHARDS = 9
MAX_STORED_SECONDS = 60.0
MAX_STORED_SAMPLES = int(SAMPLE_RATE * MAX_STORED_SECONDS)


def _decode_and_resample(audio_bytes: bytes, target_sr: int):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # downmix multi-channel -> mono
    orig_duration_s = len(wav) / sr

    t = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
    if sr != target_sr:
        t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=target_sr)

    if t.shape[-1] > MAX_STORED_SAMPLES:
        start = (t.shape[-1] - MAX_STORED_SAMPLES) // 2
        t = t[..., start:start + MAX_STORED_SAMPLES]

    return t.squeeze(0).numpy(), orig_duration_s, sr


def prepare(data_root: Path = DATA_ROOT, force: bool = False, repo: str = HF_REPO) -> None:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    audio_dir = data_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifest.csv"

    existing_rows = []
    if manifest_path.exists() and not force:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        print(f"found existing manifest with {len(existing_rows)} rows "
              f"(pass --force to rebuild from scratch)")

    written = {row["record_number"] for row in existing_rows}
    rows = list(existing_rows)
    n_failed = 0
    t0 = time.time()

    for shard in range(NUM_SHARDS):
        filename = f"data/train-{shard:05d}-of-{NUM_SHARDS:05d}.parquet"
        print(f"[{shard + 1}/{NUM_SHARDS}] downloading {filename} ...")
        local_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename)
        pf = pq.ParquetFile(local_path)
        for batch in tqdm(pf.iter_batches(batch_size=64), desc=f"shard {shard}"):
            table = batch.to_pylist()
            for rec in table:
                record_number = rec["record_number"]
                if record_number in written:
                    continue
                class_id = rec["class_id"]
                audio = rec["audio"]
                out_dir = audio_dir / str(class_id)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{record_number}.wav"
                try:
                    wav, orig_duration_s, orig_sr = _decode_and_resample(audio["bytes"], SAMPLE_RATE)
                    sf.write(str(out_path), wav, SAMPLE_RATE, subtype="PCM_16")
                except Exception as e:  # noqa: BLE001 -- log and skip malformed clips
                    n_failed += 1
                    print(f"  skipping {record_number} (class {class_id}): {e}")
                    continue
                rows.append(dict(
                    record_number=record_number, class_id=class_id,
                    path=str(out_path.relative_to(data_root)),
                    orig_sample_rate=orig_sr, orig_duration_s=f"{orig_duration_s:.3f}",
                ))
                written.add(record_number)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["record_number", "class_id", "path",
                                                "orig_sample_rate", "orig_duration_s"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"wrote {len(rows)} clips ({n_failed} skipped) to {manifest_path} in {elapsed:.0f}s")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redo even if a manifest already exists")
    parser.add_argument("--repo", default=HF_REPO)
    args = parser.parse_args()
    prepare(force=args.force, repo=args.repo)


if __name__ == "__main__":
    _cli()
