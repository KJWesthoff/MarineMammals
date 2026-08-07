# Orientation for Claude

A 7-notebook curriculum teaching passive-sonar signal analysis and
classification on the Watkins Marine Mammal Sound Database: 54 species,
15,248 clips, severe long-tail imbalance. `README.md` is thorough and is
the source of truth for *what* the project is, how to set it up, and what
the results are. **Read it before answering questions about the science
or the pipeline.** This file covers only what the README doesn't: working
conventions, traps, and current state.

## The one-paragraph version

Everything in `notebooks/` is a thin layer over the `watkins` package in
`src/` -- notebooks call the same functions the CLI does, so a lesson and
`python -m watkins.train` execute identical code. Development is CPU-only
(the machine's Quadro P520 is Pascal-generation; `torch.cuda.is_available()`
returns True but kernel launches fail). The published results came from
five `configs/gpu/` runs on a rented RTX 4090, ~2 hours total.

## Run naming -- do not blur these

| suffix | what it is | tracked? |
|---|---|---|
| `*_gpu` | the published results, `configs/gpu/` on a 4090 | yes |
| `*_demo` | notebook smoke tests, 5 epochs / 30% subset | **no** (gitignored) |
| bare name | a full `configs/` run on CPU | yes |

A `*_demo` number is a smoke test that the code path works, **not** a
smaller version of a published number -- different training budget
entirely. Notebook 05 deliberately excludes `*_demo` from its comparison
table. Never present demo output as a result.

Report **macro-F1, not accuracy**: with three orders of magnitude of class
imbalance, accuracy mostly measures performance on killer whale and sperm
whale. The macro-F1 denominator is pinned to classes present in the split
(`watkins.train._macro_f1`, `watkins.evaluate`) -- see the README section
on why.

## What ships and what doesn't

Committed (~1.7MB): `results/metrics/*_gpu_eval.json`,
`*_gpu_robustness.csv`, `results/logs/*_gpu.csv`,
`results/figures/*_gpu_*.png`. This is what lets notebooks 05 and 06
render the real comparison and SNR curves on a fresh clone with no
training.

Not committed: `results/checkpoints/*.pt` (~820MB; the two AST
checkpoints are 330MB each), the materialized `Watkins/` dataset
(re-derivable via `python -m watkins.prepare_data`), and all `*_demo`
artifacts.

## The notebook playground

`notebooks/local/` and `results-local/` are gitignored copies for
experimenting without dirtying the repo. Managed by
`scripts/sync_local_notebooks.py`:

```bash
python scripts/sync_local_notebooks.py            # refresh copies from tracked
python scripts/sync_local_notebooks.py --status   # what has diverged
python scripts/sync_local_notebooks.py --promote 03   # local edit -> tracked
python scripts/sync_local_notebooks.py --retrofit     # re-apply transforms in place
python scripts/sync_local_notebooks.py --seed-results # refresh results-local/
```

The copies sit one directory deeper, so two bootstrap lines get rewritten
(`../src` -> `../../src`, `Path("..")` -> `Path("../..")`) plus a marked
block injecting `WATKINS_RESULTS_ROOT=results-local`. `--promote`
reverses all three. `results-local/` is seeded with a copy of the tracked
results so 05/06 still read the real GPU numbers; its `checkpoints/` is a
symlink to `results/checkpoints` (those were already gitignored, so
duplicating 820MB buys nothing).

A copy edited since its last sync is never overwritten without `--force`.
That guard is the only thing protecting work that exists solely in the
playground -- **check `--status` before any `--force`.**

## Traps that have already bitten

- **`micro avg` in classification reports.** `evaluate.py` passes an
  explicit `labels=`, which makes sklearn emit `micro avg` *instead of*
  `accuracy`. Its support is the whole split, so it sorts to the top of
  any by-support ranking and steals a species slot. Filter with
  `name != "accuracy" and not name.endswith(" avg")`. This bug appeared
  independently in `summarize.py` and notebook 05 -- check any new
  by-support ranking for it.
- **`--promote` carries stored cell outputs.** If you have run cells in a
  playground copy, promoting brings the epoch logs and absolute
  `/home/kj/...` paths with them. Review `git diff` before committing;
  strip outputs if the notebook didn't have them.
- **Output conventions differ by notebook.** 00, 01, 05, 06 store cell
  outputs; 02, 03, 04 do not. Preserve whichever a file already uses.
- Notebook 06 is cache-first (`{run}_robustness.csv` exists -> load and
  skip) and CPU-guarded (`ALLOW_CPU_SWEEP = False`). A sweep is 13 full
  test-set passes over 3,072 clips. Notebook 05 writes nothing at all.
- Cell `source` fields are a mix of JSON string and list-of-lines, so
  notebook diffs look noisier than the real change. Harmless.

## State as of 2026-08-07

Clean tree, `main` in sync with `origin/main` at `6fabd1a`. All seven
playground copies verified cell-identical to their tracked counterparts
(`--status` reads `current` for all), so the user has a matched baseline
and was about to start working in `notebooks/local/`.

Open items, both raised with the user and deferred:

1. **The RunPod checkpoints exist only on this machine** -- gitignored,
   ~820MB, no backup. The committed metrics defend every README number,
   so this is not a correctness gap, but reproducing the AST fine-tune
   means renting a GPU again. Offered a backup script; not yet wanted.
2. Notebooks 05 and 06 have `/home/kj/...` absolute paths embedded in
   stored cell outputs (5 each). Cosmetic; clearing those cells' outputs
   would remove them.

## Working with this user

Commits land directly on `main` and are pushed explicitly, one step at a
time -- do not push unless asked. They read diffs and care about accuracy
in claims: verify numbers against the actual files rather than asserting
from memory, and say plainly when a check changes an earlier statement.
