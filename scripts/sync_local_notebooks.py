#!/usr/bin/env python3
"""Keep a private, gitignored playground copy of the notebooks.

`notebooks/local/` is in .gitignore, so anything you do there -- running
cells, leaving outputs in, scribbling experiments -- never shows up in
`git status` and never gets pushed. The tracked `notebooks/*.ipynb` stay
clean and reviewable.

    python scripts/sync_local_notebooks.py              # refresh local copies
    python scripts/sync_local_notebooks.py --only 02 05  # just those two
    python scripts/sync_local_notebooks.py --from-head   # ignore working-tree edits
    python scripts/sync_local_notebooks.py --force       # overwrite local edits
    python scripts/sync_local_notebooks.py --status      # what's diverged
    python scripts/sync_local_notebooks.py --retrofit    # re-apply transforms in place
    python scripts/sync_local_notebooks.py --seed-results
    python scripts/sync_local_notebooks.py --promote 03  # local edit -> tracked

Runs launched from a local copy write to `results-local/` (also
gitignored) rather than the tracked `results/` tree, so demo checkpoints,
metrics and figures stay out of `git status` too. That root is seeded
with a copy of the tracked results, which is what keeps notebooks 05 and
06 able to read the RunPod GPU runs they don't produce themselves.

The copies live one directory deeper than the originals, so the two
depth-dependent lines in the bootstrap cell get rewritten on the way in
(and back again on `--promote`). Everything else resolves through
`repo_root()` in `watkins.utils`, which keys off the package location
rather than the notebook's, so data/checkpoint paths are unaffected by
the extra level.

A copy you've edited yourself is never silently clobbered: the script
records a hash of what it wrote, and refuses to overwrite anything that
has changed since. Use --force when you really do want the tracked
version back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "notebooks"
LOCAL_DIR = SOURCE_DIR / "local"
STATE_FILE = LOCAL_DIR / ".sync_state.json"
RESULTS_DIR = REPO_ROOT / "results"
LOCAL_RESULTS_DIR = REPO_ROOT / "results-local"

# Subdirectories seeded by copy. Small (~4MB all told), and copies mean a
# playground re-run that lands on an existing name overwrites the copy
# rather than the tracked file. `checkpoints/` is deliberately not in this
# list -- see seed_results().
SEEDED = ["metrics", "logs", "figures"]

BANNER = (
    "> **Local playground copy -- not tracked by git.**\n"
    ">\n"
    "> Edits and outputs here stay on this machine. The reviewable version\n"
    "> is `notebooks/{name}`; refresh this copy from it with\n"
    "> `python scripts/sync_local_notebooks.py`, or push an edit back the\n"
    "> other way with `--promote {stem}`."
)


def notebook_paths() -> list[Path]:
    return sorted(p for p in SOURCE_DIR.glob("*.ipynb") if p.is_file())


def _root_prefix(nb_dir: Path) -> str:
    """The `..`-style path from a notebook directory back to the repo root."""
    depth = len(nb_dir.resolve().relative_to(REPO_ROOT).parts)
    return "/".join([".."] * depth)


# The bootstrap cell's only depth-dependent lines. Each entry is a
# format string with a {root} placeholder; anything that doesn't appear
# in a given notebook is fine (00/01/05/06 have no PROJECT_ROOT), but a
# rewrite that matches *nothing at all* means the bootstrap cell changed
# shape and this script needs updating -- so that gets reported.
REWRITES = [
    'sys.path.insert(0, "{root}/src")',
    'PROJECT_ROOT = Path(PROJECT_DRIVE_PATH) if IN_COLAB else Path("{root}")',
]


def retarget(nb: dict, from_root: str, to_root: str, label: str) -> dict:
    """Rewrite depth-dependent paths in code cells. Returns the notebook."""
    matched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        joined = "".join(source) if isinstance(source, list) else source
        original = joined
        for template in REWRITES:
            old = template.format(root=from_root)
            new = template.format(root=to_root)
            if old in joined:
                joined = joined.replace(old, new)
        if joined != original:
            matched = True
            cell["source"] = joined.splitlines(keepends=True)
    if not matched:
        print(
            f"  ! {label}: no path lines rewritten -- the bootstrap cell may have "
            f"changed. Check that it still runs from {to_root}/.",
            file=sys.stderr,
        )
    return nb


# Injected into the non-Colab branch of the bootstrap cell, right after the
# sys.path line and before any `from watkins...` import -- `results_root()`
# reads the environment at call time, so it has to be set before the module
# that calls it is imported. `os` is already imported at the top of that cell;
# `pathlib` is not yet, hence os.path.abspath. That resolves against the
# kernel's working directory, which for a notebook is its own directory --
# the same assumption the neighbouring "{root}/src" line already makes.
RESULTS_BEGIN = "# >>> local playground results root"
RESULTS_END = "# <<< local playground results root"
RESULTS_BLOCK = """\
{indent}{begin} (scripts/sync_local_notebooks.py)
{indent}# Runs launched from this copy write to results-local/ instead of the
{indent}# tracked results/ tree, so demo checkpoints, metrics and figures never
{indent}# turn up in `git status`. Seeded from results/ (so notebooks 05/06 can
{indent}# still read the GPU runs) by:
{indent}#     python scripts/sync_local_notebooks.py --seed-results
{indent}os.environ["WATKINS_RESULTS_ROOT"] = os.path.abspath("{root}/results-local")
{indent}{end}
"""


def inject_results_root(nb: dict, root: str, label: str) -> dict:
    """Point the local copy's results root at results-local/."""
    anchor = f'sys.path.insert(0, "{root}/src")'
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        joined = "".join(source) if isinstance(source, list) else source
        if anchor not in joined or RESULTS_BEGIN in joined:
            continue

        out = []
        for line in joined.splitlines(keepends=True):
            out.append(line)
            if line.strip() == anchor:
                indent = line[: len(line) - len(line.lstrip())]
                out.append(RESULTS_BLOCK.format(
                    indent=indent, root=root, begin=RESULTS_BEGIN, end=RESULTS_END))
        cell["source"] = "".join(out).splitlines(keepends=True)
        return nb

    if not any(RESULTS_BEGIN in "".join(c["source"]) for c in nb.get("cells", [])
               if c.get("cell_type") == "code"):
        print(f"  ! {label}: no `{anchor}` line to anchor the results root to; "
              f"this copy will write into the tracked results/ tree.",
              file=sys.stderr)
    return nb


def strip_results_root(nb: dict) -> dict:
    """Remove the injected block, for the promote direction."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        joined = "".join(source) if isinstance(source, list) else source
        if RESULTS_BEGIN not in joined:
            continue
        out, skipping = [], False
        for line in joined.splitlines(keepends=True):
            if RESULTS_BEGIN in line:
                skipping = True
            if not skipping:
                out.append(line)
            if RESULTS_END in line:
                skipping = False
        cell["source"] = "".join(out).splitlines(keepends=True)
    return nb


def seed_results(force: bool = False) -> None:
    """Give results-local/ a starting copy of the tracked results.

    Notebooks 05 and 06 read metrics/robustness sweeps that they don't
    produce themselves, so an empty results root would make them look like
    nothing had ever been trained. Seeding copies the small artifacts
    (~4MB) so those notebooks keep working against the real GPU numbers.

    `checkpoints/` is a symlink rather than a copy: the .pt files are
    ~820MB, and they are already fully gitignored, so they were never a
    source of git noise -- the isolation this function provides is about
    the *tracked* metrics/logs/figures. The symlink also means notebook 06
    can load the RunPod checkpoints without duplicating them.
    """
    LOCAL_RESULTS_DIR.mkdir(exist_ok=True)
    copied = 0

    for sub in SEEDED:
        src_dir = RESULTS_DIR / sub
        if not src_dir.is_dir():
            continue
        (LOCAL_RESULTS_DIR / sub).mkdir(exist_ok=True)
        for item in src_dir.iterdir():
            if not item.is_file():
                continue
            dest = LOCAL_RESULTS_DIR / sub / item.name
            if dest.exists() and not force:
                continue
            dest.write_bytes(item.read_bytes())
            copied += 1

    for item in RESULTS_DIR.iterdir():
        if item.is_file():
            dest = LOCAL_RESULTS_DIR / item.name
            if not dest.exists() or force:
                dest.write_bytes(item.read_bytes())
                copied += 1

    link = LOCAL_RESULTS_DIR / "checkpoints"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(Path("..") / "results" / "checkpoints", target_is_directory=True)
        print(f"  link results-local/checkpoints -> ../results/checkpoints")
    elif not link.is_symlink():
        print(f"  ! results-local/checkpoints is a real directory, not the expected "
              f"symlink; leaving it alone.", file=sys.stderr)

    print(f"  seed results-local/  ({copied} file(s) copied"
          f"{', overwriting' if force else ', existing left alone'})")


def add_banner(nb: dict, name: str) -> dict:
    text = BANNER.format(name=name, stem=name.split("_")[0])
    banner_cell = {
        "cell_type": "markdown",
        "metadata": {"tags": ["local-copy-banner"]},
        "source": text.splitlines(keepends=True),
    }
    nb["cells"].insert(0, banner_cell)
    return nb


def strip_banner(nb: dict) -> dict:
    cells = nb.get("cells", [])
    if cells and "local-copy-banner" in cells[0].get("metadata", {}).get("tags", []):
        cells.pop(0)
    return nb


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def read_notebook_from_head(path: Path) -> dict:
    rel = path.resolve().relative_to(REPO_ROOT)
    blob = subprocess.run(
        ["git", "show", f"HEAD:{rel.as_posix()}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if blob.returncode != 0:
        raise SystemExit(f"{rel} is not in HEAD: {blob.stderr.strip()}")
    return json.loads(blob.stdout)


def write_notebook(path: Path, nb: dict) -> str:
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(paths: list[Path], only: list[str] | None) -> list[Path]:
    if not only:
        return paths
    picked = [p for p in paths if any(p.name.startswith(o) or p.stem == o for o in only)]
    if not picked:
        raise SystemExit(f"No notebooks matched {only!r}. Available: "
                         + ", ".join(p.stem for p in paths))
    return picked


def to_local(nb: dict, name: str, from_root: str, to_root: str) -> dict:
    """Everything that turns a tracked notebook into its playground copy."""
    nb = retarget(nb, from_root, to_root, name)
    nb = inject_results_root(nb, to_root, name)
    return add_banner(nb, name)


def cmd_sync(only: list[str] | None, from_head: bool, force: bool) -> int:
    state = load_state()
    from_root = _root_prefix(SOURCE_DIR)
    to_root = _root_prefix(LOCAL_DIR)
    copied = skipped = 0

    for src in selected(notebook_paths(), only):
        dest = LOCAL_DIR / src.name
        if dest.exists() and not force:
            recorded = state.get(src.name)
            if recorded is None or file_hash(dest) != recorded:
                print(f"  skip {src.name}  (edited locally -- --force to overwrite)")
                skipped += 1
                continue

        nb = read_notebook_from_head(src) if from_head else read_notebook(src)
        nb = to_local(nb, src.name, from_root, to_root)
        state[src.name] = write_notebook(dest, nb)
        print(f"  sync {src.name}  ->  {dest.relative_to(REPO_ROOT)}")
        copied += 1

    save_state(state)
    seed_results()
    source_desc = "HEAD" if from_head else "the working tree"
    print(f"\n{copied} copied from {source_desc}, {skipped} left alone.")
    print(f"Playground: {LOCAL_DIR.relative_to(REPO_ROOT)}/ (gitignored), "
          f"writing results to {LOCAL_RESULTS_DIR.relative_to(REPO_ROOT)}/")
    return 0


def cmd_retrofit(only: list[str] | None) -> int:
    """Apply the local-copy transforms to existing copies, in place.

    Unlike a sync this pulls no content from the tracked notebooks, so
    playground edits survive. It is how an already-populated
    notebooks/local/ picks up a transform added later -- the results-root
    injection, say. Each transform is a no-op if already applied.
    """
    state = load_state()
    to_root = _root_prefix(LOCAL_DIR)
    changed = 0

    for src in selected(notebook_paths(), only):
        local = LOCAL_DIR / src.name
        if not local.exists():
            print(f"  ! no local copy of {src.name}", file=sys.stderr)
            continue

        before = file_hash(local)
        nb = read_notebook(local)
        nb = inject_results_root(nb, to_root, src.name)
        if not (nb.get("cells") and "local-copy-banner"
                in nb["cells"][0].get("metadata", {}).get("tags", [])):
            nb = add_banner(nb, src.name)
        after = write_notebook(local, nb)

        if after == before:
            print(f"  ok   {src.name}  (already current)")
        else:
            print(f"  fix  {src.name}")
            changed += 1
        # Only re-record for copies the script already considered unedited;
        # a hash we never recorded means local-only content worth protecting.
        if state.get(src.name) == before:
            state[src.name] = after

    save_state(state)
    seed_results()
    print(f"\n{changed} notebook(s) updated in place.")
    return 0


def cmd_seed(force: bool) -> int:
    seed_results(force=force)
    return 0


def cmd_status(only: list[str] | None) -> int:
    state = load_state()
    if not LOCAL_DIR.exists():
        print("No local copies yet. Run: python scripts/sync_local_notebooks.py")
        return 0

    from_root = _root_prefix(SOURCE_DIR)
    to_root = _root_prefix(LOCAL_DIR)

    for src in selected(notebook_paths(), only):
        dest = LOCAL_DIR / src.name
        if not dest.exists():
            print(f"  missing   {src.name}  (never synced)")
            continue

        edited = state.get(src.name) != file_hash(dest)
        # What a sync would produce right now, to tell "the tracked notebook
        # has moved on" apart from "nothing has changed anywhere".
        would_write = json.dumps(
            to_local(read_notebook(src), src.name, from_root, to_root),
            indent=1, ensure_ascii=False,
        ) + "\n"
        stale = hashlib.sha256(would_write.encode()).hexdigest() != state.get(src.name)

        if edited and stale:
            print(f"  BOTH      {src.name}  (local edits + tracked has changed; "
                  f"sync would need --force)")
        elif edited:
            print(f"  edited    {src.name}  (local-only changes; --promote to keep them)")
        elif stale:
            print(f"  stale     {src.name}  (tracked has changed; sync will update it)")
        else:
            print(f"  current   {src.name}")
    return 0


def cmd_promote(only: list[str], force: bool) -> int:
    """Copy a local notebook back over its tracked counterpart."""
    state = load_state()
    from_root = _root_prefix(LOCAL_DIR)
    to_root = _root_prefix(SOURCE_DIR)

    for src in selected(notebook_paths(), only):
        local = LOCAL_DIR / src.name
        if not local.exists():
            print(f"  ! no local copy of {src.name}", file=sys.stderr)
            continue

        dirty = subprocess.run(
            ["git", "diff", "--quiet", "--", f"notebooks/{src.name}"],
            cwd=REPO_ROOT,
        ).returncode != 0
        if dirty and not force:
            print(f"  skip {src.name}  (tracked copy has uncommitted changes "
                  f"that promoting would overwrite -- commit them, or --force)")
            continue

        nb = strip_banner(read_notebook(local))
        nb = strip_results_root(nb)
        nb = retarget(nb, from_root, to_root, src.name)
        write_notebook(src, nb)
        # The local copy is now the source of truth for that file; re-record
        # its hash so the next plain sync doesn't call it "edited".
        state[src.name] = file_hash(local)
        print(f"  promote {local.relative_to(REPO_ROOT)}  ->  notebooks/{src.name}")
        print(f"          review it: git diff notebooks/{src.name}")

    save_state(state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", nargs="+", metavar="PREFIX",
                        help="only these notebooks, by number or stem (e.g. 02 05)")
    parser.add_argument("--from-head", action="store_true",
                        help="copy the last-committed version rather than the working tree")
    parser.add_argument("--force", action="store_true",
                        help="overwrite even if the destination has been edited")
    parser.add_argument("--status", action="store_true",
                        help="report which local copies have diverged, change nothing")
    parser.add_argument("--seed-results", action="store_true",
                        help="populate results-local/ from results/ and exit")
    parser.add_argument("--retrofit", action="store_true",
                        help="re-apply the local-copy transforms in place, keeping edits")
    parser.add_argument("--promote", nargs="+", metavar="PREFIX",
                        help="copy local edits back onto the tracked notebooks")
    args = parser.parse_args()

    if args.status:
        return cmd_status(args.only)
    if args.seed_results:
        return cmd_seed(args.force)
    if args.retrofit:
        return cmd_retrofit(args.only)
    if args.promote:
        return cmd_promote(args.promote, args.force)
    return cmd_sync(args.only, args.from_head, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
