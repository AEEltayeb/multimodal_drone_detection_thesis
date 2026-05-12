"""dryrun.py — Run an eval command at high stride + low frame limit, validate.

Wraps any of: eval/eval_pipeline.py, eval/eval_model.py, eval/eval_video.py,
eval/eval_video_temporal.py, eval/cache_inference.py.

What it does:
  1. Parses the user's command, finds --stride / --limit / --output-dir.
  2. Re-runs with stride bumped 30× and limit capped at 30, output redirected
     to a per-run temp dir under eval/results/_dryrun/.
  3. Validates: subprocess exit 0, manifest.json written somewhere under the
     temp dir, at least one CSV produced (skipped if running cache_inference).
  4. Estimates the full run cost from the dry-run timing, prints OK/FAIL.

Convention:
  python eval/dryrun.py <full eval command, including 'python eval/<script>.py ...'>

Examples:
  python eval/dryrun.py python eval/eval_pipeline.py --dataset svanstrom --stride 1
  python eval/dryrun.py python eval/eval_model.py --weights best.pt --dataset G:/.. --stride 5
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
DRYRUN_ROOT = EVAL_DIR / "results" / "_dryrun"


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[33m{s}\033[0m"


def _bump_args(argv: list[str], temp_out: Path) -> tuple[list[str], int, int, int]:
    """Return (modified argv, original_stride, dryrun_stride, dryrun_limit)."""
    new = list(argv)
    orig_stride = 1
    # --stride
    for i, a in enumerate(new):
        if a == "--stride" and i + 1 < len(new):
            try:
                orig_stride = int(new[i + 1])
            except ValueError:
                orig_stride = 1
            new[i + 1] = str(max(orig_stride * 30, 30))
            break
    else:
        new += ["--stride", "30"]
    dryrun_stride = int(new[new.index("--stride") + 1])

    # --limit (cap at 30; insert if absent)
    if "--limit" in new:
        i = new.index("--limit")
        new[i + 1] = "30"
    else:
        new += ["--limit", "30"]
    dryrun_limit = 30

    # --output-dir → temp
    if "--output-dir" in new:
        i = new.index("--output-dir")
        new[i + 1] = str(temp_out)
    else:
        new += ["--output-dir", str(temp_out)]

    return new, orig_stride, dryrun_stride, dryrun_limit


def _validate(temp_out: Path, is_cache: bool) -> tuple[bool, str]:
    """Return (ok, message). For cache_inference we just check the cache file
    landed in eval/cache and is non-empty. For everything else, check
    manifest + at least one CSV."""
    if is_cache:
        # cache_inference writes to eval/cache, not output_dir. Just check
        # that the latest *.json in eval/cache is non-empty.
        cache_dir = EVAL_DIR / "cache"
        if not cache_dir.exists():
            return False, "no eval/cache directory"
        jsons = sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        if not jsons:
            return False, "no cache files produced"
        if jsons[0].stat().st_size < 100:
            return False, f"newest cache file is empty: {jsons[0].name}"
        return True, f"cache: {jsons[0].name} ({jsons[0].stat().st_size:,}B)"

    if not temp_out.exists():
        return False, f"output dir does not exist: {temp_out}"
    manifests = list(temp_out.rglob("manifest.json"))
    if not manifests:
        return False, "no manifest.json under output_dir (Phase-0 wiring missing?)"
    csvs = list(temp_out.rglob("*.csv"))
    if not csvs:
        return False, "no CSV results produced"
    # Check at least one CSV is non-empty
    nonempty = [c for c in csvs if c.stat().st_size > 0]
    if not nonempty:
        return False, "all CSV results are empty"
    return True, f"{len(manifests)} manifest(s), {len(nonempty)} non-empty CSV(s)"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1:]

    # Identify which script
    script = ""
    for tok in cmd:
        if tok.endswith(".py"):
            script = Path(tok).name
            break
    is_cache = script == "cache_inference.py"

    # Temp output dir
    DRYRUN_ROOT.mkdir(parents=True, exist_ok=True)
    temp_out = DRYRUN_ROOT / f"{script.replace('.py', '')}_{uuid.uuid4().hex[:8]}"

    new_cmd, orig_stride, dryrun_stride, dryrun_limit = _bump_args(cmd, temp_out)

    print(f"[dryrun] script:        {script or '<unknown>'}")
    print(f"[dryrun] orig stride:   {orig_stride}")
    print(f"[dryrun] dryrun stride: {dryrun_stride}  (cap limit: {dryrun_limit})")
    print(f"[dryrun] temp out:      {temp_out}")
    print(f"[dryrun] cmd:           {' '.join(shlex.quote(t) for t in new_cmd)}")
    print()

    t0 = time.time()
    try:
        proc = subprocess.run(new_cmd, cwd=str(REPO), check=False)
    except FileNotFoundError as e:
        print(_red(f"[dryrun] FAILED: {e}"))
        return 3
    elapsed = time.time() - t0

    print()
    if proc.returncode != 0:
        print(_red(f"[dryrun] FAILED: subprocess exited {proc.returncode} "
                   f"after {elapsed:.1f}s"))
        return proc.returncode

    ok, msg = _validate(temp_out, is_cache)
    if not ok:
        print(_red(f"[dryrun] FAILED validation: {msg}  ({elapsed:.1f}s)"))
        return 4

    # Estimate full-run cost: linear in 1/stride. dryrun_stride / orig_stride.
    speedup = dryrun_stride / max(orig_stride, 1)
    est_full = elapsed * speedup
    print(_green(f"[dryrun] OK in {elapsed:.1f}s — {msg}"))
    print(f"[dryrun] estimated full-run cost @ stride {orig_stride}: "
          f"~{est_full:.0f}s ({est_full/60:.1f} min)")

    # Best-effort cleanup of dry-run temp dir if it's harmless to remove.
    # Keep the manifest path printed so the user can inspect on failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
