#!/usr/bin/env python3
"""
reorg_2026_06_sweep.py — execute the green-lit 2026-06-11 script-sprawl sweep
(docs/analysis/2026-06-11_sweep_list.md, categories A+B+C+D approved; E kept).

- RESCUES: 7 useful tools re-homed (+ recorded via kb.py record).
- gui/flet_app/ KEPT: pyside_app.py imports flet_app.settings_dialog (verified).
- Everything else -> archive/2026-06-11/<original path>: recorded rows via
  kb.py mv + lifecycle=archived; unrecorded via git mv (tracked) or move.

Usage: py knowledge/_tools/reorg_2026_06_sweep.py [--dry-run]
"""
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KB = REPO / "knowledge" / "_tools" / "kb.py"
ARCH = "archive/2026-06-11"
DRY = "--dry-run" in sys.argv

RESCUES = [  # (old, new, purpose, role)
    ("classifier/convert_svanstrom_paired.py", "scripts/dataset_preparation/convert_svanstrom_paired.py",
     "Convert Svanstrom IR+Visible videos to paired YOLO dataset (frames + MATLAB GT parse)", "library"),
    ("classifier/extract_antiuav_crops.py", "scripts/dataset_preparation/extract_antiuav_crops.py",
     "Mine drone crops from Anti-UAV for confuser-filter training", "library"),
    ("classifier/extract_background_crops.py", "scripts/dataset_preparation/extract_background_crops.py",
     "Mine background/other crops from Svanstrom+Anti-UAV avoiding GT boxes", "library"),
    ("classifier/clean_patches_consensus.py", "scripts/dataset_preparation/clean_patches_consensus.py",
     "Consensus-of-filters cleaning of noisy crops in the patch manifest", "library"),
    ("classifier/check_gt_alignment.py", "scripts/check_gt_alignment.py",
     "RGB-vs-IR GT bbox alignment diagnostic (calibration check)", "library"),
    ("classifier/check_offset_per_seq.py", "scripts/check_offset_per_seq.py",
     "Fit per-sequence IR->RGB bbox transform (camera-rig calibration)", "library"),
    ("eval/render_example_images.py", None,  # record-in-place, no move
     "Composite per-dataset example PNGs (happy/sad panels) for the eval dashboard", "library"),
]

GUI_LEGACY = ["gui/app.py", "gui/fusion_app.py", "gui/run_flet.py",
              "gui/flet_theme.py", "gui/run_app.py"]

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))

def is_tracked(rel):
    return run(["git", "ls-files", "--error-unmatch", rel]).returncode == 0

def do_move(rel, dest_rel):
    src, dst = REPO / rel, REPO / dest_rel
    if not src.exists():
        print(f"  SKIP missing: {rel}"); return False
    if DRY:
        print(f"  would move: {rel} -> {dest_rel}"); return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_tracked(rel):
        r = run(["git", "mv", rel, dest_rel])
        if r.returncode != 0:
            print(f"  !! git mv failed ({rel}): {r.stderr.strip()[:120]}"); return False
    else:
        shutil.move(str(src), str(dst))
    print(f"  moved: {rel} -> {dest_rel}")
    return True

def main():
    rows = list(csv.DictReader(open(REPO / "knowledge/scripts.csv", encoding="utf-8")))
    by_path = {r["path"].replace("\\", "/"): r for r in rows}

    doc = (REPO / "docs/analysis/2026-06-11_sweep_list.md").read_text(encoding="utf-8")
    cands = []
    for m in re.finditer(r"^## ([ABCD]) .*?\n(.*?)(?=^## |\Z)", doc, flags=re.M | re.S):
        cands += [(m.group(1), c) for c in re.findall(r"^- (.+)$", m.group(2), flags=re.M)]

    rescue_old = {old for old, *_ in RESCUES}
    print("=== RESCUES ===")
    for old, new, purpose, role in RESCUES:
        if new:
            do_move(old, new)
        target = new or old
        if not DRY:
            r = run([sys.executable, str(KB), "record", "scripts",
                     f"path={target}", f"purpose={purpose}", f"role={role}", "lifecycle=active"])
            print(f"  record: {target} -> {'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")

    print("=== ARCHIVE ===")
    n = 0
    for sec, rel in cands:
        rel = rel.rstrip("/")
        if rel in rescue_old or rel == "eval/render_example_images.py":
            continue
        if sec == "D":
            continue  # handled by GUI_LEGACY below (flet_app/ dir is KEPT)
        dest = f"{ARCH}/{rel}"
        if rel.endswith((".py", ".ipynb")) is False and (REPO / rel).is_dir():
            continue
        row = by_path.get(rel)
        if not do_move(rel, dest):
            continue
        n += 1
        if row and not DRY:
            run([sys.executable, str(KB), "set", "scripts", row["id"], f"path={dest}"])
            run([sys.executable, str(KB), "set", "scripts", row["id"], "lifecycle=archived"])
            print(f"    kb: {row['id']} -> archived")
    for rel in GUI_LEGACY:
        if do_move(rel, f"{ARCH}/{rel}"):
            n += 1
    print(f"=== done: {n} archived, {len(RESCUES)} rescued ===")
    if not DRY:
        v = run([sys.executable, str(KB), "validate"])
        print("validate:", (v.stdout or v.stderr).strip()[:200])

if __name__ == "__main__":
    main()
