#!/usr/bin/env python3
"""
reorg_2026_06.py — one-shot path rewrite for the 2026-06-11 repo reorganization.

Applies the move-map prefixes (see knowledge/DECISIONS.md 2026-06-11 reorg entry)
to LIVE code + configs only. Historical artifacts (archive/, eval/results/,
thesis_eval/{cache,results}/, docs/analysis caches, *.md, *.tex, result
manifests) are deliberately untouched — they are provenance records.

Idempotent: re-running after success finds nothing to replace.
Usage: py knowledge/_tools/reorg_2026_06.py [--dry-run]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# Ordered longest-prefix-first. Pure ASCII so we can operate on bytes.
MAP = [
    ("RGB model/dataset preparation/", "training/dataset_preparation/"),
    ("eval/results/_v5_selcom_pure_1x8/classifiers/", "models/verifiers/rgb_v5/"),
    ("eval/results/_routing_pipeline_cmp/robust8", "models/routers/robust8"),
    ("eval/results/_routing_pipeline_cmp/new_router", "models/routers/new_router"),
    ("mri/results/ir_aligned/classifiers/", "models/verifiers/ir_aligned/"),
    ("classifier/fusion_models/", "models/routers/"),
    ("classifier/runs/patches/", "models/patches/"),
    ("runs/corrective_finetune/", "models/ir/corrective_finetune/"),
    ("models/IR_", "models/ir/IR_"),
    ("RGB model/Yolo26n_", "models/rgb/Yolo26n_"),
    ("RGB model/", "training/"),
    ("scripts/review_labels_gui.py", "label_reviewer/review_labels_gui.py"),
    ("scripts/label_reviewer", "label_reviewer"),
    ("ir_gui/", "gui/"),
    ("ir_gui.", "gui."),  # module-style refs (comments/import strings)
]

def variants(old: str, new: str):
    """fwd-slash, single-backslash, and JSON-escaped double-backslash forms."""
    yield old.encode(), new.encode()
    o_bs, n_bs = old.replace("/", "\\"), new.replace("/", "\\")
    if o_bs != old:
        yield o_bs.encode(), n_bs.encode()
        yield o_bs.replace("\\", "\\\\").encode(), n_bs.replace("\\", "\\\\").encode()

EXCLUDE_PARTS = {".git", "archive", "__pycache__", ".cache", "node_modules",
                 "models", "datasets"}
EXCLUDE_UNDER = [("eval", "results"), ("thesis_eval", "results"),
                 ("thesis_eval", "cache"), ("thesis_eval", "cache_conf005"),
                 ("docs", "analysis", "full_pipeline_ablations")]
EXTS = {".py", ".ps1", ".yaml", ".yml"}
EXTRA_FILES = [REPO / "gui" / "fusion_settings.json", REPO / "gui" / "settings.json"]

def in_scope(p: Path) -> bool:
    if p.resolve() == Path(__file__).resolve():
        return False  # never rewrite our own MAP strings
    rel = p.relative_to(REPO).parts
    if any(part in EXCLUDE_PARTS for part in rel[:-1]):
        return False
    for pre in EXCLUDE_UNDER:
        if rel[:len(pre)] == pre:
            return False
    return p.suffix.lower() in EXTS

def main():
    dry = "--dry-run" in sys.argv
    targets = [p for p in REPO.rglob("*") if p.is_file() and in_scope(p)]
    targets += [p for p in EXTRA_FILES if p.exists()]
    pairs = [v for old, new in MAP for v in variants(old, new)]
    changed, total = [], 0
    for p in targets:
        data = p.read_bytes()
        orig = data
        hits = 0
        for ob, nb in pairs:
            n = data.count(ob)
            if n:
                data = data.replace(ob, nb)
                hits += n
        if hits:
            changed.append((p.relative_to(REPO), hits))
            total += hits
            if not dry:
                p.write_bytes(data)
    print(f"{'DRY-RUN ' if dry else ''}rewrote {total} occurrences in {len(changed)} files:")
    for rel, hits in sorted(changed, key=lambda x: -x[1]):
        print(f"  {hits:4d}  {rel}")

if __name__ == "__main__":
    main()
