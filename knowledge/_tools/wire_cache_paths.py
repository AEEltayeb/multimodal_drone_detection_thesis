#!/usr/bin/env python3
"""wire_cache_paths.py — one-time, idempotent. Sets cache_path on eval rows to the
result/cache artifact named in EVIDENCE_LEDGER's source columns, so kb.py check-eval
can report cache liveness (LIVE/MISSING) and the rerun-guard works.

Only fills cache_path where currently blank. Paths are repo-relative; many point to
gitignored result dirs (local-only) — check-eval will mark those MISSING if absent,
which is the honest signal.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

# eval id prefix / exact id -> cache_path
CACHE = {
    "rgb_svan_": "eval/results/_failure_diagnosis/svanstrom_1280_by_category.csv",
    "rgb_antiuav_": "eval/results/_ablation/2026-05-18T22-48-19/master.csv",
    "selcom_baseline_1280": "runs/rgb_finetune_eval/Yolo26n_selcom_mixed_ft2_1280/comparison.json",
    "selcom_ft2_1280": "runs/rgb_finetune_eval/Yolo26n_selcom_mixed_ft2_1280/comparison.json",
    "selcom_ft3_1280": "runs/rgb_finetune_eval/compare_selcom_ft2_vs_ft3/Yolo26n_selcom_mixed_ft3_1280.json",
    "ir_final_": "eval/results/ir_version_comparison/ir_comparison_test_640_2026-05-16T20-04-39.csv",
    "v5_": "eval/results/_v5_head_to_head_pure_1x8/comparison.md",
    "clf3_": "docs/analysis/full_pipeline_ablations/csv/classifier_3way.csv",
    "clfzoo_fnfn": "eval/results/_cumulative_halluc/confuser_fusion_no_fn_model_v1.1/summary.json",
    "clfzoo_sa32": "eval/results/_cumulative_halluc/confuser_sa32/summary.json",
}


def _match(eid):
    if eid in CACHE:
        return CACHE[eid]
    for k, v in CACHE.items():
        if k.endswith("_") and eid.startswith(k):
            return v
    return None


def main():
    rows = kb._read("evals")
    n = live = 0
    for r in rows:
        if (r.get("cache_path") or "").strip():
            continue
        cp = _match(r["id"])
        if cp:
            r["cache_path"] = cp
            n += 1
            if (kb.REPO / cp).exists():
                live += 1
    kb._write("evals", rows)
    kb._regen_views()
    print(f"wired cache_path on {n} eval rows ({live} LIVE on disk, {n - live} MISSING/local-only)")


if __name__ == "__main__":
    main()
