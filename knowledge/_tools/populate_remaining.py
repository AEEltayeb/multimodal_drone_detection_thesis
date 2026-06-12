#!/usr/bin/env python3
"""populate_remaining.py — one-time, idempotent. Closes audit threads #9 (un-ledgered +
archive models) and #10 (superseded-script overlap groups), from Agent A/B census.

- IR detector version lineage V2-V6 (+ their §4.1 test-split F1) so ir_v4 eval target
  resolves and the IR progression is complete.
- Un-ledgered classifiers (optimal_v1, lean19_v2 sweep, split_v2/v3) and the RGB_M0 orphan.
- Superseded overlap-group scripts -> lifecycle=superseded + absorbed_into the winner, so
  /sweep has concrete safe-to-archive candidates later. Nothing is moved/deleted here.
Idempotent: skips existing ids.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402


def M(id, name, type, weights, prov, tags="drone-detection", train="", lifecycle="active",
      script="", prod="no"):
    return dict(id=id, name=name, type=type, purpose_tags=tags, trained_from_script=script,
                train_dataset=train, weights_path=weights, provenance_notes=prov,
                production=prod, lifecycle=lifecycle)

MODELS = [
    # IR detector version lineage (§4.1)
    M("ir_v2", "IR_dsetV2_merged_300ep", "ir_yolo", "archive/models/ir/IR_dsetV2_merged_300ep/best.pt",
      "IR V2 (merged/noisy corpus); F1 0.430. Only copy lives in archive/. §4.1", lifecycle="archived"),
    M("ir_v3", "IR finetune_v3", "ir_yolo", "models/ir/corrective_finetune/finetune_v3/weights/best.pt",
      "IR V3 (curated); F1 0.611; parent of v3b. §4.1", lifecycle="superseded"),
    M("ir_v4", "IR_dsetV4_300ep", "ir_yolo", "models/ir/IR_dsetV4_300ep/best.pt",
      "IR V4; F1 0.765 (precision jump from FP review). §4.1", lifecycle="superseded"),
    M("ir_v5", "IR_dsetV5_269ep", "ir_yolo", "models/ir/IR_dsetV5_269ep/best.pt",
      "IR V5; F1 0.737 (regression: new data added noise). §4.1", lifecycle="superseded"),
    M("ir_v6", "IR_dsetV6_118ep", "ir_yolo", "models/ir/IR_dsetV6_118ep/best.pt",
      "IR V6; F1 0.931 (comprehensive split cleanup). §4.1", lifecycle="superseded"),
    # Un-ledgered classifiers (Agent B orphans)
    M("optimal_v1", "optimal_v1 (8-feat)", "classifier", "models/routers/optimal_v1/model.joblib",
      "Un-ledgered: 'most promising 8-feature direction' from 2026-05-26 ft4 analysis doc. Not yet eval-rowed.",
      tags="confusion-filter;trust", lifecycle="active"),
    M("lean19_v2", "lean19_v2 (A/B/C/ABC sweep)", "classifier", "models/routers/lean19_v2_ABC/model.joblib",
      "Un-ledgered 4-way lean19 retrain sweep (variants A,B,C,ABC). No ledger row; needs eval.",
      tags="confusion-filter;trust", lifecycle="active"),
    M("split_v2v3", "split_v2/v3 paired+grayscale", "classifier", "models/routers/split_v3/paired/model_sa32_feats.joblib",
      "Un-ledgered paired-vs-grayscale split classifiers (eval-dashboard work); v2+v3 iterations.",
      tags="confusion-filter;trust", lifecycle="active"),
    M("rgb_m0_baseline", "RGB_M0_baseline", "rgb_yolo", "models/RGB_M0_baseline/best.pt",
      "ORPHAN: unreferenced RGB baseline snapshot under models/; naming predates Yolo26n_trained. VERIFY it is not a stale duplicate of baseline before any action.",
      lifecycle="active"),
]


def E(id, target, f1, note):
    return dict(id=id, target=target, config_id="ir_final_640", f1=f1, extra=note,
                source_script="eval/ir_version_comparison.py", date="2026-05-30")

EVALS = [
    E("ir_final_v2", "ir_v2", "0.430", "P0.458 R0.406; §4.1"),
    E("ir_final_v3", "ir_v3", "0.611", "§4.1"),
    E("ir_final_v5", "ir_v5", "0.737", "regression §4.1"),
    E("ir_final_v6", "ir_v6", "0.931", "§4.1"),
]


def SS(id, path, purpose, winner):
    return dict(id=id, path=path, purpose=purpose, role="one-off", lifecycle="superseded",
                absorbed_into=winner, last_run="")

SUPERSEDED = [
    SS("eval_distill_v2", "eval/distill_v2_domain_mixed.py", "V2 domain-mixed distillation (early lineage)", "eval_distill_v5_p3p5_ft4"),
    SS("eval_distill_v3", "eval/distill_v3_p3_features.py", "V3 P3-only distillation", "eval_distill_v5_p3p5_ft4"),
    SS("eval_distill_v4", "eval/distill_v4_p3p5_ft4.py", "V4 P3+P5 distillation (pre-V5)", "eval_distill_v5_p3p5_ft4"),
    SS("rgb_finetune_run", "training/finetune_run.py", "Early RGB fine-tune driver", "rgb_finetune_run_v2"),
    SS("rgb_finetune_unfrozen", "training/finetune_unfrozen_run.py", "Unfrozen-backbone fine-tune probe", "rgb_finetune_run_v2"),
    SS("rgb_finetune_and_eval", "training/finetune_and_eval.py", "Combined finetune+eval one-off", "rgb_finetune_selcom"),
    SS("rgb_unfinish_ckpt", "training/unfinish_ckpt.py", "Clear early-stop marker to resume training (ft3 hack)", "rgb_compare_selcom_ft"),
    SS("clf_extract_patches_v1", "classifier/extract_patches.py", "v1 crop extraction", "clf_extract_patches_v2"),
    SS("clf_train_lean13", "classifier/train_lean13_classifier.py", "13-feat trainer (deprecated: brightness fingerprint)", "clf_train_lean19"),
    SS("clf_train_lean17", "classifier/train_lean17_classifier.py", "17-feat trainer (deprecated: pos_x fingerprint)", "clf_train_lean19"),
    SS("eval_eval_video", "eval/eval_video.py", "Older generic video runner", "eval_eval_video_tests"),
    SS("nb_model_results_comparison_v2", "notebooks/model_results_comparison_v2.ipynb", "Dead: lacks IR Final + cross-eval matrix", "nb_model_results_comparison_v3"),
]


def _append(table, rows):
    cur = kb._read(table); existing = {r["id"] for r in cur}; added = 0
    for row in rows:
        if row["id"] in existing:
            continue
        errs = kb._validate_row(table, row)
        if errs:
            print(f"SKIP {table}.{row['id']}: {'; '.join(errs)}"); continue
        cur.append(row); existing.add(row["id"]); added += 1
    kb._write(table, cur)
    print(f"{table}: +{added} ({len(cur)} total)")


def main():
    _append("models", MODELS)
    _append("evals", EVALS)
    _append("scripts", SUPERSEDED)
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
