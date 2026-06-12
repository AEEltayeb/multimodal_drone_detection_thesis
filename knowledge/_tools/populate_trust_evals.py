#!/usr/bin/env python3
"""populate_trust_evals.py — one-time, idempotent. Adds trust-classifier evals so the
'trust' and 'confusion-filter' rankings become meaningful.

Sources (verified against docs/EVIDENCE_LEDGER.md 2026-05-30):
  - §5.1 unified 300-frame 3-way eval: acc + macro-F1 (F1m) per classifier
  - §7 confuser zoo (imgsz=1280, n=2633): classifier-stage (S2) fire rate = halluc proxy
Idempotent: skips existing ids.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

NEW_MODELS = [
    dict(id="retrained_v2_32feat", name="retrained_v2_32feat", type="classifier",
         purpose_tags="confusion-filter;trust", trained_from_script="classifier/train_classifier.py",
         train_dataset="retrained_v2 RGB fusion (32 feat)",
         weights_path="models/routers/retrained_v2_32feat/model.joblib",
         provenance_notes="Superseded: retrain calibration mismatch; collapses on OOD drone video (3way video_drone 0.280). §5",
         production="no", lifecycle="superseded"),
]

NEW_CONFIGS = [
    dict(id="clf_3way_300", dataset="3way_300frame", n_samples="300", imgsz="",
         scoring_rule="", conf_thr="",
         notes="100 each antiuav/svanstrom/video; 3-way drone/confuser/bg classifier eval. f1=macro-F1(F1m), extra=accuracy. §5.1"),
]

def _e(id, target, cfg, f1="", halluc="", acc="", src="", note=""):
    extra = note
    if acc:
        extra = (f"acc={acc}; " + note).strip()
    return dict(id=id, target=target, config_id=cfg, f1=f1, halluc_rate=halluc,
                extra=extra, source_script=src, date="2026-05-30")

NEW_EVALS = [
    # §5.1 3-way (f1 = macro-F1, acc in extra)
    _e("clf3_lean19", "lean19", "clf_3way_300", "0.978", acc="0.990", src="eval/eval_classifier_3way.py", note="§5.1"),
    _e("clf3_lean13", "lean13", "clf_3way_300", "0.979", acc="0.987", src="eval/eval_classifier_3way.py", note="§5.1 (best F1m but scene-fingerprint overfit)"),
    _e("clf3_lean10", "lean10", "clf_3way_300", "0.963", acc="0.980", src="eval/eval_classifier_3way.py", note="§5.1"),
    _e("clf3_lean17", "lean17", "clf_3way_300", "0.958", acc="0.977", src="eval/eval_classifier_3way.py", note="§5.1"),
    _e("clf3_control40", "control40", "clf_3way_300", "0.920", acc="0.953", src="eval/eval_classifier_3way.py", note="§5.1"),
    _e("clf3_retrainedv2", "retrained_v2_32feat", "clf_3way_300", "0.842", acc="0.923", src="eval/eval_classifier_3way.py", note="§5.1 video_drone collapse 0.280"),
    # §7 confuser zoo S2 (classifier-stage) fire rate as halluc proxy
    _e("clfzoo_fnfn", "fusion_no_fn_v1.1", "confuser_zoo_1280", halluc="0.016", src="eval/cumulative_halluc.py", note="S2 classifier-stage fire; S3=0.008. §7 (open-world safest)"),
    _e("clfzoo_sa32", "sa32", "confuser_zoo_1280", halluc="0.205", src="eval/cumulative_halluc.py", note="S2 classifier-stage fire; S3=0.103; 13x fnfn on OOD zoo. §7"),
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
    _append("models", NEW_MODELS)
    _append("eval_configs", NEW_CONFIGS)
    _append("evals", NEW_EVALS)
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
