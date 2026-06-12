#!/usr/bin/env python3
"""migrate_evidence_ledger.py — one-time, idempotent migration of the headline,
decision-relevant content of docs/EVIDENCE_LEDGER.md into the knowledge/ tables.

Scope: the discriminating metrics + key findings that drive the thesis and the
ranking/comparison views — NOT every historical sub-table. EVIDENCE_LEDGER.md stays
`active` until a fuller pass is done. Numbers verified against the ledger 2026-05-30.

Idempotent: skips any row whose id already exists, so it can be re-run safely and
will not clobber rows added by hand (e.g. the mri-* rows).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

L = "docs/EVIDENCE_LEDGER.md"

EVAL_CONFIGS = [
    dict(id="svan_iop_1280", dataset="svanstrom", n_samples="28710", imgsz="1280",
         scoring_rule="iop", conf_thr="0.25", notes="IoP@0.5; full stride=1; drone-class. src "+L+" §3,§7"),
    dict(id="svan_iop_1280_s9", dataset="svanstrom", n_samples="3190", imgsz="1280",
         scoring_rule="iop", conf_thr="0.25", notes="stride=9 sample; used for head-to-head comparisons. §7,§13.1"),
    dict(id="antiuav_iou_1280_s5", dataset="antiuav", n_samples="", imgsz="1280",
         scoring_rule="iou", conf_thr="", notes="stride=5; 2026-05-18 ablation. §3.2"),
    dict(id="antiuav_iou_640_s5", dataset="antiuav", n_samples="17075", imgsz="640",
         scoring_rule="iou", conf_thr="", notes="stride=5; V5 head-to-head surface. §13.1"),
    dict(id="ir_final_640", dataset="ir_dset_final", n_samples="9612", imgsz="640",
         scoring_rule="iou", conf_thr="", notes="test split; ir_version_comparison 2026-05-16. §4.1"),
    dict(id="selcom_iop_1280", dataset="selcom_mixed_ft2_val", n_samples="311", imgsz="1280",
         scoring_rule="iop", conf_thr="0.25", notes="295 GT boxes; pure-selcom held-out val. §3.4,§13.1"),
    dict(id="confuser_zoo_1280", dataset="rgb_confusers_merged", n_samples="2633", imgsz="1280",
         scoring_rule="", conf_thr="", notes="no GT; every detection=FP (fire rate). §7"),
    dict(id="confuser_test_640", dataset="confuser_test", n_samples="2633", imgsz="640",
         scoring_rule="", conf_thr="", notes="no GT; halluc/img. V5 head-to-head. §13.1"),
    dict(id="rgb_dataset_iou_640", dataset="rgb_dataset_test", n_samples="507", imgsz="640",
         scoring_rule="iou", conf_thr="", notes="stride=34 sample; photo-style content. §13.1,§13.4"),
]

MODELS = [
    # RGB YOLO
    dict(id="baseline", name="Yolo26n_trained", type="rgb_yolo", purpose_tags="drone-detection",
         trained_from_script="", train_dataset="general RGB drone corpus (+ drone-vs-bird subset)",
         weights_path="models/rgb/Yolo26n_trained/weights/best.pt",
         provenance_notes="Base RGB; best small-drone recall @1280; base model for selcom fine-tunes. §1,§3.1",
         production="yes", lifecycle="active"),
    dict(id="hardneg_v3more", name="Yolo26n_hardneg_v3more", type="rgb_yolo",
         purpose_tags="drone-detection;confusion-filter", trained_from_script="",
         train_dataset="+ airplane/heli/bird hard-negs (Svanstrom split)", weights_path="",
         provenance_notes="Lower confuser halluc than baseline; near-baseline recall. §3.1,§3.3",
         production="no", lifecycle="active"),
    dict(id="retrained_v2", name="Yolo26n_retrained_v2", type="rgb_yolo", purpose_tags="drone-detection",
         trained_from_script="", train_dataset="aggressive confuser negatives", weights_path="",
         provenance_notes="Recall collapse on Svanstrom (R=0.306 @1280); disqualified for production. §3.1",
         production="no", lifecycle="active"),
    dict(id="selcom_mixed_ft2_1280", name="Yolo26n_selcom_mixed_ft2_1280", type="rgb_yolo",
         purpose_tags="drone-detection", trained_from_script="training/finetune_selcom.py",
         train_dataset="80% general RGB + 20% selcom CCTV (pure-selcom val)", weights_path="",
         provenance_notes="CCTV production weights; F1 0.580 @1280 selcom_val. §3.4", production="no",
         lifecycle="active"),
    dict(id="selcom_mixed_ft3_1280", name="Yolo26n_selcom_mixed_ft3_1280", type="rgb_yolo",
         purpose_tags="drone-detection", trained_from_script="training/compare_selcom_ft.py",
         train_dataset="ft2 train + 50/50 selcom-baseline val", weights_path="",
         provenance_notes="Candidate CCTV weights; F1 0.619; baseline regression closed. §3.4", production="no",
         lifecycle="active"),
    dict(id="ft4", name="Yolo26n_selcom_confuser_ft4_1280 (FT4 R3)", type="rgb_yolo",
         purpose_tags="drone-detection", trained_from_script="", train_dataset="selcom + confuser hard-negs",
         weights_path="models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt",
         provenance_notes="Current detector in the V5 verifier stack. §13", production="yes", lifecycle="active"),
    # IR YOLO
    dict(id="ir_v3b", name="IR finetune_v3b", type="ir_yolo", purpose_tags="drone-detection",
         trained_from_script="", train_dataset="IR_dset_final (+ HITL hard-negs)",
         weights_path="models/ir/corrective_finetune/finetune_v3b/weights/best.pt",
         provenance_notes="Production IR; F1 0.967 on IR_dset_final; 2-epoch corrective FT on Final. §1,§4.1",
         production="yes", lifecycle="active"),
    dict(id="ir_final", name="IR Final", type="ir_yolo", purpose_tags="drone-detection",
         trained_from_script="eval/ir_version_comparison.py", train_dataset="IR_dset_final", weights_path="",
         provenance_notes="Base for v3b; F1 0.967 (≈ v3b). §4.1", production="no", lifecycle="active"),
    # Trust classifiers
    dict(id="sa32", name="scene_aware_v3more_32feat", type="classifier",
         purpose_tags="confusion-filter;trust", trained_from_script="", train_dataset="fusion features (32)",
         weights_path="", provenance_notes="Production trust classifier; best on Svanstrom-distribution; fires 13x more than fnfn on OOD zoo. §1,§7",
         production="yes", lifecycle="active"),
    dict(id="fusion_no_fn_v1.1", name="fusion_no_fn_model_v1.1", type="classifier",
         purpose_tags="confusion-filter;trust", trained_from_script="", train_dataset="baseline RGB fusion (40)",
         weights_path="", provenance_notes="Open-world fallback; most conservative on OOD confuser zoo (S2=0.016). §1,§7",
         production="no", lifecycle="active"),
    dict(id="control40", name="control_v3more_40feat", type="classifier",
         purpose_tags="confusion-filter;trust", trained_from_script="", train_dataset="40 feat (v3more scene-aware)",
         weights_path="", provenance_notes="Deprecated: best raw drone metrics but worst confuser rejection. §5.2,§7",
         production="no", lifecycle="active"),
    dict(id="lean19", name="lean19", type="classifier", purpose_tags="confusion-filter;trust",
         trained_from_script="classifier/train_lean19_classifier.py", train_dataset="lean19 fusion dataset (19 feat)",
         weights_path="models/routers/lean19", provenance_notes="Smallest train-test gap (41pp); selcom_1280 RGB + v3b IR. §5.1,§5.2",
         production="no", lifecycle="active"),
    dict(id="lean10", name="lean10", type="classifier", purpose_tags="confusion-filter;trust",
         trained_from_script="classifier/train_lean10_classifier.py", train_dataset="lean fusion (10 feat, no brightness)",
         weights_path="models/routers/lean10", provenance_notes="OOD-robust with ~half compute; no scene-fingerprint features. §5.2",
         production="no", lifecycle="active"),
    dict(id="lean13", name="lean13", type="classifier", purpose_tags="confusion-filter;trust",
         trained_from_script="classifier/train_lean13_classifier.py", train_dataset="lean fusion (13 feat)",
         weights_path="", provenance_notes="Deprecated: brightness scalars memorize scene fingerprints (74pp gap). §5.3",
         production="no", lifecycle="superseded"),
    dict(id="lean17", name="lean17", type="classifier", purpose_tags="confusion-filter;trust",
         trained_from_script="classifier/train_lean17_classifier.py", train_dataset="lean fusion (17 feat, +pos_x)",
         weights_path="", provenance_notes="Deprecated: pos_x acts as scene fingerprint (top feature). §5.3",
         production="no", lifecycle="superseded"),
    # Verifiers
    dict(id="patch_v2", name="confuser_filter4 v2 (rgb+ir)", type="verifier",
         purpose_tags="confusion-filter", trained_from_script="",
         train_dataset="45,917 RGB+IR patches (drone/bird/airplane/heli; NO CCTV)",
         weights_path="models/patches/confuser_filter4_rgb_v2_backup.pt",
         provenance_notes="Production patch verifier (fallback under V5 proposal); neutral on selcom; 70-112 ms/det. v3 over-aggressive, v4≈v2. §6,§13",
         production="no", lifecycle="active"),
    dict(id="mlp_v5", name="mlp_v5 pure_1x8", type="mlp", purpose_tags="confusion-filter",
         trained_from_script="eval/distill_v5_swap_selcom.py",
         train_dataset="distilled FT4 p3+p5 ROI features (517-D, pure selcom CCTV)",
         weights_path="models/verifiers/rgb_v5/mlp_v5.pt",
         provenance_notes="PRODUCTION-CANDIDATE verifier; +8-10pp Svan F1, 7.5x cleaner confuser, 46-72x faster/det; ship per-frame. §13",
         production="yes", lifecycle="active"),
]

# evals: (id, target, config_id, P, R, F1, halluc, source_script, extra)
def _ev(id, target, cfg, p="", r="", f1="", halluc="", src="", extra=""):
    return dict(id=id, target=target, config_id=cfg, precision=p, recall=r, f1=f1,
                halluc_rate=halluc, source_script=src, extra=extra, date="2026-05-30")

EVALS = [
    # RGB on Svanstrom @1280 (drone-class)
    _ev("rgb_svan_baseline", "baseline", "svan_iop_1280", "0.940", "0.961", "0.950",
        src="eval/diagnose_failures.py", extra="S1 RGB-alone, full 28710 §7"),
    _ev("rgb_svan_hardneg", "hardneg_v3more", "svan_iop_1280", "0.941", "0.950", "0.946",
        src="eval/diagnose_failures_all.py", extra="det rate 95.7% §3.1"),
    _ev("rgb_svan_retrainedv2", "retrained_v2", "svan_iop_1280", "0.943", "0.306", "0.462",
        src="eval/diagnose_failures_all.py", extra="det rate 30.8%; recall collapse §3.1"),
    # RGB on Anti-UAV @1280
    _ev("rgb_antiuav_baseline", "baseline", "antiuav_iou_1280_s5", "0.9922", "0.9950", "0.9936",
        src="eval/ablate.py", extra="TP3178 FP25 FN16 §3.2"),
    _ev("rgb_antiuav_retrainedv2", "retrained_v2", "antiuav_iou_1280_s5", "0.9922", "0.9950", "0.9936",
        src="eval/ablate.py", extra="identical to baseline; AntiUAV saturated §3.2"),
    # selcom CCTV fine-tunes @1280
    _ev("selcom_baseline_1280", "baseline", "selcom_iop_1280", "0.413", "0.088", "0.145",
        src="training/finetune_selcom.py", extra="pre-finetune baseline §3.4"),
    _ev("selcom_ft2_1280", "selcom_mixed_ft2_1280", "selcom_iop_1280", "0.762", "0.468", "0.580",
        src="training/finetune_selcom.py", extra="CCTV prod weights §3.4"),
    _ev("selcom_ft3_1280", "selcom_mixed_ft3_1280", "selcom_iop_1280", "0.847", "0.488", "0.619",
        src="training/compare_selcom_ft.py", extra="candidate; +3.4pp F1 vs ft2 §3.4"),
    # IR version comparison @640 test split
    _ev("ir_final_final", "ir_final", "ir_final_640", "0.955", "0.980", "0.967",
        src="eval/ir_version_comparison.py", extra="mAP50=0.977 §4.1"),
    _ev("ir_final_v3b", "ir_v3b", "ir_final_640", "0.957", "0.977", "0.967",
        src="eval/ir_version_comparison.py", extra="mAP50=0.972; production §4.1"),
    _ev("ir_final_v4", "ir_v4", "ir_final_640", "0.895", "0.669", "0.765",
        src="eval/ir_version_comparison.py", extra="superseded §4.1"),
    # V5 head-to-head §13.1: bare ft4 / patch v2 / mlp v5 across 5 surfaces
    _ev("v5_svan_bare", "ft4", "svan_iop_1280_s9", "0.4425", "0.9140", "0.5963", "0.4699",
        src="eval/eval_v4_vs_patch.py", extra="bare FT4 §13.1"),
    _ev("v5_svan_patch", "patch_v2", "svan_iop_1280_s9", "0.6858", "0.8717", "0.7677", "0.1630",
        src="eval/eval_v4_vs_patch.py", extra="FT4+patch_v2 @0.5 §13.1"),
    _ev("v5_svan_mlp", "mlp_v5", "svan_iop_1280_s9", "0.9031", "0.8379", "0.8693", "0.0367",
        src="eval/eval_v4_vs_patch.py", extra="FT4+V5 @0.5 §13.1"),
    _ev("v5_antiuav_bare", "ft4", "antiuav_iou_640_s5", "0.9881", "0.9836", "0.9859", "0.0111",
        src="eval/eval_v4_vs_patch.py", extra="bare FT4 §13.1"),
    _ev("v5_antiuav_patch", "patch_v2", "antiuav_iou_640_s5", "0.9881", "0.9836", "0.9859", "0.0111",
        src="eval/eval_v4_vs_patch.py", extra="=bare §13.1"),
    _ev("v5_antiuav_mlp", "mlp_v5", "antiuav_iou_640_s5", "0.9894", "0.9811", "0.9853", "0.0098",
        src="eval/eval_v4_vs_patch.py", extra="ties §13.1"),
    _ev("v5_selcom_bare", "ft4", "selcom_iop_1280", "0.8581", "0.4508", "0.5911", "0.0707",
        src="eval/eval_v4_vs_patch.py", extra="bare FT4 §13.1"),
    _ev("v5_selcom_patch", "patch_v2", "selcom_iop_1280", "0.8581", "0.4508", "0.5911", "0.0707",
        src="eval/eval_v4_vs_patch.py", extra="=bare (no CCTV exposure) §13.3"),
    _ev("v5_selcom_mlp", "mlp_v5", "selcom_iop_1280", "0.9562", "0.4441", "0.6065", "0.0193",
        src="eval/eval_v4_vs_patch.py", extra="+1.5pp vs bare §13.1"),
    _ev("v5_rgbds_bare", "ft4", "rgb_dataset_iou_640", "0.9650", "0.8956", "0.9290", "0.0276",
        src="eval/eval_v4_vs_patch.py", extra="bare FT4 §13.1"),
    _ev("v5_rgbds_patch", "patch_v2", "rgb_dataset_iou_640", "0.9657", "0.8492", "0.9037", "0.0256",
        src="eval/eval_v4_vs_patch.py", extra="§13.1"),
    _ev("v5_rgbds_mlp", "mlp_v5", "rgb_dataset_iou_640", "0.9828", "0.6636", "0.7922", "0.0099",
        src="eval/eval_v4_vs_patch.py", extra="recall ceiling carve-out §13.4"),
    _ev("v5_confuser_bare", "ft4", "confuser_test_640", halluc="0.3171",
        src="eval/eval_v4_vs_patch.py", extra="no GT; FP=835 §13.1"),
    _ev("v5_confuser_patch", "patch_v2", "confuser_test_640", halluc="0.1071",
        src="eval/eval_v4_vs_patch.py", extra="FP=282 §13.1"),
    _ev("v5_confuser_mlp", "mlp_v5", "confuser_test_640", halluc="0.0080",
        src="eval/eval_v4_vs_patch.py", extra="FP=21; 97% cleaner §13.1"),
]

def _f(id, claim, outcome, evidence="", condition="", contradicts="", contrib="", status="confirmed", notes=""):
    return dict(id=id, date="2026-05-30", claim=claim, outcome=outcome, condition=condition,
                evidence_evals=evidence, contradicts=contradicts, thesis_contribution=contrib,
                status=status, notes=notes)

LEDGER = [
    _f("selcom-imgsz-win", "selcom CCTV recall doubles from imgsz 640->1280 (0.244->0.468) while precision rises (0.59->0.76); imgsz crosses YOLO's small-object floor, preprocessing does not help",
       "supported", "selcom_ft2_1280", contrib="resolution-driven small-drone recovery", notes="§3.4; preprocess sweep REPORT.md"),
    _f("retrainedv2-recall-collapse", "aggressive confuser-negative retraining (retrained_v2) collapses drone recall on small-drone Svanstrom (R=0.306 @1280); disqualified for production",
       "supported", "rgb_svan_retrainedv2;rgb_antiuav_retrainedv2", condition="visible only on small-drone Svanstrom; Anti-UAV saturation hides it", notes="§3.1"),
    _f("antiuav-saturated", "Anti-UAV is saturated (F1~0.994 for both RGB variants, identical TP/FP/FN) - a sanity floor, not a discriminating benchmark",
       "supported", "rgb_antiuav_baseline;rgb_antiuav_retrainedv2", notes="§3.2; Svanstrom is the discriminating surface"),
    _f("scene-fingerprint-overfit", "per-clip brightness scalars and pos_x act as scene fingerprints under sequence-split; dropping them (lean13->lean10) recovers +18-26pp on held-out drone clips",
       "supported", "", contrib="scene-fingerprint overfitting in fusion classifiers", notes="§5.2,§5.3; lean13/17 deprecated"),
    _f("trust-classifier-conditional", "no single trust classifier dominates: sa32 wins Svanstrom-distribution metrics but fires 13x more than fusion_no_fn on the OOD confuser zoo",
       "conditional", "", condition="ship sa32 for calibrated/known-scene deploy; fusion_no_fn for open-world", contrib="condition-dependent trust calibration", notes="§7"),
    _f("v5-beats-patch", "V5 distillation MLP verifier beats patch v2 on 3/5 surfaces (Svan +8.6pp, selcom +1.5pp, confuser 7.5x cleaner), ties Anti-UAV, suppresses halluc on all 5, and is 46-72x faster per detection",
       "supported", "v5_svan_mlp;v5_svan_patch;v5_selcom_mlp;v5_confuser_mlp;v5_antiuav_mlp", contrib="distilled feature-space verifier (production candidate)", notes="§13.1,§13.8"),
    _f("v5-rgbds-ceiling", "V5 has a structural recall ceiling (~0.77) on rgb_dataset_test (photo-style content), -7.8pp F1 vs bare; the Svanstrom-weight rebalance is a measured no-op",
       "supported", "v5_rgbds_mlp;v5_rgbds_bare", contradicts="v5-beats-patch", notes="§13.4,§13.8; photo-style carve-out -> patch v2 fallback there"),
    _f("patch-v2-neutral-selcom", "patch v2 is identical to bare FT4 on selcom (trained with no CCTV exposure -> votes 'other'); V5 must beat bare, not patch, on this surface - and it does (+1.5pp)",
       "supported", "v5_selcom_patch;v5_selcom_bare;v5_selcom_mlp", contrib="patch-verifier blind spot motivates distillation", notes="§13.3"),
    _f("v5-ship-per-frame", "V5 verifier ships per-frame, not alert-gated: alert-gating saves <0.2 ms/frame but costs -4.0pp Svanstrom F1",
       "supported", "", notes="§13.7"),
    _f("ir-version-progression", "IR detector improved V2->Final from F1 0.430->0.967 via FP review and split cleanup; v3b ~ Final (numerically indistinguishable on the test split)",
       "supported", "ir_final_final;ir_final_v3b;ir_final_v4", notes="§4.1"),
    _f("v5-lda-separability", "FT4 fused p3+p5 ROI features are linearly separable: LDA 0.949 binary (drone vs confuser) and 0.954 4-class by confuser category - grounding the V5 verifier in feature geometry",
       "supported", "", contrib="linear separability of detector ROI features", notes="§13.9"),
    _f("svanstrom-category-in-filename", "Svanstrom confuser category lives in the filename prefix (IR_BIRD_/IR_AIRPLANE_/IR_HELICOPTER_), NOT the GT class (GT is drone-only); mine confusers by prefix with no IoU/IoP filter",
       "supported", "", notes="§13.9 methodology; corrects an earlier wrong 'Svanstrom cannot supply confusers' claim"),
    _f("ir-grayscale-fallback", "IR model on grayscale-RGB beats IR-on-raw-RGB on drone video (F1 0.664 vs 0.298) and beats every RGB model on the seagull-attack clip; cross-modal grayscale fallback validated on drone-positive data",
       "supported", "", contrib="cross-modal grayscale fallback", notes="§9; project_ir_grayscale_video_eval memory"),
    _f("cascade-confuser-collapse", "the full cascade (RGB->trust classifier->patch v2) collapses Svanstrom confuser FPs ~16149->271 (98.3%) at a 14.4pp drone-recall cost; patch_thr 0.5->0.9 monotonically trades recall for FPs",
       "supported", "", notes="§7; operating-point choice = recall (defense) vs precision (alarm fatigue)"),
]


def _apply(table, rows):
    existing = {r["id"] for r in kb._read(table)}
    cur = kb._read(table)
    added = 0
    for row in rows:
        if row["id"] in existing:
            continue
        errs = kb._validate_row(table, row)
        if errs:
            print(f"SKIP {table}.{row['id']}: {'; '.join(errs)}")
            continue
        cur.append(row)
        existing.add(row["id"])
        added += 1
    kb._write(table, cur)
    print(f"{table}: +{added} rows ({len(cur)} total)")


def main():
    _apply("eval_configs", EVAL_CONFIGS)
    _apply("models", MODELS)
    _apply("evals", EVALS)
    _apply("ledger", LEDGER)
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
