#!/usr/bin/env python3
"""apply_classifier_census.py — one-time, idempotent. Applies the classifier-zoo census
agent's COMPLETE inventory: phase1/2 ablations, reliability + failure sub-models, the
fusion_no_fn family, lean_yt + lean19_v2 arms, patch-CNN lineage, V5/V4 MLP variants.

Metrics go on a dedicated `clf_own_holdout` eval_config = EACH classifier's own train-time
test split, explicitly NOT cross-comparable (kept out of the comparison views' meaning).
Dedups flagged by the agent are marked superseded+absorbed_into. No-weight stubs skipped.
Idempotent: skips existing ids.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402


def M(id, name, weights, prov, type="classifier", tags="confusion-filter;trust",
      lifecycle="active", absorbed=""):
    # models schema has no absorbed_into; fold dedup pointer into the note
    if absorbed:
        prov = f"{prov} [absorbed_into={absorbed}]"
    return dict(id=id, name=name, type=type, purpose_tags=tags, trained_from_script="",
                train_dataset="", weights_path=weights, provenance_notes=prov,
                production="no", lifecycle=lifecycle)


MODELS = [
    # A. phase-1 dataset-ablation trust classifiers (14-feat)
    M("clf_phase1_baseline", "phase1 baseline", "classifier/runs/phase1/classifier_baseline.joblib",
      "14-feat fusion full corpus; test F1 0.9935; max_conf_ir imp 0.81"),
    M("clf_phase1_anti_uav_only", "phase1 anti_uav_only", "classifier/runs/phase1/classifier_anti_uav_only.joblib",
      "Anti-UAV only; test F1 0.9968 (in-domain)"),
    M("clf_phase1_svanstrom_only", "phase1 svanstrom_only", "classifier/runs/phase1/classifier_svanstrom_only.joblib",
      "Svanstrom only; test F1 1.0 (overfit/in-domain)"),
    M("clf_phase1_no_svan_drones", "phase1 no_svan_drones", "classifier/runs/phase1/classifier_no_svan_drones.joblib",
      "Svan drones removed; test F1 0.9892"),
    # B. phase-2 IR-suppression
    M("clf_phase2_ir_suppressed", "phase2 ir_suppressed", "classifier/runs/phase2/classifier_ir_suppressed.joblib",
      "trained w/ 30% IR-dropout aug; F1 0.9845; degrades gracefully under IR-suppression (sup100 0.588 vs baseline 0.0)"),
    # C. per-modality reliability gates
    M("rgb_reliability", "rgb_reliability_model", "classifier/runs/reliability/rgb_reliability_model.joblib",
      "12-feat RGB trust gate; F1 0.888 AUC 0.964; conf=92% imp; collapses on Svan (0.085) -> motivated fusion router", tags="trust"),
    M("ir_reliability", "ir_reliability_model", "classifier/runs/reliability/ir_reliability_model.joblib",
      "12-feat IR trust gate; F1 0.917 AUC 0.981; strong Svan IR (0.966)", tags="trust"),
    # D. failure models (FP/FN predictors)
    M("rgb_fp_model", "rgb_fp_model", "classifier/runs/reliability/failure_models/rgb_fp_model.joblib",
      "predicts RGB FPs from scene stats; F1 0.677 AUC 0.950; blur+edge dominate", tags="trust"),
    M("rgb_fn_model", "rgb_fn_model", "classifier/runs/reliability/failure_models/rgb_fn_model.joblib",
      "predicts RGB missed-detections; F1 0.796 AUC 0.965", tags="trust"),
    M("ir_fp_model", "ir_fp_model", "classifier/runs/reliability/failure_models/ir_fp_model.joblib",
      "predicts IR FPs; F1 0.392 AUC 0.781 (thermal FP harder)", tags="trust"),
    M("ir_fn_model", "ir_fn_model", "classifier/runs/reliability/failure_models/ir_fn_model.joblib",
      "predicts IR missed-detections; F1 0.766 AUC 0.947", tags="trust"),
    # E. fusion_no_fn family (4-class routers)
    M("fusion_no_fn_base", "fusion_no_fn (unversioned)", "classifier/runs/reliability/fusion/fusion_no_fn_model.joblib",
      "DUPLICATE of fusion_no_fn_v1.1 (identical f1_macro 0.9684)", lifecycle="superseded", absorbed="fusion_no_fn_v1.1"),
    M("fusion_no_fn_original", "fusion_no_fn_original", "classifier/runs/reliability/fusion/fusion_no_fn_model_original.joblib",
      "pre-v1.1 snapshot; superseded by v1.1", lifecycle="superseded", absorbed="fusion_no_fn_v1.1"),
    M("fusion_no_fn_v3more", "fusion_no_fn_v3more", "classifier/runs/reliability/fusion/fusion_no_fn_v3more_model.joblib",
      "40-feat on v3more dets; f1_macro 0.9490; Svan 0.799; det-presence flags dominate"),
    M("fusion_no_fn_v3more_no_det", "fusion_no_fn_v3more_no_det_signals", "classifier/runs/reliability/fusion/fusion_no_fn_v3more_no_det_signals_model.joblib",
      "32-feat (det flags removed); f1_macro 0.9493 (no loss) -> det-flags redundant"),
    M("fusion_no_fn_v3more_capped30k", "fusion_no_fn_v3more_capped30k", "classifier/runs/reliability/fusion/fusion_no_fn_v3more_capped30k_model.joblib",
      "antiuav capped ~30k (rebalanced); f1_macro 0.9383; Svan 0.743"),
    M("fusion_no_fn_v3more_capped_no_flags", "fusion_no_fn_v3more_capped30k_no_flags", "classifier/runs/reliability/fusion/fusion_no_fn_v3more_capped30k_no_flags_model.joblib",
      "capped + drops detected/only-detect flags; f1_macro 0.9370"),
    M("fusion_no_fn_v3more_gray_aug", "fusion_no_fn_v3more_gray_aug", "classifier/runs/reliability/fusion/fusion_no_fn_v3more_gray_aug_model.joblib",
      "32-feat + synthetic-grayscale aug; f1_macro 0.9369; Svan 0.739"),
    M("fusion_no_fn_v3more_realgray", "fusion_no_fn_v3more_realgray", "classifier/runs/reliability/fusion/fusion_no_fn_v3more_realgray_model.joblib",
      "32-feat + REAL grayscale; f1_macro 0.9092 (lowest) -> real-gray distribution shift hurts"),
    # F. lean *_yt
    M("lean10_yt", "lean10_yt", "models/routers/lean10_yt/model.joblib",
      "lean10 + YouTube rows; acc 0.9649 f1_macro 0.8998"),
    M("lean13_yt", "lean13_yt", "models/routers/lean13_yt/model.joblib",
      "lean13 + YouTube; acc 0.9692 f1_macro 0.9121"),
    # G. lean19_v2 arms (ABC already = registered lean19_v2; add A/B/C)
    M("lean19_v2_A", "lean19_v2_A", "models/routers/lean19_v2_A/model.joblib",
      "19-feat class-weighted; f1_macro 0.9115"),
    M("lean19_v2_B", "lean19_v2_B", "models/routers/lean19_v2_B/model.joblib",
      "19-feat no class weights; f1_macro 0.8881 -> weighting matters"),
    M("lean19_v2_C", "lean19_v2_C", "models/routers/lean19_v2_C/model.joblib",
      "22-feat (+3 cross-modal) no weights; f1_macro 0.9154"),
    # H. optimal_v1 full + split_v1
    M("optimal_v1_all56", "optimal_v1 model_all56", "models/routers/optimal_v1/model_all56.joblib",
      "56-feat superset the 8-feat optimal_v1 was distilled from; 8-feat is -2.3pp f1_macro vs sa32"),
    M("split_v1", "split_v1 paired+grayscale (sa32_lite)", "models/routers/split_v1/paired/model_sa32_lite+.joblib",
      "first split iteration (sa32_lite+); predecessor of split_v2v3"),
    # I. patch-CNN lineage
    M("confuser_filter_v0", "confuser_filter (binary v0)", "models/patches/confuser_filter_rgb.pt",
      "earliest patch verifier (binary); val AUC 0.998 acc 0.972; IR sibling confuser_filter_ir.pt", type="verifier", tags="confusion-filter", lifecycle="superseded"),
    M("patch_verifier_v1", "patch_verifier (binary v1)", "models/patches/patch_verifier_rgb.pt",
      "v1 binary drone-vs-confuser CNN; val AUC 0.989 acc 0.949; IR sibling", type="verifier", tags="confusion-filter", lifecycle="superseded"),
    M("confuser_filter4_v1", "confuser_filter4 v1 (4-class)", "models/patches/confuser_filter4_rgb_v1_backup.pt",
      "first 4-class patch CNN; acc 0.975; predecessor of production v2", type="verifier", tags="confusion-filter", lifecycle="superseded", absorbed="patch_v2"),
    M("confuser_filter4_v3", "confuser_filter4 v3 (over-aggressive)", "models/patches/confuser_filter4_rgb_v3_backup.pt",
      "v3 backup; acc 0.978 vs v2 0.984; bird precision 0.90->0.78; over-aggressive, NOT shipped", type="verifier", tags="confusion-filter", lifecycle="superseded"),
    M("confuser_filter4_live", "confuser_filter4 (live/current)", "models/patches/confuser_filter4_rgb.pt",
      "non-suffixed live 4-class weights (rgb+ir); verify vs v2_backup which GUI loads", type="verifier", tags="confusion-filter"),
    M("confuser_filter4_ckpt", "confuser_filter4 ckpt", "models/patches/confuser_filter4_rgb_ckpt.pt",
      "mid-training checkpoint (rgb+ir); not an inference artifact", type="verifier", tags="confusion-filter", lifecycle="safe-to-archive"),
    # J. early trust classifiers (pre-reliability line)
    M("clf_early_v1", "classifier (early 21-feat)", "classifier/runs/classifier.joblib",
      "earliest trust head; 21-feat incl time-of-day; test F1 0.9984 in-domain", tags="trust", lifecycle="superseded"),
    M("clf_early_merged", "classifier_merged", "classifier/runs/classifier_merged.joblib",
      "DUPLICATE of clf_early_v1 (identical metrics)", tags="trust", lifecycle="superseded", absorbed="clf_early_v1"),
    M("clf_aerial_v1.1", "classifier_v1.1 (aerial)", "classifier/runs/classifier_v1.1.joblib",
      "v1.1 aerial classifier (adds YouTube); superseded by reliability/fusion line", tags="trust", lifecycle="superseded"),
    # K. V5/V4 MLP + prototype variants
    M("mlp_v5_pure_3x5", "mlp_v5 pure_3x5", "eval/results/_v5_selcom_pure_3x5/classifiers/mlp_v5.pt",
      "sibling of production mlp_v5 (pure_1x8) w/ stronger selcom up-weight (3.5/2.5); selcom F1 0.6097; pure_1x8 chosen", type="mlp", tags="confusion-filter"),
    M("mlp_v5_rebalance_svan", "mlp_v5 v5.1 rebalance_svan", "eval/results/_v5_rebalance_svan/classifiers/mlp_v5.pt",
      "v5.1 experiment: Svan drone weight 2.5->1.5; measured no-op on rgb_dataset recall; not shipped", type="mlp", tags="confusion-filter"),
    M("mlp_v5_p3p5_ft4", "mlp_v5 p3p5_ft4 (mixed-domain progenitor)", "eval/results/_v5_p3p5_ft4_distill/classifiers/mlp_v5.pt",
      "original mixed-domain V5 MLP before pure-selcom swap; CV F1 0.986; Svan F1 0.839; parent npz feeds pure/rebalance/remine", type="mlp", tags="confusion-filter"),
    M("prototype_v1", "prototype_v1 (centroid verifier)", "eval/results/_v5_p3p5_ft4_distill/classifiers/prototype_v1.pt",
      "non-MLP V5 verifier: 32-feat prototype/centroid-distance (tau 7.52); drone_kept@p90 0.90, confuser_kept@p90 0.476", type="verifier", tags="confusion-filter"),
    M("mlp_v4", "mlp_v4 p3p5_ft4", "eval/results/_v4_p3p5_ft4_distill/classifiers/mlp_v4.pt",
      "V4 distillation pilot (1093 samples); CV F1 0.880; ~=v2 patch; superseded by V5", type="mlp", tags="confusion-filter", lifecycle="superseded", absorbed="mlp_v5"),
]

CONFIG = [dict(id="clf_own_holdout", dataset="per-classifier", n_samples="", imgsz="",
               scoring_rule="", conf_thr="",
               notes="EACH row = that classifier's OWN train-time test split. NOT cross-comparable; excluded from comparison-view meaning. f1 = test F1 or f1_macro as noted.")]


def E(id, target, f1, note):
    return dict(id=id, target=target, config_id="clf_own_holdout", f1=f1, extra=note,
                source_script="(classifier metrics.json)", date="2026-05-30")

EVALS = [
    E("own_phase1_baseline", "clf_phase1_baseline", "0.9935", "test F1; full corpus"),
    E("own_phase1_antiuav", "clf_phase1_anti_uav_only", "0.9968", "in-domain"),
    E("own_phase1_svan", "clf_phase1_svanstrom_only", "1.0", "overfit/in-domain"),
    E("own_phase1_nosvan", "clf_phase1_no_svan_drones", "0.9892", ""),
    E("own_phase2_irsup", "clf_phase2_ir_suppressed", "0.9845", "normal; sup100 0.588 vs baseline 0.0"),
    E("own_rgb_reliability", "rgb_reliability", "0.888", "AUC 0.964; Svan 0.085"),
    E("own_ir_reliability", "ir_reliability", "0.917", "AUC 0.981"),
    E("own_rgb_fp", "rgb_fp_model", "0.677", "AUC 0.950"),
    E("own_rgb_fn", "rgb_fn_model", "0.796", "AUC 0.965"),
    E("own_ir_fp", "ir_fp_model", "0.392", "AUC 0.781"),
    E("own_ir_fn", "ir_fn_model", "0.766", "AUC 0.947"),
    E("own_fnfn_v3more", "fusion_no_fn_v3more", "0.9490", "f1_macro; Svan 0.799"),
    E("own_fnfn_v3more_nodet", "fusion_no_fn_v3more_no_det", "0.9493", "f1_macro"),
    E("own_fnfn_capped30k", "fusion_no_fn_v3more_capped30k", "0.9383", "f1_macro"),
    E("own_fnfn_capped_noflags", "fusion_no_fn_v3more_capped_no_flags", "0.9370", "f1_macro"),
    E("own_fnfn_grayaug", "fusion_no_fn_v3more_gray_aug", "0.9369", "f1_macro"),
    E("own_fnfn_realgray", "fusion_no_fn_v3more_realgray", "0.9092", "f1_macro; lowest"),
    E("own_lean10_yt", "lean10_yt", "0.8998", "f1_macro; acc 0.9649"),
    E("own_lean13_yt", "lean13_yt", "0.9121", "f1_macro; acc 0.9692"),
    E("own_lean19v2_a", "lean19_v2_A", "0.9115", "f1_macro"),
    E("own_lean19v2_b", "lean19_v2_B", "0.8881", "f1_macro; no class weights"),
    E("own_lean19v2_c", "lean19_v2_C", "0.9154", "f1_macro; +xmodal"),
    E("own_optimal_all56", "optimal_v1_all56", "0.9493", "f1_macro; 8-feat is -2.3pp vs sa32"),
    E("own_clf_early", "clf_early_v1", "0.9984", "test F1; in-domain 21-feat"),
    E("own_mlp_v5_p3p5", "mlp_v5_p3p5_ft4", "0.986", "5-fold CV f1 (meta+yolo); Svan F1 0.839"),
    E("own_mlp_v4", "mlp_v4", "0.880", "5-fold CV f1 (meta+yolo)"),
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
    _append("eval_configs", CONFIG)
    _append("evals", EVALS)
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
