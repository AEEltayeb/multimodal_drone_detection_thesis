#!/usr/bin/env python3
"""apply_ledger_sweep.py — one-time, idempotent. Applies the ledger detail-sweep agent's
extraction of the NOT-yet-migrated EVIDENCE_LEDGER sections (§5.2/5.3, §6/6.1, §8.3/8.4,
§9.5, §10, §11, §12, dual-classifier §[2026-05-26], embedding-distillation §[2026-05-27],
§13.2). After this, EVIDENCE_LEDGER.md is fully extractable.

Handling: ir-ood-recall-ceiling finding ALREADY exists -> its §8.3 evals link via ledger_ids,
finding NOT re-added. A few missing eval targets get minimal model rows. Numbers faithful;
placeholders/UNKNOWN/open-provenance flagged as status=open. Idempotent: skips existing ids.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402

# --- missing eval targets as minimal model rows ---
MODELS = [
    dict(id="dual_classifier_v3", name="dual classifier v3 (paired+grayscale)", type="classifier",
         purpose_tags="confusion-filter;trust", trained_from_script="", train_dataset="paired sa32_lite+ 24feat / grayscale sa32_feats 32feat",
         weights_path="models/routers/split_v1/", provenance_notes="§[2026-05-26] dual-architecture: paired core + grayscale fallback; strictly dominates single sa32 on RGB-fallback surfaces",
         production="no", lifecycle="active"),
    dict(id="mlp_v5_mixed", name="mlp_v5 mixed (original)", type="mlp", purpose_tags="confusion-filter",
         trained_from_script="eval/distill_v5_p3p5_ft4.py", train_dataset="80% general/20% CCTV mixed (ft2)",
         weights_path="eval/results/_v5_head_to_head_mixed/", provenance_notes="original mixed-source V5; selcom collapse 0.243 -> train-deploy mismatch (fixed by pure CCTV). §13.2",
         production="no", lifecycle="superseded"),
    dict(id="distill_mlp_261feat", name="distill MLP 261-feat (CV)", type="mlp", purpose_tags="confusion-filter",
         trained_from_script="eval/overnight_confuser_distill.py", train_dataset="3351 dets (664 drone/2687 confuser), 5 meta + 256 p5 embeddings",
         weights_path="eval/results/_overnight_distill/classifiers.pkl", provenance_notes="Phase-2 embedding-distillation CV winner (F1 0.9955 CV-only); conceptual ancestor of 517-D V5. §[2026-05-27]",
         production="no", lifecycle="superseded"),
]

# --- new eval_configs ---
def C(id, dataset, notes, n="", imgsz="", scoring="", conf=""):
    return dict(id=id, dataset=dataset, n_samples=n, imgsz=imgsz, scoring_rule=scoring, conf_thr=conf, notes=notes)

CONFIGS = [
    C("svan_iop_640_may10", "svanstrom", "May-10 ablation default; imgsz UNKNOWN (§11 open). §5,§6", imgsz="640", scoring="iop", conf="0.25"),
    C("patch_catch_svan_1280_s9", "svanstrom", "stride=9; per-bucket patch-verifier catch-rate audit. §6.1", n="3190", imgsz="1280", scoring="iop", conf="0.25"),
    C("roboflow_ir_drone_640", "ir_drone_night;ir_mixed_cbam", "OOD IR drone; ir_drone_night heavily sensor-augmented (worst-case probe). §8.3", imgsz="640", scoring="iou", conf="0.25"),
    C("roboflow_ir_confuser_640", "ir_airplane_hors2;ir_airplane_plane;ir_bird", "OOD IR confusers; no GT every det=FP. §8.4", imgsz="640", scoring="iou", conf="0.25"),
    C("pipe_video_drone_iop", "drone_video_tests", "9 drone videos 1234 GT; FULL pipeline RGB->IRgray->sa32->temporal->patch_thr0.7; per-frame + 3-frame segment. §9.5", n="1359", imgsz="per-model", scoring="iop", conf="0.25"),
    C("pipe_video_confuser", "confuser_video_tests", "10 confuser videos; full pipeline; per-frame + segment FPR. §9.5", n="1250", imgsz="per-model", conf="0.25"),
    C("svan_persize_1280", "svanstrom", "per-size drone+confuser buckets; check n_gt>0. §12", imgsz="1280", scoring="iop", conf="0.25"),
    C("selcom_holdout_persize", "selcom_mixed_ft2_val", "selcom held-out per-size, 6 RGB models. §12", n="311", imgsz="per-model", scoring="iop", conf="0.25"),
    C("antiuav_per_model", "antiuav", "per-model RGB on Anti-UAV (saturated). §12", imgsz="per-model", scoring="iou", conf="0.25"),
    C("dual_clf_v3_surfaces", "multi", "dual classifier v3 vs sa32 across confuser/video/rgb/antiuav. §[2026-05-26]", imgsz="per-surface"),
    C("distill_cv_5fold", "distill_train_3351", "5-fold CV; 664 drone TP/2687 confuser FP; 261-D (5 meta+256 p5). Phase-2 CV only. §[2026-05-27]", n="3351"),
    C("selcom_iop_1280_persrc", "selcom_mixed_ft2_val", "selcom_val for V5 selcom-source ablation (mixed vs pure). §13.2", n="311", imgsz="1280", scoring="iop", conf="0.5"),
]

def E(id, target, cfg, f1="", r="", p="", fpr="", halluc="", note="", led=""):
    return dict(id=id, target=target, config_id=cfg, f1=f1, recall=r, precision=p, fpr=fpr,
                halluc_rate=halluc, extra=note, ledger_ids=led, source_script="(EVIDENCE_LEDGER sweep)", date="2026-05-30")

EVALS = [
    # §6 patch versions (Svanstrom filter_then_classifier F1)
    E("patch_v1_svan640", "patch_v1", "svan_iop_640_may10", f1="0.9241", note="§6; v3 path UNKNOWN", led="patch-version-ranking"),
    E("patch_v2_svan640", "patch_v2", "svan_iop_640_may10", f1="0.9311", note="production choice §6", led="patch-version-ranking"),
    E("patch_v3_svan640", "patch_v3", "svan_iop_640_may10", f1="0.8781", note="over-aggressive §6", led="patch-version-ranking"),
    E("patch_v4_svan640", "patch_v4", "svan_iop_640_may10", f1="0.9331", note="current code default ~=v2 §6", led="patch-version-ranking"),
    # §6.1 catch audit
    E("patch_catch_v2_svan", "patch_v2", "patch_catch_svan_1280_s9", note="catch@0.5: BIRD0.638 AIRPLANE0.517 HELI0.709; DRONE_TP_veto0.054; all<0.90 bar; bimodal", led="patch-catch-below-bar"),
    # §7 control40 (not previously rowed)
    E("clfzoo_control40", "control40", "confuser_zoo_1280", halluc="0.212", note="S2 zoo fire; S3=0.094; ties sa32 §7", led="control40-deprecated"),
    E("svan_s3_control40_thr09", "control40", "svan_iop_1280_s9", p="0.925", r="0.893", f1="0.909", note="S3 patch_thr0.9; wins every Svan drone metric §7", led="control40-deprecated"),
    E("svan_s3_sa32_thr08", "sa32", "svan_iop_1280_s9", r="0.868", f1="0.896", note="S3 patch_thr0.8; confuser S3 FP 41 §7"),
    # §8.3/8.4 IR OOD (link to existing ir-ood-recall-ceiling finding)
    E("ir_ood_night_raw", "ir_final", "roboflow_ir_drone_640", p="0.494", r="0.264", f1="0.344", note="raw; sensor-augmented probe §8.3", led="ir-ood-recall-ceiling"),
    E("ir_ood_night_patch", "ir_final", "roboflow_ir_drone_640", p="0.474", r="0.210", f1="0.291", note="+patch_v2 §8.3", led="ir-ood-recall-ceiling"),
    E("ir_ood_cbam_raw", "ir_final", "roboflow_ir_drone_640", p="0.821", r="0.519", f1="0.636", note="ir_mixed_cbam raw §8.3", led="ir-ood-recall-ceiling"),
    E("ir_ood_cbam_patch", "ir_final", "roboflow_ir_drone_640", p="0.874", r="0.491", f1="0.628", note="ir_mixed_cbam +patch §8.3", led="ir-ood-recall-ceiling"),
    E("ir_ood_conf_airhors2", "ir_final", "roboflow_ir_confuser_640", note="airplane_hors2 FP 1128->1048 (7.1% supp) §8.4", led="patch-verifier-distribution-bound"),
    E("ir_ood_conf_airplane", "ir_final", "roboflow_ir_confuser_640", note="airplane_plane FP 386->368 (4.7%) §8.4", led="patch-verifier-distribution-bound"),
    E("ir_ood_conf_bird", "ir_final", "roboflow_ir_confuser_640", note="bird FP 95->61 (35.8%) §8.4", led="patch-verifier-distribution-bound"),
    # §9.5 full-pipeline video per-frame (+classifier)
    E("pipe_vid_baseline_pf", "baseline", "pipe_video_drone_iop", p="0.520", r="0.672", f1="0.586", note="+clf per-frame; RGB-alone 0.760 (dF1 -0.174) §9.5.2", led="cascade-perframe-misleading"),
    E("pipe_vid_retrainedv2_pf", "retrained_v2", "pipe_video_drone_iop", f1="0.615", note="+clf; RGB 0.605; ONLY variant that gains §9.5.2", led="cascade-perframe-misleading"),
    E("pipe_vid_selcom1280_pf", "selcom_1280", "pipe_video_drone_iop", f1="0.537", note="+clf; RGB 0.721 (dF1 -0.184) §9.5.2", led="cascade-perframe-misleading"),
    E("pipe_vid_selcom640_pf", "selcom_640", "pipe_video_drone_iop", f1="0.568", note="+clf; RGB 0.730 §9.5.2", led="cascade-perframe-misleading"),
    # §9.5 segment-level (+temporal+patch)
    E("pipe_vid_baseline_seg", "baseline", "pipe_video_drone_iop", p="0.987", r="0.711", f1="0.826", note="segment; temporal-only 0.833 §9.5.4", led="cascade-segment-recovers"),
    E("pipe_vid_retrainedv2_seg", "retrained_v2", "pipe_video_drone_iop", f1="0.770", note="segment §9.5.4", led="cascade-segment-recovers"),
    E("pipe_vid_selcom1280_seg", "selcom_1280", "pipe_video_drone_iop", f1="0.814", note="segment §9.5.4", led="cascade-segment-recovers"),
    E("pipe_vid_selcom640_seg", "selcom_640", "pipe_video_drone_iop", f1="0.816", note="segment §9.5.4", led="cascade-segment-recovers"),
    # §9.5 confuser FPR segment
    E("pipe_vidconf_baseline_seg", "baseline", "pipe_video_confuser", fpr="0.162", note="segment; RGB-raw 0.512 (68% cut) §9.5.5", led="cascade-tightens-variance"),
    E("pipe_vidconf_selcom1280_seg", "selcom_1280", "pipe_video_confuser", fpr="0.136", note="segment; RGB-raw 0.709 (81% cut) §9.5.5", led="cascade-tightens-variance"),
    # §9.5.8 three-classifier (segment drone F1, baseline RGB variant)
    E("pipe3clf_sa32", "sa32", "pipe_video_drone_iop", f1="0.826", note="segment drone F1 (baseline RGB); TP retention 0.90 §9.5.8", led="three-classifier-realvideo"),
    E("pipe3clf_control40", "control40", "pipe_video_drone_iop", f1="0.644", note="segment drone F1 (baseline RGB); TP retention 0.60 §9.5.8", led="three-classifier-realvideo"),
    E("pipe3clf_fnfn", "fusion_no_fn_v1.1", "pipe_video_drone_iop", f1="0.219", r="0.128", note="segment drone F1 (baseline RGB); rejects 85% of drone TPs §9.5.8", led="three-classifier-realvideo"),
    # §9.5.9 per-category
    E("pipe_percat_sa32", "sa32", "pipe_video_confuser", note="seg FPR by cat (baseline): birds0.017 airplanes0.225 helis0.216; PF FP-supp birds~80% airplanes~0 §9.5.9", led="cascade-bird-vs-airplane-asymmetry"),
    # §12 cross-surface
    E("xsurf_selcom_holdout", "selcom_960", "selcom_holdout_persize", f1="0.585", r="0.44", p="0.88", note="#1 selcom holdout (vs selcom1280 0.580, baseline 0.15) §12", led="selcom960-cross-surface-winner"),
    E("xsurf_roboflow_drone", "selcom_960", "roboflow_rgb_drone_640", f1="0.84", note="#1 OOD drone +patch (vs baseline 0.79) §12", led="selcom960-cross-surface-winner"),
    E("xsurf_antiuav_selcom1280", "selcom_1280", "antiuav_per_model", f1="0.902", p="0.84", note="849 FP; bleeds FP at high recall §12.1", led="selcom1280-bleeds-fp"),
    # §[2026-05-26] dual classifier
    E("dualclf_v3_vs_sa32", "dual_classifier_v3", "dual_clf_v3_surfaces", note="F1: BIRD sa32 0.247->v3 1.000; seagull 0.461->0.942; rgb_test 0.690->0.960; paired core neutral; residual airplane/heli grayscale ~0.50", led="dual-classifier-v3"),
    # §[2026-05-27] embedding distillation CV (provisional)
    E("distill_cv_mlp", "distill_mlp_261feat", "distill_cv_5fold", f1="0.9955", note="CV winner; meta-only 0.939; CV ONLY (Phase3 pending) §[2026-05-27]", led="embedding-distillation-cv"),
    E("distill_cv_xgb", "distill_xgb_261feat", "distill_cv_5fold", f1="0.9917", note="meta-only 0.966 already strong; CV ONLY", led="embedding-distillation-cv"),
    # §13.2 selcom-source ablation
    E("v5_selcom_mixed_src", "mlp_v5_mixed", "selcom_iop_1280_persrc", f1="0.2426", note="mixed 80/20 source; collapsed (train-deploy mismatch) §13.2", led="v5-selcom-train-deploy-mismatch"),
    E("v5_selcom_pure3x5", "mlp_v5_pure_3x5", "selcom_iop_1280_persrc", f1="0.6097", note="pure CCTV drone-wt3.5 ~= pure_1x8; weight bump redundant §13.2", led="v5-selcom-train-deploy-mismatch"),
]

def F(id, claim, outcome, evidence="", condition="", contradicts="", contrib="", status="confirmed", notes=""):
    return dict(id=id, date="2026-05-30", claim=claim, outcome=outcome, condition=condition,
                evidence_evals=evidence, contradicts=contradicts, thesis_contribution=contrib, status=status, notes=notes)

FINDINGS = [
    F("patch-version-ranking", "patch verifier versions rank v4(0.9331)~=v2(0.9311)>v1(0.9241)>v3(0.8781) on Svanstrom F1; v3 over-aggressive; production ships v2",
      "supported", "patch_v1_svan640;patch_v2_svan640;patch_v3_svan640;patch_v4_svan640", notes="§6; imgsz likely 640 (§11 open)"),
    F("patch-catch-below-bar", "baseline RGB x v2 patch catch rates at thr=0.5 are bird 64%/airplane 52%/heli 71% - all below the 0.90 bar; drone-TP veto only 5.4%; misses are 'other'-class crops",
      "supported", "patch_catch_v2_svan", contrib="patch verifier alone insufficient on baseline confuser output; motivates retraining/distillation", notes="§6.1"),
    F("control40-deprecated", "control_v3more_40feat wins every Svanstrom drone metric at S3 (R0.893 F10.909 vs sa32 0.868/0.896) and ties sa32 on OOD zoo, but loses 18-22pp segment drone F1 on real video; no regime favors it",
      "supported", "svan_s3_control40_thr09;clfzoo_control40;pipe3clf_control40", contradicts="", contrib="", notes="§7,§9.5.8"),
    F("cascade-perframe-misleading", "per-frame metrics misrepresent the cascade: classifier stage drops per-frame drone F1 -17 to -18pp for 3/4 RGB variants; only retrained_v2 gains (+1pp)",
      "supported", "pipe_vid_baseline_pf;pipe_vid_selcom1280_pf;pipe_vid_retrainedv2_pf", contrib="segment/alert grain is the production-relevant unit, not per-frame", notes="§9.5.2/9.5.6"),
    F("cascade-segment-recovers", "at segment grain (+temporal+patch) the cascade recovers classifier-stage loss: baseline 0.760->0.586->0.826; temporal smoother is the load-bearing recovery step, patch veto small",
      "supported", "pipe_vid_baseline_seg;pipe_vid_selcom1280_seg", contrib="temporal voting is the recovery mechanism", notes="§9.5.4/9.5.6"),
    F("cascade-tightens-variance", "the cascade tightens RGB-variant variance: stage-1 F1 spans 15.5pp -> post-cascade segment spans 5.6pp; selcom_1280 confuser FPR collapses 0.709->0.136 (81%) at -0.7pp drone F1",
      "supported", "pipe_vidconf_baseline_seg;pipe_vidconf_selcom1280_seg", contrib="downstream stack absorbs detector variance", notes="§9.5.5"),
    F("three-classifier-realvideo", "on real video sa32 dominates: highest segment drone F1 for all RGB variants (+18-22pp vs control40, +59-71pp vs fnfn); fnfn rejects 85% of correct drone TPs (seg R 0.128) - too conservative to ship",
      "supported", "pipe3clf_sa32;pipe3clf_control40;pipe3clf_fnfn", condition="fnfn valid only when confuser cost dominates and missed drones tolerable", contradicts="trust-classifier-conditional", contrib="Svanstrom classifier ordering predicts real-video; basis for sa32 production flip", notes="§9.5.8"),
    F("cascade-bird-vs-airplane-asymmetry", "cascade value is concentrated on birds (per-frame FP suppression ~80%) but essentially inert on airplanes (sometimes ADDS FPs; segment FPR ~0.21 across all RGB)",
      "supported", "pipe_percat_sa32", contrib="cascade handles exactly the category (birds) where detector-level mining hits a wall", notes="§9.5.9; strongest argument for cascade design intent"),
    F("selcom960-cross-surface-winner", "selcom_960 is the cross-surface drone-first winner (composite rank 2.5): #1 selcom holdout F1 (0.585), #1 Roboflow OOD drone +patch (0.84); NOT in production stack",
      "partial", "xsurf_selcom_holdout;xsurf_roboflow_drone", condition="loses to baseline on Svanstrom medium + clean video; untested in full pipeline", contrib="selcom_960 as candidate production RGB - revisit stack", status="open", notes="§12"),
    F("selcom1280-bleeds-fp", "selcom_1280 has highest raw recall but bleeds FPs: Anti-UAV P=0.84 (849 FP, F1 0.902), worst Roboflow confuser FP total (4311 vs hardneg 785); imgsz=640 inference trades recall for far fewer FPs",
      "supported", "xsurf_antiuav_selcom1280", contrib="selcom recall-vs-FP tradeoff is resolution-tunable", notes="§12.1"),
    F("dual-classifier-v3", "splitting sa32 into a dual architecture (paired 24feat + grayscale 32feat) strictly dominates single sa32: bird-confuser RGB-fallback 0.247->1.000 F1, seagull 0.461->0.942, rgb_test 0.690->0.960, paired core neutral",
      "supported", "dualclf_v3_vs_sa32", condition="grayscale-fallback airplane/heli F1 still ~0.50; needs more hard-neg mining", contrib="dual-classifier architecture for missing-modality fallback", notes="§[2026-05-26]; leakage fix: seagull held out"),
    F("embedding-distillation-cv", "YOLO p5 backbone embeddings are highly informative for drone/confuser separation: LogReg 0.71->0.99 F1 with 256 embedding feats; MLP 0.9955 CV; meta-alone already 0.966 (XGB)",
      "partial", "distill_cv_mlp;distill_cv_xgb", condition="CV ONLY - Phase 3 held-out test-surface eval not completed (re-run needed)", contrib="distill YOLO internal features instead of reprocessing pixels (precursor to V5)", status="open", notes="§[2026-05-27]; ancestor of 517-D V5 MLP"),
    F("v5-selcom-train-deploy-mismatch", "V5's initial selcom collapse (F1 0.243) was a train-deploy distribution mismatch not a feature ceiling: pure-CCTV source recovers to 0.607 (+37pp); the drone-weight bump is redundant once source matches",
      "supported", "v5_selcom_mixed_src;v5_selcom_pure3x5", contrib="train-source distribution match > loss-reweighting for verifier deploy", notes="§13.2"),
    F("latency-edge-unmeasured", "end-to-end pipeline latency on edge hardware is UNMEASURED (§10 placeholders); only verifier-stage latency measured (V5 1.3-2.1ms/det vs patch 59-112ms, 37-72x faster; V5 PF overhead 1-4% vs patch 48-191%)",
      "partial", "", condition="verifier-stage measured (§13.7/13.8); full e2e on target hardware not run", contrib="deployment latency budget", status="open", notes="§10 + §13.7/13.8"),
    # §11 open data-provenance
    F("prov-patch-v3-path", "patch verifier v3 weights file path is unknown/deleted - blocks reproducing the v3 0.8781 F1", "partial", "patch_v3_svan640", status="open", notes="§11"),
    F("prov-may10-imgsz", "the imgsz of the May-10 ablation (Anti-UAV + Svanstrom) is unconfirmed (640 vs 1280) - affects comparability of May-10 classifier/patch rows", "partial", "", status="open", notes="§11; affects ir_final §4 svanstrom row too"),
    F("prov-fnfn-training", "fusion_no_fn_v1.1 training data composition + val accuracy are NOT recoverable - no metrics.json saved at train time", "partial", "", status="open", notes="§11; provenance gap on a production-candidate classifier"),
    F("prov-ir-gray-confuser-src", "IR-on-grayscale-confuser halluc rates (airplane 16.2%/other 22.2% @1280) have source=UNKNOWN", "partial", "", status="open", notes="§4 table source UNKNOWN; verify before citing"),
    F("prov-ir-antiuav-gray", "IR on antiuav_rgb_gray (real R/P with GT) at imgsz 640+1280 is NOT YET RUN (placeholder)", "partial", "", status="open", notes="§4; the only labeled grayscale-cross-domain dataset"),
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
    _append("eval_configs", CONFIGS)
    _append("evals", EVALS)
    _append("ledger", FINDINGS)
    kb._regen_views()
    print("views regenerated.")


if __name__ == "__main__":
    main()
