"""Read-only audit: verify the thesis's headline cells against the replay JSONs."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
T = json.load(open(REPO / "thesis_eval/results/tier1_results.json"))
V = json.load(open(REPO / "thesis_eval/results/temporal_results.json"))
W = json.load(open(REPO / "thesis_eval/results/video_thr_sweep.json"))
N = json.load(open(REPO / "thesis_eval/results/notes_round1_results.json"))
F = json.load(open(REPO / "thesis_eval/results/failure_profile_results.json"))
G = json.load(open(REPO / "thesis_eval/results/negative_frame_fire.json"))
# no-reject (shipped robust8-nr) replay outputs
TN = json.load(open(REPO / "thesis_eval/results_noreject/tier1_results.json"))
VN = json.load(open(REPO / "thesis_eval/results_noreject/temporal_results.json"))
NN = json.load(open(REPO / "thesis_eval/results_noreject/notes_round1_results.json"))

CHECKS = [
    # (label, claimed, actual)
    ("svan bare F1", 0.7415, T["svanstrom"]["B_pipeline"]["bare"]["f1"]),
    ("svan composed F1", 0.948, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
    ("svan composed P", 0.941, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["precision"]),
    ("svan composed R", 0.9552, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["recall"]),
    ("svan bare R", 0.9481, T["svanstrom"]["B_pipeline"]["bare"]["recall"]),
    ("svan filt->clf F1", 0.9636, T["svanstrom"]["B_pipeline"]["filt->clf[robust8]"]["f1"]),
    ("svan clf[robust6] F1", 0.9514, T["svanstrom"]["B_pipeline"]["clf[robust6]"]["f1"]),
    ("svan clf[sa32] F1", 0.9666, T["svanstrom"]["B_pipeline"]["clf[sa32]"]["f1"]),
    ("antiuav bare F1", 0.9728, T["antiuav"]["B_pipeline"]["bare"]["f1"]),
    ("antiuav composed F1", 0.9844, T["antiuav"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
    ("antiuav ft4 bare F1", 0.9853, T["antiuav"]["A_bare"]["ft4/rgb"]["f1"]),
    ("antiuav ft4 FP", 41, T["antiuav"]["A_bare"]["ft4/rgb"]["FP"]),
    ("antiuav v3b F1", 0.961, T["antiuav"]["A_bare"]["v3b/ir"]["f1"]),
    ("rgbconf bare fire", 0.3035, T["rgb_confuser"]["C_confuser"]["bare"]["fire_rate"]),
    ("rgbconf clf[r8] fire", 0.049, T["rgb_confuser"]["C_confuser"]["clf[robust8]"]["fire_rate"]),
    ("rgbconf mlp fire", 0.0144, T["rgb_confuser"]["C_confuser"]["filt_mlp"]["fire_rate"]),
    ("rgbconf mlp FP", 39, T["rgb_confuser"]["C_confuser"]["filt_mlp"]["FP"]),
    ("rgbconf patch fire", 0.1022, T["rgb_confuser"]["C_confuser"]["filt_patch"]["fire_rate"]),
    ("rgbconf composed fire", 0.0015, T["rgb_confuser"]["C_confuser"]["clf->filt[robust8]"]["fire_rate"]),
    ("rgbconf composed FP", 3, T["rgb_confuser"]["C_confuser"]["clf->filt[robust8]"]["FP"]),
    ("irconf bare fire", 0.2943, T["ir_confusers"]["C_confuser"]["bare"]["fire_rate"]),
    ("irconf composed r8 fire", 0.0243, T["ir_confusers"]["C_confuser"]["clf->filt[robust8]"]["fire_rate"]),
    ("irconf composed r6 fire", 0.0192, T["ir_confusers"]["C_confuser"]["clf->filt[robust6]"]["fire_rate"]),
    ("3way RGB F1", 0.6067, T["svanstrom"]["A_bare"]["ft4/rgb"]["f1"]),
    ("3way rawrgb F1", 0.1874, T["svanstrom_rawrgb"]["A_bare"]["v3b/ir"]["f1"]),
    ("3way gray F1", 0.5796, T["svanstrom_gray"]["A_bare"]["v3b/ir"]["f1"]),
    ("rgbtest bare F1", 0.9259, T["rgb_dataset_test"]["S4_verifier"]["bare"]["f1"]),
    ("rgbtest mlp F1", 0.9222, T["rgb_dataset_test"]["S4_verifier"]["filt_mlp"]["f1"]),
    ("rgbtest patch F1", 0.8898, T["rgb_dataset_test"]["S4_verifier"]["filt_patch"]["f1"]),
    ("selcom bare F1", 0.5911, T["selcom_val"]["S4_verifier"]["bare"]["f1"]),
    ("selcom mlp F1", 0.6115, T["selcom_val"]["S4_verifier"]["filt_mlp"]["f1"]),
    ("irtest bare F1", 0.961, T["ir_dset_final"]["S4_verifier"]["bare"]["f1"]),
    ("irtest mlp F1", 0.9421, T["ir_dset_final"]["S4_verifier"]["filt_mlp"]["f1"]),
    ("irtest patch F1", 0.9398, T["ir_dset_final"]["S4_verifier"]["filt_patch"]["f1"]),
    ("video bare win F1", 0.843, V["video_drone"]["bare"]["window"][2]),
    ("video r6 win F1", 0.7368, V["video_drone"]["clf[robust6]"]["window"][2]),
    ("video composed r8 F1", 0.5436, V["video_drone"]["clf->filt[robust8]"]["window"][2]),
    ("video replica F1", 0.6891, V["video_drone"]["clf->filt_patch[sa32]"]["window"][2]),
    ("video bare fire", 0.3504, V["video_confuser"]["bare"]["window_fire"]),
    ("video r6 fire", 0.0813, V["video_confuser"]["clf[robust6]"]["window_fire"]),
    ("video composed r8 fire", 0.0756, V["video_confuser"]["clf->filt[robust8]"]["window_fire"]),
    ("sweep r8@0.01 R", 0.507, W["robust8@0.01"][1]),
    ("sweep r6@0.01 F1", 0.7309, W["robust6@0.01"][2]),
    # notes-round-1 additions (per-size, per-category, background profile). Modality A/B (coverage)
    # cells removed 2026-06-17: the thesis scores own-GT/trust-aware only; the coverage table was deleted.
    ("SZ rgbtest <16 bare R", 0.7824, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"]["<16px"]["recall"]),
    ("SZ rgbtest <16 filt R", 0.7672, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["filt"]["<16px"]["recall"]),
    ("SZ rgbtest 16-32 bare R", 0.8649, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"]["16-32px"]["recall"]),
    ("SZ rgbtest 16-32 filt R", 0.8435, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["filt"]["16-32px"]["recall"]),
    ("SZ rgbtest >=64 bare R", 0.9555, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"][">=64px"]["recall"]),
    ("SZ rgbtest >=64 filt R", 0.9513, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["filt"][">=64px"]["recall"]),
    ("SZ svan rgb median px", 29.8, N["svanstrom"]["SZ_per_size"]["ft4/rgb"]["median_gt_sqrt_area_px"]),
    ("SZ svan ir median px", 14.8, N["svanstrom"]["SZ_per_size"]["v3b/ir"]["median_gt_sqrt_area_px"]),
    ("SZ svan rgb <16 bare R", 0.6296, N["svanstrom"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"]["<16px"]["recall"]),
    ("SZ svan rgb 16-32 bare R", 0.8974, N["svanstrom"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"]["16-32px"]["recall"]),
    ("CAT rgbconf bird bare", 0.3896, N["rgb_confuser"]["CAT_confuser"]["bird"]["bare"]["fire_rate"]),
    ("CAT rgbconf heli bare", 0.5802, N["rgb_confuser"]["CAT_confuser"]["helicopter"]["bare"]["fire_rate"]),
    ("CAT rgbconf airplane bare", 0.2339, N["rgb_confuser"]["CAT_confuser"]["airplane"]["bare"]["fire_rate"]),
    ("FP svan conf-sky fire", 0.6669, F["svanstrom"]["ft4/rgb"]["confuser-seqs/sky"]["fp_frame_rate"]),
    ("FP svan conf-horizon fire", 0.741, F["svanstrom"]["ft4/rgb"]["confuser-seqs/horizon"]["fp_frame_rate"]),
    ("FP svan conf-ground fire", 0.6463, F["svanstrom"]["ft4/rgb"]["confuser-seqs/ground"]["fp_frame_rate"]),
    ("FP svan drone-sky R", 0.8274, F["svanstrom"]["ft4/rgb"]["drone-seqs/sky"]["recall"]),
    ("FP svan drone-horizon R", 0.9444, F["svanstrom"]["ft4/rgb"]["drone-seqs/horizon"]["recall"]),
    ("FP svan ir conf-horizon fire", 0.0169, F["svanstrom"]["v3b/ir"]["confuser-seqs/horizon"]["fp_frame_rate"]),
    ("FP svan ir drone-ground fire", 0.1534, F["svanstrom"]["v3b/ir"]["drone-seqs/ground"]["fp_frame_rate"]),
    # notes-round-2 additions (negative-frame fire, IR per-category)
    ("NEG ir fire", 0.0164, G["ir_dset_final"]["bare"]["fire_rate"]),
    ("NEG ir FP dets", 38, G["ir_dset_final"]["bare"]["FP"]),
    ("NEG ir n", 1400, G["ir_dset_final"]["n_negative_frames"]),
    ("CAT irconf airplane bare", 0.352, N["ir_confusers"]["CAT_confuser"]["airplane"]["bare"]["fire_rate"]),
    ("CAT irconf bird bare", 0.1217, N["ir_confusers"]["CAT_confuser"]["bird"]["bare"]["fire_rate"]),
    ("CAT irconf heli bare", 0.0, N["ir_confusers"]["CAT_confuser"]["helicopter"]["bare"]["fire_rate"]),
]

# round-3: current scoring swing (sec:scoring_audit). The cached-subset LEAK cells were removed
# 2026-06-12 with the thesis rollback — clean-subset metrics return only after the FULL clean-split
# eval (cache surfaces svanstrom_clean/antiuav_clean) lands; leakage_controlled.json remains the
# artifact for the overlap derivation in the meantime.
CHECKS += [
    # sec:scoring_audit current swing: dual (= coverage routed+filt) vs trust-aware, identical P
    ("SWING dual routed+filt F1", 0.9206, N["svanstrom"]["M_modality_ab"]["routed[robust8] +filt"]["f1"]),
    ("SWING trust-aware pipeline F1", 0.948, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
    ("SWING identical precision", 0.941, N["svanstrom"]["M_modality_ab"]["routed[robust8] +filt"]["precision"]),
    ("SWING trust-aware precision", 0.941, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["precision"]),
]

# CBAM held-out FP for the THERMAL-NATIVE filter (mlp_aligned_thermalonly) is stated in two places
# (methodology sec:ir_xmodal_verifier prose + empirical tab:ir_aligned). Canonical = the saved held-out
# JSON eval/results/ir_heldout_results.json (cbam@0.05 FP=6, R=0.967); kb evals cbam_heldout_thermalonly.
# Pin both .tex restatements to the canonical value and to each other (catches silent prose drift).
import re
_CBAM_CANON_FP = json.load(open(REPO / "eval/results/ir_heldout_results.json"))["cbam"]["results"]["cbam@0.05"]["confuser_FP"]
_meth = (REPO / "docs/thesis_working_distilling_overleaf/chapters/methodology.tex").read_text(encoding="utf-8")
_emp = (REPO / "docs/thesis_working_distilling_overleaf/chapters/empirical.tex").read_text(encoding="utf-8")
_m_meth = re.search(r"cutting false positives to \$(\d+)\$ \(bare \$48\$\)", _meth)
_m_emp = re.search(r"0\.905/0\.967/0\.935\}\$\s*\(\\textbf\{(\d+)\}\)", _emp)
CHECKS += [
    ("CBAM aligned FP (methodology prose)", _CBAM_CANON_FP, int(_m_meth.group(1)) if _m_meth else -1),
    ("CBAM aligned FP (empirical table)", _CBAM_CANON_FP, int(_m_emp.group(1)) if _m_emp else -1),
]

# round-4b: deferred-suppression rationale (sec:design_rationale) quotes the conf sweep
S = json.load(open(REPO / "thesis_eval/results/conf_sweep/conf_sweep_results.json"))

def sweep_row(surface, conf):
    blk = next(v for v in S[surface].values() if isinstance(v, dict) and "rows" in v)
    return next(r for r in blk["rows"]
                if r.get("slice", "rgb") == "rgb" and r.get("rgb_conf", r.get("conf")) == conf)

CHECKS += [
    ("SWEEP antiuav bare@0.05", 0.9592, sweep_row("antiuav", 0.05)["bare"]["f1"]),
    ("SWEEP antiuav bare@0.25", 0.9631, sweep_row("antiuav", 0.25)["bare"]["f1"]),
    ("SWEEP rgb_test bare@0.25 (peak)", 0.9259, sweep_row("rgb_dataset_test", 0.25)["bare"]["f1"]),
    ("SWEEP selcom bare@0.25", 0.5911, sweep_row("selcom_val", 0.25)["bare"]["f1"]),
    ("SWEEP selcom filt@0.05", 0.6993, sweep_row("selcom_val", 0.05)["filt"]["f1"]),
]

# clean-split integration (sec:svanstrom_audit tab:clean_split) — audited against the FROZEN copy
C = json.load(open(REPO / "runs/clean_split/clean_split_results.json"))
CHECKS += [
    ("CLEAN svan n", 5557, C["svanstrom_clean"]["meta"]["n"]),
    ("CLEAN svan ft4 solo", 0.5717, C["svanstrom_clean"]["A_bare"]["ft4/rgb"]["f1"]),
    ("CLEAN svan v3b solo", 0.8674, C["svanstrom_clean"]["A_bare"]["v3b/ir"]["f1"]),
    ("CLEAN svan bare paired", 0.6842, C["svanstrom_clean"]["B_pipeline"]["bare"]["f1"]),
    ("CLEAN svan pipeline", 0.934, C["svanstrom_clean"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
    ("CLEAN auv n", 57542, C["antiuav_clean"]["meta"]["n"]),
    ("CLEAN auv ft4 solo", 0.9878, C["antiuav_clean"]["A_bare"]["ft4/rgb"]["f1"]),
    ("CLEAN auv v3b solo", 0.9656, C["antiuav_clean"]["A_bare"]["v3b/ir"]["f1"]),
    ("CLEAN auv bare paired", 0.9765, C["antiuav_clean"]["B_pipeline"]["bare"]["f1"]),
    ("CLEAN auv pipeline", 0.9861, C["antiuav_clean"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
]
# frozen copy must equal the live replay output (no silent divergence)
try:
    CL = json.load(open(REPO / "thesis_eval/results_clean/tier1_results.json"))
    CHECKS += [("CLEAN frozen==live (svan pipeline)",
                C["svanstrom_clean"]["B_pipeline"]["clf->filt[robust8]"]["f1"],
                CL["svanstrom_clean"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
               ("CLEAN frozen==live (auv pipeline)",
                C["antiuav_clean"]["B_pipeline"]["clf->filt[robust8]"]["f1"],
                CL["antiuav_clean"]["B_pipeline"]["clf->filt[robust8]"]["f1"])]
except FileNotFoundError:
    pass

# round-5: resolution 2x2 (sec:resolution_arch, fig6_6_resolution) — one-harness sweep
R2 = json.load(open(REPO / "eval/results/svan_resolution_sweep.json"))
CHECKS += [
    ("RES baseline@640 R", 0.6838, R2["baseline@640"]["recall"]),
    ("RES baseline@1280 R", 0.9641, R2["baseline@1280"]["recall"]),
    ("RES retrained_v2@640 R", 0.0699, R2["retrained_v2@640"]["recall"]),
    ("RES retrained_v2@1280 R", 0.3234, R2["retrained_v2@1280"]["recall"]),
    ("RES n_frames", 4102, R2["baseline@640"]["n_frames"]),
]

# round-6: §3.8.8 IR-MRI separability table (tab:ir_mri_sep) vs mri/results/ir_v3b_report/stats.json
MRI = json.load(open(REPO / "mri/results/ir_v3b_report/stats.json"))
CHECKS += [
    ("MRI ir LDA", 0.981, round(MRI["separability"]["lda_train_accuracy"], 3)),
    ("MRI ir maxF", 5370, round(MRI["separability"]["max_anova_F"])),
    ("MRI ir medianF", 256, round(MRI["separability"]["median_anova_F"])),
    ("MRI ir n_drone", 14697, MRI["separability"]["n_drone"]),
    ("MRI ir n_confuser", 1386, MRI["separability"]["n_confuser"]),
    ("MRI ir halluc", 0.018, round(MRI["diagnosis"]["raw_halluc_rate"], 3)),
    ("MRI ir fp_cut", 0.89, round(MRI["diagnosis"]["fp_reduction"], 2)),
    ("MRI ir recall_ret", 0.997, round(MRI["diagnosis"]["classifier_recall_retention"], 3)),
]

# round-7: DUT Anti-UAV test-split 960 ablation (tab:ablation_dut) vs runs/results_dut frozen copy
D = json.load(open(REPO / "runs/results_dut/tier1_results.json"))["dut_antiuav_960"]
DB = D["B_pipeline"]; DA = D["A_bare"]
CHECKS += [
    ("DUT ft4 bare F1", 0.899, round(DA["ft4/rgb"]["f1"], 3)),
    ("DUT v3b-gray bare F1", 0.596, round(DA["v3b/ir"]["f1"], 3)),
    ("DUT fused bare F1", 0.758, round(DB["bare"]["f1"], 3)),
    ("DUT fused bare P", 0.864, round(DB["bare"]["precision"], 3)),
    ("DUT clf[robust8] F1", 0.763, round(DB["clf[robust8]"]["f1"], 3)),
    ("DUT clf[robust8] P", 0.895, round(DB["clf[robust8]"]["precision"], 3)),
    ("DUT clf->filt[robust8] P", 0.9, round(DB["clf->filt[robust8]"]["precision"], 3)),
    ("DUT filt->clf[robust8] P", 0.901, round(DB["filt->clf[robust8]"]["precision"], 3)),
]

# SHIPPED no-reject router robust8-nr: the production cells across every classifier-bearing table
_NR = "robust8_nr_drop"
DBN = TN["dut_antiuav_960"]["B_pipeline"]
CHECKS += [
    ("NR svan composed F1",  0.931, round(TN["svanstrom"]["B_pipeline"][f"clf->filt[{_NR}]"]["f1"], 3)),
    ("NR svan composed R",   0.957, round(TN["svanstrom"]["B_pipeline"][f"clf->filt[{_NR}]"]["recall"], 3)),
    ("NR svan composed P",   0.906, round(TN["svanstrom"]["B_pipeline"][f"clf->filt[{_NR}]"]["precision"], 3)),
    ("NR svan filt->clf F1", 0.946, round(TN["svanstrom"]["B_pipeline"][f"filt->clf[{_NR}]"]["f1"], 3)),
    ("NR antiuav composed F1", 0.984, round(TN["antiuav"]["B_pipeline"][f"clf->filt[{_NR}]"]["f1"], 3)),
    ("NR dut composed F1",   0.79, round(DBN[f"clf->filt[{_NR}]"]["f1"], 3)),
    ("NR dut composed R",    0.721, round(DBN[f"clf->filt[{_NR}]"]["recall"], 3)),
    ("NR dut filt->clf F1",  0.835, round(DBN[f"filt->clf[{_NR}]"]["f1"], 3)),
    ("NR rgb_conf fire",     0.0144, round(TN["rgb_confuser"]["C_confuser"][f"clf->filt[{_NR}]"]["fire_rate"], 4)),
    ("NR ir_conf fire",      0.028, round(TN["ir_confusers"]["C_confuser"][f"clf->filt[{_NR}]"]["fire_rate"], 3)),
    # shipped composition filt->clf on confusers: identical fire (no-reject router never vetoes a confuser frame)
    ("NR rgb_conf fire filt->clf", 0.0144, round(TN["rgb_confuser"]["C_confuser"][f"filt->clf[{_NR}]"]["fire_rate"], 4)),
    ("NR ir_conf fire filt->clf",  0.028, round(TN["ir_confusers"]["C_confuser"][f"filt->clf[{_NR}]"]["fire_rate"], 3)),
    ("NR rgb_test clf F1",   0.926, round(TN["rgb_dataset_test"]["S4_verifier"][f"clf[{_NR}]"]["f1"], 3)),
    ("NR rgb_test composed F1", 0.922, round(TN["rgb_dataset_test"]["S4_verifier"][f"clf->filt[{_NR}]"]["f1"], 3)),
    ("NR selcom composed F1", 0.612, round(TN["selcom_val"]["S4_verifier"][f"clf->filt[{_NR}]"]["f1"], 3)),
    ("NR ir_dset clf F1",    0.961, round(TN["ir_dset_final"]["S4_verifier"][f"clf[{_NR}]"]["f1"], 3)),
    ("NR video composed F1", 0.646, round(VN["video_drone"][f"clf->filt[{_NR}]"]["window"][2], 3)),
    ("NR video_conf fire",   0.213, round(VN["video_confuser"][f"clf->filt[{_NR}]"]["window_fire"], 4)),
    # session-8: svan IR-only row (the others — svan/antiuav RGB-only, antiuav IR-only, DUT — already covered)
    ("svan v3b IR-only F1", 0.940, round(T["svanstrom"]["A_bare"]["v3b/ir"]["f1"], 3)),
    # session-8b: split mlp filter-only rows (filt_mlp_rgb / filt_mlp_ir) in the 3 paired tables
    ("svan filt_mlp_rgb F1",   0.907, round(T["svanstrom"]["B_pipeline"]["filt_mlp_rgb"]["f1"], 3)),
    ("svan filt_mlp_ir F1",    0.742, round(T["svanstrom"]["B_pipeline"]["filt_mlp_ir"]["f1"], 3)),
    ("antiuav filt_mlp_rgb F1",0.973, round(T["antiuav"]["B_pipeline"]["filt_mlp_rgb"]["f1"], 3)),
    ("antiuav filt_mlp_ir F1", 0.973, round(T["antiuav"]["B_pipeline"]["filt_mlp_ir"]["f1"], 3)),
    ("dut filt_mlp_rgb F1",    0.722, round(DB["filt_mlp_rgb"]["f1"], 3)),
    ("dut filt_mlp_ir F1",     0.728, round(DB["filt_mlp_ir"]["f1"], 3)),
]

# session-8c: filter operating-point figure (fig:filter_operating) caption numbers vs the sweep JSON
FS = json.load(open(REPO / "eval/results/filter_operating_sweep.json"))
CHECKS += [
    ("FIG rgb recall@0.25",  0.956, round(FS["RGB mlp_v5"]["shipped"][0], 3)),
    ("FIG rgb fire@0.25",    0.011, round(FS["RGB mlp_v5"]["shipped"][1], 3)),
]

# Validate ALL numeric cells in one pass (must run AFTER every `CHECKS +=` block above; a loop
# placed earlier silently skipped the SWEEP/CLEAN/RES/CBAM cells while still counting them).
bad = 0
for label, claimed, actual in CHECKS:
    ok = abs(float(claimed) - float(actual)) < 5e-4
    if not ok:
        bad += 1
        print(f"  MISMATCH {label}: thesis={claimed}  json={actual}")

# PATH-EXISTENCE block (round 4): every artifact path the thesis prose/source-comments cite must
# exist in THIS repo (ES_Drone_Thesis is the final directory; a citation into the frozen archive
# or a moved file is an audit failure).
CITED_PATHS = [
    "eval/run_manifest.py",
    "eval/results/antiuav_per_model/baseline/manifest.json",
    "mri/results/v5_report_regen/stats.json",
    "mri/results/v5_report_regen/report.md",
    "mri/docs/mlp_v5_report_regen.md",
    "mri/modality_align.py",
    "mri/classifier.py",
    "mri/holdout.py",
    "docs/analysis/2026-05-26_classifier_ft4_analysis.md",
    "docs/analysis/2026-06-01_mlp_v5_recall_drop_mri.md",
    "label_reviewer/gui.py",
    "label_reviewer/core.py",
    "classifier/train_routing_robust.py",
    "models/routers/lean_ft4/fusion_dataset_lean19.csv",
    "thesis_eval/pipeline_cache_unified.py",
    "thesis_eval/pipeline_eval_unified.py",
    "thesis_eval/results/clean_split_manifest.json",
    "runs/README.md",
    "runs/tier1_results.json",
    "runs/leakage_controlled.json",
    "runs/clean_split/README.md",
    "runs/clean_split/clean_split_results.json",
    "runs/clean_split/svanstrom_clean_sequences.txt",
    "runs/clean_split/antiuav_clean_sequences.txt",
    "runs/clean_split/clean_split_manifest.json",
    "eval/results/svan_resolution_sweep.json",
    "runs/svan_resolution_sweep.json",
    "eval/svan_resolution_sweep.py",
    "thesis_eval/results_dut/tier1_results.json",
    "runs/results_dut/tier1_results.json",
    "eval/filter_operating_sweep.py",
    "eval/results/filter_operating_sweep.json",
    "docs/thesis_working_distilling_overleaf/figures/fig_filter_operating.pdf",
    "docs/thesis_working_distilling_overleaf/figures/fig_pipeline.tex",
    # shipped no-reject router (robust8-nr) artifacts
    "models/routers/robust8_noreject_drop/model.joblib",
    "classifier/train_robust8_noreject.py",
    "thesis_eval/results_noreject/tier1_results.json",
    "thesis_eval/results_noreject/temporal_results.json",
    "thesis_eval/results_noreject/notes_round1_results.json",
    "docs/analysis/2026-06-14_robust8_noreject.md",
]
missing = [p for p in CITED_PATHS if not (REPO / p).exists()]
for p in missing:
    print(f"  MISSING cited artifact: {p}")
bad += len(missing)
n_total = len(CHECKS) + len(CITED_PATHS)
print(f"{n_total - bad}/{n_total} checks pass ({len(CHECKS)} headline cells + {len(CITED_PATHS)} cited paths); {bad} failures")
