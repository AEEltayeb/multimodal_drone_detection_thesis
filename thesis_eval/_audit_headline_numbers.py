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

CHECKS = [
    # (label, claimed, actual)
    ("svan bare F1", 0.7415, T["svanstrom"]["B_pipeline"]["bare"]["f1"]),
    ("svan composed F1", 0.9485, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
    ("svan composed P", 0.9388, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["precision"]),
    ("svan composed R", 0.9584, T["svanstrom"]["B_pipeline"]["clf->filt[robust8]"]["recall"]),
    ("svan bare R", 0.9481, T["svanstrom"]["B_pipeline"]["bare"]["recall"]),
    ("svan filt->clf F1", 0.9629, T["svanstrom"]["B_pipeline"]["filt->clf[robust8]"]["f1"]),
    ("svan clf[robust6] F1", 0.9514, T["svanstrom"]["B_pipeline"]["clf[robust6]"]["f1"]),
    ("svan clf[sa32] F1", 0.9666, T["svanstrom"]["B_pipeline"]["clf[sa32]"]["f1"]),
    ("antiuav bare F1", 0.9728, T["antiuav"]["B_pipeline"]["bare"]["f1"]),
    ("antiuav composed F1", 0.9844, T["antiuav"]["B_pipeline"]["clf->filt[robust8]"]["f1"]),
    ("antiuav ft4 bare F1", 0.9853, T["antiuav"]["A_bare"]["ft4/rgb"]["f1"]),
    ("antiuav ft4 FP", 41, T["antiuav"]["A_bare"]["ft4/rgb"]["FP"]),
    ("antiuav v3b F1", 0.961, T["antiuav"]["A_bare"]["v3b/ir"]["f1"]),
    ("rgbconf bare fire", 0.3035, T["rgb_confuser"]["C_confuser"]["bare"]["fire_rate"]),
    ("rgbconf clf[r8] fire", 0.049, T["rgb_confuser"]["C_confuser"]["clf[robust8]"]["fire_rate"]),
    ("rgbconf mlp fire", 0.0106, T["rgb_confuser"]["C_confuser"]["filt_mlp"]["fire_rate"]),
    ("rgbconf mlp FP", 29, T["rgb_confuser"]["C_confuser"]["filt_mlp"]["FP"]),
    ("rgbconf patch fire", 0.1022, T["rgb_confuser"]["C_confuser"]["filt_patch"]["fire_rate"]),
    ("rgbconf composed fire", 0.0015, T["rgb_confuser"]["C_confuser"]["clf->filt[robust8]"]["fire_rate"]),
    ("rgbconf composed FP", 4, T["rgb_confuser"]["C_confuser"]["clf->filt[robust8]"]["FP"]),
    ("irconf bare fire", 0.2943, T["ir_confusers"]["C_confuser"]["bare"]["fire_rate"]),
    ("irconf composed r8 fire", 0.2167, T["ir_confusers"]["C_confuser"]["clf->filt[robust8]"]["fire_rate"]),
    ("irconf composed r6 fire", 0.1792, T["ir_confusers"]["C_confuser"]["clf->filt[robust6]"]["fire_rate"]),
    ("grayconf bare fire", 0.2378, T["gray_confuser"]["C_confuser"]["bare"]["fire_rate"]),
    ("grayconf mlp fire", 0.0076, T["gray_confuser"]["C_confuser"]["filt_mlp"]["fire_rate"]),
    ("grayconf mlp FP", 21, T["gray_confuser"]["C_confuser"]["filt_mlp"]["FP"]),
    ("3way RGB F1", 0.6067, T["svanstrom"]["A_bare"]["ft4/rgb"]["f1"]),
    ("3way rawrgb F1", 0.1874, T["svanstrom_rawrgb"]["A_bare"]["v3b/ir"]["f1"]),
    ("3way gray F1", 0.5796, T["svanstrom_gray"]["A_bare"]["v3b/ir"]["f1"]),
    ("rgbtest bare F1", 0.9259, T["rgb_dataset_test"]["S4_verifier"]["bare"]["f1"]),
    ("rgbtest mlp F1", 0.8092, T["rgb_dataset_test"]["S4_verifier"]["filt_mlp"]["f1"]),
    ("rgbtest patch F1", 0.8898, T["rgb_dataset_test"]["S4_verifier"]["filt_patch"]["f1"]),
    ("selcom bare F1", 0.5911, T["selcom_val"]["S4_verifier"]["bare"]["f1"]),
    ("selcom mlp F1", 0.6115, T["selcom_val"]["S4_verifier"]["filt_mlp"]["f1"]),
    ("irtest bare F1", 0.961, T["ir_dset_final"]["S4_verifier"]["bare"]["f1"]),
    ("irtest mlp F1", 0.9578, T["ir_dset_final"]["S4_verifier"]["filt_mlp"]["f1"]),
    ("irtest patch F1", 0.9398, T["ir_dset_final"]["S4_verifier"]["filt_patch"]["f1"]),
    ("video bare win F1", 0.843, V["video_drone"]["bare"]["window"][2]),
    ("video r6 win F1", 0.7368, V["video_drone"]["clf[robust6]"]["window"][2]),
    ("video composed r8 F1", 0.56, V["video_drone"]["clf->filt[robust8]"]["window"][2]),
    ("video replica F1", 0.6891, V["video_drone"]["clf->filt_patch[sa32]"]["window"][2]),
    ("video bare fire", 0.3504, V["video_confuser"]["bare"]["window_fire"]),
    ("video r6 fire", 0.0813, V["video_confuser"]["clf[robust6]"]["window_fire"]),
    ("video composed r8 fire", 0.0732, V["video_confuser"]["clf->filt[robust8]"]["window_fire"]),
    ("sweep r8@0.01 R", 0.507, W["robust8@0.01"][1]),
    ("sweep r6@0.01 F1", 0.7309, W["robust6@0.01"][2]),
    # notes-round-1 additions (modality A/B coverage, per-size, per-category, background profile)
    ("AB svan rgb_only F1", 0.4579, N["svanstrom"]["M_modality_ab"]["rgb_only bare"]["f1"]),
    ("AB svan ir_only F1", 0.6316, N["svanstrom"]["M_modality_ab"]["ir_only bare"]["f1"]),
    ("AB svan both bare F1", 0.7415, N["svanstrom"]["M_modality_ab"]["both bare"]["f1"]),
    ("AB svan both+filt F1", 0.9071, N["svanstrom"]["M_modality_ab"]["both +filt"]["f1"]),
    ("AB svan routed bare F1", 0.9148, N["svanstrom"]["M_modality_ab"]["routed[robust8] bare"]["f1"]),
    ("AB svan routed+filt F1", 0.9206, N["svanstrom"]["M_modality_ab"]["routed[robust8] +filt"]["f1"]),
    ("AB svan rgb_only+filt F1", 0.5818, N["svanstrom"]["M_modality_ab"]["rgb_only +filt"]["f1"]),
    ("AB antiuav both bare F1", 0.9728, N["antiuav"]["M_modality_ab"]["both bare"]["f1"]),
    ("AB antiuav routed+filt F1", 0.9733, N["antiuav"]["M_modality_ab"]["routed[robust8] +filt"]["f1"]),
    ("AB antiuav rgb_only F1", 0.6432, N["antiuav"]["M_modality_ab"]["rgb_only bare"]["f1"]),
    ("AB antiuav ir_only F1", 0.6519, N["antiuav"]["M_modality_ab"]["ir_only bare"]["f1"]),
    ("SZ rgbtest <16 bare R", 0.7824, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"]["<16px"]["recall"]),
    ("SZ rgbtest <16 filt R", 0.2562, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["filt"]["<16px"]["recall"]),
    ("SZ rgbtest 16-32 bare R", 0.8649, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"]["16-32px"]["recall"]),
    ("SZ rgbtest 16-32 filt R", 0.4465, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["filt"]["16-32px"]["recall"]),
    ("SZ rgbtest >=64 bare R", 0.9555, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["bare"][">=64px"]["recall"]),
    ("SZ rgbtest >=64 filt R", 0.9506, N["rgb_dataset_test"]["SZ_per_size"]["ft4/rgb"]["buckets"]["filt"][">=64px"]["recall"]),
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

bad = 0
for label, claimed, actual in CHECKS:
    ok = abs(float(claimed) - float(actual)) < 5e-4
    if not ok:
        bad += 1
        print(f"  MISMATCH {label}: thesis={claimed}  json={actual}")
print(f"{len(CHECKS) - bad}/{len(CHECKS)} headline cells verified against replay JSONs; {bad} mismatches")
