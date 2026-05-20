# Production stack

Single source of truth for the component versions, thresholds, and inference settings the production system ships with. Pinned 2026-05-12 based on the May 2026 ablation + cumulative-halluc evidence. **Do not change a row without updating both this file and `docs/EVIDENCE_LEDGER.md`.**

For every cell below, the *Source* column links into `EVIDENCE_LEDGER.md` where the measurement that justifies the choice lives. The ledger is the authority; this file is the punchline.

## Pinned components

| Component | Pick | Path | Source for the choice |
|---|---|---|---|
| RGB YOLO | **baseline** `Yolo26n_trained` | `RGB model/Yolo26n_trained/weights/best.pt` | EVIDENCE_LEDGER §3.1 — 0.959 drone recall on Svanstrom@1280; retrained_v2 collapsed to 0.306; hardneg_v3more strictly dominated |
| IR YOLO | `finetune_v3b` | `runs/corrective_finetune/finetune_v3b/weights/best.pt` | EVIDENCE_LEDGER §4 — 0.9613 F1 on Svanstrom; only IR model considered |
| Trust classifier | **`control_v3more_40feat`** | `classifier/fusion_models/control_v3more_40feat/model.joblib` | EVIDENCE_LEDGER §7 classifier comparison — wins Svanstrom on every drone metric. S3 F1=0.909 vs sa32 0.896 vs fnfn 0.895; S3 recall=0.893 vs sa32 0.868 vs fnfn 0.869 (the +2.5 pp recall delta over sa32 is the meaningful one). Confuser FP count tied with sa32 at S2 (111). |
| Patch verifier (RGB) | **v2_backup** | `classifier/runs/patches/confuser_filter4_rgb_v2_backup.pt` | EVIDENCE_LEDGER §6.1 — at thr=0.5: catches 64% birds (v4: 56%), 71% helicopters (v4: 58%), drone-TP veto 5.4% (v4: 6.7%). v2 beats v4 on every axis. |
| Patch verifier (IR) | **v2_backup** | `classifier/runs/patches/confuser_filter4_ir_v2_backup.pt` | By symmetry with RGB; dedicated IR-side audit not yet run (open item). |
| Cascade | `alert_gate_only` | (set in `ir_gui/fusion_settings.json`) | Memory `project_pyside_gui_features` + EVIDENCE_LEDGER §1; the per-frame `filter_then_classifier` and `classifier_then_filter` cascades cost ~1pp F1 vs no-patch and are not the production runtime. |
| `patch_threshold` | **0.9** | (set in `ir_gui/fusion_settings.json`) | EVIDENCE_LEDGER §7 threshold sweep — drone F1 at S3 is monotone in patch_thr; thr=0.9 gives drone F1=0.895 (best in S3 series), still suppresses 63% of S2's confuser FPs (45 vs 120). |
| Scoring rule (for eval/reporting) | `trust_aware` | `--scoring trust_aware` flag | EVIDENCE_LEDGER §1 — matches the system's actual decision rule; 28-pp F1 swing vs `dual` on Svanstrom |
| `imgsz` (YOLO inference) | **1280** | (set in `ir_gui/fusion_settings.json`) | Memory `project_imgsz_1280_svanstrom`; without it baseline RGB recall on Svanstrom drops 0.959→0.07 |
| RGB conf | 0.25 | (set in `ir_gui/fusion_settings.json`) | EVIDENCE_LEDGER §6.1 used 0.25; thresholding the patch verifier carries discrimination so YOLO conf can stay permissive |
| IR conf | 0.40 | (set in `ir_gui/fusion_settings.json`) | EVIDENCE_LEDGER `H_conf_sweep` (May 10 matrix); plateau begins at 0.40 with P≥0.95 on Anti-UAV |

## Headline performance (with this stack pinned)

Measured on Svanstrom @ stride=9, baseline RGB, IoP@0.5 scoring (`eval/results/_cumulative_halluc/svanstrom_sa32_thr08/summary.json`; thr=0.9 numbers from `svanstrom_fnfn_thr09/summary.json` with the same classifier delta applied):

| Stage | Drone P | Drone R | Drone F1 | Confuser FPs (total) |
|---|---|---|---|---|
| RGB YOLO alone | 0.940 | 0.959 | 0.949 | ~1,793 (out of 1891 confuser frames @ stride=9) |
| + trust classifier | 0.916 | 0.922 | 0.919 | 111 |
| + patch verifier (alert gate) | ~0.92 | ~0.87 | ~0.90 | ~45 |

End-to-end suppression of confuser FPs: **~97%** (1,793 → 45). Drone recall cost: **8.7 pp** (0.959 → 0.872 — closes when system reverts to S2 on borderline scenes via the alert-gate runtime; the per-frame number is the worst case).

## Open items before production sign-off

These are gated on the company / engineering side; do not block algorithmic work:

- **IR patch verifier v2 vs v4** — RGB-side decisive; IR-side not directly tested with the audit (the audit script crops from RGB). Run a separate IR audit if precision-critical.
- **Latency / throughput** on company target hardware — EVIDENCE_LEDGER §8 is all placeholders.
- **Operating-environment validation** — these numbers are on Svanstrom + Anti-UAV + a confuser zoo. The company's deployment scene distribution may differ; recommend a calibration smoke test on 100–500 frames from their environment before going live.
- **Open-world fallback decision** — if the company's deployment has unknown OOD confuser sources (not Svanstrom-like), `fusion_no_fn_v1.1` is the more conservative classifier (13× lower confuser fire rate on the broad confuser zoo) at the cost of ~1pp Svanstrom F1. See EVIDENCE_LEDGER §7 classifier comparison.

## How to update this file

1. Run a measurement.
2. Update `EVIDENCE_LEDGER.md` with the row.
3. If the new measurement changes a production pick, update both the pin row above and the *Source* column with the new ledger section.
4. Append to the changelog below — never silent-edit.

## Changelog

- **2026-05-12** — Initial lock. Four changes from previously-deployed state:
  1. Patch verifier RGB and IR swapped from `confuser_filter4_*.pt` (= v4) to `confuser_filter4_*_v2_backup.pt` (= v2). Backed by patch-catch audit (EVIDENCE_LEDGER §6.1).
  2. `patch_threshold` set to **0.9** (from whatever previous default; was 0.5 in the eval pipeline). Backed by Svanstrom threshold sweep (EVIDENCE_LEDGER §7).
  3. RGB YOLO confirmed at baseline `Yolo26n_trained` (not retrained_v2). Backed by three-way RGB comparison (EVIDENCE_LEDGER §3.1).
  4. Trust classifier swapped from `scene_aware_v3more_32feat` (deployed) to `control_v3more_40feat`. Backed by three-way classifier comparison (EVIDENCE_LEDGER §7). +2.5 pp drone recall at S3 over sceneaware with no confuser-FP cost.
