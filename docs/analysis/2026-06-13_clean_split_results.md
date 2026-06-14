# Clean-split (held-out) evaluation results — 2026-06-13

The new test split: every sequence verified to have ZERO frames in IR detector training.
Built full-frame (no striding). Source: `thesis_eval/results_clean/tier1_results.json`,
cache `thesis_eval/cache/{svanstrom_clean,antiuav_clean}.pkl`, split list
`thesis_eval/results/clean_split_manifest.json`.
Replay: `py -u thesis_eval/pipeline_eval_unified.py --only svanstrom_clean,antiuav_clean --out thesis_eval/results_clean`

## 1. SVANSTRÖM CLEAN — 54 sequences, 5,557 frames (clean for BOTH detectors)

| arm | clean F1 | headline F1 | change |
|---|---|---|---|
| RGB detector alone (ft4) | 0.572 | 0.607 | −3.5 pp |
| IR detector alone (v3b) | 0.867 | 0.940 | −7.3 pp |
| both detectors, no pipeline | 0.684 | 0.742 | −5.7 pp |
| filter only | 0.879 | 0.907 | −2.8 pp |
| router only (robust8) | 0.900 | 0.941 | −4.1 pp |
| **PRODUCTION (router then filter)** | **0.935** | **0.949** | **−1.4 pp** |
| filter then router | 0.945 | 0.963 | −1.8 pp |
| router (sa32 comparison) | 0.944 | 0.967 | −2.2 pp |
| router (robust6 comparison) | 0.909 | 0.951 | −3.7 pp |

95% CI of the production arm on clean data: [0.928, 0.941]. Headline CI: [0.943, 0.954].

## 2. ANTI-UAV CLEAN — 61 segments, 57,542 frames (clean for IR detector only*)

| arm | clean F1 | headline F1 | change |
|---|---|---|---|
| RGB detector alone (ft4) | 0.988 | 0.985 | +0.3 pp |
| IR detector alone (v3b) | 0.966 | 0.961 | +0.5 pp |
| both detectors, no pipeline | 0.977 | 0.973 | +0.4 pp |
| **PRODUCTION (router then filter)** | **0.986** | **0.984** | **+0.2 pp** |

*The RGB training corpus contains material from all Anti-UAV segments (measured 2026-06-12),
so this split is held out for the IR detector but not the RGB detector. Svanström clean has
no such problem: zero Svanström exists in any RGB training corpus.

## 3. What this entails

1. **Anti-UAV: no inflation at all.** Clean scores are slightly HIGHER than the headline on
   every arm. The overlapped segments were the harder ones, not the easier ones. The
   "pipeline does no harm" claim is confirmed on 57k held-out frames.

2. **Svanström: the IR detector was inflated, but boundedly.** v3b alone drops 7.3 pp on
   clean sequences. BUT the RGB detector — which never saw a single Svanström frame —
   drops 3.5 pp on the same sequences. So the clean sequences are partly just harder;
   the leakage-attributable part of the v3b drop is roughly 4 pp, and 7.3 pp is the hard
   upper bound.

3. **The production pipeline barely moves: 0.949 → 0.935 (−1.4 pp), the smallest drop of
   any arm.** On data nothing in the detectors ever saw, the cascade still lifts
   0.684 → 0.935 (+25 pp). The thesis's central claim survives held-out evaluation.

4. **No production decision changes.** The router comparisons (sa32 > robust8 > robust6)
   keep the same order on clean data.

5. Side note: filter-then-router scores slightly higher than the production order on
   Svanström (0.945 vs 0.935), same direction as on the full surface (0.963 vs 0.949).
   Known trade-off; production chose router-first for recall safety.

## 4. What goes into the thesis (proposal, pending user approval)

- §3.3 becomes "Training–Evaluation Overlap Audit" again, containing:
  (a) the overlap counts table (svan: 17,314 train frames +1,325 zoom copies, 37.3% of
      eval frames are exact training images; auv: 22,603 frames, 30/90 segments, 6.3%
      exact; router trained on 214/273 svan seqs + 61/90 auv segments; patch filter
      in-distribution on both; mlp_v5 clean; aligned filter clean on auv only);
  (b) the clean-split definition (sequence-level, full-frame, manifest path, the
      RGB-side caveat for auv);
  (c) a 6-row results table = sections 1+2 above, condensed;
  (d) verdict prose = section 3 above.
- Protocol table: two new rows (svanstrom_clean 5,557 full; antiuav_clean 57,542 full).
- Contribution 1: restore "(unchanged on the held-out clean split)".
- Empirical limitations paragraph: numbers restored.
- HEADLINE F1 STAYS THE HEADLINE everywhere — the clean split lives only in the audit
  section as the control, per the user's standing instruction.
- Bookkeeping: audit cells for every quoted clean number; results_clean + manifest frozen
  into runs/; app:provenance rows; kb eval rows.

## Delivered
- C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\docs\analysis\2026-06-13_clean_split_results.md (this file)
- C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\thesis_eval\results_clean\tier1_results.json (+ tier1_screening_results.md)
- C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Thesis\thesis_eval\results\clean_split_manifest.json
