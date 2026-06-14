# The held-out CLEAN SPLIT — definition, derivation, and traceability

This directory is the fixed, citable record of the leakage-controlled test split used in the
thesis's Training–Evaluation Overlap Audit (§3.3). Everything needed to know *exactly which
frames* the clean numbers were computed on, and to re-derive the split from scratch, is here.

## What the split is

**Definition: every evaluation sequence with ZERO frames in `G:/drone/IR_dset_final/train/images`
(the IR detector's training split).** Exclusion is at SEQUENCE level, not frame level, because
the training corpus holds stride-sampled siblings of most evaluation frames (a frame 3 frames
away from a training frame is near-identical; excluding individual frames would still leak).

| surface | sequences kept | frames | evaluated | clean for |
|---|---|---|---|---|
| `svanstrom_clean` | 54 of 279 | 5,557 of 28,710 | ALL (full-frame, no striding) | **both detectors** (zero Svanström exists in any RGB training corpus) |
| `antiuav_clean`   | 61 of 91  | 57,542 of 85,374 | ALL (full-frame, no striding) | **IR detector only** — the composite RGB corpus contains `anti_uav_*` material from all 91 eval segments (59,226 train frames; 13,676 exact visible frames = 16.0% of the eval test dir) |

Known residual overlap (disclosed, not removed): the trust router's training rows were mined
from both surfaces (214/273 svan seqs, 61/90 auv segments are router-training sequences);
sequence-stratified splitting protects the router's own train/test protocol but cascade
evaluation still runs on router-training sequences.

## Files in this directory

- `svanstrom_clean_sequences.txt` — the 54 sequence IDs (e.g. `IR_DRONE_046`), one per line.
- `antiuav_clean_sequences.txt` — the 61 segment IDs (e.g. `20190925_124000_1_2`), one per line.
- `clean_split_manifest.json` — machine-readable: lists + counts + definition string
  (auto-written by the cache builder at split-construction time).
- `clean_split_results.json` / `.md` — frozen replay results on the split (all pipeline arms,
  bootstrap CIs). The thesis §3.3 numbers are audited against this JSON.

## How to re-derive the split from scratch

1. The contaminated-ID scan (shared function, used by both the audit and the cache builder):
   `thesis_eval/leakage_controlled_replay.py :: train_contaminated_ids()` — one pass over the
   `IR_dset_final/train/images` listing; a sequence is contaminated if ANY filename matches it.
   - Svanström sequence ID regex: `IR_[A-Z]+_\d+`   (catches `svan_*` and `czoom_svan_*` copies)
   - Anti-UAV segment ID regex: `20\d{6}_\d{6}_\d+_\d+` (catches `dv5_auv_*` and `czoom_dv5_auv_*`)
2. The clean split = (all sequences in the eval corpus) MINUS (contaminated set). Implemented in
   `thesis_eval/pipeline_cache_unified.py :: iter_clean()` for the two surfaces
   `svanstrom_clean` / `antiuav_clean`; it writes `clean_split_manifest.json` whenever it runs.
3. Build the detection cache (GPU; run FROM the ES_Drone_Thesis root):
   `py -u thesis_eval/pipeline_cache_unified.py --only svanstrom_clean,antiuav_clean --full --no-patch --overwrite`
   (the `--no-patch` cache stores zero patch scores; `filt_patch` cells are NOT valid on it)
4. Replay all pipeline arms (zero-GPU, ~2.5 min):
   `py -u thesis_eval/pipeline_eval_unified.py --only svanstrom_clean,antiuav_clean --out thesis_eval/results_clean`
5. Verify: `py thesis_eval/_audit_headline_numbers.py` asserts the thesis's §3.3 cells equal
   `clean_split_results.json` (frozen here) and that this directory's files exist.

## Headline outcome (details in clean_split_results.md)

- Anti-UAV clean: bare 0.977 / production pipeline 0.986 — at or slightly above the full-surface
  headline (0.973 / 0.984). No inflation.
- Svanström clean: IR detector solo 0.867 vs 0.940 headline (≤7.3pp inflation; the leakage-free
  RGB detector drops 3.5pp on the same sequences, so the sequences are partly just harder);
  production pipeline 0.935 vs 0.949 headline; lift preserved 0.684 → 0.935.
