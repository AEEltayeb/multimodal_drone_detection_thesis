# Background failure profile (manual seq-level tags, coarse 3-class)
Tags: sky / horizon (sky + ground strip) / ground-dominant. One representative frame per sequence tagged by one annotator (2026-06-11); the tag propagates to all cached frames of the sequence. Bare detectors, per modality vs own GT, Tier-1 rules.


## svanstrom (n=4000, rule=iop)

**ft4/rgb**

| background | n_frames | n_gt | P | R | F1 | FP-frame rate |
|---|---|---|---|---|---|---|
| confuser-seqs/ground | 147 | 0 | 0.0 | — | — | 0.6463 |
| confuser-seqs/horizon | 888 | 0 | 0.0 | — | — | 0.741 |
| confuser-seqs/sky | 1294 | 0 | 0.0 | — | — | 0.6669 |
| drone-seqs/ground | 163 | 163 | 1.0 | 0.9325 | 0.9651 | 0.0 |
| drone-seqs/horizon | 1114 | 1116 | 0.9661 | 0.9444 | 0.9551 | 0.0332 |
| drone-seqs/sky | 394 | 394 | 0.9183 | 0.8274 | 0.8705 | 0.0736 |

**v3b/ir**

| background | n_frames | n_gt | P | R | F1 | FP-frame rate |
|---|---|---|---|---|---|---|
| confuser-seqs/ground | 147 | 0 | 0.0 | — | — | 0.0 |
| confuser-seqs/horizon | 888 | 0 | 0.0 | — | — | 0.0169 |
| confuser-seqs/sky | 1294 | 0 | 0.0 | — | — | 0.0062 |
| drone-seqs/ground | 163 | 163 | 0.8512 | 0.8773 | 0.864 | 0.1534 |
| drone-seqs/horizon | 1114 | 1098 | 0.9275 | 0.9909 | 0.9582 | 0.0727 |
| drone-seqs/sky | 394 | 380 | 0.9024 | 0.9974 | 0.9475 | 0.1015 |


## video_drone (n=1359, rule=iop)

**ft4/rgb**

| background | n_frames | n_gt | P | R | F1 | FP-frame rate |
|---|---|---|---|---|---|---|
| drone-seqs/ground | 792 | 707 | 0.915 | 0.6393 | 0.7527 | 0.0518 |
| drone-seqs/horizon | 182 | 183 | 0.6562 | 0.3443 | 0.4516 | 0.1813 |
| drone-seqs/sky | 385 | 344 | 0.5968 | 0.5465 | 0.5706 | 0.2597 |
