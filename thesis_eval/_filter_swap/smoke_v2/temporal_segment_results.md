# Temporal / segment-level replay — real-video clips, production stack
2026-06-17 20:36 | grayrgb_paired regime (ft4 + v3b-on-gray, is_grayscale=1) | window 2-of-3 per clip | patch_thr=0.7 (per-frame approx of alert gate)
Old tab:cascade_segment rows used baseline RGB + ALERT-gated patch — detector & gate point differ; compare directions, not decimals.


## video_drone  (n=1359 frames, 9 clips, consecutive)

| cell | frame P/R/F1 | window P/R/F1 (2-of-3) | ΔR (win−frame) |
|---|---|---|---|
| bare | 0.9516/0.7502/0.839 | 0.9574/0.7531/0.843 | +0.003 |
| filt_mlp | 0.9505/0.5766/0.7178 | 0.9578/0.5756/0.7191 | -0.001 |
| filt_patch | 0.9525/0.6829/0.7955 | 0.9569/0.6893/0.8013 | +0.006 |
| clf[robust8] | 0.9586/0.5255/0.6789 | 0.9666/0.52/0.6762 | -0.005 |
| clf->filt[robust8] | 0.9611/0.4412/0.6048 | 0.9762/0.4358/0.6026 | -0.005 |
| clf[robust6] | 0.9873/0.5653/0.7189 | 0.9986/0.5838/0.7368 | +0.018 |
| clf->filt[robust6] | 0.9846/0.468/0.6344 | 0.9983/0.4759/0.6445 | +0.008 |
| clf[sa32] | 0.958/0.5734/0.7174 | 0.9674/0.5822/0.7269 | +0.009 |
| clf->filt[sa32] | 0.9668/0.4485/0.6127 | 0.9823/0.4538/0.6208 | +0.005 |
| clf[robust8_nr_drop] | 0.9516/0.7502/0.839 | 0.9574/0.7531/0.843 | +0.003 |
| clf->filt[robust8_nr_drop] | 0.9505/0.5766/0.7178 | 0.9578/0.5756/0.7191 | -0.001 |
| clf[robust8_nr_both] | 0.9516/0.7502/0.839 | 0.9574/0.7531/0.843 | +0.003 |
| clf->filt[robust8_nr_both] | 0.9505/0.5766/0.7178 | 0.9578/0.5756/0.7191 | -0.001 |
| clf->filt_patch[sa32] | 0.9585/0.5239/0.6775 | 0.9689/0.5348/0.6891 | +0.011 |

## video_confuser  (n=1250 frames, 10 clips, consecutive)

| cell | frame fire | window fire (2-of-3) |
|---|---|---|
| bare | 0.3632 | 0.3504 |
| filt_mlp | 0.2664 | 0.248 |
| filt_patch | 0.2264 | 0.1821 |
| clf[robust8] | 0.1288 | 0.1008 |
| clf->filt[robust8] | 0.108 | 0.0837 |
| clf[robust6] | 0.1016 | 0.0813 |
| clf->filt[robust6] | 0.0832 | 0.0683 |
| clf[sa32] | 0.1376 | 0.1203 |
| clf->filt[sa32] | 0.0864 | 0.0707 |
| clf[robust8_nr_drop] | 0.3632 | 0.3504 |
| clf->filt[robust8_nr_drop] | 0.2664 | 0.248 |
| clf[robust8_nr_both] | 0.3632 | 0.3504 |
| clf->filt[robust8_nr_both] | 0.2664 | 0.248 |
| clf->filt_patch[sa32] | 0.1 | 0.0797 |