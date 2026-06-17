# Temporal / segment-level replay — real-video clips, production stack
2026-06-17 23:58 | grayrgb_paired regime (ft4 + v3b-on-gray, is_grayscale=1) | window 2-of-3 per clip | patch_thr=0.7 (per-frame approx of alert gate)
Old tab:cascade_segment rows used baseline RGB + ALERT-gated patch — detector & gate point differ; compare directions, not decimals.


## video_drone  (n=1359 frames, 9 clips, consecutive)

| cell | frame P/R/F1 | window P/R/F1 (2-of-3) | ΔR (win−frame) |
|---|---|---|---|
| bare | 0.9516/0.7502/0.839 | 0.9574/0.7531/0.843 | +0.003 |
| filt_mlp | 0.9475/0.5126/0.6653 | 0.9537/0.489/0.6465 | -0.024 |
| filt_patch | 0.9525/0.6829/0.7955 | 0.9569/0.6893/0.8013 | +0.006 |
| clf[robust8] | 0.9586/0.5255/0.6789 | 0.9666/0.52/0.6762 | -0.005 |
| clf->filt[robust8] | 0.9606/0.3958/0.5606 | 0.9746/0.3769/0.5436 | -0.019 |
| clf[robust6] | 0.9873/0.5653/0.7189 | 0.9986/0.5838/0.7368 | +0.018 |
| clf->filt[robust6] | 0.9868/0.4242/0.5933 | 0.998/0.4121/0.5833 | -0.012 |
| clf[sa32] | 0.958/0.5734/0.7174 | 0.9674/0.5822/0.7269 | +0.009 |
| clf->filt[sa32] | 0.9675/0.4104/0.5763 | 0.9838/0.3982/0.5669 | -0.012 |
| clf[robust8_nr_drop] | 0.9516/0.7502/0.839 | 0.9574/0.7531/0.843 | +0.003 |
| clf->filt[robust8_nr_drop] | 0.9475/0.5126/0.6653 | 0.9537/0.489/0.6465 | -0.024 |
| clf[robust8_nr_both] | 0.9516/0.7502/0.839 | 0.9574/0.7531/0.843 | +0.003 |
| clf->filt[robust8_nr_both] | 0.9475/0.5126/0.6653 | 0.9537/0.489/0.6465 | -0.024 |
| clf->filt_patch[sa32] | 0.9585/0.5239/0.6775 | 0.9689/0.5348/0.6891 | +0.011 |

## video_confuser  (n=1250 frames, 10 clips, consecutive)

| cell | frame fire | window fire (2-of-3) |
|---|---|---|
| bare | 0.3632 | 0.3504 |
| filt_mlp | 0.236 | 0.213 |
| filt_patch | 0.2264 | 0.1821 |
| clf[robust8] | 0.1288 | 0.1008 |
| clf->filt[robust8] | 0.0976 | 0.0756 |
| clf[robust6] | 0.1016 | 0.0813 |
| clf->filt[robust6] | 0.0752 | 0.0585 |
| clf[sa32] | 0.1376 | 0.1203 |
| clf->filt[sa32] | 0.0808 | 0.0634 |
| clf[robust8_nr_drop] | 0.3632 | 0.3504 |
| clf->filt[robust8_nr_drop] | 0.236 | 0.213 |
| clf[robust8_nr_both] | 0.3632 | 0.3504 |
| clf->filt[robust8_nr_both] | 0.236 | 0.213 |
| clf->filt_patch[sa32] | 0.1 | 0.0797 |