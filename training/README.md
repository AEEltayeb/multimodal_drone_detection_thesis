# training/

One folder per model in the architecture, each holding (or pointing to) the script that trains it. The
trained weights for all of these live under [`models/`](../models/) and are committed; retraining needs a
GPU and the datasets, which are not in the repo.

| Model | Trainer | Folder |
|---|---|---|
| RGB detector `ft4` | `scripts/auto_confuser_ft4.py` (main confuser fine-tune) and `finetune_selcom.py` | [`rgb_detector_ft4/`](rgb_detector_ft4/) |
| IR detector `v3b` | the IR corrective fine-tune family | [`ir_detector_v3b/`](ir_detector_v3b/) |
| Trust router `robust8-nr` | `train_robust8_noreject.py` | [`trust_router_robust8nr/`](trust_router_robust8nr/) |
| RGB confuser filter `mlp_v5_v4` | `eval/build_balanced_v4_birdsplit.py` + `eval/distill_v5_p3p5_ft4.py` | [`rgb_filter_mlp_v5/`](rgb_filter_mlp_v5/) |
| IR / grayscale confuser filter | `mri/train_aligned.py` | [`ir_filter_aligned/`](ir_filter_aligned/) |
| Patch verifier | `train_patch_verifier.py` | [`patch_verifier/`](patch_verifier/) |

Some trainers physically live in this directory; others stay where they are because they share code with
the eval pipeline or the operator GUI, and moving them would break those imports. Each per-model folder
says exactly where its trainer is. The dataset-preparation scripts shared across the detectors are in
[`dataset_preparation/`](dataset_preparation/).
