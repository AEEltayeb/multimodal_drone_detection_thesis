# RGB confuser filter: mlp_v5_v4

The production RGB confuser filter, a small MLP distilled from the RGB detector's P3 and P5 feature maps.
It vetoes detections that look like birds, airplanes, or helicopters.

- Weights: `models/verifiers/rgb_v5/mlp_v5_v4.pt` (operating point P(drone) 0.25)
- Trainer: `eval/build_balanced_v4_birdsplit.py`
- Feature extractor: `eval/distill_v5_p3p5_ft4.py`

The trainer and its feature extractor stay under `eval/` because `distill_v5_p3p5_ft4.py` is the same
feature extractor the eval pipeline imports; moving it would break that import web. This folder is the
pointer to it.
