# Patch verifier

The patch-CNN confuser verifier (a MobileNetV3-small crop classifier), the predecessor to the distilled
MLP filter. It is kept for the ablations in the thesis.

- Weights: `models/patches/` (the deployable patch weights; training crops are excluded as regenerable)
- Trainer: `train_patch_verifier.py` (in this folder)

The trainer is self-contained (PyTorch, torchvision, scikit-learn) and has no in-repo imports.
