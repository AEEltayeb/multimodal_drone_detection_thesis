# RGB detector: ft4

The production RGB detector, a YOLO26n fine-tuned on the SelCom surveillance set plus mined confusers.

- Weights: `models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt`
- Main confuser fine-tune: `scripts/auto_confuser_ft4.py` (kept under `scripts/` because it imports a
  helper from that folder)
- SelCom fine-tune step: `finetune_selcom.py` (in this folder)
- Dataset preparation: `training/dataset_preparation/build_selcom_confuser_ft4.py`
