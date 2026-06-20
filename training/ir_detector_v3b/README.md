# IR detector: v3b

The production thermal-IR detector, the corrective fine-tune `v3b`.

- Weights: `models/ir/corrective_finetune/finetune_v3b/weights/best.pt`
- Fine-tune scripts (the IR detector fine-tune family that produced v3b): `finetune_v3_more.py`,
  `finetune_run_v2.py`, `finetune_freeze8_after.py` (in this folder)

These are standard YOLO fine-tune drivers (Ultralytics). The v3b weights are the committed result; the
exact fine-tune invocation is documented in the thesis methodology chapter.
