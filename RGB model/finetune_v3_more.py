"""
finetune_v3_more.py — Continue training from v3 best weights for 7 more epochs.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = Path(r"G:/drone/finetune_dataset_v2/data.yaml")
BASE_MODEL = ROOT / "RGB model" / "Yolo26n_hardneg_v3" / "weights" / "best.pt"

def main():
    if not BASE_MODEL.exists():
        print(f"[fatal] Base model not found: {BASE_MODEL}")
        sys.exit(1)
        
    print("=" * 72)
    print(f"Continuing training from {BASE_MODEL}")
    print("=" * 72)
    
    model = YOLO(str(BASE_MODEL))
    
    train_kwargs = dict(
        data=str(DATA_YAML), 
        epochs=7, 
        patience=5,
        batch=4, 
        imgsz=640, 
        device=0, 
        amp=True,
        optimizer="AdamW", 
        lr0=0.0001, 
        lrf=0.01,
        freeze=10, 
        cos_lr=True, 
        close_mosaic=2,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
        save_period=1, 
        workers=2, 
        cache=False,
        project=str(ROOT / "RGB model"), 
        name="Yolo26n_hardneg_v3_more",
        pretrained=True, 
        exist_ok=True, 
        verbose=True,
    )
    
    for k, v in train_kwargs.items():
        print(f"  {k} = {v}")
        
    model.train(**train_kwargs)
    
    print("\n" + "=" * 72)
    print("ALL DONE.")
    print(f"Best checkpoint: {ROOT / 'RGB model' / 'Yolo26n_hardneg_v3_more' / 'weights' / 'best.pt'}")
    print("=" * 72)

if __name__ == "__main__":
    main()
