"""Regenerate the LEGACY 'drone detected' example frames with the PRODUCTION stack.

Production stack drawn here:
  - RGB detector : ft4  (models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt)
  - RGB filter   : mlp_v5_v4 (models/verifiers/rgb_v5/mlp_v5_v4.pt), per-frame, P(drone) >= 0.25

The original B_drone_detected_field.jpg / B_drone_detected_sky.jpg came from
training/drone_detection.py, which loads the LEGACY baseline detector
(Yolo26n_trained) at imgsz=640 with NO confuser filter / trust router. This script
re-runs the production detector + filter on the CLEAN source frame and draws an
emerald box for every detection the filter KEEPS (P(drone) >= 0.25), with the
P(drone) score, so the shown box is a real production-pipeline output.

Run (from repo root):
  py site/public/media/examples/_regen_prod_frames.py
"""
import os
import sys
import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
os.chdir(REPO)
sys.path.insert(0, "classifier")

from ultralytics import YOLO                                   # noqa: E402
from mlp_verifier import DetectInputHook, MLPVerifier          # noqa: E402

RGB_W = "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt"
MLP_W = "models/verifiers/rgb_v5/mlp_v5_v4.pt"
RGB_THR = 0.25          # production deploy threshold for mlp_v5_v4
DET_CONF = 0.25         # detector confidence floor (matches the cache / GUI)
IMGSZ = 1280            # CCTV / sky-surveillance canonical imgsz
OUTDIR = "site/public/media/examples"

EMERALD = (90, 220, 120)   # BGR ~ tailwind emerald
ROSE = (90, 90, 235)       # BGR ~ tailwind rose (for vetoed, if ever shown)


def grab_frame(video_path, frame_idx):
    c = cv2.VideoCapture(video_path)
    c.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, fr = c.read()
    c.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {video_path}")
    return fr


def annotate(model, hook, ver, img, out_name, label_keep="DRONE"):
    H, W = img.shape[:2]
    res = model.predict(img, imgsz=IMGSZ, conf=DET_CONF, verbose=False)[0]
    dets = [[*b.xyxy[0].tolist(), float(b.conf[0])] for b in res.boxes]
    out = img.copy()
    kept = 0
    if dets:
        probs = ver.score_dets(hook, dets, (H, W))
        for (x1, y1, x2, y2, conf), p in zip(dets, probs):
            keep = p >= RGB_THR
            color = EMERALD if keep else ROSE
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 3)
            tag = f"{label_keep} P(drone)={p:.2f}" if keep else f"VETO P(drone)={p:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ty = max(0, y1i - 10)
            cv2.rectangle(out, (x1i, ty - th - 6), (x1i + tw + 6, ty + 2), color, -1)
            cv2.putText(out, tag, (x1i + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (20, 20, 20), 2, cv2.LINE_AA)
            kept += int(keep)
            print(f"  det conf={conf:.3f} P(drone)={p:.3f} -> {'KEEP' if keep else 'VETO'} "
                  f"box=({x1i},{y1i},{x2i},{y2i})")
    op = os.path.join(OUTDIR, out_name)
    cv2.imwrite(op, out)
    print(f"  wrote {op}  ({kept}/{len(dets)} kept by filter)")
    return kept, len(dets)


def main():
    print(f"loading detector {RGB_W}")
    model = YOLO(RGB_W)
    hook = DetectInputHook()
    hook.register(model)
    print(f"loading filter {MLP_W}  (threshold {RGB_THR})")
    ver = MLPVerifier(MLP_W)

    # --- (1) FIELD frame: clean source present in repo ---
    field_video = "training/video_test/gopro_006.mp4"
    if os.path.exists(field_video):
        print("[field] gopro_006.mp4 frame 444")
        img = grab_frame(field_video, 444)
        annotate(model, hook, ver, img, "prod_B_drone_detected_field.jpg")
    else:
        print(f"[field] SKIP - source video not found: {field_video}")

    # --- (2) SKY frame: source clip GOPR5844_002.mp4 NOT in repo ---
    # If the user supplies it (set SKY_VIDEO env or drop the file), regenerate too.
    sky_video = os.environ.get("SKY_VIDEO", "training/video_test/GOPR5844_002.mp4")
    if os.path.exists(sky_video):
        print(f"[sky] {sky_video} frame 72")
        img = grab_frame(sky_video, 72)
        annotate(model, hook, ver, img, "prod_B_drone_detected_sky.jpg")
    else:
        print(f"[sky] SKIP - source clip not in repo: {sky_video} "
              f"(set SKY_VIDEO to its path to regenerate)")


if __name__ == "__main__":
    main()
