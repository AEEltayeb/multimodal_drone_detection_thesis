"""Re-export the 3 corrupted video-test label dirs from their review sessions.

Background: a "Merge ALL sessions" export in the label reviewer concatenated
same-stem files across unrelated datasets, inflating GT counts in some folders
and leaving `flock_of_birds_attack_drone` empty (its session ran after the
merge). The true labels still live in the per-session dirs under
`label_reviewer/`.

This script:
  1. Backs up the current `labels/test` dir.
  2. Deletes every `frame_*.txt` in `labels/test` (leaves any non-frame files).
  3. Copies `frame_*.txt` from the source session into `labels/test`.
  4. Re-creates empty placeholders for image stems without a session label so
     every image still has a matching label file (eval expects this).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DS_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb" / "drone"
LR_ROOT = REPO / "label_reviewer"

FIXES = {
    "drone_over_mountain_attacked_by_birds": "review__model_test_2026-05-17_1944",
    "flock_of_seagulls_attack_drone_beach":  "review__model_test_2026-05-17_1936",
    "flock_of_birds_attack_drone":           "review__model_test_2026-05-17_1915",
}


def fix(dataset: str, session: str) -> None:
    img_dir = DS_ROOT / dataset / "images" / "test"
    lbl_dir = DS_ROOT / dataset / "labels" / "test"
    sess_dir = LR_ROOT / session

    assert img_dir.is_dir(), f"missing {img_dir}"
    assert sess_dir.is_dir(), f"missing {sess_dir}"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = lbl_dir.parent / f"test_backup_fix_{stamp}"
    shutil.copytree(lbl_dir, backup)

    for f in lbl_dir.glob("frame_*.txt"):
        f.unlink()

    copied = 0
    for src in sess_dir.glob("frame_*.txt"):
        shutil.copy2(src, lbl_dir / src.name)
        copied += 1

    image_stems = {p.stem for p in img_dir.glob("*.jpg")}
    label_stems = {p.stem for p in lbl_dir.glob("frame_*.txt")}
    missing = image_stems - label_stems
    for stem in missing:
        (lbl_dir / f"{stem}.txt").write_text("")

    total_boxes = sum(
        1 for p in lbl_dir.glob("frame_*.txt")
        for line in p.read_text().splitlines() if line.strip()
    )
    print(f"  {dataset}")
    print(f"    session:  {session}")
    print(f"    backup:   {backup.name}")
    print(f"    copied:   {copied} session files")
    print(f"    padded:   {len(missing)} empty placeholders for unlabeled images")
    print(f"    images:   {len(image_stems)}")
    print(f"    labels:   {len(list(lbl_dir.glob('frame_*.txt')))}")
    print(f"    boxes:    {total_boxes}")


def main() -> None:
    print(f"Fixing {len(FIXES)} dataset label dirs from review sessions\n")
    for ds, sess in FIXES.items():
        fix(ds, sess)
        print()
    print("Done. Re-run: python eval/eval_video_tests.py --categories drone")


if __name__ == "__main__":
    main()
