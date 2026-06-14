import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


IN_IMG_DIR = Path(r"E:\Dataset\DUT Anti-UAV Detection and Tracking\train\img")
IN_XML_DIR = Path(r"E:\Dataset\DUT Anti-UAV Detection and Tracking\train\xml")

OUT_ROOT = Path(r"E:\Dataset\YOLOv11_ready_rgb\dut_anti_uav_det")
OUT_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_LBL_DIR = OUT_ROOT / "labels" / "train"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def voc_to_yolo(xml_path: Path) -> tuple[str, int, int, list[tuple[int, float, float, float, float]]]:
    # returns: filename, width, height, list of (cls, cx, cy, w, h)
    root = ET.parse(str(xml_path)).getroot()
    filename = root.findtext("filename", default=xml_path.stem + ".jpg")

    size = root.find("size")
    if size is None:
        raise RuntimeError(f"Missing <size> in {xml_path}")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))

    yolo_objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        # DUT uses "UAV" in your sample
        cls = 0  # drone

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        if bw <= 0 or bh <= 0:
            continue

        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        nw = bw / w
        nh = bh / h

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        yolo_objs.append((cls, cx, cy, nw, nh))

    return filename, w, h, yolo_objs


def main() -> None:
    ap = argparse.ArgumentParser(description="DUT Anti-UAV VOC XML -> YOLO labels (class 0 = drone).")
    ap.add_argument("--xml-dir", default=str(IN_XML_DIR), help="dir of VOC .xml files")
    ap.add_argument("--img-dir", default=str(IN_IMG_DIR), help="dir of source images (for stem matching / optional copy)")
    ap.add_argument("--out-lbl-dir", default=str(OUT_LBL_DIR), help="dir to write YOLO .txt labels")
    ap.add_argument("--out-img-dir", default=str(OUT_IMG_DIR), help="dir to copy images into (only if --copy-images)")
    ap.add_argument("--labels-only", action="store_true",
                    help="write labels only; do NOT copy images (use when images are already in place)")
    args = ap.parse_args()

    in_xml = Path(args.xml_dir); in_img = Path(args.img_dir)
    out_lbl = Path(args.out_lbl_dir); out_img = Path(args.out_img_dir)
    ensure_dir(out_lbl)
    if not args.labels_only:
        ensure_dir(out_img)

    xml_files = sorted(in_xml.glob("*.xml"))
    n_lbl = n_box = n_empty = 0
    for xml_path in tqdm(xml_files, desc="DUT VOC->YOLO", unit="file"):
        filename, w, h, objs = voc_to_yolo(xml_path)

        # resolve the matching image (by <filename>, else by stem)
        src_img = in_img / filename
        if not src_img.exists():
            jpg = in_img / (xml_path.stem + ".jpg")
            src_img = jpg if jpg.exists() else None

        if not args.labels_only:
            if src_img is None:
                continue
            shutil.copy2(src_img, out_img / src_img.name)

        # label stem follows the image stem when known, else the xml stem
        stem = (src_img.stem if src_img is not None else xml_path.stem)
        dst_lbl = out_lbl / (stem + ".txt")
        if objs:
            lines = [f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}" for cls, cx, cy, nw, nh in objs]
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            n_box += len(objs)
        else:
            dst_lbl.write_text("", encoding="utf-8")
            n_empty += 1
        n_lbl += 1

    print(f"\nDONE. {n_lbl} labels ({n_box} boxes, {n_empty} empty) -> {out_lbl}")


if __name__ == "__main__":
    main()
