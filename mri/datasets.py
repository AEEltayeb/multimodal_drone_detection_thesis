"""
mri.datasets — turn "a directory of images" into a scannable DatasetSpec.

The user only has to point at folders. This module:
  * resolves the YOLO labels dir for a positive (drone) folder, handling both
    the `images/labels` sibling layout and the `images/<split>` mirrored layout;
  * applies project-rule defaults (Svanstrom/Selcom -> imgsz=1280 + IoP) inferred
    from the path name, so the user can't silently squash small drones;
  * parses inline overrides of the form  PATH:imgsz=1280,rule=iop,stride=2,max=5000.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Path-name fragments that imply native-low-res / loose-GT datasets needing
# imgsz=1280 + IoP matching (project memory: imgsz_1280_svanstrom, svanstrom_iop).
_HIRES_IOP_HINTS = ("svan", "selcom")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def resolve_labels_dir(img_dir: Path) -> Path:
    """Find the YOLO labels dir for an images dir (both common layouts)."""
    sibling = img_dir.parent / "labels"
    if sibling.exists():
        return sibling
    mirrored = img_dir.parent.parent / "labels" / img_dir.name
    if mirrored.exists():
        return mirrored
    return sibling  # default; missing files handled per-image


@dataclass
class DatasetSpec:
    name: str
    path: Path
    role: str                       # "pos" (drones present) or "neg" (confusers)
    imgsz: int = 640
    stride: int = 1
    match_rule: str = "iou"         # "iou" or "iop"
    max_drones: int = 0             # 0 -> unlimited
    max_confusers: int = 0          # 0 -> unlimited
    weight_drone: float = 1.0
    weight_confuser: float = 1.0
    filter_prefixes: tuple = ()
    main_class: int = 0             # YOLO label class id of the POSITIVE object
                                    # (e.g. CBAM names=['B','D','P'] -> drone=1).
                                    # Detections matching a main_class GT box are
                                    # positives; other-class GT are not matched.

    @property
    def has_gt(self) -> bool:
        return self.role == "pos"

    def list_images(self) -> list[Path]:
        if not self.path.exists():
            return []
        imgs = sorted(p for p in self.path.iterdir() if is_image(p))
        if self.filter_prefixes:
            imgs = [p for p in imgs
                    if any(p.name.startswith(pre) for pre in self.filter_prefixes)]
        return imgs[::self.stride]


def _auto_defaults(path: Path) -> dict:
    """Project-rule defaults inferred from the path name."""
    name = path.as_posix().lower()
    if any(h in name for h in _HIRES_IOP_HINTS):
        return {"imgsz": 1280, "match_rule": "iop"}
    return {}


def _spec_name(path: Path) -> str:
    """Readable name: <parent>_<dir>, e.g. images dir -> 'svanstrom_paired_images'."""
    parts = [p for p in (path.parent.name, path.name) if p]
    return "_".join(parts) or path.name


def parse_dataset_arg(arg: str, role: str,
                      global_imgsz: int, global_stride: int,
                      global_rule: str, global_max: int) -> DatasetSpec:
    """Parse a `PATH` or `PATH:key=val,key=val` CLI argument into a DatasetSpec.

    Recognized override keys: imgsz, stride, rule, max, name, prefixes
    (prefixes is '|'-separated). Precedence: inline override > path auto-rule >
    global flag default.
    """
    # Split PATH from optional :key=val overrides, but don't be fooled by a
    # Windows drive-letter colon ("G:/..."). If the arg starts with a drive
    # letter, peel it off, split the remainder on the FIRST ":", then re-attach.
    if len(arg) >= 2 and arg[1] == ":" and arg[0].isalpha():
        drive, rest = arg[:2], arg[2:]
        path_part, _, override_str = rest.partition(":")
        raw_path = drive + path_part
    else:
        raw_path, _, override_str = arg.partition(":")
    path = Path(raw_path)

    overrides: dict[str, str] = {}
    if override_str:
        for kv in override_str.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()

    auto = _auto_defaults(path)
    imgsz = int(overrides.get("imgsz", auto.get("imgsz", global_imgsz)))
    stride = int(overrides.get("stride", global_stride))
    rule = overrides.get("rule", auto.get("match_rule", global_rule))
    cap = int(overrides.get("max", global_max))
    name = overrides.get("name", _spec_name(path))
    prefixes = tuple(overrides["prefixes"].split("|")) if "prefixes" in overrides else ()
    main_class = int(overrides.get("main_class", 0))

    return DatasetSpec(
        name=name, path=path, role=role,
        imgsz=imgsz, stride=stride, match_rule=rule,
        max_drones=cap if role == "pos" else 0,
        max_confusers=cap,
        filter_prefixes=prefixes,
        main_class=main_class,
    )


def specs_from_config(cfg: dict, global_imgsz: int, global_stride: int,
                      global_rule: str, global_max: int) -> list[DatasetSpec]:
    """Build specs from a YAML dict with `pos:` and `neg:` lists of entries.

    Each entry is either a string path or a mapping with `path` + override keys.
    """
    specs: list[DatasetSpec] = []
    for role in ("pos", "neg"):
        for entry in cfg.get(role, []) or []:
            if isinstance(entry, str):
                specs.append(parse_dataset_arg(
                    entry, role, global_imgsz, global_stride, global_rule, global_max))
                continue
            path = Path(entry["path"])
            auto = _auto_defaults(path)
            specs.append(DatasetSpec(
                name=entry.get("name", _spec_name(path)),
                path=path, role=role,
                imgsz=int(entry.get("imgsz", auto.get("imgsz", global_imgsz))),
                stride=int(entry.get("stride", global_stride)),
                match_rule=entry.get("rule", auto.get("match_rule", global_rule)),
                max_drones=int(entry.get("max", global_max)) if role == "pos" else 0,
                max_confusers=int(entry.get("max", global_max)),
                weight_drone=float(entry.get("weight_drone", 1.0)),
                weight_confuser=float(entry.get("weight_confuser", 1.0)),
                filter_prefixes=tuple(entry.get("prefixes", [])),
                main_class=int(entry.get("main_class", 0)),
            ))
    return specs
