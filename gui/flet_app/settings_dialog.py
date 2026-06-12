"""Settings dialog — Flet 0.84 compatible."""
import json
from pathlib import Path
import flet as ft

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "fusion_settings.json"

DEFAULTS = {
    "rgb_model": "", "ir_model": "", "fusion_model": "",
    "rgb_conf": 0.30, "ir_conf_real": 0.40, "ir_conf_gray": 0.05,
    "nms_iou": 0.45, "imgsz": 640, "gpu_device": 0, "infer_fps": 5,
    "feature_stride": 5, "feature_max_height": 480,
    "warning_window_frames": 10, "warning_require_hits": 9,
    "alert_window_frames": 10, "alert_require_hits": 9,
    "alert_avg_conf_threshold": 0.0,
    "warning_cooldown_s": 3.0, "alert_cooldown_s": 3.0,
    "roi_ttl": 5, "roi_expand": 1.5,
    "show_troi": True, "show_gate": True, "show_source_tags": True,
    "rgb_patch_weights": "", "ir_patch_weights": "",
    "use_patch_verifier": True, "patch_threshold": 0.70,
    "use_mlp_verifier": False, "mlp_verifier_weights": "", "mlp_threshold": 0.25,
    "mlp_filter_mode": "per_frame", "mlp_alert_gate_conf": 0.4,
    "use_ir_mlp_verifier": False, "ir_mlp_verifier_weights": "", "ir_mlp_threshold": 0.25,
    "ir_mlp_filter_mode": "per_frame", "ir_mlp_alert_gate_conf": 0.4,
    "cascade_order": "filter_then_classifier",
    "grayscale_run_ir_filter": True,
    "confuser_suppress_mode": "primary_only",
    "confuser_filter_history": False,
    "suppress_helicopter": True, "suppress_airplane": True, "suppress_bird": True,
    # Output
    "save_output_dir": "", "save_output_enabled": False,
    # Persisted UI state (not shown in settings dialog)
    "last_rgb_path": "", "last_ir_path": "", "last_mode": "Single Model",
}

FLOAT_KEYS = {"rgb_conf", "ir_conf_real", "ir_conf_gray", "nms_iou",
              "alert_avg_conf_threshold", "warning_cooldown_s",
              "alert_cooldown_s", "roi_expand", "patch_threshold",
              "mlp_threshold", "mlp_alert_gate_conf",
              "ir_mlp_threshold", "ir_mlp_alert_gate_conf"}
INT_KEYS = {"gpu_device", "infer_fps", "warning_window_frames",
            "warning_require_hits", "alert_window_frames",
            "alert_require_hits", "roi_ttl",
            "feature_stride"}
BOOL_KEYS = {"show_troi", "show_gate", "show_source_tags",
             "use_patch_verifier", "use_mlp_verifier", "use_ir_mlp_verifier",
              "grayscale_run_ir_filter",
              "save_output_enabled", "confuser_filter_history",
              "suppress_helicopter", "suppress_airplane", "suppress_bird"}
CHOICE_KEYS = {
    "imgsz": ["320", "480", "640", "960", "1280", "1920"],
    "feature_max_height": ["240", "320", "480", "720", "1080"],
    "confuser_suppress_mode": ["primary_only", "primary_and_avg", "any_above"],
    "cascade_order": ["filter_then_classifier", "classifier_then_filter"],
    "mlp_filter_mode": ["per_frame", "alert_gate"],
    "ir_mlp_filter_mode": ["per_frame", "alert_gate"],
}

# Path-valued keys, used by the PySide settings dialog to show a browse button.
PATH_FILE_KEYS = {"rgb_model", "ir_model", "fusion_model",
                  "rgb_patch_weights", "ir_patch_weights",
                  "mlp_verifier_weights", "ir_mlp_verifier_weights"}
PATH_DIR_KEYS = {"save_output_dir"}

# CHOICE_KEYS that should be saved as int, not string
CHOICE_INT_KEYS = {"imgsz", "feature_max_height"}

LABELS = {
    "rgb_model": "RGB Model Path", "ir_model": "IR Model Path",
    "fusion_model": "Fusion Classifier", "rgb_conf": "RGB Confidence",
    "ir_conf_real": "IR Confidence (paired)", "ir_conf_gray": "IR Confidence (gray)",
    "nms_iou": "NMS IoU", "imgsz": "YOLO Input Size", "gpu_device": "GPU Device",
    "feature_stride": "Scene Feature Stride (frames)",
    "feature_max_height": "Scene Feature Resolution (px)",
    "infer_fps": "Temporal Infer FPS",
    "warning_window_frames": "Warning Window (M)",
    "warning_require_hits": "Warning Hits (N)",
    "alert_window_frames": "Alert Window (M)",
    "alert_require_hits": "Alert Hits (N)",
    "alert_avg_conf_threshold": "Alert Avg Conf Thr",
    "warning_cooldown_s": "Warning Cooldown (s)",
    "alert_cooldown_s": "Alert Cooldown (s)",
    "roi_ttl": "ROI TTL", "roi_expand": "ROI Expand",
    "show_troi": "Show TROI boxes", "show_gate": "Show TC Gate",
    "show_source_tags": "Show Source Tags",
    "rgb_patch_weights": "RGB Patch Verifier",
    "ir_patch_weights": "IR Patch Verifier",
    "use_patch_verifier": "Enable Patch Verifier",
    "patch_threshold": "Patch Threshold",
    "use_mlp_verifier": "Enable MLP Verifier (V5)",
    "mlp_verifier_weights": "MLP Verifier Weights",
    "mlp_threshold": "MLP Drone-Prob Threshold",
    "mlp_filter_mode": "MLP Filter Mode",
    "mlp_alert_gate_conf": "MLP Alert-Gate Conf Threshold",
    "use_ir_mlp_verifier": "Enable IR MLP Verifier (V5)",
    "ir_mlp_verifier_weights": "IR MLP Verifier Weights",
    "ir_mlp_threshold": "IR MLP Drone-Prob Threshold",
    "ir_mlp_filter_mode": "IR MLP Filter Mode",
    "ir_mlp_alert_gate_conf": "IR MLP Alert-Gate Conf Threshold",
    "cascade_order": "Cascade Order",
    "grayscale_run_ir_filter": "Gray: run IR filter",
    "confuser_suppress_mode": "Suppress Mode",
    "confuser_filter_history": "Use Confuser History",
    "suppress_helicopter": "Suppress Helicopters",
    "suppress_airplane": "Suppress Airplanes",
    "suppress_bird": "Suppress Birds",
    "save_output_dir": "Save Output Folder",
    "save_output_enabled": "Save Output",
}

SECTIONS = [
    ("Models", ["rgb_model", "ir_model", "fusion_model",
                "rgb_patch_weights", "ir_patch_weights"]),
    ("Detection", ["rgb_conf", "ir_conf_real", "ir_conf_gray",
                    "nms_iou", "imgsz", "gpu_device",
                    "feature_stride", "feature_max_height"]),
    ("Temporal", ["infer_fps", "warning_window_frames", "warning_require_hits",
                  "alert_window_frames", "alert_require_hits",
                  "alert_avg_conf_threshold", "warning_cooldown_s",
                  "alert_cooldown_s", "roi_ttl", "roi_expand"]),
    ("Overlays", ["show_troi", "show_gate", "show_source_tags"]),
    ("Confuser Filter", ["use_patch_verifier", "patch_threshold",
                         "cascade_order",
                         "grayscale_run_ir_filter",
                         "confuser_suppress_mode",
                         "confuser_filter_history"]),
    ("MLP Verifier (V5)", ["use_mlp_verifier", "mlp_verifier_weights",
                           "mlp_threshold", "mlp_filter_mode",
                           "mlp_alert_gate_conf"]),
    ("IR MLP Verifier (V5)", ["use_ir_mlp_verifier", "ir_mlp_verifier_weights",
                              "ir_mlp_threshold", "ir_mlp_filter_mode",
                              "ir_mlp_alert_gate_conf"]),
    ("Confuser Classes", ["suppress_helicopter", "suppress_airplane",
                           "suppress_bird"]),
    ("Output", ["save_output_enabled", "save_output_dir"]),
]


def load_settings():
    s = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text()))
        except Exception:
            pass
    return s


def save_settings(s):
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))


def open_settings_dialog(page: ft.Page, current: dict, on_save, t):
    refs = {}
    rows = []
    for sec_name, keys in SECTIONS:
        rows.append(ft.Text(sec_name, size=14, weight=ft.FontWeight.W_700,
                            color=t["accent"]))
        rows.append(ft.Divider(height=1, color=t["divider"]))
        for k in keys:
            label = LABELS.get(k, k)
            val = current.get(k, DEFAULTS.get(k, ""))
            if k in BOOL_KEYS:
                sw = ft.Checkbox(value=bool(val), active_color=t["accent"])
                refs[k] = sw
                rows.append(ft.Row([
                    ft.Text(label, size=12, color=t["text"], expand=True), sw,
                ]))
            else:
                tf = ft.TextField(
                    value=str(val), dense=True, text_size=12,
                    bgcolor=t["input_bg"], color=t["text"],
                    border_color=t["input_border"], border_radius=8,
                    content_padding=ft.Padding(10, 6, 10, 6), expand=True,
                )
                refs[k] = tf
                rows.append(ft.Row([
                    ft.Text(label, size=12, color=t["text"], width=200), tf,
                ]))
        rows.append(ft.Container(height=8))

    def do_save(e):
        new = dict(current)
        for k, ctrl in refs.items():
            if k in BOOL_KEYS:
                new[k] = ctrl.value
            elif k in FLOAT_KEYS:
                try: new[k] = float(ctrl.value)
                except ValueError: pass
            elif k in INT_KEYS:
                try: new[k] = int(ctrl.value)
                except ValueError: pass
            else:
                new[k] = ctrl.value
        save_settings(new)
        on_save(new)
        dlg.open = False
        page.update()

    def do_close(e):
        dlg.open = False
        page.update()

    dlg = ft.BottomSheet(
        content=ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("Settings", size=18, weight=ft.FontWeight.W_700,
                            color=t["text"]),
                    ft.IconButton(ft.Icons.CLOSE, icon_color=t["text_dim"],
                                 on_click=do_close),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(color=t["divider"]),
                ft.Column(rows, scroll=ft.ScrollMode.AUTO, expand=True, spacing=6),
                ft.Row([
                    ft.Button("Save", color=ft.Colors.WHITE,
                              bgcolor=t["accent"], on_click=do_save),
                ], alignment=ft.MainAxisAlignment.END),
            ], expand=True, spacing=10),
            bgcolor=t["card"], padding=20,
            border_radius=ft.BorderRadius.only(top_left=16, top_right=16),
            width=600, height=550,
        ),
        open=True,
    )
    page.overlay.append(dlg)
    page.update()
