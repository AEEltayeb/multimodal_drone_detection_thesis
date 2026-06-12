"""TALOS Drone Detection — Flet 0.84 Desktop App."""
import os
import sys
import threading
from pathlib import Path

import flet as ft

import base64 as _b64
_PIXEL = _b64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB"
    "Nl7BcQAAAABJRU5ErkJggg=="
)

_WS = Path(__file__).resolve().parents[2]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from .theme import DARK, LIGHT, TRUST_COLORS, TRUST_LABELS, MODES
from .widgets import card, status_chip, section_title, sidebar_item, CENTER
from .settings_dialog import load_settings, open_settings_dialog
from .engine import DetectionEngine
from .analytics import SessionStats, build_analytics_view

LOGO_DIR = Path(__file__).resolve().parents[1] / "ui" / "public"
P = ft.Padding


def main(page: ft.Page):
    page.title = "TALOS — Drone Detection System"
    page.window.width = 1440
    page.window.height = 900
    page.window.min_width = 1100
    page.window.min_height = 700
    page.padding = 0
    page.spacing = 0
    page.fonts = {"Inter": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"}
    page.theme = ft.Theme(font_family="Inter")

    # ── State ────────────────────────────────────────
    is_dark = True
    settings = load_settings()
    current_mode = MODES[0]
    engine = DetectionEngine()
    session = SessionStats()
    temporal_on = False
    speed_val = 1.0
    active_tab = "detection"  # detection | youtube | analytics

    def t():
        return DARK if is_dark else LIGHT

    # ── Glass card helper ────────────────────────────
    def glass(content, th, padding=16, expand=False, elevated=False, **kw):
        bg = th["card_elevated"] if elevated else th["card"]
        border_col = th["card_border"]
        return ft.Container(
            content=content, bgcolor=bg,
            border=ft.Border.all(1, border_col),
            border_radius=14, padding=padding, expand=expand,
            shadow=ft.BoxShadow(
                spread_radius=0, blur_radius=20 if elevated else 8,
                color="#00000040" if is_dark else "#00000008",
                offset=ft.Offset(0, 6 if elevated else 2)),
            **kw,
        )

    # ── Persistent widgets ───────────────────────────
    video_img = ft.Image(src=_PIXEL, fit=ft.BoxFit.FILL, expand=True, visible=False)
    video_placeholder = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.VIDEOCAM_OFF_ROUNDED, size=56, color=t()["text_muted"]),
            ft.Text("Load a video to begin detection", size=14, color=t()["text_muted"]),
        ], alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        expand=True, alignment=CENTER,
    )
    video_area = ft.Stack(controls=[video_placeholder, video_img], expand=True)

    progress_bar = ft.ProgressBar(value=0, bgcolor=t()["divider"], color=t()["accent"],
                                   bar_height=3)
    stats_text = ft.Text("Idle", size=11, color=t()["text_dim"], font_family="Consolas")

    trust_value = ft.Text("—", size=32, weight=ft.FontWeight.W_700, color=t()["text_muted"])
    trust_label_txt = ft.Text("Waiting", size=12, color=t()["text_dim"])
    fps_value = ft.Text("0", size=24, weight=ft.FontWeight.W_700, color=t()["cyan"])
    frame_value = ft.Text("0/0", size=14, weight=ft.FontWeight.W_600, color=t()["text"])
    frame_pct = ft.Text("0%", size=11, color=t()["text_dim"])
    status_dot = ft.Container(width=8, height=8, border_radius=4, bgcolor=t()["text_muted"])
    status_label = ft.Text("Idle", size=12, weight=ft.FontWeight.W_600, color=t()["text_dim"])

    alert_chip = ft.Container(visible=False)
    warning_chip = ft.Container(visible=False)

    def _tf(label):
        return ft.TextField(
            label=label, dense=True, text_size=13,
            bgcolor=t()["input_bg"], color=t()["text"],
            border_color=t()["input_border"], border_radius=10,
            content_padding=P(12, 10, 12, 10), expand=True,
        )

    rgb_path_field = _tf("RGB / Video path")
    ir_path_field = _tf("IR Video path")
    ir_path_field.visible = False
    save_path_field = _tf("Save output to")
    yt_field = _tf("YouTube URL")
    log_list = ft.ListView(spacing=2, auto_scroll=True, height=130)
    temporal_switch = ft.Switch(value=False, active_color=t()["accent"], scale=0.8)

    def add_log(msg):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        log_list.controls.append(
            ft.Text(f"[{ts}] {msg}", size=11, color=t()["text_dim"],
                    font_family="Consolas"))
        if len(log_list.controls) > 150:
            log_list.controls.pop(0)
        try:
            log_list.update()
        except Exception:
            pass

    # ── File Picker ──────────────────────────────────
    file_picker = ft.FilePicker()

    async def pick_rgb_file(e):
        result = await file_picker.pick_files(
            dialog_title="Select Video",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4", "avi", "mkv", "mov"],
        )
        if result:
            rgb_path_field.value = result[0].path
            base, ext = os.path.splitext(result[0].path)
            save_path_field.value = base + "_output" + ext
            page.update()

    async def pick_ir_file(e):
        result = await file_picker.pick_files(
            dialog_title="Select IR Video",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["mp4", "avi", "mkv", "mov"],
        )
        if result:
            ir_path_field.value = result[0].path
            page.update()

    # ── YouTube ──────────────────────────────────────
    def download_yt(e):
        url = yt_field.value.strip() if yt_field.value else ""
        if not url:
            add_log("Enter a YouTube URL first"); return
        add_log(f"Downloading: {url}")
        def _dl():
            try:
                import re, yt_dlp
                m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
                vid_id = m.group(1) if m else "video"
                dl_dir = LOGO_DIR.parent.parent / "demo_outputs"
                dl_dir.mkdir(exist_ok=True)
                dl_path = str(dl_dir / f"yt_{vid_id}.mp4")
                if os.path.exists(dl_path) and os.path.getsize(dl_path) > 100000:
                    _done(dl_path, True); return
                opts = {'format': 'best[height<=720]', 'outtmpl': dl_path,
                        'quiet': True, 'overwrites': True}
                for br in ['chrome', 'opera', 'edge']:
                    try:
                        with yt_dlp.YoutubeDL({**opts, 'cookiesfrombrowser': (br,),
                                               'extract_flat': True}) as ydl:
                            ydl.extract_info(url, download=False)
                        opts['cookiesfrombrowser'] = (br,); break
                    except Exception: continue
                with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
                _done(dl_path, False)
            except Exception as ex:
                add_log(f"YouTube error: {ex}")
        def _done(path, cached):
            rgb_path_field.value = path
            base, ext = os.path.splitext(path)
            save_path_field.value = base + "_output" + ext
            add_log(f"{'Cached' if cached else 'Downloaded'}: {os.path.basename(path)}")
            try: page.update()
            except Exception: pass
        threading.Thread(target=_dl, daemon=True).start()

    # ── Mode ─────────────────────────────────────────
    mode_buttons = []
    def set_mode(m):
        nonlocal current_mode
        current_mode = m
        ir_path_field.visible = (m == "Paired Fusion")
        for i, btn in enumerate(mode_buttons):
            active = MODES[i] == m
            btn.bgcolor = t()["accent"] if active else "transparent"
            btn.content.color = t()["accent_fg"] if active else t()["text_dim"]
        page.update()

    for m in MODES:
        btn = ft.Container(
            content=ft.Text(m, size=12, weight=ft.FontWeight.W_600,
                            color=t()["accent_fg"] if m == current_mode else t()["text_dim"]),
            bgcolor=t()["accent"] if m == current_mode else "transparent",
            border_radius=8, padding=P(14, 8, 14, 8),
            on_click=lambda e, mode=m: set_mode(mode), ink=True,
        )
        mode_buttons.append(btn)

    mode_selector = ft.Container(
        content=ft.Row(mode_buttons, spacing=2),
        bgcolor=t()["surface"], border=ft.Border.all(1, t()["card_border"]),
        border_radius=10, padding=4,
    )

    # ── Speed ────────────────────────────────────────
    speed_btns = []
    def set_speed(v):
        nonlocal speed_val
        speed_val = v; engine.playback_speed = v
        for sb in speed_btns:
            act = sb.data == v
            sb.bgcolor = t()["accent"] if act else t()["btn_surface"]
            sb.content.color = t()["accent_fg"] if act else t()["text_dim"]
        page.update()

    for lbl, val in [("1x", 1.0), ("2x", 2.0), ("4x", 4.0), ("Max", 0.0)]:
        sb = ft.Container(
            content=ft.Text(lbl, size=11, weight=ft.FontWeight.W_700,
                            color=t()["accent_fg"] if val == 1.0 else t()["text_dim"]),
            bgcolor=t()["accent"] if val == 1.0 else t()["btn_surface"],
            border_radius=6, padding=P(10, 6, 10, 6),
            on_click=lambda e, v=val: set_speed(v), ink=True, data=val,
        )
        speed_btns.append(sb)

    # ── Playback ─────────────────────────────────────
    def on_play(e):
        nonlocal temporal_on
        if engine.running and engine.paused:
            engine.toggle_pause(); add_log("Resumed"); return
        path = rgb_path_field.value.strip() if rgb_path_field.value else ""
        if not path or not os.path.isfile(path):
            add_log("Select a valid video file"); return
        ir_path = None
        if current_mode == "Paired Fusion":
            ir_path = ir_path_field.value.strip() if ir_path_field.value else ""
            if not ir_path or not os.path.isfile(ir_path):
                add_log("Select a valid IR video file"); return
        temporal_on = temporal_switch.value
        session.reset()
        session.video_name = os.path.basename(path)
        session.mode = current_mode
        try:
            engine.load_models(settings, current_mode, status_cb=add_log)
        except Exception as ex:
            add_log(f"Model load failed: {ex}"); return
        engine.on_frame = on_engine_frame
        sp = save_path_field.value.strip() if save_path_field.value else None
        engine.start(path, ir_path, current_mode, settings, temporal_on, sp)
        status_dot.bgcolor = t()["green"]; status_label.value = "Processing"
        status_label.color = t()["green"]
        add_log(f"Playing: {os.path.basename(path)} [{current_mode}]")

    def on_pause(e):
        engine.toggle_pause()
        p = engine.paused
        status_dot.bgcolor = t()["amber"] if p else t()["green"]
        status_label.value = "Paused" if p else "Processing"
        status_label.color = t()["amber"] if p else t()["green"]
        add_log("Paused" if p else "Resumed")

    def on_stop(e):
        engine.stop()
        status_dot.bgcolor = t()["text_muted"]; status_label.value = "Idle"
        status_label.color = t()["text_dim"]
        add_log("Stopped")

    def on_skip(e):
        engine.skip_forward(30); add_log("Skipped 30s")

    def on_engine_frame(jpeg_bytes, stats):
        if jpeg_bytes is None:
            add_log(f"Done. {stats.get('frame', 0)} frames processed.")
            status_dot.bgcolor = t()["text_muted"]; status_label.value = "Done"
            status_label.color = t()["text_dim"]
            session.total_frames = stats.get("frame", 0)
            return
        session.record_frame(stats)
        # Pass raw JPEG bytes directly — Flet 0.84 Image.src accepts bytes
        video_img.src = jpeg_bytes
        video_img.visible = True
        f, tot = stats["frame"], stats["total"]
        progress_bar.value = f / max(tot, 1)
        stats_text.value = f"Frame {f}/{tot}"
        fps_value.value = str(stats['fps'])
        frame_value.value = f"{f}/{tot}"
        frame_pct.value = f"{f / max(tot, 1) * 100:.1f}%"
        trust = stats.get("trust")
        if trust is not None:
            trust_value.value = TRUST_LABELS.get(trust, "—")
            trust_value.color = TRUST_COLORS.get(trust, t()["text_muted"])
            tp = stats.get("trust_prob")
            trust_label_txt.value = f"{tp * 100:.1f}%" if tp else ""
        alert_chip.visible = bool(stats.get("alert"))
        warning_chip.visible = bool(stats.get("warning")) and not stats.get("alert")
        try: page.update()
        except Exception: pass

    # ── Nav / Theme / Settings ───────────────────────
    def switch_tab(tab):
        nonlocal active_tab
        active_tab = tab
        rebuild()

    def toggle_theme(e):
        nonlocal is_dark
        is_dark = not is_dark
        rebuild()

    def open_settings(e):
        def on_save(ns):
            nonlocal settings
            settings = ns; add_log("Settings saved")
        open_settings_dialog(page, settings, on_save, t())

    # ── BUILD ────────────────────────────────────────
    def rebuild():
        th = t()
        page.bgcolor = th["bg"]

        # Update persisted widget colors
        for fld in (rgb_path_field, ir_path_field, save_path_field, yt_field):
            fld.bgcolor = th["input_bg"]; fld.color = th["text"]
            fld.border_color = th["input_border"]
        progress_bar.bgcolor = th["divider"]; progress_bar.color = th["accent"]
        stats_text.color = th["text_dim"]
        temporal_switch.active_color = th["accent"]
        for sb in speed_btns:
            act = sb.data == speed_val
            sb.bgcolor = th["accent"] if act else th["btn_surface"]
            sb.content.color = th["accent_fg"] if act else th["text_dim"]
        for i, btn in enumerate(mode_buttons):
            active = MODES[i] == current_mode
            btn.bgcolor = th["accent"] if active else "transparent"
            btn.content.color = th["accent_fg"] if active else th["text_dim"]
        mode_selector.bgcolor = th["surface"]
        mode_selector.border = ft.Border.all(1, th["card_border"])
        alert_chip.content = status_chip("ALERT", th["red"], th)
        alert_chip.visible = False
        warning_chip.content = status_chip("WARNING", th["amber"], th)
        warning_chip.visible = False

        # ── SIDEBAR ──────────────────────────────────
        logo_path = str(LOGO_DIR / th["logo"])
        logo_w = (ft.Image(src=logo_path, height=28, fit=ft.BoxFit.CONTAIN)
                  if os.path.exists(logo_path)
                  else ft.Text("TALOS", size=18, weight=ft.FontWeight.W_700,
                               color=th["text"]))

        nav_items = [
            ("detection", ft.Icons.VIDEOCAM_ROUNDED, "Detection"),
            ("youtube", ft.Icons.DOWNLOAD_ROUNDED, "YouTube"),
            ("analytics", ft.Icons.ANALYTICS_ROUNDED, "Analytics"),
        ]

        sidebar = ft.Container(
            content=ft.Column([
                ft.Container(content=logo_w, padding=P(20, 20, 20, 12)),
                ft.Divider(height=1, color=th["divider"]),
                ft.Container(
                    content=ft.Column([
                        sidebar_item(icon, lbl, th, active=active_tab == key,
                                     on_click=lambda e, k=key: switch_tab(k))
                        for key, icon, lbl in nav_items
                    ] + [
                        ft.Container(height=12),
                        ft.Divider(height=1, color=th["divider"]),
                        ft.Container(height=8),
                        sidebar_item(ft.Icons.SETTINGS_ROUNDED, "Settings", th,
                                     on_click=open_settings),
                        sidebar_item(
                            ft.Icons.LIGHT_MODE_ROUNDED if is_dark else ft.Icons.DARK_MODE_ROUNDED,
                            "Light Mode" if is_dark else "Dark Mode", th,
                            on_click=toggle_theme),
                    ], spacing=2),
                    padding=P(8, 8, 8, 8), expand=True,
                ),
                ft.Container(
                    content=ft.Row([status_dot, status_label], spacing=8),
                    padding=P(16, 12, 16, 16),
                ),
            ], spacing=0),
            width=190, bgcolor=th["sidebar"],
            border=ft.Border.only(right=ft.BorderSide(1, th["divider"])),
        )

        # ── HEADER ───────────────────────────────────
        header = ft.Container(
            content=ft.Row([
                mode_selector,
                ft.Container(expand=True),
                ft.Row([alert_chip, warning_chip], spacing=8),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=th["surface"], padding=P(16, 8, 16, 8),
            border=ft.Border.only(bottom=ft.BorderSide(1, th["divider"])),
        )

        # ── DETECTION VIEW ───────────────────────────
        if active_tab == "detection":
            play_btn = ft.Button(
                content=ft.Row([
                    ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, size=18, color=th["accent_fg"]),
                    ft.Text("Play", size=13, weight=ft.FontWeight.W_600, color=th["accent_fg"]),
                ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
                style=ft.ButtonStyle(bgcolor=th["accent"],
                                     shape=ft.RoundedRectangleBorder(radius=8),
                                     padding=ft.Padding(16, 8, 16, 8)),
                on_click=on_play)
            pause_btn = ft.IconButton(ft.Icons.PAUSE_ROUNDED, icon_color=th["text"],
                                       tooltip="Pause", on_click=on_pause,
                                       style=ft.ButtonStyle(bgcolor=th["btn_surface"],
                                                            shape=ft.RoundedRectangleBorder(radius=8)))
            stop_btn = ft.IconButton(ft.Icons.STOP_ROUNDED, icon_color=th["red"],
                                      tooltip="Stop", on_click=on_stop,
                                      style=ft.ButtonStyle(bgcolor=th["btn_surface"],
                                                           shape=ft.RoundedRectangleBorder(radius=8)))
            skip_btn = ft.TextButton("Skip 30s", on_click=on_skip,
                                      style=ft.ButtonStyle(color=th["text_dim"]))

            controls_bar = ft.Container(
                content=ft.Row([
                    play_btn, pause_btn, stop_btn,
                    ft.VerticalDivider(width=1, color=th["divider"]),
                    *speed_btns,
                    ft.VerticalDivider(width=1, color=th["divider"]),
                    skip_btn,
                    ft.VerticalDivider(width=1, color=th["divider"]),
                    ft.Row([ft.Text("TC", size=11, color=th["text_dim"]),
                            temporal_switch], spacing=4),
                    ft.Container(expand=True),
                    stats_text,
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=th["surface"], border_radius=10,
                padding=P(12, 6, 12, 6),
                border=ft.Border.all(1, th["card_border"]),
            )

            video_card = glass(ft.Column([
                ft.Container(content=video_area, bgcolor=th["video_bg"],
                             border_radius=12, expand=True,
                             clip_behavior=ft.ClipBehavior.HARD_EDGE),
                progress_bar,
                controls_bar,
            ], spacing=6), th, padding=10, expand=True)

            # Source row — always visible, compact
            browse_rgb = ft.IconButton(ft.Icons.FOLDER_OPEN, icon_color=th["text_dim"],
                                        on_click=pick_rgb_file, tooltip="Browse video")
            browse_ir = ft.IconButton(ft.Icons.FOLDER_OPEN, icon_color=th["text_dim"],
                                       on_click=pick_ir_file, tooltip="Browse IR")

            ir_path_field.visible = current_mode == "Paired Fusion"
            browse_ir.visible = current_mode == "Paired Fusion"

            source_row = glass(ft.Row([
                ft.Icon(ft.Icons.VIDEO_FILE_ROUNDED, size=16, color=th["text_dim"]),
                rgb_path_field, browse_rgb,
                ir_path_field, browse_ir,
                ft.VerticalDivider(width=1, color=th["divider"]),
                save_path_field,
            ], spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER),
                th, padding=P(10, 6, 10, 6))

            # Event log — compact
            log_card = glass(ft.Column([
                ft.Row([
                    section_title("Events", th, ft.Icons.TERMINAL_ROUNDED),
                    ft.Container(expand=True),
                    ft.TextButton("Clear", on_click=lambda e: (
                        log_list.controls.clear(), log_list.update()),
                        style=ft.ButtonStyle(color=th["text_muted"])),
                ]),
                ft.Container(content=log_list, bgcolor=th["input_bg"],
                             border_radius=8, padding=8, height=80),
            ], spacing=4), th, padding=10)

            # Right metrics
            trust_card = glass(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SHIELD_ROUNDED, size=16, color=th["accent"]),
                    ft.Text("Fusion", size=12, weight=ft.FontWeight.W_700, color=th["text"]),
                ], spacing=6),
                ft.Container(
                    content=ft.Column([trust_value, trust_label_txt],
                                       horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                       spacing=2),
                    alignment=CENTER, padding=P(4, 12, 4, 12),
                ),
            ], spacing=6), th, elevated=True, padding=14)

            fps_card = glass(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SPEED_ROUNDED, size=14, color=th["cyan"]),
                    ft.Text("FPS", size=11, weight=ft.FontWeight.W_600, color=th["text_dim"]),
                ], spacing=6),
                ft.Row([fps_value, ft.Text("fps", size=12, color=th["text_dim"])],
                       spacing=4, vertical_alignment=ft.CrossAxisAlignment.END),
            ], spacing=6), th, padding=12)

            progress_card = glass(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.ANALYTICS_ROUNDED, size=14, color=th["green"]),
                    ft.Text("Progress", size=11, weight=ft.FontWeight.W_600,
                            color=th["text_dim"]),
                ], spacing=6),
                frame_value, frame_pct,
            ], spacing=4), th, padding=12)

            right_col = ft.Column([trust_card, fps_card, progress_card],
                                   spacing=8, width=220)

            content = ft.Column([
                source_row,
                ft.Row([
                    ft.Column([video_card], expand=True),
                    right_col,
                ], spacing=10, expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START),
                log_card,
            ], spacing=8, expand=True)

        # ── YOUTUBE VIEW ─────────────────────────────
        elif active_tab == "youtube":
            content = ft.Column([
                ft.Text("YouTube Download", size=22, weight=ft.FontWeight.W_700,
                        color=th["text"]),
                ft.Container(height=8),
                glass(ft.Column([
                    section_title("Download from YouTube", th, ft.Icons.DOWNLOAD_ROUNDED),
                    ft.Divider(height=1, color=th["divider"]),
                    ft.Text("Paste a YouTube URL to download and auto-load for detection.",
                            size=13, color=th["text_dim"]),
                    ft.Container(height=8),
                    ft.Row([yt_field,
                            ft.Button("Download",
                                      style=ft.ButtonStyle(bgcolor=th["accent"],
                                                           color=th["accent_fg"],
                                                           shape=ft.RoundedRectangleBorder(radius=8)),
                                      on_click=download_yt)], spacing=8),
                    ft.Container(height=8),
                    ft.Divider(height=1, color=th["divider"]),
                    section_title("Event Log", th, ft.Icons.TERMINAL_ROUNDED),
                    ft.Container(content=log_list, bgcolor=th["input_bg"],
                                 border_radius=8, padding=8, height=200),
                ], spacing=8), th, padding=20),
            ], spacing=8, expand=True)

        # ── ANALYTICS VIEW ───────────────────────────
        elif active_tab == "analytics":
            content = build_analytics_view(session, th)

        else:
            content = ft.Text("Unknown tab", color=th["text"])

        # ── ASSEMBLE ─────────────────────────────────
        body = ft.Container(
            content=ft.Column([header,
                                ft.Container(content=content,
                                             padding=P(16, 12, 16, 16),
                                             expand=True)],
                               spacing=0),
            expand=True,
        )

        page.controls.clear()
        page.add(ft.Row([sidebar, body], spacing=0, expand=True))
        page.update()

    rebuild()
    add_log("TALOS initialized")


def run():
    ft.app(target=main)


if __name__ == "__main__":
    run()
