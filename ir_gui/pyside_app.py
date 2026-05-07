"""TALOS PySide6 -- Step 1: Single Model + all controls.

Reuses flet_app/engine.py for detection logic.
Run:  python -B ir_gui/pyside_app.py
"""
import json, os, sys, threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, Slot, QSize, QEvent
from PySide6.QtGui import QImage, QPixmap, QFont, QShortcut, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QSlider,
    QFrame, QSizePolicy, QTextEdit, QCheckBox, QScrollArea,
    QStackedWidget, QStyle, QStyleOptionSlider,
)


class ClickJumpSlider(QSlider):
    """QSlider that jumps to the clicked position instead of paging."""

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            handle = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderHandle, self)
            if not handle.contains(event.position().toPoint()):
                if self.orientation() == Qt.Orientation.Horizontal:
                    new_val = QStyle.sliderValueFromPosition(
                        self.minimum(), self.maximum(),
                        int(event.position().x() - handle.width() / 2),
                        self.width() - handle.width())
                else:
                    new_val = QStyle.sliderValueFromPosition(
                        self.minimum(), self.maximum(),
                        int(event.position().y() - handle.height() / 2),
                        self.height() - handle.height())
                self.setValue(new_val)
                self.sliderPressed.emit()
                self.sliderReleased.emit()
                event.accept()
                return
        super().mousePressEvent(event)

_WS = Path(__file__).resolve().parents[1]
_IR = Path(__file__).resolve().parent
for _p in (_WS, _IR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pyside_engine import TalosEngine
from flet_app.settings_dialog import load_settings, save_settings, SETTINGS_PATH
from flet_app.settings_dialog import DEFAULTS, FLOAT_KEYS, INT_KEYS, BOOL_KEYS, CHOICE_KEYS, CHOICE_INT_KEYS, LABELS, SECTIONS

TRUST_COLORS = {0: "#ff4444", 1: "#44ff88", 2: "#44ddff", 3: "#66ff66"}
TRUST_LABELS = {0: "REJECT BOTH", 1: "TRUST RGB", 2: "TRUST IR", 3: "TRUST BOTH"}
MODES = ["Single Model", "Paired Fusion", "Grayscale Fusion"]
LOGO_DIR = Path(__file__).resolve().parent / "ui" / "public"
ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"


class FrameBridge(QObject):
    """Thread-safe signal bridge: engine thread -> Qt main thread."""
    frame_ready = Signal(object, dict)   # (jpeg_bytes, stats)
    finished = Signal()
    log_msg = Signal(str)


class TalosWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TALOS - Drone Detection System")
        self.setMinimumSize(1100, 700)
        self.resize(1440, 900)

        self.settings = load_settings()
        self.engine = TalosEngine()
        self.bridge = FrameBridge()
        self.bridge.frame_ready.connect(self._on_frame)
        self.bridge.finished.connect(self._on_done)
        self.bridge.log_msg.connect(self._append_log)

        self.current_mode = self.settings.get("last_mode", MODES[0])
        if self.current_mode not in MODES:
            self.current_mode = MODES[0]
        self.single_model_type = "rgb"   # "rgb" or "ir"
        self.temporal_on = False
        self._prev_alert = False
        self._prev_warning = False
        self.dark_mode = True
        self._is_fullscreen = False
        self._pre_fs_geometry = None
        self._path_history = []  # list of (frame, rgb_cx, rgb_cy, rgb_area, ir_cx, ir_cy, ir_area, trust, alert, warning)
        self._path_frame_w = 0
        self._path_frame_h = 0
        self._seeking = False

        self._build_ui()
        self._apply_theme()
        self._setup_shortcuts()
        self._restore_persisted_state()
        self._add_log("TALOS initialized")

    # -- UI BUILD ----------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_h = QHBoxLayout(central)
        root_h.setContentsMargins(0, 0, 0, 0)
        root_h.setSpacing(0)

        # -- SIDEBAR --
        self.sidebar = self._build_sidebar()
        root_h.addWidget(self.sidebar)

        # -- MAIN AREA --
        main_col = QVBoxLayout()
        main_col.setContentsMargins(0, 0, 0, 0)
        main_col.setSpacing(0)

        # Header with mode selector
        self.header = self._build_header()
        main_col.addWidget(self.header)

        # Content -- stacked widget for Detection / YouTube
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_detection_view())  # index 0
        self.stack.addWidget(self._build_youtube_view())    # index 1
        self.stack.addWidget(self._build_analytics_view()) # index 2
        self.stack.addWidget(self._build_drone_path_view()) # index 3
        main_col.addWidget(self.stack, 1)

        main_w = QWidget()
        main_w.setLayout(main_col)
        root_h.addWidget(main_w, 1)

    def _build_detection_view(self):
        page = QWidget()
        content = QVBoxLayout(page)
        content.setContentsMargins(16, 12, 16, 16)
        content.setSpacing(8)
        self.source_row = self._build_source_row()
        content.addWidget(self.source_row)
        mid = QHBoxLayout()
        mid.setSpacing(10)
        mid.addWidget(self._build_video_card(), 1)
        self.metrics_col = self._build_metrics_col()
        mid.addWidget(self.metrics_col)
        content.addLayout(mid, 1)
        self.log_card = self._build_log_card()
        content.addWidget(self.log_card)
        return page

    def _build_youtube_view(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(8)

        title = QLabel("YouTube Download")
        title.setStyleSheet("font-size:22px; font-weight:700;")
        lay.addWidget(title)

        card = QFrame(); card.setObjectName("card")
        cl = QVBoxLayout(card); cl.setContentsMargins(20, 20, 20, 20); cl.setSpacing(12)
        cl.addWidget(QLabel("Paste a YouTube URL to download and auto-load for detection."))

        url_row = QHBoxLayout()
        self.yt_url = QLineEdit(); self.yt_url.setPlaceholderText("YouTube URL...")
        self.yt_download_btn = QPushButton("Download")
        self.yt_download_btn.setObjectName("accent_btn")
        self.yt_download_btn.clicked.connect(self._download_yt)
        url_row.addWidget(self.yt_url, 1); url_row.addWidget(self.yt_download_btn)
        cl.addLayout(url_row)

        self.yt_status = QLabel("")
        self.yt_status.setStyleSheet("color:#999; font-size:12px;")
        cl.addWidget(self.yt_status)

        lay.addWidget(card)
        lay.addStretch()
        return page

    def _build_analytics_view(self):
        # Inner content widget
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 12, 16, 16); lay.setSpacing(12)

        title = QLabel("Session Analytics")
        title.setStyleSheet("font-size:22px; font-weight:700;")
        lay.addWidget(title)

        # Stats grid -- 2 columns x 3 rows, all cards same size
        grid = QHBoxLayout(); grid.setSpacing(10)

        CARD_MIN_H = 80
        CARD_STYLE = "font-size:20px; font-weight:700;"

        def make_stat_card(label_text, attr):
            c = QFrame(); c.setObjectName("card")
            c.setMinimumHeight(CARD_MIN_H)
            cl = QVBoxLayout(c); cl.setContentsMargins(16, 14, 16, 14)
            cl.addWidget(QLabel(label_text))
            val = QLabel("--")
            val.setStyleSheet(CARD_STYLE)
            cl.addWidget(val)
            setattr(self, attr, val)
            return c

        # Left column: Total Frames, Mode, Drone Path
        left = QVBoxLayout(); left.setSpacing(8)
        left.addWidget(make_stat_card("Total Frames", "an_frames"))
        left.addWidget(make_stat_card("Mode", "an_mode"))

        # Drone Path card (clickable, styled like other cards)
        path_card = QFrame(); path_card.setObjectName("card")
        path_card.setMinimumHeight(CARD_MIN_H)
        path_card.setCursor(Qt.CursorShape.PointingHandCursor)
        pcl = QVBoxLayout(path_card); pcl.setContentsMargins(16, 14, 16, 14)
        pcl.addWidget(QLabel("Drone Path"))
        path_row = QHBoxLayout()
        self.path_card_label = QLabel("→")
        self.path_card_label.setStyleSheet(CARD_STYLE)
        path_row.addWidget(self.path_card_label)
        path_row.addStretch()
        pcl.addLayout(path_row)
        path_card.mousePressEvent = lambda e: self._open_drone_path_view()
        self.path_card = path_card
        self.path_card.setEnabled(False)
        self.path_card.setStyleSheet("QFrame#card { opacity: 0.4; }")
        left.addWidget(path_card)

        left.addStretch()
        grid.addLayout(left, 1)

        # Right column: Warning Events, Alert Events, Video
        right = QVBoxLayout(); right.setSpacing(8)
        right.addWidget(make_stat_card("Warning Events", "an_warnings"))
        right.addWidget(make_stat_card("Alert Events", "an_alerts"))

        # Video card (same size as others)
        video_card = QFrame(); video_card.setObjectName("card")
        video_card.setMinimumHeight(CARD_MIN_H)
        vcl = QVBoxLayout(video_card); vcl.setContentsMargins(16, 14, 16, 14)
        vcl.addWidget(QLabel("Video"))
        self.an_video = QLabel("--")
        self.an_video.setStyleSheet(CARD_STYLE)
        self.an_video.setWordWrap(True)
        vcl.addWidget(self.an_video)
        right.addWidget(video_card)

        right.addStretch()
        grid.addLayout(right, 1)

        lay.addLayout(grid)
        lay.addStretch()
        return page

    def _build_drone_path_view(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 12, 16, 16); lay.setSpacing(10)

        # Header: back button + title + save
        hdr = QHBoxLayout()
        back_btn = QPushButton("←  Back to Analytics")
        back_btn.setObjectName("text_btn")
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        hdr.addWidget(back_btn)
        title = QLabel("Drone Path")
        title.setStyleSheet("font-size:22px; font-weight:700;")
        hdr.addWidget(title)
        hdr.addStretch()
        self.path_save_btn = QPushButton("Save")
        self.path_save_btn.setObjectName("text_btn")
        self.path_save_btn.clicked.connect(self._save_path_image)
        self.path_save_btn.setEnabled(False)
        hdr.addWidget(self.path_save_btn)
        lay.addLayout(hdr)

        # Image fills remaining space.
        self.path_image_label = QLabel("Run a video to generate the drone flight path")
        self.path_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path_image_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                            QSizePolicy.Policy.Expanding)
        self.path_image_label.setStyleSheet(
            "color:#555; font-size:12px; background:#0a0a0a; border:1px solid #1c1c1c;")
        self.path_image_label.installEventFilter(self)
        lay.addWidget(self.path_image_label, 1)
        return page

    def _open_drone_path_view(self):
        self.stack.setCurrentIndex(3)
        # Re-fit pixmap once the new page is laid out.
        self._refresh_path_pixmap()

    def eventFilter(self, obj, event):
        if obj is getattr(self, "path_image_label", None) \
                and event.type() == QEvent.Type.Resize \
                and getattr(self, "_path_canvas", None) is not None:
            self._refresh_path_pixmap()
        return super().eventFilter(obj, event)

    def _refresh_path_pixmap(self):
        """Re-scale the cached path canvas to the current label width."""
        canvas = getattr(self, "_path_canvas", None)
        if canvas is None:
            return
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch_c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch_c, QImage.Format.Format_RGB888)
        target_w = max(self.path_image_label.width() - 8, 320)
        target_h = max(self.path_image_label.height() - 8, 200)
        pm = QPixmap.fromImage(qimg).scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.path_image_label.setPixmap(pm)

    def _build_sidebar(self):
        sb = QFrame()
        sb.setFixedWidth(190)
        sb.setObjectName("sidebar")
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Logo
        logo_path = str(LOGO_DIR / "logo.png")
        logo_container = QWidget()
        logo_container.setFixedHeight(48)
        logo_lay = QHBoxLayout(logo_container)
        logo_lay.setContentsMargins(20, 0, 20, 0)
        if os.path.exists(logo_path):
            self.logo_lbl = QLabel()
            pm = QPixmap(logo_path).scaledToHeight(28, Qt.TransformationMode.SmoothTransformation)
            self.logo_lbl.setPixmap(pm)
            logo_lay.addWidget(self.logo_lbl)
        else:
            self.logo_lbl = QLabel("TALOS")
            self.logo_lbl.setStyleSheet("font-size:18px; font-weight:700; color:#fff;")
            logo_lay.addWidget(self.logo_lbl)
        logo_lay.addStretch()
        logo_container.setObjectName("logo_bar")
        lay.addWidget(logo_container)

        # Nav items
        nav_frame = QWidget()
        nav_lay = QVBoxLayout(nav_frame)
        nav_lay.setContentsMargins(8, 8, 8, 8)
        nav_lay.setSpacing(2)
        self.nav_btns = []
        for i, (lbl, action) in enumerate([("Detection", lambda: self._switch_tab(0)),
                            ("YouTube", lambda: self._switch_tab(1)),
                            ("Analytics", lambda: self._switch_tab(2)),
                            ("Settings", self._open_settings)]):
            btn = QPushButton(lbl)
            btn.setObjectName("nav_btn")
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(action)
            nav_lay.addWidget(btn)
            self.nav_btns.append(btn)

        nav_lay.addStretch()
        lay.addWidget(nav_frame, 1)

        # Status dot
        status_w = QWidget()
        sl = QHBoxLayout(status_w)
        sl.setContentsMargins(16, 12, 16, 16)
        self.status_dot = QLabel("*")
        self.status_dot.setStyleSheet("color:#555; font-size:10px;")
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color:#666; font-size:12px; font-weight:600;")
        sl.addWidget(self.status_dot)
        sl.addWidget(self.status_label)
        sl.addStretch()
        lay.addWidget(status_w)

        return sb

    def _build_header(self):
        h = QFrame()
        h.setObjectName("header")
        h.setFixedHeight(48)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(16, 8, 16, 8)

        # Left stretch to center mode buttons
        lay.addStretch()

        self.mode_btns = []
        for m in MODES:
            btn = QPushButton(m)
            btn.setCheckable(True)
            btn.setChecked(m == self.current_mode)
            btn.setObjectName("mode_btn")
            btn.clicked.connect(lambda checked, mode=m: self._set_mode(mode))
            self.mode_btns.append(btn)
            lay.addWidget(btn)

        # Single mode sub-toggle (RGB / IR)
        lay.addSpacing(12)
        self.single_rgb_btn = QPushButton("RGB")
        self.single_rgb_btn.setCheckable(True); self.single_rgb_btn.setChecked(True)
        self.single_rgb_btn.setObjectName("mode_btn")
        self.single_rgb_btn.clicked.connect(lambda: self._set_single_type("rgb"))
        self.single_ir_btn = QPushButton("IR")
        self.single_ir_btn.setCheckable(True); self.single_ir_btn.setChecked(False)
        self.single_ir_btn.setObjectName("mode_btn")
        self.single_ir_btn.clicked.connect(lambda: self._set_single_type("ir"))
        lay.addWidget(self.single_rgb_btn)
        lay.addWidget(self.single_ir_btn)

        # Right stretch to center mode buttons
        lay.addStretch()

        # Alert / Warning chips
        self.alert_chip = QLabel("ALERT")
        self.alert_chip.setStyleSheet("background:#ff4444; color:#fff; padding:4px 12px;"
                                       "border-radius:6px; font-weight:700; font-size:11px;")
        self.alert_chip.hide()
        self.warning_chip = QLabel("WARNING")
        self.warning_chip.setStyleSheet("background:#ccaa00; color:#000; padding:4px 12px;"
                                         "border-radius:6px; font-weight:700; font-size:11px;")
        self.warning_chip.hide()
        lay.addWidget(self.alert_chip)
        lay.addWidget(self.warning_chip)

        # Theme toggle icon (top-right)
        self.theme_btn = QPushButton()
        self.theme_btn.setIconSize(QSize(20, 20))
        self.theme_btn.setFixedSize(36, 36)
        self.theme_btn.setToolTip("Toggle Light/Dark Mode")
        self.theme_btn.setObjectName("theme_icon_btn")
        self.theme_btn.clicked.connect(self._toggle_theme)
        lay.addWidget(self.theme_btn)

        return h

    def _build_source_row(self):
        card = QFrame(); card.setObjectName("card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(4)

        self.rgb_path = QLineEdit()
        self.rgb_path.setPlaceholderText("RGB / Video path")
        rgb_browse = QPushButton("...")
        rgb_browse.setFixedWidth(36)
        rgb_browse.clicked.connect(self._browse_rgb)

        self.ir_path = QLineEdit()
        self.ir_path.setPlaceholderText("IR Video path")
        self.ir_browse = QPushButton("...")
        self.ir_browse.setFixedWidth(36)
        self.ir_browse.clicked.connect(self._browse_ir)
        self.ir_path.hide(); self.ir_browse.hide()

        lay.addWidget(QLabel("SRC")); lay.addWidget(self.rgb_path, 2)
        lay.addWidget(rgb_browse)
        lay.addWidget(self.ir_path, 2); lay.addWidget(self.ir_browse)
        return card

    def _build_video_card(self):
        card = QFrame(); card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        # Video display with fullscreen overlay
        video_container = QWidget()
        video_container.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        container_lay = QVBoxLayout(video_container)
        container_lay.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("Load a video to begin detection")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumHeight(300)
        self.video_label.setObjectName("video_area")
        container_lay.addWidget(self.video_label, 1)

        # Fullscreen overlay button (top-right of video)
        self.fs_btn = QPushButton()
        self.fs_btn.setIconSize(QSize(20, 20))
        self.fs_btn.setParent(self.video_label)
        self.fs_btn.setFixedSize(32, 32)
        self.fs_btn.setToolTip("Toggle Fullscreen (F11)")
        self.fs_btn.setObjectName("fs_overlay_btn")
        self.fs_btn.clicked.connect(self._toggle_fullscreen)
        self.fs_btn.raise_()

        lay.addWidget(video_container, 1)

        # Seekable progress slider
        self.progress = ClickJumpSlider(Qt.Orientation.Horizontal)
        self.progress.setMinimum(0); self.progress.setMaximum(1000)
        self.progress.setValue(0)
        self.progress.setFixedHeight(16)
        self.progress.setObjectName("seek_slider")
        self.progress.sliderPressed.connect(self._on_seek_start)
        self.progress.sliderReleased.connect(self._on_seek_end)
        lay.addWidget(self.progress)

        # Controls
        lay.addWidget(self._build_controls())
        return card

    def _build_controls(self):
        bar = QFrame(); bar.setObjectName("controls_bar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 6, 12, 6); lay.setSpacing(6)

        # Unified play/pause button
        self.play_btn = QPushButton()
        self.play_btn.setIconSize(QSize(20, 20))
        self.play_btn.setFixedSize(42, 42)
        self.play_btn.setObjectName("accent_btn")
        self.play_btn.setToolTip("Play / Pause")
        self.play_btn.clicked.connect(self._on_play_pause)

        self.stop_btn = QPushButton()
        self.stop_btn.setIconSize(QSize(20, 20))
        self.stop_btn.setFixedSize(42, 42)
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self._on_stop)

        lay.addWidget(self.play_btn)
        lay.addWidget(self.stop_btn)

        skip = QPushButton("Skip 30s"); skip.clicked.connect(self._on_skip)
        lay.addWidget(skip)

        # Temporal toggle
        self.tc_check = QCheckBox("TC")
        self.tc_check.setStyleSheet("color:#999;")
        self.tc_check.setChecked(True)
        lay.addWidget(self.tc_check)

        lay.addStretch()
        self.stats_text = QLabel("Idle")
        self.stats_text.setObjectName("stats_text")
        lay.addWidget(self.stats_text)
        return bar

    def _build_metrics_col(self):
        col = QWidget(); col.setFixedWidth(270)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)

        # 1. Fusion Decision card
        tc = QFrame(); tc.setObjectName("card")
        tl = QVBoxLayout(tc); tl.setContentsMargins(14, 14, 14, 14)
        tl.addWidget(QLabel("FUSION DECISION"))
        self.trust_value = QLabel("--")
        self.trust_value.setStyleSheet("font-size:20px; font-weight:700; color:#555;")
        self.trust_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trust_pct = QLabel("")
        self.trust_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trust_pct.setStyleSheet("color:#666; font-size:12px;")
        tl.addWidget(self.trust_value); tl.addWidget(self.trust_pct)
        lay.addWidget(tc)

        # 2. Confuser Filter card (after classifier)
        flt = QFrame(); flt.setObjectName("card")
        fl2 = QVBoxLayout(flt); fl2.setContentsMargins(12, 12, 12, 12)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("CONFUSER FILTER"))
        hdr.addStretch()
        self.gate_badge = QLabel("ALERT GATE")
        self.gate_badge.setStyleSheet(
            "background:#333; color:#aaa; font-size:9px; font-weight:700; "
            "padding:2px 6px; border-radius:4px;")
        hdr.addWidget(self.gate_badge)
        fl2.addLayout(hdr)
        self.filter_status = QLabel("--")
        self.filter_status.setStyleSheet("font-size:13px; font-weight:600;")
        self.filter_detail = QLabel("")
        self.filter_detail.setStyleSheet("color:#666; font-size:10px;")
        self.filter_detail.setWordWrap(True)
        fl2.addWidget(self.filter_status); fl2.addWidget(self.filter_detail)
        lay.addWidget(flt)

        # 3. Detections card -- 2-column RGB/IR + events
        dc = QFrame(); dc.setObjectName("card")
        dl_lay = QVBoxLayout(dc); dl_lay.setContentsMargins(12, 12, 12, 16)
        det_row = QHBoxLayout()
        # RGB dets
        rgb_col = QVBoxLayout()
        self.dets_rgb_lbl = QLabel("0")
        self.dets_rgb_lbl.setStyleSheet("font-size:16px; font-weight:700; color:#44ff88;")
        self.dets_rgb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rgb_sub = QLabel("RGB Dets")
        rgb_sub.setStyleSheet("color:#666; font-size:10px;")
        rgb_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rgb_col.addWidget(self.dets_rgb_lbl); rgb_col.addWidget(rgb_sub)
        det_row.addLayout(rgb_col)
        # IR dets
        ir_col = QVBoxLayout()
        self.dets_ir_lbl = QLabel("0")
        self.dets_ir_lbl.setStyleSheet("font-size:16px; font-weight:700; color:#44ddff;")
        self.dets_ir_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ir_sub = QLabel("IR Dets")
        ir_sub.setStyleSheet("color:#666; font-size:10px;")
        ir_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ir_col.addWidget(self.dets_ir_lbl); ir_col.addWidget(ir_sub)
        det_row.addLayout(ir_col)
        dl_lay.addLayout(det_row)
        # Events row
        ev_row = QHBoxLayout()
        w_col = QVBoxLayout()
        self.warn_count = QLabel("0")
        self.warn_count.setStyleSheet("font-size:14px; font-weight:700;")
        self.warn_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        w_sub = QLabel("Warnings")
        w_sub.setStyleSheet("color:#666; font-size:10px;")
        w_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        w_col.addWidget(self.warn_count); w_col.addWidget(w_sub)
        ev_row.addLayout(w_col)
        a_col = QVBoxLayout()
        self.alert_count = QLabel("0")
        self.alert_count.setStyleSheet("font-size:14px; font-weight:700;")
        self.alert_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        a_sub = QLabel("Alerts")
        a_sub.setStyleSheet("color:#666; font-size:10px;")
        a_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        a_col.addWidget(self.alert_count); a_col.addWidget(a_sub)
        ev_row.addLayout(a_col)
        dl_lay.addLayout(ev_row)
        lay.addWidget(dc)

        # 4. Progress + Performance card (combined)
        pc = QFrame(); pc.setObjectName("card")
        pl = QVBoxLayout(pc); pl.setContentsMargins(12, 12, 12, 12)
        pl.addWidget(QLabel("PROGRESS"))
        self.frame_value = QLabel("0/0")
        self.frame_value.setStyleSheet("font-size:14px; font-weight:600;")
        self.frame_pct = QLabel("0%")
        self.frame_pct.setStyleSheet("color:#666; font-size:11px;")
        pl.addWidget(self.frame_value); pl.addWidget(self.frame_pct)
        # FPS + latency row
        perf_row = QHBoxLayout()
        self.fps_value = QLabel("0")
        self.fps_value.setStyleSheet("font-size:14px; font-weight:700;")
        fps_unit = QLabel("fps")
        fps_unit.setStyleSheet("color:#666; font-size:11px;")
        perf_row.addWidget(self.fps_value); perf_row.addWidget(fps_unit); perf_row.addStretch()
        self.infer_ms_lbl = QLabel("")
        self.infer_ms_lbl.setStyleSheet("color:#666; font-size:11px;")
        perf_row.addWidget(self.infer_ms_lbl)
        pl.addLayout(perf_row)
        lay.addWidget(pc)

        lay.addStretch()
        return col

    def _build_log_card(self):
        card = QFrame(); card.setObjectName("card")
        lay = QVBoxLayout(card); lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(4)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("EVENTS"))
        hdr.addStretch()
        clr = QPushButton("Clear"); clr.setObjectName("text_btn")
        clr.clicked.connect(lambda: self.log_text.clear())
        hdr.addWidget(clr)
        lay.addLayout(hdr)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True); self.log_text.setFixedHeight(90)
        self.log_text.setObjectName("log_area")
        lay.addWidget(self.log_text)
        return card

    # -- THEME -------------------------------------------
    def _apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget { background:#000; color:#fff; font-family:'Segoe UI'; }
                QLabel { background:transparent; }
                #sidebar { background:#0a0a0a; border-right:1px solid #1a1a1a; }
                #logo_bar { background:transparent; border-bottom:1px solid #222; }
                #header { background:transparent; border-bottom:1px solid #222; }
                #card { background:#111; border:1px solid #282828; border-radius:14px; }
                #video_area { background:#0a0a0a; border-radius:12px; color:#555; }
                #controls_bar { background:#0a0a0a; border:1px solid #333; border-radius:10px; }
                QLineEdit { background:#0a0a0a; color:#fff; border:1px solid #333;
                             border-radius:10px; padding:6px 10px; font-size:13px; }
                QPushButton { background:#222; color:#fff; border:1px solid #333;
                              border-radius:8px; padding:6px 14px; font-size:12px; font-weight:600; }
                QPushButton:hover { background:#333; }
                QPushButton:checked { background:#fff; color:#000; }
                #accent_btn { background:#fff; color:#000; border-radius:10px; font-size:16px; }
                #accent_btn:hover { background:#ddd; }
                #stop_btn { background:#222; color:#ff4444; border:1px solid #333;
                            border-radius:10px; font-size:14px; }
                #stop_btn:hover { background:#331111; }
                #mode_btn { background:#0a0a0a; border:1px solid #333; border-radius:10px;
                            padding:5px 14px; font-size:11px; font-weight:600; }
                #mode_btn:checked { background:#fff; color:#000; border-color:#fff; }
                #mode_btn:hover { background:#1a1a1a; }
                #nav_btn { text-align:left; padding:8px 14px; border:none; background:transparent;
                           border-radius:8px; margin:2px 6px; color:#fff; }
                #nav_btn:hover { background:#1a1a1a; }
                #nav_btn:checked { background:#1a1a1a; color:#fff; }
                #text_btn { border:none; background:transparent; color:#555; }
                #stats_text { color:#666; font-family:Consolas; font-size:11px; }
                #seek_slider { background:transparent; }
                #seek_slider::groove:horizontal { background:#222; height:4px; border-radius:2px; }
                #seek_slider::handle:horizontal { background:#fff; width:12px; height:12px;
                    margin:-4px 0; border-radius:6px; }
                #seek_slider::sub-page:horizontal { background:#fff; border-radius:2px; }
                #fs_overlay_btn { background:rgba(0,0,0,0.5); color:#fff; border:none;
                    border-radius:6px; font-size:16px; }
                #fs_overlay_btn:hover { background:rgba(255,255,255,0.2); }
                #theme_icon_btn { background:transparent; border:none; font-size:18px; color:#fff; }
                #theme_icon_btn:hover { background:#1a1a1a; border-radius:8px; }
                #log_area { background:#0a0a0a; border:1px solid #222; border-radius:8px;
                            color:#999; font-family:Consolas; font-size:11px; padding:8px; }
                #speed_btn { padding:4px 8px; border-radius:6px; }
                QCheckBox { color:#999; }
                QCheckBox::indicator { width:16px; height:16px; border-radius:4px;
                                       border:1px solid #555; background:#222; }
                QCheckBox::indicator:checked { background:#fff; }
                QTextEdit { border-radius:8px; }
                QScrollArea { border:none; border-radius:10px; }
            """)
            check_b = str(ICON_DIR / 'check-black.svg').replace('\\', '/')
            self.setStyleSheet(self.styleSheet() + f"\nQCheckBox::indicator:checked {{ image: url({check_b}); padding: 1px; }}")
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background:#fafafa; color:#111; font-family:'Segoe UI'; }
                QLabel { background:transparent; }
                #sidebar { background:#fff; border-right:1px solid #e0e0e0; }
                #logo_bar { background:transparent; border-bottom:1px solid #e0e0e0; }
                #header { background:transparent; border-bottom:1px solid #e0e0e0; }
                #card { background:#fff; border:1px solid #ccc; border-radius:14px; }
                #video_area { background:#f0f0f0; border:1px solid #ccc; border-radius:12px; color:#888; }
                #controls_bar { background:#f5f5f5; border:1px solid #ccc; border-radius:10px; }
                QLineEdit { background:#fff; color:#000; border:1px solid #ccc;
                             border-radius:10px; padding:6px 10px; font-size:13px; }
                QPushButton { background:#f0f0f0; color:#000; border:1px solid #ccc;
                              border-radius:8px; padding:6px 14px; font-size:12px; font-weight:600; }
                QPushButton:hover { background:#e0e0e0; }
                QPushButton:checked { background:#000; color:#fff; }
                #accent_btn { background:#000; color:#fff; border:1px solid #ccc;
                              border-radius:10px; font-size:16px; }
                #accent_btn:hover { background:#333; }
                #stop_btn { background:#f0f0f0; color:#ff4444; border:1px solid #ccc;
                            border-radius:10px; font-size:14px; }
                #stop_btn:hover { background:#ffeeee; }
                #mode_btn { background:#fff; border:1px solid #ccc; border-radius:10px;
                            padding:5px 14px; font-size:11px; font-weight:600; }
                #mode_btn:checked { background:#000; color:#fff; border-color:#000; }
                #mode_btn:hover { background:#eee; }
                #nav_btn { text-align:left; padding:8px 14px; border:none; background:transparent;
                           border-radius:8px; margin:2px 6px; color:#000; }
                #nav_btn:hover { background:#e5e5e5; }
                #nav_btn:checked { background:#e0e0e0; color:#000; }
                #text_btn { border:none; background:transparent; color:#666; }
                #stats_text { color:#555; font-family:Consolas; font-size:11px; }
                #seek_slider { background:transparent; }
                #seek_slider::groove:horizontal { background:#ddd; height:4px; border-radius:2px; }
                #seek_slider::handle:horizontal { background:#000; width:12px; height:12px;
                    margin:-4px 0; border-radius:6px; }
                #seek_slider::sub-page:horizontal { background:#000; border-radius:2px; }
                #fs_overlay_btn { background:rgba(0,0,0,0.4); color:#fff; border:none;
                    border-radius:6px; font-size:16px; }
                #fs_overlay_btn:hover { background:rgba(0,0,0,0.6); }
                #theme_icon_btn { background:transparent; border:none; font-size:18px; color:#111; }
                #theme_icon_btn:hover { background:#e5e5e5; border-radius:8px; }
                #log_area { background:#fff; border:1px solid #ccc; border-radius:8px;
                            color:#333; font-family:Consolas; font-size:11px; padding:8px; }
                QTextEdit { border-radius:8px; }
                #speed_btn { padding:4px 8px; border-radius:6px; }
                QCheckBox { color:#333; }
                QCheckBox::indicator { width:16px; height:16px; border-radius:4px;
                                       border:1px solid #ccc; background:#fff; }
                QCheckBox::indicator:checked { background:#000; }
                QScrollArea { border:none; border-radius:10px; }
            """)
            check_w = str(ICON_DIR / 'check-white.svg').replace('\\', '/')
            self.setStyleSheet(self.styleSheet() + f"\nQCheckBox::indicator:checked {{ image: url({check_w}); padding: 1px; }}")
        self._update_icons()

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()
        # Swap logo
        logo_file = "logo.png" if self.dark_mode else "logo-black.png"
        logo_path = str(LOGO_DIR / logo_file)
        if os.path.exists(logo_path):
            pm = QPixmap(logo_path).scaledToHeight(28, Qt.TransformationMode.SmoothTransformation)
            self.logo_lbl.setPixmap(pm)
    # -- FULLSCREEN / SHORTCUTS ---------------------------
    def _update_icons(self):
        inv = "black" if self.dark_mode else "white"
        std = "white" if self.dark_mode else "black"
        
        if hasattr(self, 'engine') and self.engine.running and not self.engine.paused:
            self.play_btn.setIcon(QIcon(str(ICON_DIR / f"pause-{inv}.svg")))
        else:
            self.play_btn.setIcon(QIcon(str(ICON_DIR / f"play-{inv}.svg")))
            
        self.stop_btn.setIcon(QIcon(str(ICON_DIR / f"stop-{std}.svg")))
        self.fs_btn.setIcon(QIcon(str(ICON_DIR / f"fullscreen-{std}.svg")))
        
        theme_icon = "sun" if self.dark_mode else "moon"
        self.theme_btn.setIcon(QIcon(str(ICON_DIR / f"{theme_icon}-{std}.svg")))

    def _setup_shortcuts(self):
        sc = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        sc.activated.connect(self._toggle_fullscreen)
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self._exit_fullscreen)
        self.video_label.mouseDoubleClickEvent = lambda e: self._toggle_fullscreen()

    def _toggle_fullscreen(self):
        if self._is_fullscreen:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        if self._is_fullscreen:
            return
        self._is_fullscreen = True
        self._pre_fs_geometry = self.geometry()
        self.sidebar.hide()
        self.header.hide()
        self.source_row.hide()
        self.metrics_col.hide()
        self.log_card.hide()
        self.showFullScreen()

    def _exit_fullscreen(self):
        if not self._is_fullscreen:
            return
        self._is_fullscreen = False
        self.sidebar.show()
        self.header.show()
        self.source_row.show()
        self.metrics_col.show()
        self.log_card.show()
        self.showNormal()
        if self._pre_fs_geometry:
            self.setGeometry(self._pre_fs_geometry)

    # -- ACTIONS -----------------------------------------
    def _set_mode(self, mode):
        self.current_mode = mode
        for btn in self.mode_btns:
            btn.setChecked(btn.text() == mode)
        paired = mode == "Paired Fusion"
        self.ir_path.setVisible(paired)
        self.ir_browse.setVisible(paired)
        single = mode == "Single Model"
        self.single_rgb_btn.setVisible(single)
        self.single_ir_btn.setVisible(single)

    def _set_single_type(self, t):
        self.single_model_type = t
        self.single_rgb_btn.setChecked(t == "rgb")
        self.single_ir_btn.setChecked(t == "ir")

    def _switch_tab(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_btns):
            btn.setChecked(i == idx)

    def _download_yt(self):
        url = self.yt_url.text().strip()
        if not url:
            self.yt_status.setText("Enter a YouTube URL first"); return
        self.yt_status.setText("Downloading...")
        self.yt_download_btn.setEnabled(False)

        def _dl():
            try:
                import re, yt_dlp
                # Robust video ID extraction -- covers /watch?v=, youtu.be/, /shorts/, /embed/
                m = re.search(r'(?:v=|youtu\.be/|/shorts/|/embed/)([a-zA-Z0-9_-]{11})', url)
                vid_id = m.group(1) if m else None
                if not vid_id:
                    # Fallback: use yt-dlp to extract ID
                    with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                        info = ydl.extract_info(url, download=False)
                        vid_id = info.get('id', 'unknown')
                dl_dir = Path(__file__).resolve().parent / "demo_outputs"
                dl_dir.mkdir(exist_ok=True)
                dl_path = str(dl_dir / f"yt_{vid_id}.mp4")
                self.bridge.log_msg.emit(f"Video ID: {vid_id}")
                if os.path.exists(dl_path) and os.path.getsize(dl_path) > 100_000:
                    self.bridge.log_msg.emit(f"Cached: {os.path.basename(dl_path)}")
                    self._yt_done(dl_path); return
                opts = {'format': 'best[height<=720]', 'outtmpl': dl_path,
                        'quiet': True, 'overwrites': True,
                        'remote_components': ['ejs:github']}
                for br in ['chrome', 'opera', 'edge', 'firefox']:
                    try:
                        test_opts = {**opts, 'cookiesfrombrowser': (br,)}
                        with yt_dlp.YoutubeDL(test_opts) as ydl:
                            ydl.extract_info(url, download=False)
                        opts['cookiesfrombrowser'] = (br,)
                        self.bridge.log_msg.emit(f"Using {br} cookies")
                        break
                    except Exception:
                        continue
                with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
                self.bridge.log_msg.emit(f"Downloaded: {os.path.basename(dl_path)}")
                self._yt_done(dl_path)
            except Exception as ex:
                self.bridge.log_msg.emit(f"YouTube error: {ex}")
                self.yt_download_btn.setEnabled(True)

        threading.Thread(target=_dl, daemon=True).start()

    def _yt_done(self, path):
        self.rgb_path.setText(path)
        self.yt_status.setText(f"Ready: {os.path.basename(path)}")
        self.yt_download_btn.setEnabled(True)
        self._switch_tab(0)  # auto-switch to Detection view

    def _browse_rgb(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                    "Video (*.mp4 *.avi *.mkv *.mov);;All (*)")
        if p:
            self.rgb_path.setText(p)

    def _browse_ir(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select IR Video", "",
                    "Video (*.mp4 *.avi *.mkv *.mov);;All (*)")
        if p:
            self.ir_path.setText(p)

    def _add_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

    @Slot(str)
    def _append_log(self, msg):
        self._add_log(msg)

    # -- PLAYBACK ----------------------------------------
    def _on_play_pause(self):
        """Unified play/pause toggle."""
        # If running and not paused -> pause
        if self.engine.running and not self.engine.paused:
            self.engine.toggle_pause()
            self.status_label.setText("Paused")
            self._add_log("Paused")
            self._update_icons()
            return

        # If running and paused -> resume
        if self.engine.running and self.engine.paused:
            self.engine.toggle_pause()
            self.status_label.setText("Processing")
            self._add_log("Resumed")
            self._update_icons()
            return

        # Not running -> start playback
        path = self.rgb_path.text().strip()
        if not path or not os.path.isfile(path):
            self._add_log("Select a valid video file"); return

        ir_path = None
        if self.current_mode == "Paired Fusion":
            ir_path = self.ir_path.text().strip()
            if not ir_path or not os.path.isfile(ir_path):
                self._add_log("Select a valid IR video file"); return

        self.temporal_on = self.tc_check.isChecked()

        # Persist paths for next restart
        self.settings["last_rgb_path"] = path
        self.settings["last_ir_path"] = ir_path or ""
        self.settings["last_mode"] = self.current_mode
        save_settings(self.settings)

        # For single IR mode, temporarily swap model path
        run_settings = dict(self.settings)
        if self.current_mode == "Single Model" and self.single_model_type == "ir":
            run_settings["rgb_model"] = run_settings.get("ir_model", run_settings["rgb_model"])
            run_settings["rgb_conf"] = run_settings.get("ir_conf_real", 0.40)

        try:
            self.engine.load_engine(run_settings, self.current_mode,
                                     status_cb=lambda m: self.bridge.log_msg.emit(m))
        except Exception as ex:
            self._add_log(f"Model load failed: {ex}"); return

        # Wire engine callback -> Qt signal
        def on_frame(jpeg_bytes, stats):
            if jpeg_bytes is None:
                self.bridge.finished.emit(); return
            self.bridge.frame_ready.emit(jpeg_bytes, stats)

        self.engine.on_frame = on_frame

        # Compute save path from settings
        sp = None
        if self.settings.get("save_output_enabled"):
            out_dir = self.settings.get("save_output_dir", "").strip()
            if out_dir:
                base = os.path.splitext(os.path.basename(path))[0]
                sp = os.path.join(out_dir, base + "_output.mp4")

        # Reset path tracking for new run
        self._path_history = []
        self._path_frame_w = 0
        self._path_frame_h = 0

        self.engine.start(path, ir_path, self.current_mode,
                          run_settings, self.temporal_on, sp)

        self.status_dot.setStyleSheet("color:#fff; font-size:10px;")
        self.status_label.setText("Processing")
        self._add_log(f"Playing: {os.path.basename(path)} [{self.current_mode}]")
        self._update_icons()

    def _on_stop(self):
        self.engine.stop()
        self.status_dot.setStyleSheet("color:#555; font-size:10px;")
        self.status_label.setText("Idle")
        self._add_log("Stopped")
        self._update_icons()

    def _on_skip(self):
        self.engine.skip_forward(30); self._add_log("Skipped 30s")

    def _on_seek_start(self):
        """User started dragging the seek slider."""
        self._seeking = True

    def _on_seek_end(self):
        """User released the seek slider — seek to that position."""
        self._seeking = False
        if self.engine.running and self.engine.total_frames > 0:
            ratio = self.progress.value() / 1000.0
            target_frame = int(ratio * self.engine.total_frames)
            self.engine.seek_to_frame(target_frame)
            self._add_log(f"Seeked to frame {target_frame}")

    def _restore_persisted_state(self):
        """Restore video paths and mode from settings on startup."""
        last_rgb = self.settings.get("last_rgb_path", "")
        last_ir = self.settings.get("last_ir_path", "")
        last_mode = self.settings.get("last_mode", MODES[0])
        if last_rgb:
            self.rgb_path.setText(last_rgb)
        if last_ir:
            self.ir_path.setText(last_ir)
        if last_mode in MODES:
            self._set_mode(last_mode)

    def resizeEvent(self, event):
        """Reposition fullscreen overlay button on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'fs_btn') and hasattr(self, 'video_label'):
            vw = self.video_label.width()
            self.fs_btn.move(vw - 40, 8)

    @Slot(object, dict)
    def _on_frame(self, jpeg_bytes, stats):
        """Main thread: decode JPEG -> QPixmap -> display."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None: return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        lw, lh = self.video_label.width(), self.video_label.height()
        pixmap = QPixmap.fromImage(qimg).scaled(
            lw, lh, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

        # Reposition fullscreen overlay button
        vw = self.video_label.width()
        self.fs_btn.move(vw - 40, 8)

        f, tot = stats["frame"], stats["total"]
        pct = f / max(tot, 1)
        if not self._seeking:
            self.progress.setValue(int(pct * 1000))
        self.stats_text.setText(f"Frame {f}/{tot}")
        self.fps_value.setText(str(stats["fps"]))
        self.frame_value.setText(f"{f}/{tot}")
        self.frame_pct.setText(f"{pct * 100:.1f}%")

        # Detections count per modality (2-col layout -- just numbers)
        self.dets_rgb_lbl.setText(str(stats.get("n_rgb", 0)))
        self.dets_ir_lbl.setText(str(stats.get("n_ir", 0)))
        self.warn_count.setText(str(stats.get("w_events", 0)))
        self.alert_count.setText(str(stats.get("a_events", 0)))
        self.infer_ms_lbl.setText(f"{stats.get('infer_ms', 0):.0f}ms")

        # Live analytics update
        self.an_frames.setText(str(f))
        self.an_warnings.setText(str(stats.get("w_events", 0)))
        self.an_alerts.setText(str(stats.get("a_events", 0)))
        self.an_mode.setText(self.current_mode)

        # Accumulate drone path data
        rgb_c = stats.get("rgb_center")
        ir_c = stats.get("ir_center")
        trust = stats.get("trust", 0)
        is_alert = bool(stats.get("alert"))
        is_warn = bool(stats.get("warning")) and not is_alert
        self._path_history.append((
            f,
            rgb_c[0] if rgb_c else None, rgb_c[1] if rgb_c else None,
            rgb_c[2] if rgb_c and len(rgb_c) > 2 else None,
            ir_c[0] if ir_c else None, ir_c[1] if ir_c else None,
            ir_c[2] if ir_c and len(ir_c) > 2 else None,
            trust, is_alert, is_warn,
        ))
        fw = stats.get("frame_w", 0)
        fh = stats.get("frame_h", 0)
        if fw > 0 and fh > 0:
            self._path_frame_w = fw
            self._path_frame_h = fh

        # Confuser filter -- alert-gate architecture
        v = stats.get("verifier") or {}
        suppressed = stats.get("confuser_suppressed", False)
        if suppressed:
            labels = stats.get("confuser_labels", {})
            if not getattr(self, '_last_suppressed', False):
                reason_parts = []
                for k in ("rgb", "ir"):
                    if labels.get(k): reason_parts.append(f"{k.upper()}: {labels[k][0]}")
                reason = ", ".join(reason_parts)
                self._add_log(f"Alert Suppressed ({reason})" if reason else "Alert Suppressed")
            self._last_suppressed = True
            
            parts = []
            for k in ("rgb", "ir"):
                if labels.get(k): parts.append(f"{k.upper()}:{labels[k][0]}")
            self.filter_status.setText("ALERT SUPPRESSED")
            self.filter_status.setStyleSheet("font-size:12px; font-weight:700; color:#ff4444;")
            self.filter_detail.setText("\n".join(parts) if parts else "")
        elif v.get("active"):
            self._last_suppressed = False
            rp = v.get("rgb_max_p")
            ip = v.get("ir_max_p")
            parts = []
            if rp is not None: parts.append(f"RGB P={rp:.2f} - {v.get('rgb_n_boxes',0)}box")
            if ip is not None: parts.append(f"IR P={ip:.2f} - {v.get('ir_n_boxes',0)}box")
            self.filter_status.setText("PASS")
            self.filter_status.setStyleSheet("font-size:13px; font-weight:700; color:#44ff88;")
            # Show class labels if available
            lbl_parts = []
            for lbl in v.get("rgb_labels", []): lbl_parts.append(f"rgb={lbl}")
            for lbl in v.get("ir_labels", []): lbl_parts.append(f"ir={lbl}")
            detail = "\n".join(parts)
            if lbl_parts: detail += f"\n({', '.join(lbl_parts)})"
            self.filter_detail.setText(detail)
        else:
            self._last_suppressed = False
            self.filter_status.setText("-- off")
            self.filter_status.setStyleSheet("font-size:13px; font-weight:600; color:#555;")
            self.filter_detail.setText("")

        trust = stats.get("trust")
        mode = stats.get("mode", self.current_mode)
        if mode == "Single Model":
            self.trust_value.setText("SINGLE MODEL")
            self.trust_value.setStyleSheet(
                f"font-size:16px; font-weight:700; color:#999;")
            self.trust_pct.setText("")
        elif trust is not None:
            self.trust_value.setText(TRUST_LABELS.get(trust, "--"))
            self.trust_value.setStyleSheet(
                f"font-size:20px; font-weight:700; color:{TRUST_COLORS.get(trust, '#555')};")
            tp = stats.get("trust_prob")
            self.trust_pct.setText(f"{tp*100:.1f}%" if tp else "")

        # Alert / Warning chips + event logging
        is_alert = bool(stats.get("alert"))
        is_warn = bool(stats.get("warning")) and not is_alert
        self.alert_chip.setVisible(is_alert)
        self.warning_chip.setVisible(is_warn)
        if is_alert and not self._prev_alert:
            self._add_log(f"ALERT triggered at frame {f}")
        if is_warn and not self._prev_warning:
            self._add_log(f"WARNING triggered at frame {f}")
        self._prev_alert = is_alert
        self._prev_warning = is_warn

    @Slot()
    def _on_done(self):
        self.play_btn.setText("▶")
        self.status_dot.setStyleSheet("color:#555; font-size:10px;")
        self.status_label.setText("Done")
        self._add_log("Done")
        # Finalize analytics
        self.an_video.setText(os.path.basename(self.rgb_path.text()) if self.rgb_path.text() else "--")
        # Generate drone path image
        self._generate_path_image()

    # -- DRONE PATH GENERATOR ----------------------------
    def _generate_path_image(self):
        """Render the accumulated detection centers as a drone flight path image."""
        if not self._path_history or self._path_frame_w == 0:
            self.path_image_label.setText("No detection data to generate path")
            self.path_save_btn.setEnabled(False)
            return

        fw, fh = self._path_frame_w, self._path_frame_h
        # Canvas size: match source aspect ratio, capped at 1280 wide
        scale = min(1280 / fw, 720 / fh, 1.0)
        cw, ch = int(fw * scale), int(fh * scale)

        # Dark canvas
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 12)

        # Draw subtle grid
        grid_color = (30, 30, 30)
        grid_step = max(40, cw // 16)
        for gx in range(0, cw, grid_step):
            cv2.line(canvas, (gx, 0), (gx, ch), grid_color, 1)
        for gy in range(0, ch, grid_step):
            cv2.line(canvas, (0, gy), (cw, gy), grid_color, 1)

        # Collect RGB and IR path points (with bbox area for range proxy)
        rgb_pts = []  # (x, y, t_norm, is_alert, is_warning, area_px)
        ir_pts = []
        total = len(self._path_history)
        for i, (frame, rcx, rcy, rarea, icx, icy, iarea, trust, is_alert, is_warn) in enumerate(self._path_history):
            t = i / max(total - 1, 1)  # 0..1 time progress
            if rcx is not None and rcy is not None:
                x = int(rcx * scale)
                y = int(rcy * scale)
                rgb_pts.append((x, y, t, is_alert, is_warn, rarea or 0.0))
            if icx is not None and icy is not None:
                x = int(icx * scale)
                y = int(icy * scale)
                ir_pts.append((x, y, t, is_alert, is_warn, iarea or 0.0))

        def _lerp_color(c1, c2, t):
            return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

        # Thickness range (px). Far → thin, near → bold.
        T_MIN, T_MAX = 1, 6

        def _draw_path(pts, label, offset_y, color):
            if len(pts) < 2:
                return
            # Apparent radius (∝ 1/Z): the range proxy that drives thickness.
            radii_px = [max(1.0, (a ** 0.5)) for *_, a in pts]
            # Per-modality normalization so thickness actually spans the range
            # even on clips where the drone never gets very close or very far.
            r_min = min(radii_px)
            r_max = max(radii_px)
            r_span = max(1e-6, r_max - r_min)
            # Smooth radii a touch so thickness doesn't strobe per frame.
            from collections import deque
            w = max(2, len(radii_px) // 30)
            q = deque(maxlen=w)
            smoothed = []
            for r_ in radii_px:
                q.append(r_)
                smoothed.append(sum(q) / len(q))

            gap_thresh = max(cw, ch) * 0.15  # gap if points jump > 15% of frame
            for j in range(1, len(pts)):
                x0, y0, *_ = pts[j - 1]
                x1, y1, *_ = pts[j]
                dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if dist > gap_thresh:
                    continue
                # Average smoothed size across the segment.
                seg = (smoothed[j - 1] + smoothed[j]) * 0.5
                norm = (seg - r_min) / r_span
                thick = int(round(T_MIN + norm * (T_MAX - T_MIN)))
                cv2.line(canvas, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)

            # Draw markers: alerts (red circles), warnings (yellow diamonds)
            for x, y, t, is_alert, is_warn, _ in pts:
                if is_alert:
                    cv2.circle(canvas, (x, y), 6, (0, 0, 255), 2, cv2.LINE_AA)
                elif is_warn:
                    # Small diamond
                    d = 4
                    diamond = np.array([[x, y - d], [x + d, y], [x, y + d], [x - d, y]], np.int32)
                    cv2.polylines(canvas, [diamond], True, (0, 200, 255), 1, cv2.LINE_AA)

            # Start marker (green circle with S)
            sx, sy = pts[0][0], pts[0][1]
            cv2.circle(canvas, (sx, sy), 8, (0, 220, 100), 2, cv2.LINE_AA)
            cv2.putText(canvas, "S", (sx - 4, sy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 220, 100), 1, cv2.LINE_AA)
            # End marker (red circle with E)
            ex, ey = pts[-1][0], pts[-1][1]
            cv2.circle(canvas, (ex, ey), 8, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "E", (ex - 4, ey + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (80, 80, 255), 1, cv2.LINE_AA)

            # Modality label in its own colour
            cv2.putText(canvas, label, (12, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        # Modality colors. Thickness encodes range (far=thin, near=bold).
        RGB_COLOR = (255, 80, 220)   # bright magenta
        IR_COLOR  = (60, 180, 255)   # bright orange
        _draw_path(ir_pts,  "IR path",  ch - 20, color=IR_COLOR)
        _draw_path(rgb_pts, "RGB path", ch - 40, color=RGB_COLOR)

        # Closing-rate panel + apparent-size sparkline (bottom-right of canvas).
        # Use whichever modality has more samples for the readout.
        primary_is_rgb = len(rgb_pts) >= len(ir_pts)
        primary_pts = rgb_pts if primary_is_rgb else ir_pts
        self._draw_range_panel(canvas, primary_pts, primary_is_rgb)

        # Title overlay
        video_name = os.path.basename(self.rgb_path.text()) if self.rgb_path.text() else "video"
        title = f"Drone Path  -  {video_name}  ({total} samples)"
        cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Legend: alert/warning markers
        legend_y = 48
        cv2.circle(canvas, (18, legend_y), 5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Alert", (28, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(canvas, (80, legend_y), 4, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Warning", (90, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(canvas, (155, legend_y), 5, (0, 220, 100), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Start", (165, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(canvas, (210, legend_y), 5, (80, 80, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "End", (220, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)
        # Modality color swatches
        cv2.line(canvas, (260, legend_y), (282, legend_y), (255, 80, 220), 3, cv2.LINE_AA)
        cv2.putText(canvas, "RGB", (288, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(canvas, (322, legend_y), (344, legend_y), (60, 180, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "IR", (350, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Thickness ramp: thin (far) → bold (near)
        cv2.putText(canvas, "Far", (376, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)
        ramp_x0 = 398
        seg_w = 18
        for i, t in enumerate((1, 2, 3, 4, 5, 6)):
            x = ramp_x0 + i * seg_w
            cv2.line(canvas, (x, legend_y), (x + seg_w - 2, legend_y),
                     (200, 200, 200), t, cv2.LINE_AA)
        cv2.putText(canvas, "Near", (ramp_x0 + 6 * seg_w + 4, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Cache for save and re-scale to current label size
        self._path_canvas = canvas.copy()
        self._refresh_path_pixmap()
        self.path_save_btn.setEnabled(True)
        self.path_card.setEnabled(True)
        self.path_card.setStyleSheet("")  # Reset dimmed style
        self._add_log(f"Drone path generated ({len(rgb_pts)} RGB, {len(ir_pts)} IR points)")

    def _draw_range_panel(self, canvas, pts, is_rgb):
        """Bottom-right panel: apparent-size sparkline + closing/receding readout.

        We can't measure absolute Z without camera intrinsics, but the rate of
        change of the bbox's apparent linear size (sqrt of area) is a calibration-
        free proxy: positive slope = closing, negative = receding.
        """
        ch_, cw_ = canvas.shape[:2]
        # Filter to points with a valid area sample.
        sizes = [a ** 0.5 for *_, a in pts if a and a > 0]
        if len(sizes) < 4:
            return
        # Smooth with a short moving average to suppress per-frame jitter.
        win = max(3, len(sizes) // 20)
        smoothed = []
        acc = 0.0
        from collections import deque
        q = deque(maxlen=win)
        for s in sizes:
            q.append(s)
            smoothed.append(sum(q) / len(q))

        # Overall start to end change in apparent linear size.
        # Compare mean radius of the first ~10% of samples vs the last ~10%,
        # so the readout reflects how much closer the drone got across the whole
        # run (not just the tail — which can be flat if the drone landed).
        win_n = max(3, len(smoothed) // 10)
        head_mean = sum(smoothed[:win_n]) / win_n
        tail_mean = sum(smoothed[-win_n:]) / win_n
        rel_pct = ((tail_mean - head_mean) / head_mean * 100.0) if head_mean > 0 else 0.0

        # Panel geometry
        pw, ph = 240, 78
        pad = 12
        x0 = cw_ - pw - pad
        y0 = ch_ - ph - pad
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (24, 24, 24), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (60, 60, 60), 1)

        modality = "RGB" if is_rgb else "IR"
        # Threshold at ±5% overall change — below that, the drone effectively
        # held its distance over the run.
        if rel_pct > 5:
            label = f"CLOSING  +{rel_pct:.0f}%  start to end"
            col = (0, 255, 255)       # bright yellow (BGR)
        elif rel_pct < -5:
            label = f"RECEDING  {rel_pct:.0f}%  start to end"
            col = (255, 0, 200)       # bright magenta
        else:
            label = f"STEADY  {rel_pct:+.0f}%  start to end"
            col = (200, 200, 200)
        cv2.putText(canvas, f"Range change ({modality})", (x0 + 8, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 + 8, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Sparkline of smoothed apparent size across the full run.
        sx0 = x0 + 8
        sy0 = y0 + 44
        sw = pw - 16
        sh = ph - 50
        s_min = min(smoothed)
        s_max = max(smoothed)
        rng = max(1e-6, s_max - s_min)
        prev = None
        for i, s in enumerate(smoothed):
            px = sx0 + int(i * (sw - 1) / max(1, len(smoothed) - 1))
            py = sy0 + sh - 1 - int((s - s_min) / rng * (sh - 1))
            if prev is not None:
                cv2.line(canvas, prev, (px, py), col, 1, cv2.LINE_AA)
            prev = (px, py)

    def _save_path_image(self):
        """Save the generated drone path image to file."""
        if not hasattr(self, '_path_canvas') or self._path_canvas is None:
            return
        default_name = "drone_path.png"
        if self.rgb_path.text():
            base = os.path.splitext(os.path.basename(self.rgb_path.text()))[0]
            default_name = f"{base}_drone_path.png"
        # Save to demo_outputs by default
        out_dir = str(Path(__file__).resolve().parent / "demo_outputs")
        os.makedirs(out_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Drone Path", os.path.join(out_dir, default_name),
            "PNG Image (*.png);;All Files (*)")
        if path:
            cv2.imwrite(path, self._path_canvas)
            self._add_log(f"Path image saved: {os.path.basename(path)}")

    # -- SETTINGS ----------------------------------------
    def _open_settings(self):
        from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Settings"); dlg.resize(550, 500)
        if self.dark_mode:
            dlg.setStyleSheet("QDialog{background:#111; border-radius:12px;} QLabel{color:#fff;} "
                              "QLineEdit{background:#0a0a0a;color:#fff;border:1px solid #333;"
                              "border-radius:6px;padding:4px 8px;}")
        else:
            dlg.setStyleSheet("QDialog{background:#fff; border-radius:12px;} QLabel{color:#000;} "
                              "QLineEdit{background:#fff;color:#000;border:1px solid #000;"
                              "border-radius:6px;padding:4px 8px;}")
        form = QFormLayout()
        refs = {}
        for sec, keys in SECTIONS:
            form.addRow(QLabel(f"-- {sec} --"))
            for k in keys:
                if k in CHOICE_KEYS:
                    from PySide6.QtWidgets import QComboBox
                    cb = QComboBox()
                    cb.addItems(CHOICE_KEYS[k])
                    cur = str(self.settings.get(k, DEFAULTS.get(k, "")))
                    idx = cb.findText(cur)
                    if idx >= 0: cb.setCurrentIndex(idx)
                    refs[k] = cb
                    form.addRow(LABELS.get(k, k), cb)
                elif k in BOOL_KEYS:
                    from PySide6.QtWidgets import QCheckBox
                    cb = QCheckBox()
                    val = self.settings.get(k, DEFAULTS.get(k, False))
                    cb.setChecked(str(val).lower() in ("true", "1", "yes"))
                    refs[k] = cb
                    form.addRow(LABELS.get(k, k), cb)
                else:
                    le = QLineEdit(str(self.settings.get(k, DEFAULTS.get(k, ""))))
                    refs[k] = le
                    form.addRow(LABELS.get(k, k), le)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                 QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(lambda: self._save_settings(refs, dlg))
        btns.rejected.connect(dlg.reject)
        outer = QVBoxLayout(dlg)
        scroll = QScrollArea(); sw = QWidget(); sw.setLayout(form)
        scroll.setWidget(sw); scroll.setWidgetResizable(True)
        outer.addWidget(scroll); outer.addWidget(btns)
        dlg.exec()

    def _save_settings(self, refs, dlg):
        for k, ctrl in refs.items():
            if k in CHOICE_KEYS:
                val = ctrl.currentText()
                self.settings[k] = int(val) if k in CHOICE_INT_KEYS else val
            elif k in FLOAT_KEYS:
                try: self.settings[k] = float(ctrl.text())
                except: pass
            elif k in INT_KEYS:
                try: self.settings[k] = int(ctrl.text())
                except: pass
            elif k in BOOL_KEYS:
                self.settings[k] = ctrl.isChecked()
            else:
                self.settings[k] = ctrl.text()
        save_settings(self.settings)
        self._add_log("Settings saved")
        dlg.accept()

    def closeEvent(self, event):
        self.engine.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TalosWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
