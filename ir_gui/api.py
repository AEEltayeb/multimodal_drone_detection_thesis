"""
FastAPI backend for drone detection.
Uses DroneDetector class from drone_detector.py 
"""

import asyncio
import base64
import cv2
import numpy as np
import os
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local detection module (copy of drone_detection.py)
import detection as drone_detection
import threading
import queue
import time
import json

app = FastAPI(title="Drone Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint for connection verification
@app.get("/api/health")
async def health_check():
    """Health check endpoint for startup verification."""
    return {"status": "ok", "service": "drone-detection-api"}

# Global state for Legacy Integration
detector_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
pause_event = threading.Event()
# Queue for frames (producer=legacy script, consumer=websocket)
# We use a small maxsize to drop frames if WS is slow (simulates real-time)
frame_queue = queue.Queue(maxsize=2)
latest_stats = {}

# Model weights path (set via /api/model endpoint)
model_weights_path = ""

# Video state (set via /api/source/* endpoints)
video_path = None
video_cap = None
video_fps = 0
total_frames = 0
detector = None
is_streaming = False

# ESP32 state (None until explicitly connected via /api/esp32/connect)
esp32_address = None

# Stream quality settings
QUALITY_PRESETS = {
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": None
}
stream_quality = "1080p"

# Settings Persistence
# Settings Persistence
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def load_settings():
    """Load settings from JSON and apply to drone_detection."""
    if not os.path.exists(SETTINGS_FILE):
        return
    
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        
        print(f"[INFO] Loading settings from {SETTINGS_FILE}: {data}")
        
        # Apply strict mapping because JSON keys match API keys (mostly), 
        # but apply_runtime_config needs CONSTANT_CASE names.
        
        # We reuse the logic from update_settings by creating a dummy update object?
        # Or just map manually.
        updates = {}
        # Map JSON keys (which match SettingsUpdate model) to Script Globals
        mapping = {
            "infer_fps": "INFER_FPS",
            "temporal_roi_enabled": "TEMPORAL_ROI_PROP_ENABLED",
            "show_gate": "SHOW_GATE",
            "show_troi": "SHOW_TROI",
            "show_cascade": "SHOW_CASCADE",
            "detect_conf": "DETECT_CONF",
            "cascade_mode": "CASCADED_ROI_CONFIRM_MODE",
            "warning_cooldown": "WARNING_COOLDOWN_S",
            "alert_cooldown": "ALERT_COOLDOWN_S",
            "save_video": "SAVE_VIDEO",
            "save_alert_frames": "SAVE_ALERT_WINDOW_FRAMES",
            "log_mode": "TOPLEFT_LOG_MODE",
            "roi_size": "ROI_SIZE",
            "cascade_trigger_conf": "CASCADE_TRIGGER_CONF",
            "cascade_accept_conf": "CASCADE_ACCEPT_CONF",
            "warning_window_size": "WARNING_WINDOW_FRAMES",
            "warning_require_hits": "WARNING_REQUIRE_HITS",
            "alert_window_size": "ALERT_WINDOW_FRAMES",
            "alert_require_hits": "ALERT_REQUIRE_HITS",
            "alert_min_area": "ALERT_MIN_AREA_PX2",
            "simple_mode": "SIMPLE_MODE",
            "alert_avg_conf_threshold": "ALERT_AVG_CONF_THRESHOLD"
        }
        
        for json_key, script_key in mapping.items():
            if json_key in data:
                updates[script_key] = data[json_key]
                # Special case: Sync TROI conf
                if json_key == "detect_conf":
                    updates["TROI_DETECT_CONF"] = data[json_key]

        if updates:
            drone_detection.apply_runtime_config(**updates)
            
    except Exception as e:
        print(f"[ERROR] Failed to load settings: {e}")

def save_settings(new_data: dict):
    """Merge new_data into existing settings file."""
    try:
        current = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                try:
                    current = json.load(f)
                except:
                    pass
        
        # Update current with new data
        current.update(new_data)
        
        with open(SETTINGS_FILE, "w") as f:
            json.dump(current, f, indent=2)
            
    except Exception as e:
        print(f"[ERROR] Failed to save settings: {e}")

# Load immediately on startup
load_settings()



def frame_callback_handler(frame, stats):
    """Called by legacy script for every frame (when headless)."""
    global latest_stats
    latest_stats = stats
    
    # Non-blocking put; if full, drop frame (real-time behavior)
    try:
        frame_queue.put_nowait((frame, stats))
    except queue.Full:
        pass

def init_detector_thread():
    """Start the legacy script in a separate thread."""
    global detector_thread, stop_event, pause_event, is_streaming
    
    if detector_thread and detector_thread.is_alive():
        return # Already running
        
    stop_event.clear()
    pause_event.clear()
    
    # reset queue
    with frame_queue.mutex:
        frame_queue.queue.clear()

    def run_wrapper():
        print("[INFO] Starting Legacy Drone Detection Script...")
        try:
            # Inject video path and model weights before starting
            if video_path and video_path != "webcam":
                drone_detection.apply_runtime_config(VIDEO_PATH=video_path)
            if model_weights_path:
                drone_detection.apply_runtime_config(MODEL_WEIGHTS=model_weights_path)
                
            # Inject ESP32 hooks (fire-and-forget — never block the detection loop)
            def send_alert_real():
                def _send():
                    if esp32_address:
                        try:
                            import requests
                            print(f"[ESP32] Sending POST /alert/drone to {esp32_address}...")
                            resp = requests.post(f"http://{esp32_address}/alert/drone", timeout=1.0)
                            print(f"[ESP32] ALERT Response: {resp.status_code} {resp.text}")
                        except Exception as e:
                            print(f"[ESP32] Failed to send ALERT: {e}")
                threading.Thread(target=_send, daemon=True).start()

            def send_warning_real():
                def _send():
                    if esp32_address:
                        try:
                            import requests
                            print(f"[ESP32] Sending POST /alert/warning to {esp32_address}...")
                            resp = requests.post(f"http://{esp32_address}/alert/warning", timeout=1.0)
                            print(f"[ESP32] WARNING Response: {resp.status_code} {resp.text}")
                        except Exception as e:
                            print(f"[ESP32] Failed to send WARNING: {e}")
                threading.Thread(target=_send, daemon=True).start()

            drone_detection.send_alert_to_esp = send_alert_real
            drone_detection.send_warning_to_esp = send_warning_real
            
            drone_detection.main(
                headless=True,
                frame_callback=frame_callback_handler,
                stop_event=stop_event,
                pause_event=pause_event
            )
        except Exception as e:
            print(f"[ERROR] Script Crashed: {e}")
        finally:
            print("[INFO] Legacy Script Finished")
            is_streaming = False

    detector_thread = threading.Thread(target=run_wrapper, daemon=True)
    detector_thread.start()
    is_streaming = True



# Request models
class FileSource(BaseModel):
    path: str

class YouTubeSource(BaseModel):
    url: str
    mode: Optional[str] = None  # "paired" | "grayscale" | "single"; YT-only hint

class ESP32Connect(BaseModel):
    address: str

class SettingsUpdate(BaseModel):
    cascade_mode: Optional[str] = None
    temporal_roi_enabled: Optional[bool] = None
    infer_fps: Optional[int] = None
    show_gate: Optional[bool] = None
    show_troi: Optional[bool] = None
    show_cascade: Optional[bool] = None
    log_mode: Optional[str] = None
    save_video: Optional[bool] = None
    save_alert_frames: Optional[bool] = None
    warning_cooldown: Optional[float] = None
    alert_cooldown: Optional[float] = None
    detect_conf: Optional[float] = None
    roi_size: Optional[int] = None
    # Window settings
    warning_window_size: Optional[int] = None
    warning_require_hits: Optional[int] = None
    alert_window_size: Optional[int] = None
    alert_require_hits: Optional[int] = None
    cascade_trigger_conf: Optional[float] = None
    cascade_accept_conf: Optional[float] = None
    alert_min_area: Optional[int] = None
    simple_mode: Optional[bool] = None
    alert_avg_conf_threshold: Optional[float] = None

# Device setting (cpu/gpu)
current_device = "gpu"

@app.post("/api/device")
async def set_device(device: str = "gpu"):
    """Switch between CPU and GPU for inference."""
    global current_device
    if device not in ["cpu", "gpu"]:
        raise HTTPException(status_code=400, detail="Device must be 'cpu' or 'gpu'")
    
    current_device = device
    # Update fusion config so the next fusion stream uses this device
    fusion_config["device"] = 0 if device == "gpu" else "cpu"
    print(f"[Device] Switched to {device}. Fusion config device={fusion_config['device']}. Restart stream to apply.")
    return {"device": device, "status": "ok"}


# Status
@app.get("/api/status")
async def get_status():
    global detector, is_streaming, video_cap
    if detector is None:
        return {
            "warning_active": False,
            "alert_active": False,
            "is_streaming": False,
            "video_loaded": False,
        }
    # Build status from detector state
    return {
        "warning_active": latest_stats.get("warning_active", False),
        "alert_active": latest_stats.get("alert_active", False),
        "warning_events": latest_stats.get("warning_events", 0),
        "alert_events": latest_stats.get("alert_events", 0),
        "is_streaming": is_streaming and (detector_thread and detector_thread.is_alive()),
        "video_loaded": True, # Legacy script handles its own loading
        "esp32_connected": esp32_address is not None,
        "total_frames": 0, # Script logic
        "frame_id": latest_stats.get("frame_id", 0),
    }


# Source endpoints
@app.post("/api/source/file")
async def open_file(data: FileSource):
    global video_cap, video_path, video_fps, total_frames, detector
    
    video_path = data.path
    # We delay start until 'start_detection' is called, OR we just set the path config
    # The script opens the video in main(). So we just set the path variable.
    # Note: If thread is running, we might need to restart it.
    
    if detector_thread and detector_thread.is_alive():
        stop_event.set()
        detector_thread.join(timeout=2.0)
    
    return {"success": True, "frames": 0, "fps": 0}


@app.post("/api/source/webcam")
async def open_webcam():
    global video_cap, video_path, video_fps, total_frames, detector
    
    # Webcam not fully supported in simple legacy wrapper without changing VIDEO_PATH to int(0)
    # Use path "0" or 0
    video_path = 0 
    
    if detector_thread and detector_thread.is_alive():
        stop_event.set()
        detector_thread.join(timeout=2.0)

    return {"success": True, "fps": 30}


@app.post("/api/source/youtube")
async def open_youtube(data: YouTubeSource):
    global video_cap, video_path, video_fps, total_frames, detector
    
    try:
        import yt_dlp
        import re as _re
        
        # Extract video ID for unique filename
        vid_match = _re.search(r'(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([a-zA-Z0-9_-]{11})', data.url)
        vid_id = vid_match.group(1) if vid_match else "video"
        
        # Download to local temp file (streaming URLs expire mid-playback)
        dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_outputs")
        os.makedirs(dl_dir, exist_ok=True)
        dl_path = os.path.join(dl_dir, f"yt_{vid_id}.mp4")
        
        # If already downloaded, reuse it
        if os.path.exists(dl_path) and os.path.getsize(dl_path) > 100_000:
            print(f"[YT] Reusing cached download: {dl_path}")
            video_path = dl_path
            if detector_thread and detector_thread.is_alive():
                stop_event.set()
                detector_thread.join(timeout=2.0)
            return {"success": True, "frames": 0, "fps": 0}
        
        cookie_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_cookies.txt")
        
        # Remove partial file to prevent HTTP 416
        if os.path.exists(dl_path):
            os.remove(dl_path)
        
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': dl_path,
            'quiet': True,
            'no_warnings': True,
            'overwrites': True,
            'continuedl': False,
            'socket_timeout': 30,
            'retries': 5,
            'fragment_retries': 5,
        }
        
        # Auto-extract cookies from browser (try Chrome first, then Opera GX)
        cookie_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_cookies.txt")
        browser_cookie_set = False
        for browser in ['chrome', 'opera', 'edge']:
            try:
                test_opts = {**ydl_opts, 'cookiesfrombrowser': (browser,), 'extract_flat': True}
                with yt_dlp.YoutubeDL(test_opts) as test_ydl:
                    test_ydl.extract_info(data.url, download=False)
                ydl_opts['cookiesfrombrowser'] = (browser,)
                print(f"[YT] Using cookies from {browser}")
                browser_cookie_set = True
                break
            except Exception:
                continue
        
        if not browser_cookie_set and os.path.isfile(cookie_file):
            ydl_opts['cookiefile'] = cookie_file
            print(f"[YT] Falling back to cookie file: {cookie_file}")
            
        print(f"[YT] Downloading {vid_id} to {dl_path}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([data.url])
        print(f"[YT] Download complete: {dl_path}")

        video_path = dl_path

        if detector_thread and detector_thread.is_alive():
            stop_event.set()
            detector_thread.join(timeout=2.0)

        return {"success": True, "frames": 0, "fps": 0}
    except ImportError:
        raise HTTPException(400, "yt-dlp not installed")
    except Exception as e:
        raise HTTPException(400, f"Failed to download YouTube video: {str(e)}")


# Control endpoints
@app.post("/api/control/start")
async def start_detection():
    # Start the thread
    init_detector_thread()
    # Unpause if paused
    pause_event.clear()
    return {"success": True}


@app.post("/api/control/stop")
async def stop_detection():
    stop_event.set()
    return {"success": True}

@app.post("/api/control/pause")
async def pause_detection():
    if pause_event.is_set():
        pause_event.clear() # Resume
    else:
        pause_event.set() # Pause
    return {"success": True, "paused": pause_event.is_set()}


@app.post("/api/control/seek")
async def seek_video(data: dict):
    frames = data.get("frames", 0)
    absolute = data.get("absolute", False)
    
    if absolute:
        # Absolute seek: jump to specific frame
        drone_detection.REQUESTED_SEEK_ABS = int(frames)
    else:
        # Relative seek
        drone_detection.REQUESTED_SEEK_REL = frames
    
    return {"success": True, "frames": frames, "absolute": absolute}


@app.post("/api/control/speed")
async def set_speed(data: dict):
    speed = float(data.get("speed", 1.0))
    speed = max(0.25, min(speed, 8.0))  # clamp to reasonable range
    drone_detection.PLAYBACK_SPEED = speed
    return {"success": True, "speed": speed}


@app.post("/api/control/reset")
async def reset_detection():
    """Reset detector state for replay."""
    global detector, video_cap
    if detector:
        detector.reset()
    if video_cap and video_cap.isOpened():
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return {"success": True}


# Settings endpoints
@app.get("/api/settings")
async def get_settings():
    # Return subset of globals from script
    return {
        "infer_fps": drone_detection.INFER_FPS,
        "temporal_roi_enabled": drone_detection.TEMPORAL_ROI_PROP_ENABLED,
        "show_gate": drone_detection.SHOW_GATE,
        "show_troi": drone_detection.SHOW_TROI,
        "show_cascade": drone_detection.SHOW_CASCADE,
        "detect_conf": drone_detection.DETECT_CONF,
        "cascade_mode": drone_detection.CASCADED_ROI_CONFIRM_MODE,
        "warning_cooldown": drone_detection.WARNING_COOLDOWN_S,
        "alert_cooldown": drone_detection.ALERT_COOLDOWN_S,
        "save_video": drone_detection.SAVE_VIDEO,
        "save_alert_frames": drone_detection.SAVE_ALERT_WINDOW_FRAMES,
        "log_mode": drone_detection.TOPLEFT_LOG_MODE,
        "cascade_trigger_conf": drone_detection.CASCADE_TRIGGER_CONF,
        "cascade_accept_conf": drone_detection.CASCADE_ACCEPT_CONF,
        "roi_size": drone_detection.ROI_SIZE,
        "warning_window_size": drone_detection.WARNING_WINDOW_FRAMES,
        "warning_require_hits": drone_detection.WARNING_REQUIRE_HITS,
        "alert_window_size": drone_detection.ALERT_WINDOW_FRAMES,
        "alert_require_hits": drone_detection.ALERT_REQUIRE_HITS,
        "alert_min_area": drone_detection.ALERT_MIN_AREA_PX2,
        "simple_mode": drone_detection.SIMPLE_MODE,
        "alert_avg_conf_threshold": drone_detection.ALERT_AVG_CONF_THRESHOLD,
    }


@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    # Map frontend keys to script globals (CAPS)
    updates = {}
    if settings.infer_fps is not None: updates["INFER_FPS"] = settings.infer_fps
    if settings.temporal_roi_enabled is not None: updates["TEMPORAL_ROI_PROP_ENABLED"] = settings.temporal_roi_enabled
    if settings.show_gate is not None: updates["SHOW_GATE"] = settings.show_gate
    if settings.show_troi is not None: updates["SHOW_TROI"] = settings.show_troi
    if settings.show_cascade is not None: updates["SHOW_CASCADE"] = settings.show_cascade
    if settings.detect_conf is not None: 
        updates["DETECT_CONF"] = settings.detect_conf
        # Sync TROI conf? User request "change playback settings appropriately"
        updates["TROI_DETECT_CONF"] = settings.detect_conf
        
    # Additional mappings
    if settings.cascade_mode is not None: updates["CASCADED_ROI_CONFIRM_MODE"] = settings.cascade_mode
    if settings.warning_cooldown is not None: updates["WARNING_COOLDOWN_S"] = settings.warning_cooldown
    if settings.alert_cooldown is not None: updates["ALERT_COOLDOWN_S"] = settings.alert_cooldown
    if settings.save_video is not None: updates["SAVE_VIDEO"] = settings.save_video
    if settings.save_alert_frames is not None: updates["SAVE_ALERT_WINDOW_FRAMES"] = settings.save_alert_frames
    if settings.log_mode is not None: updates["TOPLEFT_LOG_MODE"] = settings.log_mode
    if settings.roi_size is not None: updates["ROI_SIZE"] = settings.roi_size
    if settings.cascade_trigger_conf is not None: updates["CASCADE_TRIGGER_CONF"] = settings.cascade_trigger_conf
    if settings.cascade_accept_conf is not None: updates["CASCADE_ACCEPT_CONF"] = settings.cascade_accept_conf
    if settings.warning_window_size is not None: updates["WARNING_WINDOW_FRAMES"] = settings.warning_window_size
    if settings.warning_require_hits is not None: updates["WARNING_REQUIRE_HITS"] = settings.warning_require_hits
    if settings.alert_window_size is not None: updates["ALERT_WINDOW_FRAMES"] = settings.alert_window_size
    if settings.alert_require_hits is not None: updates["ALERT_REQUIRE_HITS"] = settings.alert_require_hits
    if settings.alert_min_area is not None: updates["ALERT_MIN_AREA_PX2"] = settings.alert_min_area
    if settings.simple_mode is not None: updates["SIMPLE_MODE"] = settings.simple_mode
    if settings.alert_avg_conf_threshold is not None: updates["ALERT_AVG_CONF_THRESHOLD"] = settings.alert_avg_conf_threshold
    
    # Apply
    drone_detection.apply_runtime_config(**updates)

    # PERSIST: Save the values that were handled
    # We reconstruct a dict of {json_key: value} based on what was in 'settings'
    # 'settings' is a SettingsUpdate object, containing .exclude_unset=True equivalent?
    # We iterate typical keys.
    raw_dict = settings.dict(exclude_unset=True)
    if raw_dict:
        save_settings(raw_dict)
    
    # RELIABLE SETTINGS: Restart the detection thread if it's running
    if detector_thread and detector_thread.is_alive():
        print("Settings changed: Restarting detection thread...")
        stop_event.set()
        # Wait briefly for it to stop (it checks stop_event every frame)
        detector_thread.join(timeout=2.0)
        
        # Restart
        init_detector_thread()
        pause_event.clear() # Ensure we don't start paused
    
    return {"success": True, "settings": updates}


# Model endpoints
class ModelSource(BaseModel):
    path: str

@app.get("/api/model")
async def get_model():
    return {"path": drone_detection.MODEL_WEIGHTS}

@app.post("/api/model")
async def set_model(data: ModelSource):
    global model_weights_path
    path = data.path
    if not os.path.isfile(path):
        raise HTTPException(400, f"Model file not found: {path}")

    model_weights_path = path
    drone_detection.apply_runtime_config(MODEL_WEIGHTS=path)

    # Restart detection thread if running
    if detector_thread and detector_thread.is_alive():
        stop_event.set()
        detector_thread.join(timeout=2.0)
        init_detector_thread()
        pause_event.clear()

    return {"success": True, "path": path}


# Alert endpoints
@app.post("/api/alert/dismiss")
async def dismiss_alert():
    """
    GLOBAL SUPPRESSION: Clears BOTH warning AND alert states.
    
    This is NOT alert-only. It resets all detection windows and triggers
    cooldowns for both warning and alert levels. Use this as the single
    "dismiss all" action - do NOT add separate warning-only dismiss.
    """
    # Direct global reset for legacy script
    drone_detection.win_warning.clear()
    drone_detection.win_alert.clear()
    drone_detection.inference_rows.clear()
    drone_detection.warning_active = False
    drone_detection.alert_active = False

    # Reset cooldowns? Or Trigger cooldowns? User says "trigger cooldowns".
    # Actually "trigger cooldowns" means START the cooldown so it doesn't alert again immediately.
    # So we should set cooldown_left to max.
    drone_detection.warn_cooldown_left = drone_detection.warn_cooldown_frames
    drone_detection.alert_cooldown_left = drone_detection.alert_cooldown_frames
    
    # Also send clear to ESP32 (non-blocking)
    if esp32_address:
        def _send_clear():
            try:
                import requests
                requests.post(f"http://{esp32_address}/alert/clear", timeout=1)
            except:
                pass
        threading.Thread(target=_send_clear, daemon=True).start()
    return {"success": True}


# ESP32 endpoint
@app.post("/api/esp32/connect")
async def connect_esp32(data: ESP32Connect):
    global esp32_address
    try:
        import requests
        response = requests.get(f"http://{data.address}/status", timeout=2)
        if response.status_code == 200:
            esp32_address = data.address
            return {"success": True}
    except:
        pass
    raise HTTPException(400, "Failed to connect to ESP32")


@app.get("/api/esp32/status")
async def get_esp32_status():
    return {"connected": esp32_address is not None, "address": esp32_address}


# Stream quality
@app.post("/api/stream/quality")
async def set_stream_quality(quality: str = "480p"):
    global stream_quality
    if quality in QUALITY_PRESETS:
        stream_quality = quality
        return {"success": True, "quality": quality}
    raise HTTPException(400, f"Invalid quality. Options: {list(QUALITY_PRESETS.keys())}")


@app.get("/api/stream/quality")
async def get_stream_quality():
    return {"quality": stream_quality, "options": list(QUALITY_PRESETS.keys())}


# Send alert to ESP32 (non-blocking)
def send_esp32_alert(alert_type: str, message: str = ""):
    def _send():
        if esp32_address:
            try:
                import requests
                if alert_type == "alert":
                    requests.post(f"http://{esp32_address}/alert/drone", 
                                json={"message": message}, timeout=0.5)
                elif alert_type == "warning":
                    requests.post(f"http://{esp32_address}/alert/warning",
                                json={"message": message}, timeout=0.5)
                elif alert_type == "clear":
                    requests.post(f"http://{esp32_address}/alert/clear", timeout=0.5)
            except:
                pass
    if esp32_address:
        threading.Thread(target=_send, daemon=True).start()


# WebSocket for video streaming
@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    global detector, video_cap, is_streaming
    
    await websocket.accept()
    
    # Track alert state for ESP32
    last_alert_state = False
    last_warning_state = False
    
    try:
        import time
        last_frame_time = 0
        target_fps = 25
        
        while True:
            # Consumer: Get frame from queue
            try:
                # Wait for frame (timeout to allow check for disconnect)
                frame, stats = frame_queue.get(timeout=0.1)
                
                # Resize if needed (script handles visualization resizing? No, main() creates local `vis`)
                # Let's trust the script's visual output, maybe resize for bandwidth if large
                resolution = QUALITY_PRESETS.get(stream_quality)
                if resolution is not None:
                    preview = cv2.resize(frame, resolution)
                else:
                    preview = frame
                
                # Encode
                _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                        "type": "frame",
                        "data": b64,
                        "stats": stats, # Already formatted in callback
                        "alert_active": stats['alert_active'],
                        "warning_active": stats['warning_active'],
                })
                
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WS Error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


# =========================================================================
# FUSION MODE — dual-modality YOLO + XGBoost trust classifier
# =========================================================================

from fusion.engine import FusionEngine, FusionResult, draw_fusion_frame, draw_single_frame

# Fusion state
fusion_engine: Optional[FusionEngine] = None
fusion_thread: Optional[threading.Thread] = None
fusion_stop = threading.Event()
fusion_pause = threading.Event()
fusion_queue = queue.Queue(maxsize=4)  # holds (jpeg_bytes, stats) tuples
fusion_streaming = False
fusion_stats = {}

# Fusion config (defaults)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

FUSION_DEFAULTS = {
    "rgb_weights": os.path.join(_PROJECT_ROOT, "RGB model", "Yolo26n_trained", "weights", "best.pt"),
    "ir_weights": os.path.join(_PROJECT_ROOT, "runs", "corrective_finetune", "finetune_v3b", "weights", "best.pt"),
    "fusion_model": os.path.join(_PROJECT_ROOT, "classifier", "runs", "reliability", "fusion", "fusion_no_fn_model.joblib"),
    "rgb_patch_weights": os.path.join(_PROJECT_ROOT, "classifier", "runs", "patches", "confuser_filter4_rgb.pt"),
    "ir_patch_weights": os.path.join(_PROJECT_ROOT, "classifier", "runs", "patches", "confuser_filter4_ir.pt"),
    "use_patch_verifier": True,  # Confuser filter: rejects airplanes/helicopters/birds
    "patch_threshold": 0.70,
    "rgb_conf": 0.25,
    "ir_conf": 0.40,
    "nms_iou": 0.45,
    # Grayscale test mode toggles
    "grayscale_run_ir_filter": True,
    "grayscale_disable_filter_ood": True,
    "imgsz": 640,
    "device": 0,
    # Temporal settings (matching detection.py defaults)
    "infer_fps": 5,
    "warning_window_frames": 10,
    "warning_require_hits": 9,
    "alert_window_frames": 10,
    "alert_require_hits": 9,
    "alert_avg_conf_threshold": 0.0,
    "warning_cooldown_s": 3.0,
    "alert_cooldown_s": 3.0,
    "roi_ttl": 5,
    "roi_expand": 1.5,
    "show_troi": True,
    "show_gate": True,
    "show_source_tags": True,
    "simple_mode": False,
}

fusion_config = dict(FUSION_DEFAULTS)

# Fusion request models
class FusionSourcePaired(BaseModel):
    rgb_path: str
    ir_path: str

class FusionSourceSingle(BaseModel):
    path: str

class FusionConfigUpdate(BaseModel):
    rgb_weights: Optional[str] = None
    ir_weights: Optional[str] = None
    fusion_model: Optional[str] = None
    rgb_patch_weights: Optional[str] = None
    ir_patch_weights: Optional[str] = None
    use_patch_verifier: Optional[bool] = None
    patch_threshold: Optional[float] = None
    rgb_conf: Optional[float] = None
    ir_conf: Optional[float] = None
    grayscale_run_ir_filter: Optional[bool] = None
    grayscale_disable_filter_ood: Optional[bool] = None
    nms_iou: Optional[float] = None
    imgsz: Optional[int] = None
    device: Optional[int] = None
    playback_speed: Optional[float] = None
    save_video: Optional[bool] = None
    # Temporal settings
    infer_fps: Optional[int] = None
    warning_window_frames: Optional[int] = None
    warning_require_hits: Optional[int] = None
    alert_window_frames: Optional[int] = None
    alert_require_hits: Optional[int] = None
    alert_avg_conf_threshold: Optional[float] = None
    warning_cooldown_s: Optional[float] = None
    alert_cooldown_s: Optional[float] = None
    roi_ttl: Optional[int] = None
    roi_expand: Optional[float] = None
    show_troi: Optional[bool] = None
    show_gate: Optional[bool] = None
    show_source_tags: Optional[bool] = None
    simple_mode: Optional[bool] = None


# Track current fusion source
fusion_mode = None  # "single", "paired", "grayscale"
fusion_rgb_path = None
fusion_ir_path = None
fusion_modality = "rgb"  # for single mode
fusion_playback_speed = 1.0
fusion_save_video = False
_fusion_skip_frames = 0  # set by /api/fusion/control/skip, consumed by thread


def _ensure_fusion_engine():
    """Lazily load the fusion engine with current config."""
    global fusion_engine
    if fusion_engine is not None:
        return
    fusion_engine = FusionEngine(
        rgb_weights=fusion_config["rgb_weights"],
        ir_weights=fusion_config["ir_weights"],
        fusion_model_path=fusion_config["fusion_model"],
        rgb_conf=fusion_config["rgb_conf"],
        ir_conf=fusion_config["ir_conf"],
        grayscale_run_ir_filter=bool(fusion_config.get("grayscale_run_ir_filter", True)),
        nms_iou=fusion_config["nms_iou"],
        imgsz=fusion_config["imgsz"],
        device=fusion_config["device"],
        rgb_patch_weights=fusion_config.get("rgb_patch_weights"),
        ir_patch_weights=fusion_config.get("ir_patch_weights"),
        patch_threshold=float(fusion_config.get("patch_threshold", 0.70)),
        use_patch_verifier=bool(fusion_config.get("use_patch_verifier", True)),
        cascade_order=str(fusion_config.get("cascade_order", "filter_then_classifier")),
    )


def _fusion_thread_fn():
    """Main loop for fusion detection thread — with temporal logic matching fusion_app.py."""
    global fusion_streaming, fusion_stats

    from fusion.temporal import (
        PerModalityTemporalState, draw_detections, draw_temporal_overlays,
        overlay_text_big, build_overlay_lines,
    )

    TRUST_LABELS = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}

    from fusion.engine import Detection
    def _wrap_dets(dets):
        return [Detection(box=(d[0], d[1], d[2], d[3]), conf=d[4]) for d in dets]

    def _verifier_overlay_lines(orig_trust, new_trust, rgb_probs, ir_probs,
                                ir_skipped_gray, n_rgb_dets, n_ir_dets, thr,
                                rgb_is_grayscale=False):
        """Diagnostic lines for video overlay: classifier-vs-confuser-filter."""
        lines = []
        if orig_trust != new_trust:
            lines.append(
                f"CONFUSER VETO: {TRUST_LABELS[orig_trust]} -> "
                f"{TRUST_LABELS[new_trust]}"
            )
        else:
            lines.append(f"CONFUSER FILTER: pass (thr={thr:.2f})")

        # Try to get per-box class labels from verifiers
        rgb_labels = []
        ir_labels = []
        if fusion_engine is not None:
            if fusion_engine.rgb_verifier is not None:
                rgb_labels = list(fusion_engine.rgb_verifier.last_labels)
            if fusion_engine.ir_verifier is not None:
                ir_labels = list(fusion_engine.ir_verifier.last_labels)

        if orig_trust not in (1, 3):
            lines.append("  RGB filter: classifier did not trust")
        elif rgb_is_grayscale:
            lines.append("  RGB filter: skipped (input is grayscale)")
        elif n_rgb_dets == 0:
            lines.append("  RGB filter: no detections")
        elif not rgb_probs:
            lines.append("  RGB filter: off / not loaded")
        else:
            mx = max(rgb_probs)
            tag = "VETO" if mx >= thr else "OK"
            lbl_str = ""
            if rgb_labels:
                lbl_str = " [" + ", ".join(rgb_labels[:3]) + "]"
            lines.append(
                f"  RGB: P(conf)={mx:.2f} "
                f"{len(rgb_probs)}box {tag}{lbl_str}"
            )
        if orig_trust not in (2, 3):
            lines.append("  IR filter: classifier did not trust")
        elif ir_skipped_gray:
            lines.append("  IR filter: skipped (grayscale, OOD)")
        elif n_ir_dets == 0:
            lines.append("  IR filter: no detections")
        elif not ir_probs:
            lines.append("  IR filter: off / not loaded")
        else:
            mx = max(ir_probs)
            tag = "VETO" if mx >= thr else "OK"
            lbl_str = ""
            if ir_labels:
                lbl_str = " [" + ", ".join(ir_labels[:3]) + "]"
            lines.append(
                f"  IR: P(conf)={mx:.2f} "
                f"{len(ir_probs)}box {tag}{lbl_str}"
            )
        return lines

    # Read temporal settings from fusion_config (editable via /api/fusion/config)
    temporal_settings = {
        "infer_fps": fusion_config.get("infer_fps", 5),
        "warning_window_frames": fusion_config.get("warning_window_frames", 8),
        "warning_require_hits": fusion_config.get("warning_require_hits", 5),
        "alert_window_frames": fusion_config.get("alert_window_frames", 10),
        "alert_require_hits": fusion_config.get("alert_require_hits", 9),
        "alert_avg_conf_threshold": fusion_config.get("alert_avg_conf_threshold", 0.3),
        "warning_cooldown_s": fusion_config.get("warning_cooldown_s", 3.0),
        "alert_cooldown_s": fusion_config.get("alert_cooldown_s", 3.0),
        "roi_ttl": fusion_config.get("roi_ttl", 5),
        "roi_expand": fusion_config.get("roi_expand", 1.5),
        "show_troi": fusion_config.get("show_troi", True),
        "show_gate": fusion_config.get("show_gate", True),
        "show_source_tags": fusion_config.get("show_source_tags", True),
    }

    try:
        _ensure_fusion_engine()
    except Exception as e:
        print(f"[Fusion] Engine init failed: {e}")
        fusion_streaming = False
        return

    rgb_cap = None
    ir_cap = None

    try:
        if fusion_mode == "paired":
            rgb_cap = cv2.VideoCapture(fusion_rgb_path)
            ir_cap = cv2.VideoCapture(fusion_ir_path)
            if not rgb_cap.isOpened() or not ir_cap.isOpened():
                print(f"[Fusion] Failed to open video(s)")
                return
            total = min(
                int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            fps = rgb_cap.get(cv2.CAP_PROP_FPS) or 30.0
        elif fusion_mode in ("single", "grayscale"):
            rgb_cap = cv2.VideoCapture(fusion_rgb_path)
            if not rgb_cap.isOpened():
                print(f"[Fusion] Failed to open video: {fusion_rgb_path}")
                return
            total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = rgb_cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            print(f"[Fusion] Unknown mode: {fusion_mode}")
            return

        # Stride + temporal state
        infer_fps = temporal_settings["infer_fps"]
        stride = max(1, int(round(fps / infer_fps)))
        warn_cd_infer = int(round(temporal_settings["warning_cooldown_s"] * infer_fps))
        alert_cd_infer = int(round(temporal_settings["alert_cooldown_s"] * infer_fps))

        ts_kwargs = dict(
            stride=stride,
            warning_window=temporal_settings["warning_window_frames"],
            warning_require=temporal_settings["warning_require_hits"],
            alert_window=temporal_settings["alert_window_frames"],
            alert_require=temporal_settings["alert_require_hits"],
            alert_avg_conf_thresh=temporal_settings["alert_avg_conf_threshold"],
            warning_cooldown_frames=warn_cd_infer,
            alert_cooldown_frames=alert_cd_infer,
            roi_ttl=temporal_settings["roi_ttl"],
            roi_expand=temporal_settings["roi_expand"],
        )
        rgb_temporal = PerModalityTemporalState(**ts_kwargs)
        ir_temporal = PerModalityTemporalState(**ts_kwargs)

        frame_id = 0
        frame_period = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # Video writer
        writer = None
        if fusion_save_video:
            out_dir = os.path.join(_THIS_DIR, "demo_outputs")
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"fusion_{fusion_mode}_{int(time.time())}.mp4"
            out_path = os.path.join(out_dir, out_name)

        def _run_yolo(model, frame, conf):
            """Run YOLO, return [[x1,y1,x2,y2,conf], ...]."""
            results = model.predict(
                frame, conf=conf, iou=fusion_engine.nms_iou,
                imgsz=fusion_engine.imgsz, verbose=False,
                device=fusion_engine.device,
            )[0]
            dets = []
            if results.boxes is not None and len(results.boxes) > 0:
                for i in range(len(results.boxes)):
                    x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy()
                    c = float(results.boxes.conf[i])
                    dets.append([float(x1), float(y1), float(x2), float(y2), c])
            return dets

        def _run_with_roi(model, frame, conf, temporal):
            """YOLO + TROI recovery. Returns (dets, sources, troi_rois)."""
            dets = _run_yolo(model, frame, conf)
            sources = ["full"] * len(dets)
            troi_rois = []
            if temporal is not None and len(dets) == 0:
                if temporal.last_roi is not None and temporal.roi_age > 0:
                    h, w = frame.shape[:2]
                    roi_result = temporal.get_roi_crop(frame, w, h)
                    if roi_result is not None:
                        crop, (ox, oy) = roi_result
                        troi_rois.append((ox, oy, ox + crop.shape[1], oy + crop.shape[0]))
                        crop_dets = _run_yolo(model, crop, conf * 0.8)
                        if crop_dets:
                            dets = PerModalityTemporalState.remap_dets(crop_dets, (ox, oy))
                            sources = ["troi"] * len(dets)
            return dets, sources, troi_rois

        next_deadline = time.perf_counter()
        frame_period = 1.0 / fps if fps > 0 else 1.0 / 30.0
        WS_MAX_W = 1280
        last_probs = [0, 0, 0, 1]
        last_elapsed_ms = 0.0

        def _compose_hold(rgb_frame, ir_frame):
            """Hold frame: draw cached detections on fresh video (matches fusion_app.py)."""
            show_tags = temporal_settings.get("show_source_tags", True)
            trust = rgb_temporal.last_trust if rgb_temporal.last_trust is not None else 0

            if fusion_mode == "paired":
                left = rgb_frame.copy()
                right = ir_frame.copy() if ir_frame is not None else rgb_frame.copy()
                if len(right.shape) == 2:
                    right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            elif fusion_mode == "grayscale":
                left = rgb_frame.copy()
                gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                right = cv2.merge([gray, gray, gray])
            else:  # single
                vis = rgb_frame.copy()
                if rgb_temporal.last_dets:
                    draw_detections(vis, rgb_temporal.last_dets, (0, 255, 255),
                                    sources=rgb_temporal.last_dets_sources,
                                    show_source_tags=show_tags)
                    draw_temporal_overlays(vis, rgb_temporal, temporal_settings)
                    lines = build_overlay_lines(rgb_temporal, temporal_settings)
                    overlay_text_big(vis, lines)
                return vis

            rgb_trusted = [trust in (1, 3)] * len(rgb_temporal.last_dets)
            ir_trusted = [trust in (2, 3)] * len(ir_temporal.last_dets)

            if rgb_temporal.last_dets:
                draw_detections(left, rgb_temporal.last_dets, (0, 255, 0), "RGB ",
                                rgb_trusted, rgb_temporal.last_dets_sources, show_tags)
                draw_temporal_overlays(left, rgb_temporal, temporal_settings)
            if ir_temporal.last_dets:
                draw_detections(right, ir_temporal.last_dets, (255, 200, 0), "IR ",
                                ir_trusted, ir_temporal.last_dets_sources, show_tags)
                draw_temporal_overlays(right, ir_temporal, temporal_settings)

            lh, lw = left.shape[:2]
            rh, rw = right.shape[:2]
            if rh != lh:
                right = cv2.resize(right, (int(rw * lh / rh), lh))

            vis = np.hstack([left, right])
            trust_label = TRUST_LABELS.get(trust, "?")
            trust_prob = rgb_temporal.last_trust_prob or 0.0
            tag = " [grayscale]" if fusion_mode == "grayscale" else ""
            lines = [f"FUSION: {trust_label} ({trust_prob*100:.1f}%){tag}"]
            # Confuser gate line — always visible (matches inference frame)
            suppressed = rgb_temporal.confuser_suppressed or ir_temporal.confuser_suppressed
            if suppressed:
                lines.append("CONFUSER GATE: alert suppressed")
            else:
                lines.append("CONFUSER GATE: no alert suppressed")
            lines += build_overlay_lines(rgb_temporal, temporal_settings, prefix="RGB ")
            lines += build_overlay_lines(ir_temporal, temporal_settings, prefix="IR  ")
            overlay_text_big(vis, lines)
            return vis

        while not fusion_stop.is_set():
            # Pause
            while fusion_pause.is_set() and not fusion_stop.is_set():
                time.sleep(0.05)
                next_deadline = time.perf_counter()

            if fusion_stop.is_set():
                break

            # Skip forward if requested via API
            global _fusion_skip_frames
            if _fusion_skip_frames > 0:
                skip = _fusion_skip_frames
                _fusion_skip_frames = 0
                current_pos = int(rgb_cap.get(cv2.CAP_PROP_POS_FRAMES))
                target = min(current_pos + skip, total - 1)
                rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                if fusion_mode == "paired" and ir_cap:
                    ir_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                frame_id = target
                next_deadline = time.perf_counter()
                print(f"[Fusion] Skipped to frame {target}/{total}")

            # Read ALL frames (hold + inference) — needed for smooth video
            ret_rgb, rgb_frame = rgb_cap.read()
            if not ret_rgb:
                break

            ir_frame = None
            if fusion_mode == "paired":
                ret_ir, ir_frame = ir_cap.read()
                if not ret_ir:
                    break

            frame_id += 1
            is_infer = (frame_id % stride == 0) or stride == 1

            if is_infer:
                t0 = time.perf_counter()

                # === INFERENCE FRAME ===
                if fusion_mode == "paired":
                    rgb_dets, rgb_src, rgb_troi = _run_with_roi(
                        fusion_engine.rgb_model, rgb_frame, fusion_engine.rgb_conf, rgb_temporal)
                    ir_dets, ir_src, ir_troi = _run_with_roi(
                        fusion_engine.ir_model, ir_frame, fusion_engine.ir_conf, ir_temporal)

                    rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                    ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY) if len(ir_frame.shape) == 3 else ir_frame
                    rgb_wrapped = _wrap_dets(rgb_dets)
                    ir_wrapped = _wrap_dets(ir_dets)
                    feats = fusion_engine.extract_features(
                        rgb_wrapped, ir_wrapped, rgb_gray, ir_gray)
                    trust, probs = fusion_engine.classify(feats)
                    orig_trust = trust
                    rgp, irp, _veto = [], [], False
                    rgb_is_gray = False
                    verifier_active = fusion_config.get("use_patch_verifier", True)
                    if verifier_active:
                        ir_bgr_for_verif = ir_frame if len(ir_frame.shape) == 3 else cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                        trust, rgp, irp, _veto = fusion_engine.patch_veto(
                            trust, rgb_wrapped, ir_wrapped, rgb_frame, ir_bgr_for_verif,
                            ir_is_real_thermal=True)
                        rgb_is_gray = fusion_engine.is_effectively_grayscale(rgb_frame)
                    trust_prob = float(probs[orig_trust])

                    # Store verifier diagnostics for /fusion/status
                    rgb_temporal.last_verifier = {
                        "active": verifier_active,
                        "vetoed": bool(_veto),
                        "threshold": float(fusion_config.get("patch_threshold", 0.70)),
                        "original_trust": int(orig_trust),
                        "original_trust_name": TRUST_LABELS.get(orig_trust, "?"),
                        "rgb_max_p": float(max(rgp)) if rgp else None,
                        "ir_max_p": float(max(irp)) if irp else None,
                        "rgb_n_boxes": len(rgp),
                        "ir_n_boxes": len(irp),
                        "rgb_labels": list(fusion_engine.rgb_verifier.last_labels) if fusion_engine.rgb_verifier and rgp else [],
                        "ir_labels": list(fusion_engine.ir_verifier.last_labels) if fusion_engine.ir_verifier and irp else [],
                        "rgb_skipped_reason": (
                            "classifier_untrusted" if orig_trust not in (1, 3)
                            else "grayscale_input" if rgb_is_gray
                            else "no_detections" if len(rgb_dets) == 0
                            else "verifier_off" if not rgp
                            else None
                        ),
                        "ir_skipped_reason": (
                            "classifier_untrusted" if orig_trust not in (2, 3)
                            else "no_detections" if len(ir_dets) == 0
                            else "verifier_off" if not irp
                            else None
                        ),
                    }

                    rgb_trusted = [trust in (1, 3)] * len(rgb_dets)
                    ir_trusted = [trust in (2, 3)] * len(ir_dets)
                    show_tags = temporal_settings["show_source_tags"]

                    left = rgb_frame.copy()
                    right = ir_frame.copy()
                    if len(right.shape) == 2:
                        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

                    draw_detections(left, rgb_dets, (0, 255, 0), "RGB ", rgb_trusted, rgb_src, show_tags)
                    draw_detections(right, ir_dets, (255, 200, 0), "IR ", ir_trusted, ir_src, show_tags)

                    lh, lw = left.shape[:2]
                    rh, rw = right.shape[:2]

                    fusion_simple = fusion_config.get("simple_mode", False)

                    # Gate temporal window by classifier trust (skip in simple mode)
                    # NOTE: uses orig_trust (pre-confuser) — confuser filter
                    # operates as alert-gate, not per-frame veto.
                    patch_thr = float(fusion_config.get("patch_threshold", 0.70))
                    if not fusion_simple:
                        # Shared confuser feed: paired/grayscale modes image
                        # the same physical scene, so whichever filter scored
                        # high informs both alert chains. The stronger filter
                        # (typically IR for helicopters) covers for the weaker.
                        rgb_max_p = float(max(rgp)) if rgp else None
                        ir_max_p = float(max(irp)) if irp else None
                        shared_p = (max(p for p in (rgb_max_p, ir_max_p)
                                        if p is not None)
                                    if (rgb_max_p is not None
                                        or ir_max_p is not None)
                                    else None)
                        rgb_temporal.add_confuser_prob(shared_p)
                        ir_temporal.add_confuser_prob(shared_p)
                        rgb_temporal.update(
                            rgb_dets if orig_trust in (1, 3) else [],
                            lw, lh,
                            confuser_threshold=patch_thr if verifier_active else None)
                        ir_temporal.update(
                            ir_dets if orig_trust in (2, 3) else [],
                            rw, rh,
                            confuser_threshold=patch_thr if verifier_active else None)
                    rgb_temporal.last_dets = list(rgb_dets)
                    rgb_temporal.last_dets_sources = list(rgb_src)
                    rgb_temporal.last_troi_rois = list(rgb_troi)
                    ir_temporal.last_dets = list(ir_dets)
                    ir_temporal.last_dets_sources = list(ir_src)
                    ir_temporal.last_troi_rois = list(ir_troi)
                    rgb_temporal.last_trust = orig_trust
                    rgb_temporal.last_trust_prob = trust_prob

                    temporal_settings["show_troi"] = fusion_config.get("show_troi", True)
                    temporal_settings["show_gate"] = fusion_config.get("show_gate", True)
                    temporal_settings["show_source_tags"] = fusion_config.get("show_source_tags", True)

                    if not fusion_simple:
                        draw_temporal_overlays(left, rgb_temporal, temporal_settings)
                        draw_temporal_overlays(right, ir_temporal, temporal_settings)

                    if rh != lh:
                        right = cv2.resize(right, (int(rw * lh / rh), lh))

                    vis = np.hstack([left, right])
                    trust_label = TRUST_LABELS[orig_trust]
                    lines = [f"FUSION: {trust_label} ({trust_prob*100:.1f}%)"]
                    # Confuser gate line — always visible
                    suppressed = rgb_temporal.confuser_suppressed or ir_temporal.confuser_suppressed
                    if suppressed:
                        # Build label info from last verifier run
                        rgb_lbl = ''
                        ir_lbl = ''
                        if fusion_engine.rgb_verifier and hasattr(fusion_engine.rgb_verifier, 'last_labels') and fusion_engine.rgb_verifier.last_labels:
                            rgb_lbl = fusion_engine.rgb_verifier.last_labels[0]
                        if fusion_engine.ir_verifier and hasattr(fusion_engine.ir_verifier, 'last_labels') and fusion_engine.ir_verifier.last_labels:
                            ir_lbl = fusion_engine.ir_verifier.last_labels[0]
                        parts = []
                        if rgb_lbl:
                            parts.append(f"RGB:{rgb_lbl}")
                        if ir_lbl:
                            parts.append(f"IR:{ir_lbl}")
                        detail = ' '.join(parts) if parts else ''
                        lines.append(f"CONFUSER GATE: alert suppressed ({detail})")
                    else:
                        lines.append("CONFUSER GATE: no alert suppressed")
                    # OLD per-frame verifier overlay — commented out for alert-gate architecture
                    # if verifier_active:
                    #     lines += _verifier_overlay_lines(
                    #         orig_trust, trust, rgp, irp,
                    #         ir_skipped_gray=False,
                    #         n_rgb_dets=len(rgb_dets), n_ir_dets=len(ir_dets),
                    #         thr=float(fusion_config.get("patch_threshold", 0.70)),
                    #         rgb_is_grayscale=rgb_is_gray)
                    if not fusion_simple:
                        lines += build_overlay_lines(rgb_temporal, temporal_settings, prefix="RGB ")
                        lines += build_overlay_lines(ir_temporal, temporal_settings, prefix="IR  ")
                    overlay_text_big(vis, lines)

                elif fusion_mode == "grayscale":
                    rgb_dets, rgb_src, rgb_troi = _run_with_roi(
                        fusion_engine.rgb_model, rgb_frame, fusion_engine.rgb_conf, rgb_temporal)
                    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                    gray_3ch = cv2.merge([gray, gray, gray])
                    # Grayscale uses the SHARED ir_conf (same as paired); the
                    # standalone ir_conf_grayscale knob is gone.
                    ir_dets, ir_src, ir_troi = _run_with_roi(
                        fusion_engine.ir_model, gray_3ch, fusion_engine.ir_conf, ir_temporal)

                    rgb_wrapped = _wrap_dets(rgb_dets)
                    ir_wrapped = _wrap_dets(ir_dets)
                    feats = fusion_engine.extract_features(
                        rgb_wrapped, ir_wrapped, gray, gray)
                    trust, probs = fusion_engine.classify(feats)
                    orig_trust = trust
                    rgp, irp, _veto = [], [], False
                    verifier_active = fusion_config.get("use_patch_verifier", True)
                    if verifier_active:
                        # Grayscale test mode: run IR filter on gray-replicate
                        # input per config; OOD calibration is thermal-only so
                        # the OOD gate is skipped to let the filter actually
                        # veto.
                        gs_run_ir = bool(fusion_config.get(
                            "grayscale_run_ir_filter", True))
                        gs_skip_ood = bool(fusion_config.get(
                            "grayscale_disable_filter_ood", True))
                        trust, rgp, irp, _veto = fusion_engine.patch_veto(
                            trust, rgb_wrapped, ir_wrapped, rgb_frame, gray_3ch,
                            ir_is_real_thermal=False,
                            ir_verifier_enabled=gs_run_ir,
                            skip_ir_ood_gate=gs_skip_ood)
                    trust_prob = float(probs[orig_trust])

                    # Store verifier diagnostics for /fusion/status
                    rgb_is_gray_src = fusion_engine.is_effectively_grayscale(rgb_frame)
                    rgb_temporal.last_verifier = {
                        "active": verifier_active,
                        "vetoed": bool(_veto),
                        "threshold": float(fusion_config.get("patch_threshold", 0.70)),
                        "original_trust": int(orig_trust),
                        "original_trust_name": TRUST_LABELS.get(orig_trust, "?"),
                        "rgb_max_p": float(max(rgp)) if rgp else None,
                        "ir_max_p": float(max(irp)) if irp else None,
                        "rgb_n_boxes": len(rgp),
                        "ir_n_boxes": len(irp),
                        "rgb_labels": list(fusion_engine.rgb_verifier.last_labels) if fusion_engine.rgb_verifier and rgp else [],
                        "ir_labels": list(fusion_engine.ir_verifier.last_labels) if fusion_engine.ir_verifier and irp else [],
                        "rgb_skipped_reason": (
                            "classifier_untrusted" if orig_trust not in (1, 3)
                            else "grayscale_input" if rgb_is_gray_src
                            else "no_detections" if len(rgb_dets) == 0
                            else "verifier_off" if not rgp
                            else None
                        ),
                        "ir_skipped_reason": (
                            "classifier_untrusted" if orig_trust not in (2, 3)
                            else "no_detections" if len(ir_dets) == 0
                            else "verifier_off" if not irp
                            else None
                        ),
                    }

                    rgb_trusted = [trust in (1, 3)] * len(rgb_dets)
                    ir_trusted = [trust in (2, 3)] * len(ir_dets)
                    show_tags = temporal_settings["show_source_tags"]

                    left = rgb_frame.copy()
                    right = gray_3ch.copy()
                    draw_detections(left, rgb_dets, (0, 255, 0), "RGB ", rgb_trusted, rgb_src, show_tags)
                    draw_detections(right, ir_dets, (255, 200, 0), "IR ", ir_trusted, ir_src, show_tags)

                    lh, lw = left.shape[:2]
                    rh, rw = right.shape[:2]

                    fusion_simple = fusion_config.get("simple_mode", False)

                    # Gate temporal window by classifier trust (skip in simple mode)
                    # NOTE: uses orig_trust (pre-confuser) — confuser filter
                    # operates as alert-gate, not per-frame veto.
                    patch_thr = float(fusion_config.get("patch_threshold", 0.70))
                    if not fusion_simple:
                        # Shared confuser feed: paired/grayscale modes image
                        # the same physical scene, so whichever filter scored
                        # high informs both alert chains. The stronger filter
                        # (typically IR for helicopters) covers for the weaker.
                        rgb_max_p = float(max(rgp)) if rgp else None
                        ir_max_p = float(max(irp)) if irp else None
                        shared_p = (max(p for p in (rgb_max_p, ir_max_p)
                                        if p is not None)
                                    if (rgb_max_p is not None
                                        or ir_max_p is not None)
                                    else None)
                        rgb_temporal.add_confuser_prob(shared_p)
                        ir_temporal.add_confuser_prob(shared_p)
                        rgb_temporal.update(
                            rgb_dets if orig_trust in (1, 3) else [],
                            lw, lh,
                            confuser_threshold=patch_thr if verifier_active else None)
                        ir_temporal.update(
                            ir_dets if orig_trust in (2, 3) else [],
                            rw, rh,
                            confuser_threshold=patch_thr if verifier_active else None)
                    rgb_temporal.last_dets = list(rgb_dets)
                    rgb_temporal.last_dets_sources = list(rgb_src)
                    rgb_temporal.last_troi_rois = list(rgb_troi)
                    ir_temporal.last_dets = list(ir_dets)
                    ir_temporal.last_dets_sources = list(ir_src)
                    ir_temporal.last_troi_rois = list(ir_troi)
                    rgb_temporal.last_trust = orig_trust
                    rgb_temporal.last_trust_prob = trust_prob

                    temporal_settings["show_troi"] = fusion_config.get("show_troi", True)
                    temporal_settings["show_gate"] = fusion_config.get("show_gate", True)
                    temporal_settings["show_source_tags"] = fusion_config.get("show_source_tags", True)

                    if not fusion_simple:
                        draw_temporal_overlays(left, rgb_temporal, temporal_settings)
                        draw_temporal_overlays(right, ir_temporal, temporal_settings)

                    if rh != lh:
                        right = cv2.resize(right, (int(rw * lh / rh), lh))

                    vis = np.hstack([left, right])
                    trust_label = TRUST_LABELS[orig_trust]
                    lines = [f"FUSION: {trust_label} ({trust_prob*100:.1f}%) [grayscale]"]
                    # Confuser gate line — always visible
                    suppressed = rgb_temporal.confuser_suppressed or ir_temporal.confuser_suppressed
                    if suppressed:
                        rgb_lbl = ''
                        ir_lbl = ''
                        if fusion_engine.rgb_verifier and hasattr(fusion_engine.rgb_verifier, 'last_labels') and fusion_engine.rgb_verifier.last_labels:
                            rgb_lbl = fusion_engine.rgb_verifier.last_labels[0]
                        if fusion_engine.ir_verifier and hasattr(fusion_engine.ir_verifier, 'last_labels') and fusion_engine.ir_verifier.last_labels:
                            ir_lbl = fusion_engine.ir_verifier.last_labels[0]
                        parts = []
                        if rgb_lbl:
                            parts.append(f"RGB:{rgb_lbl}")
                        if ir_lbl:
                            parts.append(f"IR:{ir_lbl}")
                        detail = ' '.join(parts) if parts else ''
                        lines.append(f"CONFUSER GATE: alert suppressed ({detail})")
                    else:
                        lines.append("CONFUSER GATE: no alert suppressed")
                    # OLD per-frame verifier overlay — commented out for alert-gate architecture
                    # if verifier_active:
                    #     lines += _verifier_overlay_lines(
                    #         orig_trust, trust, rgp, irp,
                    #         ir_skipped_gray=False,
                    #         n_rgb_dets=len(rgb_dets), n_ir_dets=len(ir_dets),
                    #         thr=float(fusion_config.get("patch_threshold", 0.70)),
                    #         rgb_is_grayscale=fusion_engine._is_effectively_grayscale(rgb_frame))
                    if not fusion_simple:
                        lines += build_overlay_lines(rgb_temporal, temporal_settings, prefix="RGB ")
                        lines += build_overlay_lines(ir_temporal, temporal_settings, prefix="IR  ")
                    overlay_text_big(vis, lines)

                else:  # single
                    rgb_dets, rgb_src, rgb_troi = _run_with_roi(
                        fusion_engine.rgb_model, rgb_frame, fusion_engine.rgb_conf, rgb_temporal)

                    vis = rgb_frame.copy()
                    draw_detections(vis, rgb_dets, (0, 255, 255), sources=rgb_src,
                                    show_source_tags=temporal_settings["show_source_tags"])
                    lh, lw = vis.shape[:2]
                    rgb_temporal.update(rgb_dets, lw, lh)
                    rgb_temporal.last_dets = list(rgb_dets)
                    rgb_temporal.last_dets_sources = list(rgb_src)
                    rgb_temporal.last_troi_rois = list(rgb_troi)
                    draw_temporal_overlays(vis, rgb_temporal, temporal_settings)
                    lines = build_overlay_lines(rgb_temporal, temporal_settings)
                    overlay_text_big(vis, lines)
                    trust = 3 if rgb_dets else 0
                    probs = [0, 0, 0, 1]
                    trust_prob = 1.0

                last_elapsed_ms = (time.perf_counter() - t0) * 1000
                last_probs = probs
                if frame_id % (stride * 10) == 0:
                    print(f"[Fusion] frame={frame_id}/{total}  infer={last_elapsed_ms:.0f}ms  stride={stride}")

            else:
                # === HOLD FRAME: draw cached detections on fresh video ===
                vis = _compose_hold(rgb_frame, ir_frame)

            # Video writer (full-res, before downscale)
            if fusion_save_video and writer is not None:
                writer.write(vis)
            elif fusion_save_video and writer is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                print(f"[Fusion] Saving video: {out_path}")

            # Build stats (updated every frame for status endpoint)
            fusion_simple = fusion_config.get("simple_mode", False)
            warn_active = False if fusion_simple else (rgb_temporal.warning_active or ir_temporal.warning_active)
            alert_active = False if fusion_simple else (rgb_temporal.alert_active or ir_temporal.alert_active)
            stats = {
                "frame_id": frame_id,
                "total_frames": total,
                "video_fps": fps,
                "trust_label": rgb_temporal.last_trust if rgb_temporal.last_trust is not None else 0,
                "trust_name": TRUST_LABELS.get(rgb_temporal.last_trust if rgb_temporal.last_trust is not None else 0, "?"),
                "trust_probs": last_probs if fusion_mode != "single" else [0, 0, 0, 1],
                "rgb_dets": len(rgb_temporal.last_dets),
                "ir_dets": len(ir_temporal.last_dets),
                "infer_ms": round(last_elapsed_ms, 1),
                "mode": fusion_mode,
                "playback_speed": fusion_playback_speed,
                "warning_active": warn_active,
                "alert_active": alert_active,
                "warning_events": rgb_temporal.count_warning_events + ir_temporal.count_warning_events,
                "alert_events": rgb_temporal.count_alert_events + ir_temporal.count_alert_events,
                "confuser_suppressed": (rgb_temporal.confuser_suppressed or ir_temporal.confuser_suppressed),
                "confuser_labels": {
                    "rgb": list(fusion_engine.rgb_verifier.last_labels)[:1] if (
                        fusion_engine.rgb_verifier and hasattr(fusion_engine.rgb_verifier, 'last_labels')
                        and fusion_engine.rgb_verifier.last_labels) else [],
                    "ir": list(fusion_engine.ir_verifier.last_labels)[:1] if (
                        fusion_engine.ir_verifier and hasattr(fusion_engine.ir_verifier, 'last_labels')
                        and fusion_engine.ir_verifier.last_labels) else [],
                },
                "verifier": getattr(rgb_temporal, "last_verifier", None),
            }
            fusion_stats = stats

            # Downscale + JPEG encode in-thread, push bytes to queue for MJPEG
            vh, vw = vis.shape[:2]
            if vw > WS_MAX_W:
                vis = cv2.resize(vis, (WS_MAX_W, int(vh * WS_MAX_W / vw)))
            _, jpeg_buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_bytes = jpeg_buf.tobytes()
            try:
                fusion_queue.put_nowait((jpeg_bytes, stats))
            except queue.Full:
                try:
                    fusion_queue.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
                try:
                    fusion_queue.put_nowait((jpeg_bytes, stats))
                except queue.Full:
                    pass

            # Pace to source video FPS
            now = time.perf_counter()
            speed = fusion_playback_speed
            if speed > 0:
                adjusted_period = frame_period / speed
                next_deadline += adjusted_period
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                else:
                    next_deadline = now
            else:
                # Max speed: no sleep
                next_deadline = time.perf_counter()

    except Exception as e:
        print(f"[Fusion] Thread error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rgb_cap:
            rgb_cap.release()
        if ir_cap:
            ir_cap.release()
        if writer is not None:
            writer.release()
            print(f"[Fusion] Video saved")
        fusion_streaming = False
        print("[Fusion] Thread finished")


# --- Fusion API endpoints ---

@app.get("/api/fusion/config")
async def get_fusion_config():
    return fusion_config


@app.post("/api/fusion/config")
async def update_fusion_config(cfg: FusionConfigUpdate):
    global fusion_engine, fusion_playback_speed, fusion_save_video
    changed = False
    raw = cfg.dict(exclude_unset=True)

    if "playback_speed" in raw:
        fusion_playback_speed = raw.pop("playback_speed")
    if "save_video" in raw:
        fusion_save_video = raw.pop("save_video")

    # Keys that can change live without reloading the fusion engine
    LIVE_KEYS = {
        "simple_mode", "use_patch_verifier", "patch_threshold",
        "show_troi", "show_gate", "show_source_tags",
        "infer_fps", "warning_window_frames", "warning_require_hits",
        "alert_window_frames", "alert_require_hits", "alert_avg_conf_threshold",
        "warning_cooldown_s", "alert_cooldown_s", "roi_ttl", "roi_expand",
    }

    for k, v in raw.items():
        if k in fusion_config and v is not None:
            if fusion_config[k] != v:
                fusion_config[k] = v
                if k not in LIVE_KEYS:
                    changed = True

    if changed:
        fusion_engine = None  # force reload on next start

    return {"success": True, "config": fusion_config}


@app.post("/api/fusion/source/paired")
async def set_fusion_paired(data: FusionSourcePaired):
    global fusion_mode, fusion_rgb_path, fusion_ir_path
    if not os.path.isfile(data.rgb_path):
        raise HTTPException(400, f"RGB file not found: {data.rgb_path}")
    if not os.path.isfile(data.ir_path):
        raise HTTPException(400, f"IR file not found: {data.ir_path}")
    fusion_mode = "paired"
    fusion_rgb_path = data.rgb_path
    fusion_ir_path = data.ir_path

    # Stop any running fusion
    if fusion_thread and fusion_thread.is_alive():
        fusion_stop.set()
        fusion_thread.join(timeout=2.0)

    return {"success": True, "mode": "paired"}


@app.post("/api/fusion/source/single")
async def set_fusion_single(data: FusionSourceSingle):
    global fusion_mode, fusion_rgb_path, fusion_ir_path
    if not os.path.isfile(data.path):
        raise HTTPException(400, f"File not found: {data.path}")
    fusion_mode = "single"
    fusion_rgb_path = data.path
    fusion_ir_path = None

    if fusion_thread and fusion_thread.is_alive():
        fusion_stop.set()
        fusion_thread.join(timeout=2.0)

    return {"success": True, "mode": "single"}


@app.post("/api/fusion/source/grayscale")
async def set_fusion_grayscale(data: FusionSourceSingle):
    global fusion_mode, fusion_rgb_path, fusion_ir_path
    if not os.path.isfile(data.path):
        raise HTTPException(400, f"File not found: {data.path}")
    fusion_mode = "grayscale"
    fusion_rgb_path = data.path
    fusion_ir_path = None

    if fusion_thread and fusion_thread.is_alive():
        fusion_stop.set()
        fusion_thread.join(timeout=2.0)

    return {"success": True, "mode": "grayscale"}


@app.post("/api/fusion/source/youtube")
async def set_fusion_youtube(data: YouTubeSource):
    """Download YouTube video and set as fusion source.

    Mode selection (from request body, defaults to grayscale):
      - "paired":    feed the same downloaded file to both RGB and IR slots
      - "grayscale": feed as RGB source; IR YOLO runs on its grayscale version
      - "single":    feed as RGB-only source
    """
    global fusion_mode, fusion_rgb_path, fusion_ir_path

    try:
        import yt_dlp
        import re as _re

        vid_match = _re.search(r'(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([a-zA-Z0-9_-]{11})', data.url)
        vid_id = vid_match.group(1) if vid_match else "video"

        dl_dir = os.path.join(_THIS_DIR, "demo_outputs")
        os.makedirs(dl_dir, exist_ok=True)
        dl_path = os.path.join(dl_dir, f"yt_{vid_id}.mp4")

        if os.path.exists(dl_path) and os.path.getsize(dl_path) > 100_000:
            print(f"[Fusion/YT] Reusing cached: {dl_path}")
        else:
            if os.path.exists(dl_path):
                os.remove(dl_path)

            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': dl_path,
                'quiet': True,
                'no_warnings': True,
                'overwrites': True,
                'continuedl': False,
                'socket_timeout': 30,
                'retries': 5,
            }

            cookie_file = os.path.join(_THIS_DIR, "youtube_cookies.txt")
            browser_cookie_set = False
            for browser in ['chrome', 'opera', 'edge']:
                try:
                    test_opts = {**ydl_opts, 'cookiesfrombrowser': (browser,), 'extract_flat': True}
                    with yt_dlp.YoutubeDL(test_opts) as test_ydl:
                        test_ydl.extract_info(data.url, download=False)
                    ydl_opts['cookiesfrombrowser'] = (browser,)
                    browser_cookie_set = True
                    break
                except Exception:
                    continue

            if not browser_cookie_set and os.path.isfile(cookie_file):
                ydl_opts['cookiefile'] = cookie_file

            print(f"[Fusion/YT] Downloading {vid_id}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([data.url])
            print(f"[Fusion/YT] Download complete: {dl_path}")

        requested_mode = (data.mode or "grayscale").lower()
        if requested_mode == "paired":
            fusion_mode = "paired"
            fusion_rgb_path = dl_path
            fusion_ir_path = dl_path  # same file both sides; YT has no IR pair
        elif requested_mode == "single":
            fusion_mode = "single"
            fusion_rgb_path = dl_path
            fusion_ir_path = None
        else:
            fusion_mode = "grayscale"
            fusion_rgb_path = dl_path
            fusion_ir_path = None

        if fusion_thread and fusion_thread.is_alive():
            fusion_stop.set()
            fusion_thread.join(timeout=2.0)

        return {"success": True, "mode": fusion_mode, "path": dl_path}
    except ImportError:
        raise HTTPException(400, "yt-dlp not installed")
    except Exception as e:
        raise HTTPException(400, f"YouTube download failed: {str(e)}")


@app.post("/api/fusion/control/start")
async def start_fusion():
    global fusion_thread, fusion_streaming
    if not fusion_mode or not fusion_rgb_path:
        raise HTTPException(400, "No fusion source set")

    if fusion_thread and fusion_thread.is_alive():
        return {"success": True, "status": "already_running"}

    fusion_stop.clear()
    fusion_pause.clear()
    with fusion_queue.mutex:
        fusion_queue.queue.clear()

    fusion_thread = threading.Thread(target=_fusion_thread_fn, daemon=True)
    fusion_thread.start()
    fusion_streaming = True
    return {"success": True}


@app.post("/api/fusion/control/stop")
async def stop_fusion():
    global fusion_streaming
    fusion_stop.set()
    fusion_streaming = False
    return {"success": True}


@app.post("/api/fusion/control/pause")
async def pause_fusion():
    if fusion_pause.is_set():
        fusion_pause.clear()
    else:
        fusion_pause.set()
    return {"success": True, "paused": fusion_pause.is_set()}


@app.post("/api/fusion/control/speed")
async def set_fusion_speed(data: dict):
    global fusion_playback_speed
    speed = float(data.get("speed", 1.0))
    if speed == 0:
        fusion_playback_speed = 0  # Max speed
    else:
        fusion_playback_speed = max(0.25, min(speed, 8.0))
    return {"success": True, "speed": fusion_playback_speed}


@app.post("/api/fusion/control/skip")
async def fusion_skip_forward(data: dict):
    """Skip forward N seconds in fusion video."""
    global fusion_rgb_cap, fusion_ir_cap
    seconds = float(data.get("seconds", 30))
    # The actual seek happens by advancing the caps in the fusion thread
    # We signal via a global
    global _fusion_skip_frames
    fps = 30  # fallback
    try:
        if fusion_stats and fusion_stats.get("video_fps"):
            fps = fusion_stats["video_fps"]
    except Exception:
        pass
    _fusion_skip_frames = int(fps * seconds)
    return {"success": True, "skip_frames": _fusion_skip_frames}


@app.get("/api/fusion/status")
async def get_fusion_status():
    return {
        "mode": fusion_mode,
        "streaming": fusion_streaming and (fusion_thread is not None and fusion_thread.is_alive()),
        "rgb_path": fusion_rgb_path,
        "ir_path": fusion_ir_path,
        "stats": fusion_stats,
    }


# Fusion MJPEG stream — browser handles natively, no JS per-frame processing
async def _mjpeg_generator():
    """Async generator yielding MJPEG multipart frames."""
    while fusion_streaming:
        try:
            jpeg_bytes, stats = fusion_queue.get_nowait()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
    yield b'--frame--\r\n'


@app.get("/api/fusion/mjpeg")
async def fusion_mjpeg_stream():
    """MJPEG video stream — use as <img src="/api/fusion/mjpeg">."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
