"""Test 2: Throttled display — only push to Flet at ~15fps, resize for display."""
import base64
import threading
import time

import cv2
import numpy as np
import flet as ft


def main(page: ft.Page):
    page.title = "Image Test v2 — Throttled"
    page.window.width = 800
    page.window.height = 600

    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB"
        "Nl7BcQAAAABJRU5ErkJggg=="
    )

    status = ft.Text("Waiting...", size=16)
    img = ft.Image(src=pixel, width=640, height=360, fit=ft.BoxFit.CONTAIN)

    page.add(status, img)
    page.update()

    # Shared state: engine writes here, UI reads at its own pace
    _latest_frame = [None]  # mutable container
    _frame_counter = [0]
    _done = [False]
    _lock = threading.Lock()

    def engine_thread():
        """Simulate fast processing — 100+ fps."""
        for i in range(500):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            hue = (i * 2) % 180
            frame[:] = (hue, 200, 200)  # HSV-ish
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            cv2.putText(frame, f"Engine Frame {i}", (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Encode JPEG at reduced quality for display
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

            with _lock:
                _latest_frame[0] = buf.tobytes()
                _frame_counter[0] = i

            time.sleep(0.01)  # ~100 fps engine

        _done[0] = True

    def ui_updater():
        """Pull latest frame at ~15fps and push to Flet."""
        target_fps = 15
        interval = 1.0 / target_fps
        last_pushed = -1

        while not _done[0]:
            with _lock:
                frame_bytes = _latest_frame[0]
                frame_num = _frame_counter[0]

            if frame_bytes is not None and frame_num != last_pushed:
                img.src = frame_bytes
                status.value = f"Displaying engine frame {frame_num}"
                try:
                    page.update()
                except Exception as e:
                    print(f"update error: {e}")
                last_pushed = frame_num

            time.sleep(interval)

        # Final frame
        status.value = f"Done! Last displayed: {last_pushed}"
        try:
            page.update()
        except Exception:
            pass

    threading.Thread(target=engine_thread, daemon=True).start()
    threading.Thread(target=ui_updater, daemon=True).start()


ft.app(target=main)
