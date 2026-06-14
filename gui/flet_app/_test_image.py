"""Minimal test: does Flet 0.84 Image update with bytes from a background thread?"""
import base64
import threading
import time

import cv2
import numpy as np
import flet as ft


def main(page: ft.Page):
    page.title = "Image Test"
    page.window.width = 800
    page.window.height = 600

    # 1-pixel transparent PNG as initial placeholder
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB"
        "Nl7BcQAAAABJRU5ErkJggg=="
    )

    status = ft.Text("Waiting...", size=16)

    # Test 1: bytes in constructor
    img = ft.Image(src=pixel, width=400, height=300, fit=ft.BoxFit.CONTAIN)

    page.add(status, img)
    page.update()

    def bg_loop():
        """Generate colored frames in a background thread."""
        for i in range(200):
            # Create a simple colored frame
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = (i % 256, (i * 3) % 256, (i * 7) % 256)
            cv2.putText(frame, f"Frame {i}", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = buf.tobytes()

            # Method: raw bytes
            img.src = jpeg_bytes
            status.value = f"Frame {i} - bytes len={len(jpeg_bytes)}"

            try:
                page.update()
            except Exception as e:
                print(f"page.update error: {e}")

            time.sleep(0.1)

        status.value = "Done!"
        try:
            page.update()
        except Exception:
            pass

    threading.Thread(target=bg_loop, daemon=True).start()


ft.app(target=main)
