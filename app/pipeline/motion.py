import cv2
import numpy as np

from app.config import settings

# ponytail: mean abs frame diff as motion score (spec-sanctioned cheap option);
# upgrade to Farneback optical flow if diff misjudges smooth pans


def motion_curve(path: str) -> list[tuple[float, float]]:
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / settings.scan_fps))
        curve: list[tuple[float, float]] = []
        prev = None
        i = 0
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h = max(1, round(frame.shape[0] * settings.scan_width / frame.shape[1]))
                    gray = cv2.cvtColor(
                        cv2.resize(frame, (settings.scan_width, h)), cv2.COLOR_BGR2GRAY
                    )
                    if prev is not None:
                        curve.append((i / fps, float(np.mean(cv2.absdiff(gray, prev)))))
                    prev = gray
            i += 1
        return curve
    finally:
        cap.release()
