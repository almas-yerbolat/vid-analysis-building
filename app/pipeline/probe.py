from dataclasses import dataclass

import cv2

# ponytail: cv2 metadata instead of ffprobe; add real ffprobe + transcode if an exotic codec shows up


@dataclass
class VideoInfo:
    duration_s: float
    fps: float
    width: int
    height: int


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok or not fps or frames <= 0:
            raise ValueError(f"cannot decode video: {path}")
        return VideoInfo(duration_s=frames / fps, fps=fps, width=width, height=height)
    finally:
        cap.release()
