from dataclasses import dataclass

import cv2
import numpy as np

from app import storage
from app.config import settings
from app.pipeline import quality


@dataclass
class ExtractedFrame:
    ts_ms: int
    media_key: str
    thumb_key: str
    width: int
    height: int
    motion_score: float
    phash: int
    low_quality: bool
    selected_reason: str


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _grab_at(cap: cv2.VideoCapture, t_s: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t_s, 0.0) * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def _best_frame_near(cap, t_s: float) -> tuple[np.ndarray, bool] | None:
    """Frame at t_s, or sharpest passing neighbor within ±neighbor_window_s.
    Returns (frame, low_quality)."""
    candidates = []
    w = settings.neighbor_window_s
    for dt in (0.0, -w / 2, w / 2, -w, w):
        frame = _grab_at(cap, t_s + dt)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok = quality.frame_ok(gray)
        candidates.append((quality.laplacian_var(gray), ok, frame))
        if dt == 0.0 and ok:
            return frame, False
    if not candidates:
        return None
    passing = [c for c in candidates if c[1]]
    if passing:
        return max(passing, key=lambda c: c[0])[2], False
    return max(candidates, key=lambda c: c[0])[2], True


def _save(img: np.ndarray, video_id: str, ts_ms: int) -> tuple[str, str]:
    q = [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
    _, buf = cv2.imencode(".jpg", img, q)
    key = storage.save_bytes(f"frames/{video_id}/{ts_ms}.jpg", buf.tobytes())
    th = settings.thumb_width
    h = max(1, round(img.shape[0] * th / img.shape[1]))
    _, tbuf = cv2.imencode(".jpg", cv2.resize(img, (th, h)), q)
    tkey = storage.save_bytes(f"frames/{video_id}/{ts_ms}_thumb.jpg", tbuf.tobytes())
    return key, tkey


def _motion_at(motion: list[tuple[float, float]], t_s: float) -> float:
    if not motion:
        return 0.0
    return min(motion, key=lambda m: abs(m[0] - t_s))[1]


def extract_frames(video_path, video_id, keyframes, motion) -> list[ExtractedFrame]:
    cap = cv2.VideoCapture(video_path)
    out: list[ExtractedFrame] = []
    prev_hash: int | None = None
    try:
        for t_s, reason in keyframes:
            picked = _best_frame_near(cap, t_s)
            if picked is None:
                continue
            frame, low_q = picked
            frame = _resize_max_side(frame, settings.max_frame_side)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = quality.phash(gray)
            if prev_hash is not None and quality.hamming(h, prev_hash) < settings.phash_max_distance:
                continue  # near-duplicate of previous kept frame
            prev_hash = h
            ts_ms = round(t_s * 1000)
            key, tkey = _save(frame, video_id, ts_ms)
            out.append(ExtractedFrame(
                ts_ms=ts_ms, media_key=key, thumb_key=tkey,
                width=frame.shape[1], height=frame.shape[0],
                motion_score=_motion_at(motion, t_s), phash=h,
                low_quality=low_q, selected_reason=reason,
            ))
        return out
    finally:
        cap.release()


def extract_photo(image_bytes: bytes, video_id: str) -> ExtractedFrame:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image")
    img = _resize_max_side(img, settings.max_frame_side)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key, tkey = _save(img, video_id, 0)
    return ExtractedFrame(
        ts_ms=0, media_key=key, thumb_key=tkey,
        width=img.shape[1], height=img.shape[0],
        motion_score=0.0, phash=quality.phash(gray),
        low_quality=not quality.frame_ok(gray), selected_reason="photo",
    )
