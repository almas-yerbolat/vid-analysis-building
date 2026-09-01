import cv2
import numpy as np

from app import storage
from app.pipeline.extract import extract_frames, extract_photo


def test_extracts_saves_and_dedups(static_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    keyframes = [(0.0, "baseline"), (5.0, "baseline"), (10.0, "baseline")]
    frames = extract_frames(static_video, "vid_t", keyframes, [(5.0, 0.2)])
    # static video → near-identical frames → dedup keeps only the first
    assert len(frames) == 1
    f = frames[0]
    assert f.ts_ms == 0 and f.selected_reason == "baseline"
    assert storage.path_for(f.media_key).exists()
    assert storage.path_for(f.thumb_key).exists()
    img = cv2.imread(str(storage.path_for(f.media_key)))
    # Exact shape, not `max(...) <= 1568`: the fixture is 320x240, so a bound alone
    # would still pass if the no-upscale guard were removed and it grew to 1568x1176.
    assert img.shape[:2] == (240, 320)


def test_moving_video_keeps_multiple(moving_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    keyframes = [(0.0, "baseline"), (5.0, "baseline"), (10.0, "baseline")]
    frames = extract_frames(moving_video, "vid_t", keyframes, [])
    assert len(frames) >= 2


def test_extract_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    img = np.random.default_rng(1).integers(0, 255, (2000, 3000, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    f = extract_photo(buf.tobytes(), "vid_p")
    assert f.selected_reason == "photo" and f.ts_ms == 0
    assert max(f.width, f.height) == 1568
