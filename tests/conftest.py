import cv2
import numpy as np
import pytest


def _base_frame(size, seg):
    """Deterministic structured frame: a sharp 40 px block grid, so Laplacian variance
    stays well above the blur threshold and pHash is stable frame to frame. The palette
    shifts per segment so PySceneDetect sees a real cut."""
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for y in range(0, size[1], 40):
        for x in range(0, size[0], 40):
            shade = 40 + ((x // 40 + y // 40 + seg * 3) % 4) * 55
            img[y:y + 40, x:x + 40] = (shade, (shade * 2) % 255, (shade + 90) % 255)
    return img


def make_video(path, seconds=12.0, fps=10, size=(320, 240), segments=1, moving=False):
    """Synthetic mp4: `segments` hard palette cuts; `moving` adds a large block that
    travels diagonally.

    Frames inside a segment are identical unless `moving`, which makes pHash dedup
    deterministic in tests. Real footage also has stable low-frequency structure —
    per-frame random noise would not, and pHash on noise is meaningless.
    """
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = int(seconds * fps)
    for i in range(n):
        seg = min(int(i / n * segments), segments - 1)
        img = _base_frame(size, seg)
        if moving:
            x = (i * 9) % (size[0] - 120)
            y = (i * 7) % (size[1] - 90)
            cv2.rectangle(img, (x, y), (x + 120, y + 90), (255, 255, 255), -1)
        w.write(img)
    w.release()
    return str(path)


@pytest.fixture
def static_video(tmp_path):
    return make_video(tmp_path / "static.mp4")


@pytest.fixture
def moving_video(tmp_path):
    return make_video(tmp_path / "moving.mp4", moving=True)


@pytest.fixture
def cut_video(tmp_path):
    return make_video(tmp_path / "cuts.mp4", segments=3)
