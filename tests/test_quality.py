import cv2
import numpy as np

from app.pipeline.quality import frame_ok, hamming, laplacian_var, mean_luma, phash


def checkerboard(shift=0):
    img = np.zeros((240, 320), np.uint8)
    img[:, :] = 30
    for y in range(0, 240, 20):
        for x in range(0, 320, 20):
            if ((x + y) // 20 + shift) % 2:
                img[y:y + 20, x:x + 20] = 220
    return img


def test_blur_detection():
    sharp = checkerboard()
    blurred = cv2.GaussianBlur(sharp, (31, 31), 10)
    assert laplacian_var(sharp) > laplacian_var(blurred)
    assert frame_ok(sharp) and not frame_ok(blurred)


def test_exposure_rejection():
    dark = np.full((240, 320), 5, np.uint8)
    bright = np.full((240, 320), 250, np.uint8)
    assert mean_luma(dark) < 15 and not frame_ok(dark)
    assert not frame_ok(bright)


def test_phash_similar_vs_different():
    a, b = checkerboard(), checkerboard()
    inverted = 255 - checkerboard()
    assert hamming(phash(a), phash(b)) < 8
    assert hamming(phash(a), phash(inverted)) >= 8
