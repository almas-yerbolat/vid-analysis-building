import cv2
import numpy as np

from app.config import settings


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_luma(gray: np.ndarray) -> float:
    return float(gray.mean())


def frame_ok(gray: np.ndarray) -> bool:
    return (
        laplacian_var(gray) >= settings.blur_threshold
        and settings.luma_min <= mean_luma(gray) <= settings.luma_max
    )


def phash(gray: np.ndarray) -> int:
    # 63 bits (not 64) so the value fits a signed BIGINT database column.
    # DCT median computed over coeffs[1:] to exclude DC, which encodes brightness not structure.
    small = cv2.resize(gray, (32, 32)).astype(np.float32)
    coeffs = cv2.dct(small)[:8, :8].flatten()
    median = np.median(coeffs[1:])
    bits = 0
    for i, v in enumerate(coeffs[:63]):
        if v > median:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
