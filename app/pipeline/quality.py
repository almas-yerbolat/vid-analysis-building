import cv2
import numpy as np

from app.config import settings


def laplacian_var(gray: np.ndarray) -> float:
    """Compute variance of Laplacian for blur detection."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_luma(gray: np.ndarray) -> float:
    """Compute mean luminance."""
    return float(gray.mean())


def frame_ok(gray: np.ndarray) -> bool:
    """Check if frame passes blur and exposure thresholds."""
    return (
        laplacian_var(gray) >= settings.blur_threshold
        and settings.luma_min <= mean_luma(gray) <= settings.luma_max
    )


def phash(gray: np.ndarray) -> int:
    """Compute 63-bit perceptual hash.

    63 bits (not 64) so the value fits in a signed BIGINT database column.
    DCT median is computed over coeffs[1:] to exclude the DC term,
    which encodes overall brightness rather than structure.
    """
    # Resize to 32x32 and compute DCT
    small = cv2.resize(gray, (32, 32)).astype(np.float32)
    coeffs = cv2.dct(small)[:8, :8].flatten()

    # Median over all coefficients except DC (index 0)
    median = np.median(coeffs[1:])

    # Compute 63-bit hash: bit i is 1 if coeffs[i] > median
    bits = 0
    for i in range(63):
        if coeffs[i] > median:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    return (a ^ b).bit_count()
