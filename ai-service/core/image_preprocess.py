"""
StyleLens V6 — Image Preprocessing
Upscaling, denoising, and enhancement for clothing images.
"""

import cv2
import numpy as np

# Constants
MIN_LONG_SIDE = 2048
MAX_LONG_SIDE = 3072
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
UNSHARP_SIGMA = 1.0
UNSHARP_AMOUNT = 1.5
UNSHARP_THRESHOLD = 0


def _unsharp_mask(image: np.ndarray, sigma: float = UNSHARP_SIGMA,
                  amount: float = UNSHARP_AMOUNT,
                  threshold: int = UNSHARP_THRESHOLD) -> np.ndarray:
    """Apply unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    if threshold == 0:
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    else:
        diff = image.astype(np.int16) - blurred.astype(np.int16)
        mask = np.abs(diff) > threshold
        sharpened = image.copy()
        sharpened[mask] = np.clip(
            image[mask].astype(np.int16) + amount * diff[mask],
            0, 255
        ).astype(np.uint8)
    return sharpened


def preprocess_clothing_image(image_bgr: np.ndarray,
                               min_long_side: int = MIN_LONG_SIDE,
                               enhance_detail: bool = True) -> np.ndarray:
    """
    Preprocess a clothing image for analysis.
    1. Lanczos upscale if below min_long_side
    2. Bilateral filter for denoising
    3. CLAHE for contrast
    4. Unsharp mask for detail sharpening
    """
    h, w = image_bgr.shape[:2]
    long_side = max(h, w)

    # Upscale if needed
    if long_side < min_long_side:
        scale = min_long_side / long_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h),
                                interpolation=cv2.INTER_LANCZOS4)

    # Cap at max size
    h, w = image_bgr.shape[:2]
    long_side = max(h, w)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)

    if not enhance_detail:
        return image_bgr

    # Bilateral filter
    denoised = cv2.bilateralFilter(
        image_bgr, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )

    # CLAHE on L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    sharpened = _unsharp_mask(enhanced)

    return sharpened


def preprocess_batch(images_bgr: list[np.ndarray],
                     min_long_side: int = MIN_LONG_SIDE,
                     enhance_detail: bool = True) -> list[np.ndarray]:
    """Preprocess a batch of clothing images."""
    return [
        preprocess_clothing_image(img, min_long_side, enhance_detail)
        for img in images_bgr
    ]
