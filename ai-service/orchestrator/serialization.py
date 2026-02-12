"""
StyleLens V6 — Data Serialization for Tier 3 ↔ Tier 4 Communication
Converts numpy arrays, images, and binary data to/from base64 for HTTP transport.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image


def ndarray_to_b64(arr: np.ndarray) -> dict:
    """Serialize a numpy array to base64 with shape/dtype metadata.

    Returns:
        {"data": "<base64 string>", "shape": [N, M, ...], "dtype": "float32"}
    """
    raw = arr.tobytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "data": b64,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def b64_to_ndarray(payload: dict) -> np.ndarray:
    """Deserialize a base64-encoded numpy array.

    Args:
        payload: {"data": "<base64>", "shape": [...], "dtype": "..."}
    """
    raw = base64.b64decode(payload["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    return arr.reshape(payload["shape"])


def image_to_b64(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR image as JPEG base64 string.

    Args:
        img_bgr: OpenCV BGR image (H, W, 3) uint8
        quality: JPEG quality 1-100

    Returns:
        base64-encoded JPEG string
    """
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def b64_to_image(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string to BGR numpy array.

    Returns:
        OpenCV BGR image (H, W, 3) uint8
    """
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


def image_to_b64_png(img_bgr: np.ndarray) -> str:
    """Encode a BGR image as lossless PNG base64 string."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parsemap_to_b64(pm: np.ndarray) -> str:
    """Encode a uint8 parse map (class labels) as lossless PNG base64.

    Parse maps contain integer class indices (0-17 for FASHN),
    so we must use lossless encoding to preserve exact values.
    """
    ok, buf = cv2.imencode(".png", pm)
    if not ok:
        raise ValueError("Failed to encode parse map as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def b64_to_parsemap(b64: str) -> np.ndarray:
    """Decode a base64 PNG parse map to uint8 numpy array.

    Returns:
        Parse map (H, W) uint8
    """
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    pm = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if pm is None:
        raise ValueError("Failed to decode base64 parse map")
    return pm


def bytes_to_b64(data: bytes) -> str:
    """Encode raw bytes (e.g., GLB file) to base64 string."""
    return base64.b64encode(data).decode("ascii")


def b64_to_bytes(b64: str) -> bytes:
    """Decode base64 string to raw bytes."""
    return base64.b64decode(b64)
