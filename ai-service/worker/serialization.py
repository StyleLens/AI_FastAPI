"""
StyleLens V6 — Worker Serialization (mirrors orchestrator/serialization.py)
Shared data conversion utilities for Tier 4 Model Worker.
"""

# Re-export from orchestrator serialization for DRY principle.
# In deployment, this module can be standalone if orchestrator is not co-located.

import base64
import io

import cv2
import numpy as np


def ndarray_to_b64(arr: np.ndarray) -> dict:
    """Serialize a numpy array to base64 with shape/dtype metadata."""
    raw = arr.tobytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "data": b64,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def b64_to_ndarray(payload: dict) -> np.ndarray:
    """Deserialize a base64-encoded numpy array."""
    raw = base64.b64decode(payload["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    return arr.reshape(payload["shape"])


def image_to_b64(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR image as JPEG base64 string."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def b64_to_image(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string to BGR numpy array."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


def parsemap_to_b64(pm: np.ndarray) -> str:
    """Encode a uint8 parse map as lossless PNG base64."""
    ok, buf = cv2.imencode(".png", pm)
    if not ok:
        raise ValueError("Failed to encode parse map as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def b64_to_parsemap(b64: str) -> np.ndarray:
    """Decode a base64 PNG parse map to uint8 numpy array."""
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
