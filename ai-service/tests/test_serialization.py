"""
Tests for orchestrator/serialization.py — data serialization round-trips.
"""

import numpy as np
import pytest

from orchestrator.serialization import (
    ndarray_to_b64, b64_to_ndarray,
    image_to_b64, b64_to_image,
    image_to_b64_png,
    parsemap_to_b64, b64_to_parsemap,
    bytes_to_b64, b64_to_bytes,
)


class TestNdarraySerialization:
    """Numpy array ↔ base64 round-trips."""

    def test_float32_vertices(self):
        """SMPL mesh vertices: (6890, 3) float32."""
        arr = np.random.randn(6890, 3).astype(np.float32)
        payload = ndarray_to_b64(arr)
        assert payload["shape"] == [6890, 3]
        assert payload["dtype"] == "float32"
        recovered = b64_to_ndarray(payload)
        np.testing.assert_array_equal(arr, recovered)

    def test_int32_faces(self):
        """Mesh faces: (13776, 3) int32."""
        arr = np.random.randint(0, 6890, (13776, 3), dtype=np.int32)
        payload = ndarray_to_b64(arr)
        recovered = b64_to_ndarray(payload)
        np.testing.assert_array_equal(arr, recovered)

    def test_float64_betas(self):
        """SMPL betas: (10,) float64."""
        arr = np.random.randn(10).astype(np.float64)
        payload = ndarray_to_b64(arr)
        recovered = b64_to_ndarray(payload)
        np.testing.assert_array_equal(arr, recovered)

    def test_1d_array(self):
        """Single-dimension array."""
        arr = np.arange(100, dtype=np.float32)
        payload = ndarray_to_b64(arr)
        recovered = b64_to_ndarray(payload)
        np.testing.assert_array_equal(arr, recovered)

    def test_empty_array(self):
        """Edge case: empty array."""
        arr = np.array([], dtype=np.float32)
        payload = ndarray_to_b64(arr)
        recovered = b64_to_ndarray(payload)
        assert recovered.shape == (0,)


class TestImageSerialization:
    """Image ↔ base64 round-trips."""

    def test_jpeg_roundtrip(self):
        """BGR image → JPEG base64 → BGR (shape preserved, decodable)."""
        # Use a smooth gradient (realistic image) rather than random noise
        # Random noise is worst-case for JPEG and produces large errors
        h, w = 512, 512
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)  # Blue gradient
        img[:, :, 1] = 128  # Constant green
        img[:, :, 2] = np.linspace(255, 0, w, dtype=np.uint8)  # Red gradient
        b64 = image_to_b64(img, quality=95)
        recovered = b64_to_image(b64)
        assert recovered.shape == img.shape
        # JPEG is lossy, but smooth images compress well
        diff = np.abs(img.astype(int) - recovered.astype(int)).mean()
        assert diff < 5, f"Mean pixel difference {diff} too large for smooth JPEG q95"

    def test_png_roundtrip(self):
        """BGR image → PNG base64 → BGR (lossless)."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        b64 = image_to_b64_png(img)
        recovered = b64_to_image(b64)
        np.testing.assert_array_equal(img, recovered)

    def test_small_image(self):
        """1x1 pixel image."""
        img = np.array([[[128, 64, 32]]], dtype=np.uint8)
        b64 = image_to_b64_png(img)
        recovered = b64_to_image(b64)
        np.testing.assert_array_equal(img, recovered)

    def test_invalid_b64_raises(self):
        """Invalid base64 should raise ValueError."""
        with pytest.raises(ValueError):
            b64_to_image("not_valid_base64!!!")


class TestParsemapSerialization:
    """Parse map ↔ base64 round-trips (must be lossless)."""

    def test_fashn_parsemap(self):
        """FASHN 18-class parse map: (512, 512) uint8 with values 0-17."""
        pm = np.random.randint(0, 18, (512, 512), dtype=np.uint8)
        b64 = parsemap_to_b64(pm)
        recovered = b64_to_parsemap(b64)
        np.testing.assert_array_equal(pm, recovered)

    def test_binary_mask(self):
        """Binary mask: values 0 and 255 only."""
        pm = np.zeros((256, 256), dtype=np.uint8)
        pm[50:200, 30:220] = 255
        b64 = parsemap_to_b64(pm)
        recovered = b64_to_parsemap(b64)
        np.testing.assert_array_equal(pm, recovered)


class TestBytesSerialization:
    """Raw bytes (GLB, etc.) ↔ base64 round-trips."""

    def test_glb_roundtrip(self):
        """Simulate GLB bytes."""
        data = b"glTF\x02\x00\x00\x00" + bytes(range(256)) * 10
        b64 = bytes_to_b64(data)
        recovered = b64_to_bytes(b64)
        assert recovered == data

    def test_empty_bytes(self):
        """Edge case: empty bytes."""
        b64 = bytes_to_b64(b"")
        assert b64_to_bytes(b64) == b""
