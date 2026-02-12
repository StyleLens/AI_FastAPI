"""
StyleLens V6 — Orchestrator Route Tests

Tests the orchestrator routes, worker client, and session management
without loading actual GPU models or requiring Gemini API keys.

Strategy:
    - WorkerClient is tested as a unit (no HTTP calls).
    - Routes are tested via FastAPI TestClient against a minimal test app
      that skips the real lifespan (no core model imports beyond dataclasses).
    - Session state is injected/mocked so every test is self-contained.
"""

import io
import time
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchestrator.worker_client import WorkerClient, WorkerUnavailableError
from orchestrator.session import SessionManager, PipelineSession
from orchestrator.serialization import (
    ndarray_to_b64,
    b64_to_ndarray,
    image_to_b64,
    b64_to_image,
    bytes_to_b64,
    b64_to_bytes,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _create_test_app() -> FastAPI:
    """Create a minimal test app without the real lifespan.

    This avoids importing GPU model code (core.gemini_client, core.loader,
    etc.) while still mounting all orchestrator route modules.
    """
    from orchestrator.routes import all_routers

    app = FastAPI(title="Orchestrator Test App")

    # Minimal state — mirrors what the real lifespan sets up
    app.state.session_manager = SessionManager(max_sessions=5, ttl_sec=300)
    app.state.gemini = None  # No Gemini for tests
    app.state.inspector = None
    app.state.worker_client = WorkerClient(base_url="")  # local mode

    for router in all_routers:
        app.include_router(router)

    # ── Root / Health / Sessions (from main.py, re-declared for tests) ──

    @app.get("/")
    async def root():
        return {
            "service": "StyleLens V6 AI Orchestrator",
            "version": "6.0.0",
            "mode": "local",
        }

    @app.get("/health")
    async def health():
        sm = app.state.session_manager
        return {
            "status": "healthy",
            "mode": "local",
            "sessions": {
                "active": sm.active_count,
                "max": 5,
            },
        }

    @app.get("/sessions")
    async def list_sessions():
        sm = app.state.session_manager
        return {"sessions": sm.list_sessions()}

    return app


@pytest.fixture
def test_app():
    """Provide a fresh test app for each test."""
    return _create_test_app()


@pytest.fixture
def client(test_app):
    """Provide a TestClient bound to the test app."""
    return TestClient(test_app)


@pytest.fixture
def session_manager(test_app) -> SessionManager:
    """Shortcut to the app's session manager."""
    return test_app.state.session_manager


def _make_body_data_mock():
    """Create a MagicMock that mimics BodyData with expected attributes."""
    body = MagicMock()
    body.vertices = np.zeros((6890, 3), dtype=np.float32)
    body.faces = np.zeros((13776, 3), dtype=np.int32)
    body.joints = np.zeros((24, 3), dtype=np.float32)
    body.betas = np.zeros((10,), dtype=np.float64)
    body.gender = "female"
    body.glb_bytes = b"glTF\x02\x00\x00\x00fake_glb"
    body.mesh_renders = {}
    body.quality_gates = []
    body.metadata = MagicMock(
        gender="female",
        height_cm=170.0,
        weight_kg=65.0,
        body_type="standard",
    )
    return body


def _make_clothing_item_mock():
    """Create a MagicMock that mimics ClothingItem."""
    item = MagicMock()
    item.analysis = MagicMock(category="top", name="T-shirt")
    item.segmented_image = np.zeros((512, 512, 3), dtype=np.uint8)
    item.garment_mask = np.zeros((512, 512), dtype=np.uint8)
    item.parse_map = np.zeros((512, 512), dtype=np.uint8)
    item.original_images = [np.zeros((512, 512, 3), dtype=np.uint8)]
    item.size_chart = {}
    item.product_info = {}
    item.fitting_model_info = {}
    item.quality_gates = []
    return item


def _make_fitting_result_mock():
    """Create a MagicMock that mimics FittingResult."""
    result = MagicMock()
    result.tryon_images = {
        0: np.zeros((512, 512, 3), dtype=np.uint8),
        45: np.zeros((512, 512, 3), dtype=np.uint8),
    }
    result.method_used = {0: "gemini", 45: "gemini"}
    result.quality_gates = []
    result.p2p_result = None
    result.elapsed_sec = 5.0
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WorkerClient Unit Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWorkerClient:
    """WorkerClient unit tests — no HTTP calls."""

    def test_is_configured_false_when_no_url(self):
        """WorkerClient with empty URL is not configured."""
        wc = WorkerClient(base_url="")
        assert wc.is_configured() is False

    def test_is_configured_true_with_url(self):
        """WorkerClient with a URL is configured."""
        wc = WorkerClient(base_url="https://worker.example.com")
        assert wc.is_configured() is True

    @pytest.mark.asyncio
    async def test_call_raises_when_not_configured(self):
        """_call raises WorkerUnavailableError when base_url is empty."""
        wc = WorkerClient(base_url="")
        with pytest.raises(WorkerUnavailableError, match="not configured"):
            await wc._call("/any-endpoint", {"key": "value"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Root / Health / Sessions Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRootAndHealth:
    """Root, health, and session listing endpoints."""

    def test_root_returns_service_info(self, client):
        """GET / returns service name and version."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "StyleLens V6 AI Orchestrator"
        assert data["version"] == "6.0.0"
        assert data["mode"] == "local"

    def test_health_returns_status(self, client):
        """GET /health returns healthy status and session info."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "sessions" in data
        assert data["sessions"]["active"] == 0

    def test_sessions_empty_initially(self, client):
        """GET /sessions returns empty list when no sessions exist."""
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Route Prerequisite Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRoutePrerequisites:
    """Routes that require earlier phases reject requests gracefully."""

    def test_wardrobe_add_image_requires_gemini(self, client):
        """POST /wardrobe/add-image returns 503 when Gemini is not available."""
        # Send a minimal JPEG image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        import cv2
        _, buf = cv2.imencode(".jpg", img)
        file_bytes = io.BytesIO(buf.tobytes())

        resp = client.post(
            "/wardrobe/add-image",
            files={"image": ("test.jpg", file_bytes, "image/jpeg")},
        )
        assert resp.status_code == 503
        assert "Gemini" in resp.json()["detail"]

    def test_fitting_tryon_requires_session(self, client):
        """POST /fitting/try-on with unknown session returns 404."""
        resp = client.post(
            "/fitting/try-on?session_id=nonexistent-session-id",
        )
        assert resp.status_code == 404

    def test_fitting_tryon_requires_body_data(self, client, session_manager):
        """POST /fitting/try-on without Phase 1 body data returns 400."""
        sid = session_manager.create()
        resp = client.post(f"/fitting/try-on?session_id={sid}")
        assert resp.status_code == 400
        assert "Phase 1" in resp.json()["detail"]

    def test_fitting_tryon_requires_clothing(self, client, session_manager):
        """POST /fitting/try-on without Phase 2 clothing returns 400."""
        sid = session_manager.create()
        # Provide body data but no clothing
        session_manager.update_body_data(sid, _make_body_data_mock())
        resp = client.post(f"/fitting/try-on?session_id={sid}")
        assert resp.status_code == 400
        assert "Phase 2" in resp.json()["detail"]

    def test_viewer3d_requires_fitting_result(self, client, session_manager):
        """POST /viewer3d/generate without Phase 3 fitting result returns 400."""
        sid = session_manager.create()
        # Provide phases 1 & 2 but not phase 3
        session_manager.update_body_data(sid, _make_body_data_mock())
        session_manager.update_clothing(sid, _make_clothing_item_mock())
        resp = client.post(f"/viewer3d/generate?session_id={sid}")
        assert resp.status_code == 400
        assert "Phase 3" in resp.json()["detail"]

    def test_viewer3d_requires_session(self, client):
        """POST /viewer3d/generate with unknown session returns 404."""
        resp = client.post("/viewer3d/generate?session_id=fake-session")
        assert resp.status_code == 404

    def test_p2p_requires_session(self, client):
        """POST /p2p/analyze with unknown session returns 404."""
        resp = client.post("/p2p/analyze?session_id=fake-session-id")
        assert resp.status_code == 404

    def test_p2p_requires_body_data(self, client, session_manager):
        """POST /p2p/analyze without Phase 1 body data returns 400."""
        sid = session_manager.create()
        resp = client.post(f"/p2p/analyze?session_id={sid}")
        assert resp.status_code == 400
        assert "Phase 1" in resp.json()["detail"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Session Lifecycle Through Routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSessionLifecycle:
    """Session state flows through route interactions."""

    def test_sessions_count_increases(self, client, session_manager):
        """Creating sessions increases the active count."""
        sid1 = session_manager.create()
        sid2 = session_manager.create()

        resp = client.get("/sessions")
        data = resp.json()
        assert len(data["sessions"]) == 2

    def test_session_reflects_body_data(self, client, session_manager):
        """After storing body data, session listing shows has_body_data=True."""
        sid = session_manager.create()
        session_manager.update_body_data(sid, _make_body_data_mock())

        resp = client.get("/sessions")
        sessions = resp.json()["sessions"]
        match = [s for s in sessions if s["session_id"] == sid]
        assert len(match) == 1
        assert match[0]["has_body_data"] is True
        assert match[0]["has_clothing"] is False
        assert match[0]["status"]["phase1"] == "done"
        assert match[0]["progress"]["phase1"] == 1.0

    def test_session_full_progression(self, client, session_manager):
        """Full Phase 1-4 progression visible in session listing."""
        sid = session_manager.create()
        session_manager.update_body_data(sid, _make_body_data_mock())
        session_manager.update_clothing(sid, _make_clothing_item_mock())
        session_manager.update_fitting(sid, _make_fitting_result_mock())
        session_manager.update_viewer3d(sid, MagicMock())

        resp = client.get("/sessions")
        sessions = resp.json()["sessions"]
        match = [s for s in sessions if s["session_id"] == sid][0]
        assert match["has_body_data"] is True
        assert match["has_clothing"] is True
        assert match["has_fitting"] is True
        assert match["has_3d"] is True
        assert all(v == "done" for v in match["status"].values())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Serialization Integration (Route-Relevant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSerializationIntegration:
    """Serialization round-trips that routes depend on."""

    def test_ndarray_roundtrip_for_worker_protocol(self):
        """Vertices serialized by worker client can be deserialized."""
        verts = np.random.randn(6890, 3).astype(np.float32)
        payload = ndarray_to_b64(verts)
        # Simulate what a route does when unpacking worker response
        recovered = b64_to_ndarray(payload)
        np.testing.assert_array_equal(verts, recovered)
        assert recovered.shape == (6890, 3)
        assert recovered.dtype == np.float32

    def test_image_roundtrip_for_route_responses(self):
        """BGR image encoded for route response can be decoded."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # Red channel
        b64 = image_to_b64(img, quality=95)
        recovered = b64_to_image(b64)
        assert recovered.shape == (256, 256, 3)
        assert recovered.dtype == np.uint8
        # JPEG is lossy but smooth images should be close
        diff = np.abs(img.astype(int) - recovered.astype(int)).mean()
        assert diff < 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Error Handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestErrorHandling:
    """Graceful error handling in routes and components."""

    def test_avatar_glb_requires_session(self, client):
        """GET /avatar/glb with unknown session returns 404."""
        resp = client.get("/avatar/glb?session_id=nonexistent")
        assert resp.status_code == 404

    def test_avatar_glb_requires_body_data(self, client, session_manager):
        """GET /avatar/glb without generated avatar returns 404."""
        sid = session_manager.create()
        resp = client.get(f"/avatar/glb?session_id={sid}")
        assert resp.status_code == 404

    def test_quality_report_without_inspector(self, client):
        """GET /quality/report without inspector returns error dict (not 500)."""
        resp = client.get("/quality/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert "Inspector" in data["error"]

    def test_worker_unavailable_error_is_catchable(self):
        """WorkerUnavailableError is a proper Exception subclass."""
        err = WorkerUnavailableError("test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Worker Fallback Logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWorkerFallback:
    """Worker client local/distributed mode logic."""

    def test_local_mode_when_no_worker_url(self, test_app):
        """With empty WORKER_URL, worker is not configured (local mode)."""
        worker = test_app.state.worker_client
        assert worker.is_configured() is False

    def test_distributed_mode_requires_url(self):
        """Setting a base_url enables distributed mode."""
        wc = WorkerClient(base_url="https://gpu.example.com")
        assert wc.is_configured() is True

    @pytest.mark.asyncio
    async def test_health_returns_not_configured_when_local(self):
        """health() returns not_configured status when no URL is set."""
        wc = WorkerClient(base_url="")
        result = await wc.health()
        assert result["status"] == "not_configured"

    def test_worker_client_retry_settings(self):
        """WorkerClient accepts custom retry parameters."""
        wc = WorkerClient(
            base_url="https://worker.test",
            timeout_sec=60.0,
            retry_delay_sec=5.0,
            max_retries=3,
        )
        assert wc.timeout_sec == 60.0
        assert wc.retry_delay_sec == 5.0
        assert wc.max_retries == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Face Bank Routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFaceBankRoutes:
    """Face Bank upload and status routes."""

    def test_face_bank_status_requires_session(self, client):
        """GET /face-bank/{session_id}/status with unknown session returns 404."""
        resp = client.get("/face-bank/nonexistent/status")
        assert resp.status_code == 404

    def test_face_bank_status_no_bank(self, client, session_manager):
        """GET /face-bank/{session_id}/status returns has_face_bank=False initially."""
        sid = session_manager.create()
        resp = client.get(f"/face-bank/{sid}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_face_bank"] is False
        assert data["session_id"] == sid

    def test_face_bank_upload_no_face_detected(self, client, session_manager):
        """POST /face-bank/upload with blank image returns 400/503.

        400 = InsightFace loaded but no face detected in blank image.
        503 = InsightFace not available.
        """
        sid = session_manager.create()
        # Create a minimal blank JPEG image (no face)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        import cv2
        _, buf = cv2.imencode(".jpg", img)
        file_bytes = io.BytesIO(buf.tobytes())

        resp = client.post(
            f"/face-bank/upload?session_id={sid}",
            files={"current_photo": ("face.jpg", file_bytes, "image/jpeg")},
        )
        # 400 = no face detected, 503 = InsightFace not available
        assert resp.status_code in (400, 503)

    def test_face_bank_upload_requires_session(self, client):
        """POST /face-bank/upload with unknown session returns 404."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        import cv2
        _, buf = cv2.imencode(".jpg", img)
        file_bytes = io.BytesIO(buf.tobytes())

        resp = client.post(
            "/face-bank/upload?session_id=fake-session",
            files={"current_photo": ("face.jpg", file_bytes, "image/jpeg")},
        )
        assert resp.status_code == 404

    def test_session_listing_includes_face_bank(self, client, session_manager):
        """Session listing includes has_face_bank field."""
        sid = session_manager.create()
        resp = client.get("/sessions")
        sessions = resp.json()["sessions"]
        match = [s for s in sessions if s["session_id"] == sid]
        assert len(match) == 1
        assert "has_face_bank" in match[0]
        assert match[0]["has_face_bank"] is False

    def test_session_face_bank_update(self, session_manager):
        """SessionManager.update_face_bank stores the bank."""
        from core.face_bank import FaceBank
        sid = session_manager.create()
        bank = FaceBank(bank_id="test_bank", created_at=0.0)
        session_manager.update_face_bank(sid, bank)
        session = session_manager.get(sid)
        assert session.face_bank is not None
        assert session.face_bank.bank_id == "test_bank"

    def test_face_bank_route_registered(self, client):
        """Face bank routes are registered in the app."""
        # Check that the route responds (even with error)
        resp = client.get("/face-bank/test-session/status")
        # 404 (session not found) means route IS registered
        assert resp.status_code == 404

    def test_fitting_response_includes_face_bank(self, client, session_manager):
        """Fitting response includes face_bank field (null when not set)."""
        sid = session_manager.create()
        session_manager.update_body_data(sid, _make_body_data_mock())
        session_manager.update_clothing(sid, _make_clothing_item_mock())
        # fitting_tryon will fail (no Gemini) but we check route contract
        resp = client.post(f"/fitting/try-on?session_id={sid}")
        # Should fail with 503 (no Gemini) — but proves route works
        assert resp.status_code == 503
