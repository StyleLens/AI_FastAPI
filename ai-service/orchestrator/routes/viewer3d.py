"""
StyleLens V6 — Orchestrator Route: Phase 4 (3D Viewer)
POST /viewer3d/generate          — 8 try-on images -> 3D GLB via Hunyuan3D
GET  /viewer3d/model/{glb_id}    — serve generated GLB file
"""

import base64
import logging
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from core.viewer3d import Viewer3DResult, generate_3d_model, get_glb_path
from core.gemini_feedback import GeminiFeedbackInspector
from orchestrator.session import SessionManager
from orchestrator.worker_client import WorkerClient, WorkerUnavailableError
from orchestrator.serialization import image_to_b64, b64_to_image, b64_to_bytes

logger = logging.getLogger("stylelens.routes.viewer3d")

router = APIRouter(prefix="/viewer3d", tags=["Phase 4 — 3D Viewer"])


# ── Helpers ─────────────────────────────────────────────────────

def _image_to_base64(img: np.ndarray) -> str:
    """Convert BGR image to base64 JPEG."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def _request_id() -> str:
    return str(uuid.uuid4())[:8]


def _get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def _get_worker(request: Request) -> WorkerClient:
    return request.app.state.worker_client


def _get_inspector(request: Request) -> GeminiFeedbackInspector | None:
    return getattr(request.app.state, "inspector", None)


# ── Routes ──────────────────────────────────────────────────────

@router.post("/generate")
async def viewer3d_generate(
    request: Request,
    session_id: str = Query("default"),
):
    """Phase 4: 8 try-on images -> 3D GLB via Hunyuan3D.

    - Local mode: full Hunyuan3D shape + paint pipeline via core.viewer3d
    - Distributed mode: delegates Hunyuan3D to GPU worker (heavy!)
    """
    sm = _get_session_manager(request)
    worker = _get_worker(request)
    inspector = _get_inspector(request)
    rid = _request_id()
    logger.info(f"[{rid}] Phase 4: 3D model generation (session={session_id})")

    # Validate prerequisites
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    if not session.fitting_result or not session.fitting_result.tryon_images:
        raise HTTPException(400, "Run Phase 3 (fitting) first")

    sm.update_status(session_id, "phase4", "running", 0.1)

    tryon_images = session.fitting_result.tryon_images

    # ── Distributed mode ────────────────────────────────────────
    if worker.is_configured():
        result = await _distributed_3d(
            rid, tryon_images, inspector, worker,
            session_id=session_id, sm=sm,
        )
    # ── Local mode ──────────────────────────────────────────────
    else:
        result = await generate_3d_model(
            tryon_images,
            inspector=inspector,
        )

    sm.update_viewer3d(session_id, result)

    # Preview renders — exact same shape as main.py
    previews = {}
    for angle, img in result.preview_renders.items():
        previews[str(angle)] = _image_to_base64(img)

    gates = [
        {"stage": g.stage, "score": g.quality_score, "pass": g.pass_check}
        for g in result.quality_gates
    ]

    return {
        "request_id": rid,
        "session_id": session_id,
        "glb_id": result.glb_id,
        "glb_size_bytes": len(result.glb_bytes),
        "glb_url": f"/viewer3d/model/{result.glb_id}",
        "previews": previews,
        "elapsed_sec": result.elapsed_sec,
        "quality_gates": gates,
    }


@router.get("/model/{glb_id}")
async def viewer3d_model(
    glb_id: str,
    session_id: str = Query("default"),
):
    """Serve generated GLB file by ID.

    The glb_id is globally unique so session_id is accepted but not strictly
    required for the lookup.
    """
    path = get_glb_path(glb_id)
    if not path:
        raise HTTPException(404, "GLB model not found")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": f"attachment; filename=model_{glb_id}.glb"},
    )


# ── Distributed helper ──────────────────────────────────────────

async def _distributed_3d(
    rid: str,
    tryon_images: dict[int, np.ndarray],
    inspector: GeminiFeedbackInspector | None,
    worker: WorkerClient,
    session_id: str = "default",
    sm: SessionManager | None = None,
) -> Viewer3DResult:
    """Run Hunyuan3D on the remote GPU worker.

    Workflow:
        1. Select best front image
        2. Encode all reference images
        3. Send to worker for Hunyuan3D shape + paint
        4. Decode GLB bytes and preview renders
        5. Save GLB locally and return Viewer3DResult
    Falls back to full local pipeline on worker failure.
    """
    import time
    from core.config import OUTPUT_DIR

    try:
        t0 = time.time()

        if sm:
            sm.update_status(session_id, "phase4", "running", 0.2)

        # Select best front image (0 > 315 > 45)
        front_img = None
        for angle in [0, 315, 45]:
            if angle in tryon_images:
                front_img = tryon_images[angle]
                break
        if front_img is None:
            front_img = next(iter(tryon_images.values()))

        front_b64 = image_to_b64(front_img)

        # Encode reference images (all angles sorted)
        ref_b64 = []
        for angle in sorted(tryon_images.keys()):
            ref_b64.append(image_to_b64(tryon_images[angle]))

        if sm:
            sm.update_status(session_id, "phase4", "running", 0.3)

        # Send to worker
        logger.info(f"[{rid}] Sending {len(ref_b64)} reference images to worker Hunyuan3D")
        worker_resp = await worker.generate_3d_full(
            front_image_b64=front_b64,
            reference_images_b64=ref_b64,
        )

        if sm:
            sm.update_status(session_id, "phase4", "running", 0.8)

        # Decode GLB
        glb_bytes = b64_to_bytes(worker_resp["glb_bytes_b64"])
        glb_id = str(uuid.uuid4())[:8]

        # Save GLB to output directory
        glb_path = OUTPUT_DIR / f"model_{glb_id}.glb"
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)

        # Decode preview renders
        preview_renders = {}
        for angle_str, img_b64 in worker_resp.get("preview_renders", {}).items():
            try:
                preview_renders[int(angle_str)] = b64_to_image(img_b64)
            except (ValueError, KeyError) as e:
                logger.warning(f"[{rid}] Failed to decode preview at {angle_str}: {e}")

        elapsed = time.time() - t0

        return Viewer3DResult(
            glb_bytes=glb_bytes,
            glb_id=glb_id,
            glb_path=str(glb_path),
            preview_renders=preview_renders,
            quality_gates=[],
            elapsed_sec=elapsed,
        )

    except WorkerUnavailableError as e:
        logger.warning(
            f"[{rid}] Worker unavailable for 3D generation, falling back to local: {e}"
        )
        return await generate_3d_model(
            tryon_images,
            inspector=inspector,
        )
