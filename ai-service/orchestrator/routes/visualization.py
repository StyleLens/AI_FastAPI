"""
StyleLens V6 — Orchestrator Route: Phase 4 (3D Visualization)
POST /visualization/generate-3d — session_id -> textured GLB 3D model
GET  /visualization/glb          — binary GLB download
"""

import base64
import logging
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from orchestrator.session import SessionManager
from orchestrator.worker_client import WorkerClient, WorkerUnavailableError
from orchestrator.serialization import image_to_b64, b64_to_bytes

logger = logging.getLogger("stylelens.routes.visualization")

router = APIRouter(prefix="/visualization", tags=["Phase 4 — 3D Visualization"])


# ── Helpers ─────────────────────────────────────────────────────

def _request_id() -> str:
    return str(uuid.uuid4())[:8]


def _get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def _get_worker(request: Request) -> WorkerClient:
    return request.app.state.worker_client


# ── Routes ──────────────────────────────────────────────────────

@router.post("/generate-3d")
async def visualization_generate_3d(
    request: Request,
    session_id: str = Query("default"),
):
    """Phase 4: Generate textured 3D GLB from try-on result.

    - Uses front view (0°) try-on image as primary input
    - Optionally includes 45° and 315° views as reference images
    - Distributed mode: Hunyuan3D on GPU worker
    - Local mode: Not supported (Hunyuan3D is CUDA-only)

    Returns:
        {
            "request_id": str,
            "session_id": str,
            "glb_bytes_b64": str,
            "num_vertices": int,
            "num_faces": int,
            "textured": bool,
            "elapsed_sec": float
        }
    """
    sm = _get_session_manager(request)
    worker = _get_worker(request)
    rid = _request_id()
    logger.info(f"[{rid}] Phase 4: 3D visualization (session={session_id})")

    # Validate prerequisites
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    if not session.body_data:
        raise HTTPException(400, "Run Phase 1 (avatar) first")
    if not session.fitting_result:
        raise HTTPException(400, "Run Phase 3 (fitting) first")

    sm.update_status(session_id, "phase4", "running", 0.1)

    fitting_result = session.fitting_result

    # Extract front-angle try-on image (0°)
    front_image = fitting_result.tryon_images.get(0)
    if front_image is None:
        raise HTTPException(400, "No front view (0°) try-on image available")

    # ── Distributed mode ────────────────────────────────────────
    if worker.is_configured():
        try:
            result = await _distributed_3d_generation(
                rid, session, worker, sm, session_id
            )

            # Store GLB in session (reuse body_data.glb_bytes for textured version)
            if "glb_bytes_b64" in result:
                session.body_data.glb_bytes = base64.b64decode(result["glb_bytes_b64"])
                logger.info(f"[{rid}] Stored textured GLB: "
                            f"{len(session.body_data.glb_bytes)} bytes")

            sm.update_status(session_id, "phase4", "done", 1.0)

            return {
                "request_id": rid,
                "session_id": session_id,
                **result,
            }

        except WorkerUnavailableError as e:
            logger.warning(f"[{rid}] Worker unavailable: {e}")
            raise HTTPException(503, f"GPU worker unavailable: {e}")

    # ── Local mode ──────────────────────────────────────────────
    else:
        raise HTTPException(
            503,
            "3D visualization requires GPU worker (Hunyuan3D is CUDA-only). "
            "Enable Modal worker in orchestrator config."
        )


@router.get("/glb")
async def visualization_glb(
    request: Request,
    session_id: str = Query("default"),
):
    """Download textured 3D GLB for the given session.

    This GLB contains the textured 3D mesh generated from the try-on result.
    To download the original avatar mesh, use /avatar/glb instead.
    """
    sm = _get_session_manager(request)

    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    if not session.body_data or not session.body_data.glb_bytes:
        raise HTTPException(404, "No 3D model generated yet. Run /visualization/generate-3d first.")

    return Response(
        content=session.body_data.glb_bytes,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=tryon_3d.glb"},
    )


# ── Distributed helper ──────────────────────────────────────────

async def _distributed_3d_generation(
    rid: str,
    session,
    worker: WorkerClient,
    sm: SessionManager,
    session_id: str,
) -> dict:
    """Hunyuan3D on GPU worker.

    Workflow:
        1. Extract front view (0°) try-on image as primary input
        2. Extract reference images (45°, 315°) for better 3D reconstruction
        3. Send to GPU worker for Hunyuan3D shape + texture generation
        4. Return GLB bytes + metadata
    """
    fitting_result = session.fitting_result

    # Step 1: Encode front view (0°) as primary input
    front_image = fitting_result.tryon_images.get(0)
    if front_image is None:
        raise HTTPException(400, "No front view try-on image")

    front_b64 = image_to_b64(front_image)
    logger.info(f"[{rid}] Front view (0°) encoded: {len(front_b64)} chars")

    sm.update_status(session_id, "phase4", "running", 0.2)

    # Step 2: Encode reference images (45°, 315° → left/right of front)
    reference_images_b64 = []
    for angle in [45, 315]:
        img = fitting_result.tryon_images.get(angle)
        if img is not None:
            reference_images_b64.append(image_to_b64(img))
            logger.info(f"[{rid}] Reference angle {angle}° added")

    # If no reference images, send None to worker
    if not reference_images_b64:
        reference_images_b64 = None
        logger.warning(f"[{rid}] No reference images available, using front view only")

    sm.update_status(session_id, "phase4", "running", 0.3)

    # Step 3: Call Hunyuan3D on GPU worker
    logger.info(f"[{rid}] Calling Hunyuan3D on GPU worker...")
    resp = await worker.generate_3d_full(
        front_image_b64=front_b64,
        reference_images_b64=reference_images_b64,
        shape_steps=50,
        paint_steps=20,
        texture_res=4096,
    )

    sm.update_status(session_id, "phase4", "running", 0.9)

    logger.info(f"[{rid}] Hunyuan3D complete: "
                f"vertices={resp.get('num_vertices', 0)}, "
                f"faces={resp.get('num_faces', 0)}, "
                f"textured={resp.get('textured', False)}, "
                f"elapsed={resp.get('elapsed_sec', 0):.2f}s")

    return resp
