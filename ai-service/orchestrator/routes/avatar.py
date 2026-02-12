"""
StyleLens V6 — Orchestrator Route: Phase 1 (Avatar)
POST /avatar/generate — video|image + metadata -> BodyData
GET  /avatar/glb     — binary GLB download
"""

import base64
import logging
import os
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from core.pipeline import Metadata, BodyData, generate_avatar
from core.config import OUTPUT_DIR
from orchestrator.session import SessionManager
from orchestrator.worker_client import WorkerClient, WorkerUnavailableError
from orchestrator.serialization import image_to_b64, b64_to_image, b64_to_ndarray

logger = logging.getLogger("stylelens.routes.avatar")

router = APIRouter(prefix="/avatar", tags=["Phase 1 — Avatar"])


# ── Helpers ─────────────────────────────────────────────────────

def _read_upload(upload: UploadFile) -> np.ndarray:
    """Read uploaded file to BGR numpy array."""
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Could not decode image: {upload.filename}")
    return img


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


def _get_inspector(request: Request):
    return getattr(request.app.state, "inspector", None)


# ── Routes ──────────────────────────────────────────────────────

@router.post("/generate")
async def avatar_generate(
    request: Request,
    video: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    gender: str = Form("female"),
    height_cm: float = Form(170.0),
    weight_kg: float = Form(65.0),
    bust_cup: str = Form(""),
    body_type: str = Form("standard"),
    session_id: str = Query("default"),
):
    """Phase 1: Generate 3D avatar from video or image.

    - Local mode: delegates to core.pipeline.generate_avatar()
    - Distributed mode: extracts frames locally, sends to GPU worker for
      SAM3D reconstruction, then assembles BodyData from the response.
    """
    sm = _get_session_manager(request)
    worker = _get_worker(request)
    inspector = _get_inspector(request)
    rid = _request_id()
    logger.info(f"[{rid}] Phase 1: Avatar generation (session={session_id})")

    # Ensure session exists
    if not sm.exists(session_id):
        session_id = sm.create()
    sm.update_status(session_id, "phase1", "running", 0.1)

    metadata = Metadata(
        gender=gender,
        height_cm=height_cm,
        weight_kg=weight_kg,
        bust_cup=bust_cup,
        body_type=body_type,
    )

    # ── Distributed mode ────────────────────────────────────────
    if worker.is_configured():
        try:
            logger.info(f"[{rid}] Using distributed worker for Phase 1")

            # Extract best frame locally (lightweight CPU work)
            if video:
                video_path = str(OUTPUT_DIR / f"temp_video_{rid}.mp4")
                with open(video_path, "wb") as f:
                    f.write(video.file.read())

                from core.pipeline import _extract_frames
                frames = _extract_frames(video_path, max_frames=10)
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                if not frames:
                    raise HTTPException(400, "Could not extract frames from video")
                # Use middle frame as best candidate
                best_frame = frames[len(frames) // 2]
            elif image:
                best_frame = _read_upload(image)
            else:
                raise HTTPException(400, "Provide either video or image")

            sm.update_status(session_id, "phase1", "running", 0.3)

            # Send to GPU worker for 3D reconstruction
            frame_b64 = image_to_b64(best_frame)
            worker_resp = await worker.reconstruct_3d_body(frame_b64)

            sm.update_status(session_id, "phase1", "running", 0.7)

            # Build BodyData from worker response
            vertices = b64_to_ndarray(worker_resp["vertices"]) if "vertices" in worker_resp else None
            faces = b64_to_ndarray(worker_resp["faces"]) if "faces" in worker_resp else None
            joints = b64_to_ndarray(worker_resp["joints"]) if "joints" in worker_resp else None
            betas = b64_to_ndarray(worker_resp["betas"]) if "betas" in worker_resp else None

            # Build GLB and renders locally (CPU-bound, cheap)
            from core.sw_renderer import render_mesh
            from core.config import FITTING_ANGLES

            mesh_renders = {}
            if vertices is not None and faces is not None:
                for angle in FITTING_ANGLES:
                    try:
                        render = render_mesh(vertices, faces, angle_deg=angle)
                        mesh_renders[angle] = render
                    except Exception as e:
                        logger.warning(f"[{rid}] Render at {angle}deg failed: {e}")

            # Build GLB bytes
            glb_bytes = b""
            if vertices is not None and faces is not None:
                try:
                    import trimesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    glb_bytes = mesh.export(file_type="glb")
                except Exception as e:
                    logger.warning(f"[{rid}] GLB export failed: {e}")

            body_data = BodyData(
                vertices=vertices,
                faces=faces,
                joints=joints,
                betas=betas,
                gender=gender,
                glb_bytes=glb_bytes,
                mesh_renders=mesh_renders,
                metadata=metadata,
            )

        except WorkerUnavailableError as e:
            logger.warning(f"[{rid}] Worker unavailable, falling back to local: {e}")
            # Fall through to local mode
            body_data = await _local_generate(
                rid, video, image, metadata, inspector,
            )

    # ── Local mode ──────────────────────────────────────────────
    else:
        body_data = await _local_generate(
            rid, video, image, metadata, inspector,
        )

    # Store result
    sm.update_body_data(session_id, body_data)

    # Prepare response — exact same shape as main.py
    renders = {}
    for angle, img in body_data.mesh_renders.items():
        renders[str(angle)] = _image_to_base64(img)

    gates = []
    for g in body_data.quality_gates:
        gates.append({
            "stage": g.stage,
            "score": g.quality_score,
            "pass": g.pass_check,
            "feedback": g.feedback,
        })

    return {
        "request_id": rid,
        "session_id": session_id,
        "gender": body_data.gender,
        "has_mesh": body_data.vertices is not None,
        "vertex_count": len(body_data.vertices) if body_data.vertices is not None else 0,
        "glb_size_bytes": len(body_data.glb_bytes),
        "renders": renders,
        "quality_gates": gates,
    }


@router.get("/glb")
async def avatar_glb(
    request: Request,
    session_id: str = Query("default"),
):
    """Download current avatar GLB for the given session."""
    sm = _get_session_manager(request)

    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    if not session.body_data or not session.body_data.glb_bytes:
        raise HTTPException(404, "No avatar generated yet")

    return Response(
        content=session.body_data.glb_bytes,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=avatar.glb"},
    )


# ── Local fallback ──────────────────────────────────────────────

async def _local_generate(
    rid: str,
    video: UploadFile | None,
    image: UploadFile | None,
    metadata: Metadata,
    inspector,
) -> BodyData:
    """Run avatar generation entirely on the local machine."""
    video_path = None
    images = None

    if video:
        video_path = str(OUTPUT_DIR / f"temp_video_{rid}.mp4")
        with open(video_path, "wb") as f:
            f.write(video.file.read())
    elif image:
        img = _read_upload(image)
        images = [img]
    else:
        raise HTTPException(400, "Provide either video or image")

    body_data = await generate_avatar(
        video_path=video_path,
        images=images,
        metadata=metadata,
        inspector=inspector,
    )

    # Clean up temp video
    if video_path and os.path.exists(video_path):
        os.remove(video_path)

    return body_data
