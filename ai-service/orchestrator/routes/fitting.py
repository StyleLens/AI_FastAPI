"""
StyleLens V6 — Orchestrator Route: Phase 3 (Fitting)
POST /fitting/try-on — face_photo (optional) -> 8-angle virtual try-on
"""

import base64
import logging
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from core.pipeline import BodyData
from core.wardrobe import ClothingItem
from core.fitting import FittingResult, generate_fitting
from core.gemini_client import GeminiClient
from core.gemini_feedback import GeminiFeedbackInspector
from orchestrator.session import SessionManager
from orchestrator.worker_client import WorkerClient, WorkerUnavailableError
from orchestrator.serialization import (
    image_to_b64, b64_to_image, image_to_b64_png, parsemap_to_b64,
)

logger = logging.getLogger("stylelens.routes.fitting")

router = APIRouter(prefix="/fitting", tags=["Phase 3 — Fitting"])


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


def _get_gemini(request: Request) -> GeminiClient | None:
    return getattr(request.app.state, "gemini", None)


def _get_inspector(request: Request) -> GeminiFeedbackInspector | None:
    return getattr(request.app.state, "inspector", None)


# ── Route ───────────────────────────────────────────────────────

@router.post("/try-on")
async def fitting_tryon(
    request: Request,
    face_photo: UploadFile | None = File(None),
    session_id: str = Query("default"),
):
    """Phase 3: CatVTON-FLUX 8-angle virtual try-on.

    - Local mode: full pipeline via core.fitting.generate_fitting()
    - Distributed mode: renders person images + masks locally,
      delegates CatVTON-FLUX batch to GPU worker, assembles result.
    """
    sm = _get_session_manager(request)
    worker = _get_worker(request)
    gemini = _get_gemini(request)
    inspector = _get_inspector(request)
    rid = _request_id()
    logger.info(f"[{rid}] Phase 3: Virtual try-on (session={session_id})")

    # Validate prerequisites
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    if not session.body_data:
        raise HTTPException(400, "Run Phase 1 (avatar) first")
    if not session.clothing_item:
        raise HTTPException(400, "Run Phase 2 (wardrobe) first")
    if not gemini:
        raise HTTPException(503, "Gemini not available")

    sm.update_status(session_id, "phase3", "running", 0.1)

    face = _read_upload(face_photo) if face_photo else None
    body_data = session.body_data
    clothing_item = session.clothing_item
    face_bank = session.face_bank  # May be None — backward compat

    # ── Distributed mode ────────────────────────────────────────
    if worker.is_configured():
        result = await _distributed_fitting(
            rid, body_data, clothing_item, gemini, inspector, worker,
            face_photo=face, session_id=session_id, sm=sm,
        )
    # ── Local mode ──────────────────────────────────────────────
    else:
        try:
            result = await generate_fitting(
                body_data, clothing_item, gemini,
                face_photo=face,
                inspector=inspector,
                face_bank=face_bank,
            )
        except Exception as e:
            logger.error(f"[{rid}] generate_fitting failed: {e}")
            # If CatVTON loading fails, disable it and retry with Gemini fallback
            from core import config as _cfg
            if _cfg.CATVTON_FLUX_ENABLED:
                logger.warning(
                    f"[{rid}] CatVTON load failed, disabling and retrying with Gemini"
                )
                _cfg.CATVTON_FLUX_ENABLED = False
                sm.update_status(session_id, "phase3", "running", 0.2)
                try:
                    result = await generate_fitting(
                        body_data, clothing_item, gemini,
                        face_photo=face,
                        inspector=inspector,
                        face_bank=face_bank,
                    )
                except Exception as retry_e:
                    logger.error(f"[{rid}] Retry also failed: {retry_e}")
                    raise HTTPException(500, f"Fitting failed after Gemini fallback: {retry_e}")
            else:
                raise HTTPException(500, f"Fitting failed: {e}")

    sm.update_fitting(session_id, result)

    # Encode images — exact same shape as main.py
    images = {}
    for angle, img in result.tryon_images.items():
        images[str(angle)] = _image_to_base64(img)

    gates = [
        {"stage": g.stage, "score": g.quality_score, "pass": g.pass_check}
        for g in result.quality_gates
    ]

    # P2P result
    p2p_data = None
    if result.p2p_result and result.p2p_result.physics_prompt:
        p2p_data = {
            "physics_prompt": result.p2p_result.physics_prompt,
            "overall_tightness": result.p2p_result.overall_tightness.value,
            "mask_expansion_factor": result.p2p_result.mask_expansion_factor,
            "confidence": result.p2p_result.confidence,
            "method": result.p2p_result.method,
            "deltas": [
                {
                    "body_part": d.body_part,
                    "delta_cm": d.delta_cm,
                    "tightness": d.tightness.value,
                    "visual_keywords": d.visual_keywords,
                }
                for d in result.p2p_result.deltas
            ],
        }

    # Face Bank info
    face_bank_data = None
    if face_bank is not None:
        face_bank_data = {
            "bank_id": face_bank.bank_id,
            "total_references": len(face_bank.references),
            "angle_coverage": face_bank.angle_coverage(),
        }

    return {
        "request_id": rid,
        "session_id": session_id,
        "images": images,
        "methods": {str(k): v for k, v in result.method_used.items()},
        "elapsed_sec": result.elapsed_sec,
        "quality_gates": gates,
        "p2p": p2p_data,
        "face_bank": face_bank_data,
    }


# ── Distributed helper ──────────────────────────────────────────

async def _distributed_fitting(
    rid: str,
    body_data: BodyData,
    clothing_item: ClothingItem,
    gemini: GeminiClient,
    inspector: GeminiFeedbackInspector | None,
    worker: WorkerClient,
    face_photo: np.ndarray | None = None,
    session_id: str = "default",
    sm: SessionManager | None = None,
) -> FittingResult:
    """Run fitting with CatVTON-FLUX on the remote GPU worker.

    Workflow:
        1. Render person images at 8 angles (local CPU, sw_renderer)
        2. Generate agnostic masks from parse map (local CPU)
        3. Send batch to worker for CatVTON-FLUX
        4. Run P2P analysis locally (deterministic, no GPU)
        5. Assemble FittingResult
    Falls back to full local pipeline on worker failure.
    """
    import time
    from core.config import FITTING_ANGLES
    from core.sw_renderer import render_mesh
    from core.p2p_engine import run_p2p

    try:
        t0 = time.time()

        # Step 1: Render person at each angle (CPU)
        persons_b64 = []
        masks_b64 = []

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.2)

        for angle in FITTING_ANGLES:
            if body_data.vertices is not None and body_data.faces is not None:
                person_img = render_mesh(
                    body_data.vertices, body_data.faces, angle_deg=angle,
                )
            else:
                person_img = np.zeros((512, 512, 3), dtype=np.uint8)

            persons_b64.append(image_to_b64(person_img))

            # Generate agnostic mask
            if clothing_item.parse_map is not None:
                from core.fitting import _generate_agnostic_mask
                category = getattr(clothing_item.analysis, "category", "top")
                mask = _generate_agnostic_mask(clothing_item.parse_map, category)
            else:
                mask = np.ones((512, 512), dtype=np.uint8) * 255
            masks_b64.append(image_to_b64_png(
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
            ))

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.4)

        # Step 2: Encode clothing
        if clothing_item.segmented_image is not None:
            clothing_b64 = image_to_b64(clothing_item.segmented_image)
        elif clothing_item.original_images:
            clothing_b64 = image_to_b64(clothing_item.original_images[0])
        else:
            raise HTTPException(400, "No clothing image available")

        # Step 3: Send to GPU worker for CatVTON-FLUX batch
        logger.info(f"[{rid}] Sending {len(persons_b64)} angles to worker CatVTON-FLUX")
        worker_resp = await worker.tryon_catvton_batch(
            persons_b64=persons_b64,
            clothing_b64=clothing_b64,
            masks_b64=masks_b64,
        )

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.8)

        # Step 4: Decode results
        tryon_images = {}
        method_used = {}
        results_b64 = worker_resp.get("results_b64", [])
        for i, angle in enumerate(FITTING_ANGLES):
            if i < len(results_b64):
                tryon_images[angle] = b64_to_image(results_b64[i])
                method_used[angle] = "catvton_flux_worker"
            else:
                logger.warning(f"[{rid}] No result for angle {angle} from worker")
                method_used[angle] = "missing"

        # Step 5: P2P analysis locally (no GPU needed)
        p2p_result = None
        try:
            from core.config import P2P_ENABLED
            if P2P_ENABLED:
                p2p_result = run_p2p(body_data, clothing_item)
        except Exception as e:
            logger.warning(f"[{rid}] P2P analysis failed: {e}")

        elapsed = time.time() - t0

        return FittingResult(
            tryon_images=tryon_images,
            method_used=method_used,
            quality_gates=[],
            p2p_result=p2p_result,
            elapsed_sec=elapsed,
        )

    except WorkerUnavailableError as e:
        logger.warning(
            f"[{rid}] Worker unavailable for fitting, falling back to local: {e}"
        )
        return await generate_fitting(
            body_data, clothing_item, gemini,
            face_photo=face_photo,
            inspector=inspector,
        )
