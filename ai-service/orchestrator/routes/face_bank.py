"""
StyleLens V6 — Orchestrator Route: Face Bank
POST /face-bank/upload — upload current + past face photos → build Face Bank
GET  /face-bank/{session_id}/status — check Face Bank status for a session
"""

import logging
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from core.config import INSIGHTFACE_ENABLED, FACE_BANK_MAX_REFERENCES
from core.face_bank import FaceBankBuilder
from orchestrator.session import SessionManager

logger = logging.getLogger("stylelens.routes.face_bank")

router = APIRouter(prefix="/face-bank", tags=["Face Bank"])


# ── Helpers ─────────────────────────────────────────────────────

def _read_upload(upload: UploadFile) -> np.ndarray:
    """Read uploaded file to BGR numpy array."""
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Could not decode image: {upload.filename}")
    return img


def _get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


# ── Routes ──────────────────────────────────────────────────────

@router.post("/upload")
async def upload_face_references(
    request: Request,
    current_photo: UploadFile = File(...),
    past_photos: list[UploadFile] = File(default=[]),
    session_id: str = Query("default"),
):
    """Upload face reference photos to build a Face Bank.

    - current_photo: The current face photo (required)
    - past_photos: Up to 10 additional reference photos (optional)
    - session_id: Session to attach the Face Bank to

    Requires InsightFace to be available for face detection + embedding.
    """
    if not INSIGHTFACE_ENABLED:
        raise HTTPException(
            503,
            "InsightFace not available — Face Bank requires InsightFace models"
        )

    sm = _get_session_manager(request)
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    # Validate total count
    total_photos = 1 + len(past_photos)
    if total_photos > FACE_BANK_MAX_REFERENCES:
        raise HTTPException(
            400,
            f"Too many photos ({total_photos}). "
            f"Maximum is {FACE_BANK_MAX_REFERENCES} (1 current + "
            f"{FACE_BANK_MAX_REFERENCES - 1} past)"
        )

    # Load InsightFace
    from core.loader import registry
    face_app = registry.load_insightface()

    builder = FaceBankBuilder(face_app)

    # Process current photo first
    current_img = _read_upload(current_photo)
    ref = builder.add_reference(current_img, label="current")

    results = []
    if ref:
        results.append({
            "label": ref.source_label,
            "face_angle": ref.face_angle,
            "det_score": round(ref.det_score, 3),
        })

    # Process past photos
    for i, photo in enumerate(past_photos):
        label = f"past_{i+1:02d}"
        past_img = _read_upload(photo)
        ref = builder.add_reference(past_img, label=label)
        if ref:
            results.append({
                "label": ref.source_label,
                "face_angle": ref.face_angle,
                "det_score": round(ref.det_score, 3),
            })

    # Build the bank
    if not results:
        raise HTTPException(
            400,
            "No faces detected in any uploaded photo. "
            "Ensure photos contain clearly visible faces."
        )

    bank = builder.build()

    # Store in session
    sm.update_face_bank(session_id, bank)

    # Unload models
    registry.unload_except()

    return {
        "bank_id": bank.bank_id,
        "session_id": session_id,
        "total_references": len(bank.references),
        "gender": bank.gender,
        "angle_coverage": bank.angle_coverage(),
        "references": results,
    }


@router.get("/{session_id}/status")
async def face_bank_status(
    request: Request,
    session_id: str,
):
    """Check Face Bank status for a session."""
    sm = _get_session_manager(request)
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    if session.face_bank is None:
        return {
            "session_id": session_id,
            "has_face_bank": False,
            "insightface_available": INSIGHTFACE_ENABLED,
        }

    bank = session.face_bank
    return {
        "session_id": session_id,
        "has_face_bank": True,
        "bank_id": bank.bank_id,
        "total_references": len(bank.references),
        "gender": bank.gender,
        "angle_coverage": bank.angle_coverage(),
        "insightface_available": INSIGHTFACE_ENABLED,
    }
