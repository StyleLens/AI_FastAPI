"""
StyleLens V6 — Orchestrator Route: P2P Physics-to-Prompt Analysis
POST /p2p/analyze — standalone P2P physics analysis
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, Request

from core.gemini_client import GeminiClient
from orchestrator.session import SessionManager

logger = logging.getLogger("stylelens.routes.p2p")

router = APIRouter(prefix="/p2p", tags=["P2P Analysis"])


# ── Helpers ─────────────────────────────────────────────────────

def _request_id() -> str:
    return str(uuid.uuid4())[:8]


def _get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def _get_gemini(request: Request) -> GeminiClient | None:
    return getattr(request.app.state, "gemini", None)


# ── Route ───────────────────────────────────────────────────────

@router.post("/analyze")
async def p2p_analyze(
    request: Request,
    session_id: str = Query("default"),
):
    """Standalone P2P analysis: body + clothing -> physics keywords.

    P2P is entirely deterministic / CPU-based, so there is no worker
    delegation. The optional Gemini ensemble is text-only (no GPU).
    Exact same response shape as main.py /p2p/analyze.
    """
    sm = _get_session_manager(request)
    gemini = _get_gemini(request)
    rid = _request_id()
    logger.info(f"[{rid}] P2P: Physics-to-Prompt analysis (session={session_id})")

    # Validate prerequisites
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session {session_id} not found")

    if not session.body_data:
        raise HTTPException(400, "Run Phase 1 (avatar) first")
    if not session.clothing_item:
        raise HTTPException(400, "Run Phase 2 (wardrobe) first")

    from core.config import P2P_ENABLED, P2P_ENSEMBLE_ENABLED
    if not P2P_ENABLED:
        raise HTTPException(503, "P2P engine not enabled")

    body_data = session.body_data
    clothing_item = session.clothing_item

    from core.p2p_engine import (
        run_p2p, extract_body_measurements, extract_garment_measurements,
    )

    # Simple P2P (deterministic)
    p2p_result = run_p2p(body_data, clothing_item)

    # Optionally run ensemble
    ensemble_data = None
    if P2P_ENSEMBLE_ENABLED and gemini:
        from core.p2p_ensemble import run_p2p_ensemble

        metadata_dict = {}
        if body_data.metadata:
            metadata_dict = {
                "gender": body_data.metadata.gender,
                "height_cm": body_data.metadata.height_cm,
                "weight_kg": body_data.metadata.weight_kg,
                "body_type": body_data.metadata.body_type,
            }
            for fld in ("shoulder_width_cm", "chest_cm", "waist_cm", "hip_cm"):
                val = getattr(body_data.metadata, fld, 0)
                if val:
                    metadata_dict[fld] = val

        body_meas = extract_body_measurements(
            body_data.vertices, body_data.joints, metadata_dict,
        )
        garment_meas = extract_garment_measurements(
            clothing_item.analysis,
            clothing_item.size_chart,
            clothing_item.fitting_model_info,
            clothing_item.product_info,
        )

        clothing_desc = (
            f"{clothing_item.analysis.category} "
            f"({clothing_item.analysis.name})"
        )
        ensemble_result = await run_p2p_ensemble(
            gemini, body_meas, garment_meas, clothing_desc,
        )
        p2p_result = ensemble_result.p2p_result
        ensemble_data = {
            "method": ensemble_result.method,
            "elapsed_sec": ensemble_result.elapsed_sec,
            "ensemble_confidence": ensemble_result.ensemble_confidence,
        }

    return {
        "request_id": rid,
        "session_id": session_id,
        "physics_prompt": p2p_result.physics_prompt,
        "overall_tightness": p2p_result.overall_tightness.value,
        "deltas": [
            {
                "body_part": d.body_part,
                "body_cm": d.body_cm,
                "garment_cm": d.garment_cm,
                "delta_cm": d.delta_cm,
                "tightness": d.tightness.value,
                "visual_keywords": d.visual_keywords,
                "prompt_fragment": d.prompt_fragment,
            }
            for d in p2p_result.deltas
        ],
        "mask_expansion_factor": p2p_result.mask_expansion_factor,
        "confidence": p2p_result.confidence,
        "method": p2p_result.method,
        "ensemble": ensemble_data,
    }
