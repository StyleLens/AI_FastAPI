"""
StyleLens V6 — Orchestrator Route: Quality Report
GET /quality/report — aggregated quality gate report from GeminiFeedbackInspector
"""

import logging

from fastapi import APIRouter, Query, Request

from core.gemini_feedback import GeminiFeedbackInspector

logger = logging.getLogger("stylelens.routes.quality")

router = APIRouter(prefix="/quality", tags=["Quality Gates"])


# ── Helpers ─────────────────────────────────────────────────────

def _get_inspector(request: Request) -> GeminiFeedbackInspector | None:
    return getattr(request.app.state, "inspector", None)


# ── Route ───────────────────────────────────────────────────────

@router.get("/report")
async def quality_report(
    request: Request,
    session_id: str = Query("default"),
):
    """Get aggregated quality gate report.

    Returns the inspector summary which includes total inspections,
    overall score, per-stage breakdowns, and failure details.
    The inspector is shared across all sessions (global quality tracking).
    Exact same response shape as main.py /quality/report.
    """
    inspector = _get_inspector(request)
    if not inspector:
        return {
            "error": "Inspector not available",
            "status": "ready",
            "total_inspections": 0,
            "overall_pass_rate": 1.0,
            "average_score": 1.0,
            "stage_breakdowns": {
                "person_detection": {"inspections": 0, "pass_rate": 1.0},
                "body_3d_reconstruction": {"inspections": 0, "pass_rate": 1.0},
                "clothing_analysis": {"inspections": 0, "pass_rate": 1.0},
                "virtual_tryon": {"inspections": 0, "pass_rate": 1.0},
            },
            "failures": [],
            "message": "Inspector disabled for fast testing",
        }
    return inspector.get_summary()
