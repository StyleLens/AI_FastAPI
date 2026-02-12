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
        return {"error": "Inspector not available"}
    return inspector.get_summary()
