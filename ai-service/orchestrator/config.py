"""
StyleLens V6 — AI Orchestrator Configuration
Environment-based config for Tier 3 orchestrator.
"""

import os

# ── GPU Worker (Tier 4) — Modal remote call ────────────────────
# Modal SDK가 설치되어 있으면 GPU 작업을 Modal H100에 위임.
# modal.Function.from_name() → .remote() 방식 (배포 없이 호출).
try:
    import modal  # noqa: F401
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# ── Session Management ────────────────────────────────────────
SESSION_MAX = int(os.getenv("SESSION_MAX", "10"))
SESSION_TTL_SEC = int(os.getenv("SESSION_TTL_SEC", "3600"))

# ── Test Data ─────────────────────────────────────────────────
# Path to IMG_Data folder (relative to project root)
IMG_DATA_DIR = os.getenv("IMG_DATA_DIR", "../IMG_Data")

# Re-export core config for convenience
from core.config import (  # noqa: E402
    BASE_DIR, OUTPUT_DIR, FITTING_ANGLES,
    GEMINI_ENABLED, GEMINI_API_KEY,
    P2P_ENABLED, P2P_ENSEMBLE_ENABLED,
    DEVICE, get_model_status,
)
