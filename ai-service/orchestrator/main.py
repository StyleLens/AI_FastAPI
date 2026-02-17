"""
StyleLens V6 — AI Orchestrator (Tier 3) FastAPI Server

Two modes of operation:
  1. Local mode (Modal not installed): All models run locally via core/ pipeline
  2. Modal mode (Modal SDK available): GPU tasks delegated to Modal H200 via .remote()

Run: python -m orchestrator.main
"""

import logging
import os
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from orchestrator.config import (
    BASE_DIR, OUTPUT_DIR, FITTING_ANGLES,
    GEMINI_ENABLED, MODAL_AVAILABLE,
    SESSION_MAX, SESSION_TTL_SEC, IMG_DATA_DIR,
    get_model_status,
)
from orchestrator.session import SessionManager
from orchestrator.worker_client import WorkerClient
from orchestrator.routes import all_routers

logger = logging.getLogger("stylelens.orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared state on startup, cleanup on shutdown."""
    logger.info("StyleLens V6 AI Orchestrator starting...")
    logger.info(f"Mode: {'MODAL' if MODAL_AVAILABLE else 'LOCAL'}")
    logger.info(f"Model status: {get_model_status()}")

    # Session manager
    app.state.session_manager = SessionManager(
        max_sessions=SESSION_MAX,
        ttl_sec=SESSION_TTL_SEC,
    )

    # Gemini
    if GEMINI_ENABLED:
        from core.gemini_client import GeminiClient
        app.state.gemini = GeminiClient()
        logger.info("Gemini client initialized")
    else:
        app.state.gemini = None
        logger.warning("Gemini disabled — quality gates will not function")

    # Feedback Inspector disabled for fast testing
    app.state.inspector = None

    # Worker client (Modal remote call)
    app.state.worker_client = WorkerClient()
    if MODAL_AVAILABLE:
        is_up = await app.state.worker_client.is_available()
        logger.info(f"Modal GPU worker: {'AVAILABLE' if is_up else 'UNAVAILABLE'}")
    else:
        logger.info("Modal not installed — running in local mode")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    yield

    # Shutdown
    await app.state.worker_client.close()
    from core.loader import registry
    registry.unload_all()
    logger.info("StyleLens V6 Orchestrator shutdown complete")


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="StyleLens V6 AI Orchestrator",
    version="6.0.0",
    description="4-Phase Virtual Try-On Pipeline: Avatar → Wardrobe → Fitting → 3D Viewer",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all route modules
for router in all_routers:
    app.include_router(router)

# Static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Root / Health / UI ────────────────────────────────────────

@app.get("/")
async def root():
    """Service info and status."""
    return {
        "service": "StyleLens V6 AI Orchestrator",
        "version": "6.0.0",
        "mode": "modal" if MODAL_AVAILABLE else "local",
        "models": get_model_status(),
        "phases": {
            "phase1_avatar": "ready",
            "phase2_wardrobe": "ready",
            "phase3_fitting": "ready",
            "phase4_3d_viewer": "ready",
        },
    }


@app.get("/health")
async def health(request: Request):
    """Detailed health check."""
    from core.loader import registry

    sm = request.app.state.session_manager
    worker = request.app.state.worker_client

    worker_status = "not_configured"
    if worker.is_configured():
        try:
            worker_health = await worker.health()
            worker_status = worker_health.get("status", "unknown")
        except Exception:
            worker_status = "unavailable"

    return {
        "status": "healthy",
        "mode": "modal" if MODAL_AVAILABLE else "local",
        "models": get_model_status(),
        "loader": registry.status_report(),
        "gemini": GEMINI_ENABLED,
        "worker": {
            "mode": "modal_remote" if MODAL_AVAILABLE else "local",
            "status": worker_status,
        },
        "sessions": {
            "active": sm.active_count,
            "max": SESSION_MAX,
        },
    }


@app.get("/sessions")
async def list_sessions(request: Request):
    """List all active pipeline sessions."""
    sm = request.app.state.session_manager
    return {"sessions": sm.list_sessions()}


@app.get("/ui")
async def ui():
    """Serve test console UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "UI not available — place index.html in static/"}


# ── Test Data Serving (IMG_Data) ──────────────────────────────

@app.get("/test-data/list")
async def list_test_data():
    """List available test data from IMG_Data directory."""
    img_data_path = Path(BASE_DIR) / IMG_DATA_DIR
    if not img_data_path.exists():
        return {"error": "IMG_Data directory not found", "path": str(img_data_path)}

    result = {}
    for category in ["User_IMG", "User_VOD", "WardrobeIMG", "wear", "wearSize"]:
        cat_path = img_data_path / category
        if cat_path.exists():
            files = sorted([f.name for f in cat_path.iterdir() if f.is_file()])
            result[category] = files
    return {"test_data": result, "base_path": str(img_data_path)}


@app.get("/test-data/{category}/{filename:path}")
async def serve_test_data(category: str, filename: str):
    """Serve individual test data files from IMG_Data."""
    # URL decode the filename (handles Korean characters)
    decoded = urllib.parse.unquote(filename)
    img_data_path = Path(BASE_DIR) / IMG_DATA_DIR / category / decoded

    if not img_data_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {category}/{decoded}"},
        )

    # Determine content type
    suffix = img_data_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        str(img_data_path),
        media_type=media_type,
    )


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "orchestrator.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
