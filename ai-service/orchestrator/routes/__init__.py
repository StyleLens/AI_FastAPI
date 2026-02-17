"""
StyleLens V6 — Orchestrator Route Modules

All route modules expose an `APIRouter` instance named `router`.
Import and include them into the main FastAPI app via `include_router()`.

Usage in app factory:
    from orchestrator.routes import all_routers
    for r in all_routers:
        app.include_router(r)
"""

from orchestrator.routes.avatar import router as avatar_router
from orchestrator.routes.wardrobe import router as wardrobe_router
from orchestrator.routes.fitting import router as fitting_router
from orchestrator.routes.viewer3d import router as viewer3d_router
from orchestrator.routes.visualization import router as visualization_router
from orchestrator.routes.p2p import router as p2p_router
from orchestrator.routes.quality import router as quality_router
from orchestrator.routes.face_bank import router as face_bank_router

all_routers = [
    avatar_router,
    wardrobe_router,
    fitting_router,
    viewer3d_router,
    visualization_router,
    p2p_router,
    quality_router,
    face_bank_router,
]

__all__ = [
    "avatar_router",
    "wardrobe_router",
    "fitting_router",
    "viewer3d_router",
    "visualization_router",
    "p2p_router",
    "quality_router",
    "face_bank_router",
    "all_routers",
]
