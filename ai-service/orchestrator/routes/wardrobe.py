"""
StyleLens V6 — Orchestrator Route: Phase 2 (Wardrobe)
POST /wardrobe/add-image          — single clothing image analysis
POST /wardrobe/add-images         — multi-image + size_chart + product_info + fitting_model
POST /wardrobe/add-url            — fetch image from URL and analyze
POST /wardrobe/extract-model-info — extract fitting model body data from photo
"""

import base64
import logging
import uuid
from dataclasses import asdict

import aiohttp
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from core.wardrobe import ClothingItem, analyze_clothing
from core.gemini_client import GeminiClient
from core.gemini_feedback import GeminiFeedbackInspector
from orchestrator.session import SessionManager
from orchestrator.worker_client import WorkerClient, WorkerUnavailableError
from orchestrator.serialization import image_to_b64, b64_to_image, b64_to_parsemap

logger = logging.getLogger("stylelens.routes.wardrobe")

router = APIRouter(prefix="/wardrobe", tags=["Phase 2 — Wardrobe"])


# ── Helpers ─────────────────────────────────────────────────────

def _read_upload(upload: UploadFile) -> np.ndarray:
    """Read uploaded file to BGR numpy array."""
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Could not decode image: {upload.filename}")
    return img


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


# ── Routes ──────────────────────────────────────────────────────

@router.post("/add-image")
async def wardrobe_add_image(
    request: Request,
    image: UploadFile = File(...),
    session_id: str = Query("default"),
):
    """Phase 2: Analyze a single clothing image.

    - Local mode: SAM3 segmentation + FASHN parsing + Gemini analysis via core.wardrobe
    - Distributed mode: delegates SAM3/FASHN GPU work to worker, Gemini runs locally
    """
    sm = _get_session_manager(request)
    worker = _get_worker(request)
    gemini = _get_gemini(request)
    inspector = _get_inspector(request)
    rid = _request_id()
    logger.info(f"[{rid}] Phase 2: Single image wardrobe analysis (session={session_id})")

    if not gemini:
        raise HTTPException(503, "Gemini not available")

    if not sm.exists(session_id):
        session_id = sm.create()
    sm.update_status(session_id, "phase2", "running", 0.1)

    img = _read_upload(image)

    if worker.is_configured():
        item = await _distributed_analyze(
            rid, [img], gemini, inspector, worker,
        )
    else:
        item = await analyze_clothing([img], gemini, inspector=inspector)

    sm.update_clothing(session_id, item)

    return {
        "request_id": rid,
        "session_id": session_id,
        "analysis": asdict(item.analysis),
        "quality_gates": [
            {"stage": g.stage, "score": g.quality_score, "pass": g.pass_check}
            for g in item.quality_gates
        ],
    }


@router.post("/add-images")
async def wardrobe_add_images(
    request: Request,
    images: list[UploadFile] = File(...),
    size_chart: UploadFile | None = File(None),
    product_info_1: UploadFile | None = File(None),
    product_info_2: UploadFile | None = File(None),
    fitting_model: UploadFile | None = File(None),
    session_id: str = Query("default"),
):
    """Phase 2: Analyze multiple clothing images with optional extras."""
    sm = _get_session_manager(request)
    worker = _get_worker(request)
    gemini = _get_gemini(request)
    inspector = _get_inspector(request)
    rid = _request_id()
    logger.info(
        f"[{rid}] Phase 2: Multi-image wardrobe analysis "
        f"({len(images)} images, session={session_id})"
    )

    if not gemini:
        raise HTTPException(503, "Gemini not available")

    if not sm.exists(session_id):
        session_id = sm.create()
    sm.update_status(session_id, "phase2", "running", 0.1)

    imgs = [_read_upload(f) for f in images]
    size_chart_img = _read_upload(size_chart) if size_chart else None
    product_info_imgs = []
    if product_info_1:
        product_info_imgs.append(_read_upload(product_info_1))
    if product_info_2:
        product_info_imgs.append(_read_upload(product_info_2))
    fitting_model_img = _read_upload(fitting_model) if fitting_model else None

    if worker.is_configured():
        item = await _distributed_analyze(
            rid, imgs, gemini, inspector, worker,
            size_chart_image=size_chart_img,
            product_info_images=product_info_imgs or None,
            fitting_model_image=fitting_model_img,
        )
    else:
        item = await analyze_clothing(
            imgs, gemini,
            size_chart_image=size_chart_img,
            product_info_images=product_info_imgs or None,
            fitting_model_image=fitting_model_img,
            inspector=inspector,
        )

    sm.update_clothing(session_id, item)

    return {
        "request_id": rid,
        "session_id": session_id,
        "analysis": asdict(item.analysis),
        "size_chart": item.size_chart,
        "product_info": item.product_info,
        "fitting_model_info": item.fitting_model_info,
        "quality_gates": [
            {"stage": g.stage, "score": g.quality_score, "pass": g.pass_check}
            for g in item.quality_gates
        ],
    }


@router.post("/add-url")
async def wardrobe_add_url(
    request: Request,
    url: str = Form(...),
    session_id: str = Query("default"),
):
    """Phase 2: Analyze clothing from URL."""
    sm = _get_session_manager(request)
    worker = _get_worker(request)
    gemini = _get_gemini(request)
    inspector = _get_inspector(request)
    rid = _request_id()
    logger.info(f"[{rid}] Phase 2: URL wardrobe analysis (session={session_id})")

    if not gemini:
        raise HTTPException(503, "Gemini not available")

    if not sm.exists(session_id):
        session_id = sm.create()
    sm.update_status(session_id, "phase2", "running", 0.1)

    # Fetch image from URL
    async with aiohttp.ClientSession() as http_session:
        async with http_session.get(url) as resp:
            if resp.status != 200:
                raise HTTPException(400, f"Failed to fetch URL: {resp.status}")
            data = await resp.read()

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image from URL")

    if worker.is_configured():
        item = await _distributed_analyze(
            rid, [img], gemini, inspector, worker,
        )
    else:
        item = await analyze_clothing([img], gemini, inspector=inspector)

    sm.update_clothing(session_id, item)

    return {
        "request_id": rid,
        "session_id": session_id,
        "analysis": asdict(item.analysis),
    }


@router.post("/extract-model-info")
async def wardrobe_extract_model_info(
    request: Request,
    image: UploadFile = File(...),
    session_id: str = Query("default"),
):
    """Phase 2: Extract fitting model info from photo.

    This is always a Gemini-only call (no GPU needed), so no worker delegation.
    """
    gemini = _get_gemini(request)
    if not gemini:
        raise HTTPException(503, "Gemini not available")
    img = _read_upload(image)
    return gemini.extract_fitting_model_info(img)


# ── Distributed helper ──────────────────────────────────────────

async def _distributed_analyze(
    rid: str,
    imgs: list[np.ndarray],
    gemini: GeminiClient,
    inspector: GeminiFeedbackInspector | None,
    worker: WorkerClient,
    size_chart_image: np.ndarray | None = None,
    product_info_images: list[np.ndarray] | None = None,
    fitting_model_image: np.ndarray | None = None,
) -> ClothingItem:
    """Run wardrobe analysis with GPU tasks on the remote worker.

    Workflow:
        1. Send first image to worker for SAM3 segmentation
        2. Send segmented image to worker for FASHN parsing
        3. Run Gemini analysis locally (text model, no GPU)
        4. Assemble ClothingItem
    Falls back to full local pipeline on worker failure.
    """
    try:
        # Step 1: SAM3 segmentation on worker
        primary_b64 = image_to_b64(imgs[0])
        sam_resp = await worker.segment_sam3(primary_b64)

        segmented_bgr = b64_to_image(sam_resp["segmented_b64"])
        mask = b64_to_image(sam_resp["mask_b64"])
        garment_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask

        # Step 2: FASHN parsing on worker
        fashn_resp = await worker.parse_fashn(sam_resp["segmented_b64"])
        parse_map = b64_to_parsemap(fashn_resp["parsemap_b64"])

        # Step 3: Gemini analysis locally (text-only, no GPU)
        # We still go through analyze_clothing for the Gemini part,
        # but the segmented/parsed data is already available.
        # For simplicity, call the full local pipeline which will skip
        # SAM3/FASHN if already computed. In practice we just run local
        # since Gemini is the bottleneck anyway.
        item = await analyze_clothing(
            imgs, gemini,
            size_chart_image=size_chart_image,
            product_info_images=product_info_images,
            fitting_model_image=fitting_model_image,
            inspector=inspector,
        )

        # Override GPU-computed fields from worker
        item.segmented_image = segmented_bgr
        item.garment_mask = garment_mask
        item.parse_map = parse_map

        return item

    except WorkerUnavailableError as e:
        logger.warning(
            f"[{rid}] Worker unavailable for wardrobe, falling back to local: {e}"
        )
        return await analyze_clothing(
            imgs, gemini,
            size_chart_image=size_chart_image,
            product_info_images=product_info_images,
            fitting_model_image=fitting_model_image,
            inspector=inspector,
        )
