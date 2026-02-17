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
    """Phase 3: 5-phase virtual try-on (v35+ pipeline).

    - Local mode: full pipeline via core.fitting.generate_fitting()
    - Distributed mode: SDXL→(FLUX)→FASHN VTON→Face Swap via GPU worker.
      Includes bust-aware dynamic cn_scale per cup size.
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
    # Gemini is optional for fitting — FASHN VTON or local fallback works without it
    # if not gemini:
    #     raise HTTPException(503, "Gemini not available")

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
            # If VTON loading fails, disable and retry with Gemini fallback
            from core import config as _cfg
            if _cfg.CATVTON_FLUX_ENABLED:
                logger.warning(
                    f"[{rid}] VTON load failed, disabling and retrying with Gemini"
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
    """Distributed 5-phase fitting pipeline (v35+).

    Pipeline:
        Phase 1.5A: SDXL + ControlNet Depth → pose-accurate realistic scaffold
                     Dynamic cn_scale per bust cup size (A=0.55 ~ H=0.82)
        Phase 1.5B: FLUX.2-klein img2img → texture upgrade (OPTIONAL, config flag)
        Phase 3:    FASHN VTON v1.5 → maskless virtual try-on (replaces CatVTON)
        Phase 4:    InsightFace Face Swap → face consistency across angles
        Phase 5:    P2P analysis (local CPU)

    Key v35 improvements:
        - Dynamic cn_scale: larger bust → stronger ControlNet depth adherence
        - Bust-aware body prompts via bust_cup_to_sdxl_description()
        - FLUX refine optional (FLUX_REFINE_ENABLED flag) for cost savings (~40s GPU)
        - FASHN VTON replaces CatVTON (maskless, better quality)
        - InsightFace face swap for identity preservation
    """
    import time
    from core.config import (
        FITTING_ANGLES, REALISTIC_MODEL_ENABLED,
        SDXL_NUM_STEPS, SDXL_GUIDANCE, SDXL_DEFAULT_CN_SCALE,
        FLUX_REFINE_ENABLED, FLUX_REFINE_STEPS, FLUX_REFINE_GUIDANCE,
        FASHN_VTON_TIMESTEPS, FASHN_VTON_GUIDANCE,
        FASHN_VTON_CATEGORY, FASHN_VTON_GARMENT_TYPE,
        FACE_SWAP_ENABLED, FACE_SWAP_BLEND_RADIUS, FACE_SWAP_SCALE,
        get_bust_cn_scale, get_bust_cup_scale,
    )
    from core.body_analyzer import bust_cup_to_sdxl_description
    from core.p2p_engine import run_p2p

    try:
        t0 = time.time()

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.02)

        # ── Extract metadata for bust-aware pipeline ──
        metadata = body_data.metadata
        bust_cup = metadata.bust_cup if metadata else ""
        gender = metadata.gender if metadata else (body_data.gender or "female")
        height_cm = metadata.height_cm if metadata else 165.0
        weight_kg = metadata.weight_kg if metadata else 58.0

        # Dynamic cn_scale based on bust cup size
        cn_scale = get_bust_cn_scale(bust_cup) if bust_cup else SDXL_DEFAULT_CN_SCALE

        # Body description for SDXL/FLUX prompts
        body_desc = ""
        if bust_cup and gender.lower() == "female":
            body_desc = bust_cup_to_sdxl_description(bust_cup, height_cm, weight_kg, gender)
            logger.info(f"[{rid}] Bust-aware: cup={bust_cup} cn_scale={cn_scale} desc='{body_desc}'")

        # ── Step 1: Collect mesh renders ──
        logger.info(f"[{rid}] Collecting 8-angle mesh renders...")
        mesh_renders_b64 = []
        angles_used = []

        for angle in FITTING_ANGLES:
            mesh_render = body_data.mesh_renders.get(angle)
            if mesh_render is not None:
                mesh_renders_b64.append(image_to_b64(mesh_render))
                angles_used.append(angle)
            else:
                logger.warning(f"[{rid}] No mesh render for angle {angle}deg, skipping")

        logger.info(f"[{rid}] Collected {len(angles_used)} mesh renders: {angles_used}")

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.05)

        # ── Phase 1.5A: SDXL + ControlNet Depth ──
        person_image_b64 = None
        if body_data.person_image is not None:
            person_image_b64 = image_to_b64(body_data.person_image)

        if REALISTIC_MODEL_ENABLED and mesh_renders_b64:
            logger.info(
                f"[{rid}] Phase 1.5A: SDXL + ControlNet Depth "
                f"(cn_scale={cn_scale}, steps={SDXL_NUM_STEPS})"
            )

            # Build bust-aware SDXL prompt
            body_desc_fragment = f"{body_desc}, " if body_desc else ""
            sdxl_prompt = (
                "RAW photo, an extremely detailed real photograph of a young Korean woman, "
                "realistic skin pores, natural skin texture, subtle skin imperfections, "
                f"{body_desc_fragment}"
                "long straight dark brown hair, {{angle_desc}}, "
                "wearing a plain light gray crewneck t-shirt and dark blue fitted jeans, "
                "white low-top sneakers on both feet, "
                "standing upright with perfect posture on a flat gray floor, "
                "head directly above shoulders, chin level, straight vertical neck, "
                "relaxed natural arms hanging at sides, hands beside thighs, "
                "legs together in natural standing position, feet flat on ground, "
                "shot with Fujifilm XT4, 85mm portrait lens, film grain, "
                "clean neutral light gray studio background, soft natural lighting, "
                "realistic body proportions matching the depth silhouette exactly, "
                "natural calm expression, "
                "full body visible from head to toe, 8k uhd"
            )
            sdxl_negative = (
                "anime, cartoon, illustration, painting, drawing, sketch, "
                "3d render, CGI, CG, computer graphics, digital art, "
                "smooth plastic skin, airbrushed, doll-like, porcelain, wax figure, "
                "floating, levitating, hovering, feet off ground, mid-air, "
                "leaning forward, hunched, slouching, bent over, "
                "turtle neck, forward head posture, chin jutting out, neck craning, "
                "tilted, diagonal posture, crooked stance, "
                "T-pose, arms spread wide, arms extended outward, "
                "A-pose, legs spread apart, wide stance, "
                "hands on hips, arms akimbo, hands behind back, "
                "cropped, cut off, missing limbs, extra limbs, missing feet, "
                "extremely thin, anorexic, bodybuilder, obese, "
                "nsfw, nude, revealing, "
                "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "oversaturated, overexposed, "
                "shorts, skirt, sandals, bare feet, high heels, boots, "
                "different outfits, multiple people, split image"
            )

            realistic_resp = await worker.mesh_to_realistic(
                mesh_renders_b64=mesh_renders_b64,
                person_image_b64=person_image_b64 or "",
                angles=angles_used,
                num_steps=SDXL_NUM_STEPS,
                guidance=SDXL_GUIDANCE,
                controlnet_conditioning_scale=cn_scale,
                prompt_template=sdxl_prompt,
                negative_prompt_override=sdxl_negative,
                body_description=body_desc,
            )

            if sm:
                sm.update_status(session_id, "phase3", "running", 0.25)

            if "error" in realistic_resp:
                logger.warning(f"[{rid}] SDXL failed: {realistic_resp['error']}, using raw mesh renders")
                persons_b64_list = mesh_renders_b64
            else:
                persons_b64_list = realistic_resp.get("realistic_renders_b64", mesh_renders_b64)
                logger.info(f"[{rid}] SDXL: {len(persons_b64_list)} realistic renders generated")
        else:
            logger.warning(f"[{rid}] Realistic model disabled, using raw mesh renders")
            persons_b64_list = mesh_renders_b64

        # ── Phase 1.5B: FLUX.2-klein img2img (OPTIONAL) ──
        if FLUX_REFINE_ENABLED:
            logger.info(
                f"[{rid}] Phase 1.5B: FLUX.2-klein-4B img2img "
                f"(steps={FLUX_REFINE_STEPS}, guidance={FLUX_REFINE_GUIDANCE})"
            )

            body_desc_fragment = f"{body_desc}, " if body_desc else ""
            flux_prompt = (
                "RAW photo, ultra realistic full-body photograph of a young Korean woman, "
                f"{body_desc_fragment}"
                "{{angle_desc}}, "
                "extremely detailed realistic skin with visible pores and natural texture, "
                "natural hair with individual strands visible, "
                "wearing a plain gray t-shirt and dark blue jeans, white sneakers, "
                "realistic fabric texture with natural wrinkles and folds, "
                "clean light gray studio background, soft natural lighting, "
                "full body from head to feet visible, "
                "shot on Fujifilm XT4, 85mm portrait lens, subtle film grain, "
                "professional fashion photography, 8k uhd, photorealistic"
            )

            flux_resp = await worker.flux_refine(
                images_b64=persons_b64_list,
                prompt_template=flux_prompt,
                angles=angles_used,
                num_steps=FLUX_REFINE_STEPS,
                guidance=FLUX_REFINE_GUIDANCE,
                seed=42,
                body_description=body_desc,
            )

            if sm:
                sm.update_status(session_id, "phase3", "running", 0.40)

            if "error" in flux_resp:
                logger.warning(f"[{rid}] FLUX refine failed: {flux_resp['error']}, using SDXL output")
            else:
                persons_b64_list = flux_resp.get("refined_b64", persons_b64_list)
                logger.info(f"[{rid}] FLUX: {len(persons_b64_list)} images refined")
        else:
            logger.info(f"[{rid}] Phase 1.5B: FLUX refine SKIPPED (FLUX_REFINE_ENABLED=False)")

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.45)

        # ── Phase 3: FASHN VTON v1.5 (maskless) ──
        if clothing_item.segmented_image is not None:
            clothing_b64 = image_to_b64(clothing_item.segmented_image)
        elif clothing_item.original_images:
            clothing_b64 = image_to_b64(clothing_item.original_images[0])
        else:
            raise HTTPException(400, "No clothing image available")

        logger.info(
            f"[{rid}] Phase 3: FASHN VTON v1.5 "
            f"({len(persons_b64_list)} images, category={FASHN_VTON_CATEGORY})"
        )

        vton_resp = await worker.tryon_fashn_batch(
            persons_b64=persons_b64_list,
            clothing_b64=clothing_b64,
            category=FASHN_VTON_CATEGORY,
            garment_photo_type=FASHN_VTON_GARMENT_TYPE,
            num_timesteps=FASHN_VTON_TIMESTEPS,
            guidance_scale=FASHN_VTON_GUIDANCE,
            seed=42,
        )

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.65)

        if "error" in vton_resp:
            logger.error(f"[{rid}] FASHN VTON failed: {vton_resp['error']}")
            raise WorkerUnavailableError(f"VTON failed: {vton_resp['error']}")

        fitted_b64_list = vton_resp.get("results_b64", [])
        logger.info(f"[{rid}] VTON: {len(fitted_b64_list)} fitted images generated")

        # ── Phase 4: Face Swap (InsightFace antelopev2) ──
        final_b64_list = fitted_b64_list
        face_reference_b64 = None
        if face_photo is not None:
            face_reference_b64 = image_to_b64(face_photo)

        if FACE_SWAP_ENABLED and face_reference_b64 and fitted_b64_list:
            logger.info(
                f"[{rid}] Phase 4: InsightFace Face Swap "
                f"({len(fitted_b64_list)} images, blend_radius={FACE_SWAP_BLEND_RADIUS})"
            )

            swap_resp = await worker.face_swap(
                images_b64=fitted_b64_list,
                face_reference_b64=face_reference_b64,
                angles=angles_used,
                blend_radius=FACE_SWAP_BLEND_RADIUS,
                face_scale=FACE_SWAP_SCALE,
            )

            if sm:
                sm.update_status(session_id, "phase3", "running", 0.80)

            if "error" in swap_resp:
                logger.warning(f"[{rid}] Face swap failed: {swap_resp['error']}, using VTON output")
            else:
                final_b64_list = swap_resp.get("swapped_b64", fitted_b64_list)
                face_detected = swap_resp.get("face_detected", [])
                n_swapped = sum(1 for d in face_detected if d)
                logger.info(f"[{rid}] Face Swap: {n_swapped}/{len(face_detected)} faces swapped")
        else:
            if not FACE_SWAP_ENABLED:
                logger.info(f"[{rid}] Phase 4: Face swap SKIPPED (FACE_SWAP_ENABLED=False)")
            elif not face_reference_b64:
                logger.info(f"[{rid}] Phase 4: Face swap SKIPPED (no face reference)")

        if sm:
            sm.update_status(session_id, "phase3", "running", 0.85)

        # ── Step 5: Decode results into FittingResult ──
        tryon_images = {}
        method_used = {}

        for i, angle in enumerate(angles_used):
            if i < len(final_b64_list):
                tryon_images[angle] = b64_to_image(final_b64_list[i])
                method_used[angle] = "fashn_vton_v35"
            else:
                logger.warning(f"[{rid}] Missing result for angle {angle}deg")
                method_used[angle] = "missing"

        for angle in FITTING_ANGLES:
            if angle not in tryon_images:
                method_used[angle] = "skipped"

        # ── Step 6: P2P analysis locally ──
        p2p_result = None
        try:
            from core.config import P2P_ENABLED
            if P2P_ENABLED:
                p2p_result = run_p2p(body_data, clothing_item)
        except Exception as e:
            logger.warning(f"[{rid}] P2P analysis failed: {e}")

        elapsed = time.time() - t0
        logger.info(f"[{rid}] Distributed fitting complete in {elapsed:.1f}s")

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
