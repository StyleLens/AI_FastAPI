"""
StyleLens V6 — Phase 3: Virtual Try-On (Fitting)
CatVTON-FLUX x 8 angles with Gemini fallback + P2P Physics-to-Prompt integration.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from core.config import (
    DEVICE, FITTING_ANGLES,
    CATVTON_FLUX_ENABLED, FASHN_PARSER_ENABLED,
    INSIGHTFACE_ENABLED,
    P2P_ENABLED, P2P_ENSEMBLE_ENABLED,
)
from core.loader import registry
from core.pipeline import BodyData
from core.wardrobe import ClothingItem
from core.gemini_client import GeminiClient
from core.gemini_feedback import GeminiFeedbackInspector, InspectionResult
from core.sw_renderer import render_mesh
from core.multiview import generate_front_view_gemini, generate_angle_with_reference
from core.p2p_engine import run_p2p, P2PResult

logger = logging.getLogger("stylelens.fitting")


def _is_blank_render(img: np.ndarray, threshold: float = 0.95) -> bool:
    """Check if a mesh render is essentially blank (capsule/placeholder).

    Returns True if the image is predominantly a single background color,
    meaning the mesh render provides no useful visual reference.
    """
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Count pixels that are near-white (bg color from sw_renderer)
    bg_pixels = np.sum(gray > 240)
    total = gray.size
    return (bg_pixels / total) > threshold


@dataclass
class FittingResult:
    """Output of Phase 3 fitting pipeline."""
    tryon_images: dict[int, np.ndarray] = field(default_factory=dict)
    method_used: dict[int, str] = field(default_factory=dict)
    quality_gates: list[InspectionResult] = field(default_factory=list)
    p2p_result: P2PResult | None = None
    elapsed_sec: float = 0.0


def _generate_agnostic_mask(parse_map: np.ndarray,
                            category: str = "top") -> np.ndarray:
    """Generate agnostic mask for try-on from FASHN parse map."""
    mask = np.zeros(parse_map.shape[:2], dtype=np.uint8)

    if category in ("top", "outerwear"):
        # Mask upper body clothing region (class 4) + arms (14, 15)
        mask[(parse_map == 4) | (parse_map == 14) | (parse_map == 15)] = 255
    elif category == "bottom":
        # Mask lower body: pants(6), skirt(5), legs(12, 13)
        mask[(parse_map == 6) | (parse_map == 5) |
             (parse_map == 12) | (parse_map == 13)] = 255
    elif category == "dress":
        # Mask full body: dress(7) + upper(4) + legs + arms
        mask[(parse_map == 7) | (parse_map == 4) |
             (parse_map == 12) | (parse_map == 13) |
             (parse_map == 14) | (parse_map == 15)] = 255
    else:
        # Default: upper body
        mask[(parse_map == 4) | (parse_map == 14) | (parse_map == 15)] = 255

    # Dilate mask slightly for cleaner edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def _apply_p2p_mask_expansion(mask: np.ndarray,
                               expansion_factor: float) -> np.ndarray:
    """Apply P2P elasticity-aware mask expansion/contraction."""
    if abs(expansion_factor - 1.0) < 0.05:
        return mask  # No significant change needed

    kernel_size = max(3, int(5 * abs(expansion_factor - 1.0) * 10))
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if expansion_factor > 1.0:
        # Loose fit → dilate mask (more inpainting area for fabric drape)
        return cv2.dilate(mask, kernel, iterations=1)
    else:
        # Tight fit + elastic → erode mask (fabric stretches to cover)
        return cv2.erode(mask, kernel, iterations=1)


def _parse_person_image(person_image: np.ndarray) -> np.ndarray:
    """Parse person image using FASHN Parser to get body segmentation."""
    import torch

    if not FASHN_PARSER_ENABLED:
        # Simple fallback: assume center region is person
        h, w = person_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h*0.1):int(h*0.9), int(w*0.2):int(w*0.8)] = 4  # upper_clothes
        return mask

    fashn = registry.load_fashn_parser()
    model = fashn["model"]
    processor = fashn["processor"]

    rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=person_image.shape[:2], mode="bilinear", align_corners=False
    )
    parse_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return parse_map


async def generate_fitting(
    body_data: BodyData,
    clothing_item: ClothingItem,
    gemini: GeminiClient,
    face_photo: np.ndarray | None = None,
    inspector: GeminiFeedbackInspector | None = None,
    face_bank=None,
) -> FittingResult:
    """
    Phase 3: CatVTON-FLUX x 8 angles virtual try-on.

    Steps:
        1. Generate base person image from body_data (front view)
        2. FASHN parse → agnostic mask (upper/lower/dress)
        2.5. P2P Physics-to-Prompt analysis (if enabled)
        3. For each of 8 angles:
           3a. Render body at angle → person image
           3b. Generate agnostic mask for angle (with P2P expansion)
           3c. CatVTON-FLUX try_on(person, clothing, mask) or Gemini fallback (+P2P prompt)
           3d. Face identity preservation (if face_photo)
        3.5. Gemini Gate 5 — virtual_tryon (with P2P physics check)
    """
    t0 = time.time()
    result = FittingResult()
    category = clothing_item.analysis.category or "top"

    # ── Step 1: Base person image ──────────────────────────────
    logger.info("Phase 3 Step 1: Preparing base person image")

    # Use mesh render as base person image for each angle
    base_person = body_data.mesh_renders.get(0)
    if base_person is None and body_data.person_image is not None:
        base_person = body_data.person_image

    # ── Step 2: FASHN Parse → Agnostic Mask ────────────────────
    logger.info("Phase 3 Step 2: FASHN parsing for agnostic mask")

    if base_person is not None:
        parse_map = _parse_person_image(base_person)
    else:
        parse_map = np.zeros((512, 512), dtype=np.uint8)

    # Prepare clothing image
    clothing_img = (clothing_item.original_images[0]
                    if clothing_item.original_images
                    else np.zeros((512, 512, 3), dtype=np.uint8))

    # ── Step 2.5: P2P Physics-to-Prompt Analysis ─────────────
    p2p_result: P2PResult | None = None
    physics_prompt: str | None = None

    if P2P_ENABLED:
        logger.info("Phase 3 Step 2.5: P2P physics analysis")
        try:
            if P2P_ENSEMBLE_ENABLED and gemini:
                from core.p2p_ensemble import run_p2p_ensemble
                from core.p2p_engine import extract_body_measurements, extract_garment_measurements

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

                clothing_desc = f"{clothing_item.analysis.category} ({clothing_item.analysis.name})"
                ensemble_result = await run_p2p_ensemble(
                    gemini, body_meas, garment_meas, clothing_desc,
                )
                p2p_result = ensemble_result.p2p_result
            else:
                p2p_result = run_p2p(body_data, clothing_item)

            if p2p_result and p2p_result.physics_prompt:
                physics_prompt = p2p_result.physics_prompt
                logger.info(f"P2P: {p2p_result.overall_tightness.value}, "
                           f"mask_factor={p2p_result.mask_expansion_factor:.2f}")

        except Exception as e:
            logger.warning(f"P2P analysis failed (continuing without): {e}")
            p2p_result = None

    result.p2p_result = p2p_result

    # ── Step 3: Try-on for each angle ──────────────────────────
    logger.info("Phase 3 Step 3: Generating try-on images for 8 angles")

    # Determine method — read dynamically to allow runtime disable
    from core import config as _runtime_cfg
    use_catvton = _runtime_cfg.CATVTON_FLUX_ENABLED
    catvton = None

    if use_catvton:
        try:
            catvton = registry.load_catvton_flux()
        except Exception as e:
            logger.warning(f"CatVTON-FLUX load failed, falling back to Gemini: {e}")
            use_catvton = False

    # Face Bank: select angle-appropriate references
    _face_bank_select = None
    if face_bank is not None:
        from core.face_bank import select_references_for_angle as _fb_select
        _face_bank_select = _fb_select
        logger.info(f"Face Bank active: {len(face_bank.references)} refs, "
                     f"coverage={face_bank.angle_coverage()}")

    # Generate front view first
    front_result = None

    for angle in FITTING_ANGLES:
        logger.info(f"  Generating angle {angle}°...")

        # Get person image at this angle
        person_at_angle = body_data.mesh_renders.get(angle, base_person)

        # Skip blank/capsule renders for Gemini — they confuse the model
        mesh_ref = person_at_angle if not _is_blank_render(person_at_angle) else None

        # Select face references for this angle (Face Bank)
        angle_face_refs = None
        if _face_bank_select is not None:
            angle_face_refs = _face_bank_select(face_bank, angle)
            if angle_face_refs:
                logger.info(f"    Face Bank: {len(angle_face_refs)} refs for {angle}° "
                           f"({[r.face_angle for r in angle_face_refs]})")

        if use_catvton and person_at_angle is not None:
            # Primary path: CatVTON-FLUX
            agnostic_mask = _generate_agnostic_mask(
                _parse_person_image(person_at_angle), category
            )

            # P2P: Apply elasticity-aware mask expansion
            if p2p_result and p2p_result.mask_expansion_factor != 1.0:
                agnostic_mask = _apply_p2p_mask_expansion(
                    agnostic_mask, p2p_result.mask_expansion_factor
                )

            pil_result = catvton.try_on(
                person_at_angle, clothing_img, agnostic_mask
            )
            tryon_bgr = cv2.cvtColor(np.array(pil_result), cv2.COLOR_RGB2BGR)
            result.tryon_images[angle] = tryon_bgr
            result.method_used[angle] = "catvton-flux"
        else:
            # Fallback: Gemini image generation (with P2P physics prompt)
            if angle == 0:
                tryon = generate_front_view_gemini(
                    gemini, face_photo, clothing_img,
                    clothing_item.analysis,
                    mesh_ref,  # None when blank capsule render
                    body_data.gender,
                    physics_prompt=physics_prompt,
                    face_references=angle_face_refs,
                )
                if tryon is not None:
                    front_result = tryon
                    result.tryon_images[angle] = tryon
                    result.method_used[angle] = "gemini-front"
            else:
                if front_result is not None:
                    tryon = generate_angle_with_reference(
                        gemini, front_result, face_photo,
                        clothing_item.analysis,
                        mesh_ref,  # None when blank capsule render
                        angle, body_data.gender,
                        physics_prompt=physics_prompt,
                        face_references=angle_face_refs,
                    )
                    if tryon is not None:
                        result.tryon_images[angle] = tryon
                        result.method_used[angle] = "gemini-angle"

            # If both failed, use mesh render (only if non-blank)
            if angle not in result.tryon_images and mesh_ref is not None:
                result.tryon_images[angle] = mesh_ref
                result.method_used[angle] = "mesh-render"

        # Face identity preservation (InsightFace post-hoc)
        # DISABLED: InsightFace face swap OVER Gemini output causes severe artifacts
        # (gray smudge on cheeks, blurred features). Gemini already handles face identity
        # via the prompt + reference photo. Only apply for CatVTON-FLUX output.
        if (use_catvton and face_photo is not None
                and angle in result.tryon_images
                and result.method_used.get(angle) == "catvton-flux"
                and INSIGHTFACE_ENABLED):
            try:
                from core.face_identity import extract_face_data, apply_face_identity
                face_app = registry.load_insightface()
                face_data = extract_face_data(face_photo, face_app)
                if face_data:
                    result.tryon_images[angle] = apply_face_identity(
                        result.tryon_images[angle], face_data, face_app, angle
                    )
            except Exception as e:
                logger.warning(f"Face identity failed for angle {angle}: {e}")

    # Unload models
    if use_catvton:
        registry.unload("catvton_flux")
    registry.unload_except()  # unload all

    # ── Step 3.25: Face Consistency Check (Face Bank) ──────────
    if face_bank is not None and INSIGHTFACE_ENABLED and result.tryon_images:
        try:
            from core.face_bank import compute_face_similarity
            from core.face_identity import extract_face_data
            face_app = registry.load_insightface()

            # Check front-facing angles against mean embedding
            check_angles = [a for a in [0, 45, 315] if a in result.tryon_images]
            for check_angle in check_angles:
                gen_face = extract_face_data(result.tryon_images[check_angle], face_app)
                if gen_face and gen_face.embedding is not None:
                    sim = compute_face_similarity(
                        face_bank.mean_embedding, gen_face.embedding
                    )
                    logger.info(f"    Face consistency @{check_angle}°: "
                               f"similarity={sim:.3f}")
                    from core.config import FACE_BANK_SIMILARITY_THRESHOLD
                    if sim < FACE_BANK_SIMILARITY_THRESHOLD:
                        logger.warning(
                            f"    ⚠ Low face consistency @{check_angle}°: "
                            f"{sim:.3f} < {FACE_BANK_SIMILARITY_THRESHOLD}"
                        )

            registry.unload_except()
        except Exception as e:
            logger.warning(f"Face consistency check failed: {e}")

    # ── Step 3.5: Gemini Gate 5 — Virtual Try-On ───────────────
    if inspector and result.tryon_images:
        # Check 2 sample angles
        sample_angles = [0, 180]
        for angle in sample_angles:
            if angle in result.tryon_images:
                person_ref = (
                    body_data.person_image
                    if body_data.person_image is not None
                    else np.zeros((512, 512, 3), np.uint8)
                )
                gate5 = inspector.inspect_virtual_tryon(
                    person_ref,
                    result.tryon_images[angle],
                    clothing_img,
                    physics_prompt=physics_prompt,
                )
                result.quality_gates.append(gate5)
                if not gate5.pass_check:
                    logger.warning(f"Gate 5 failed for angle {angle}: {gate5.feedback}")
                break  # One check is enough for sample

    # ── Step 3.75: Gemini Gate 5.5 — Face Consistency ──────────
    if inspector and face_bank is not None and result.tryon_images:
        try:
            # Collect face reference images
            ref_images = [r.image_bgr for r in face_bank.references[:2]]
            # Collect generated front-facing angles
            gen_images = [result.tryon_images[a]
                          for a in [0, 45, 315] if a in result.tryon_images][:2]

            if ref_images and gen_images:
                gate55 = inspector.inspect_face_consistency(ref_images, gen_images)
                result.quality_gates.append(gate55)
                if not gate55.pass_check:
                    logger.warning(f"Gate 5.5 face consistency failed: "
                                  f"{gate55.quality_score:.2f} — {gate55.feedback}")
                else:
                    logger.info(f"Gate 5.5 face consistency passed: "
                               f"{gate55.quality_score:.2f}")
        except Exception as e:
            logger.warning(f"Gate 5.5 face consistency check failed: {e}")

    result.elapsed_sec = time.time() - t0
    logger.info(f"Phase 3 complete in {result.elapsed_sec:.1f}s, "
                f"generated {len(result.tryon_images)} angles")
    return result
