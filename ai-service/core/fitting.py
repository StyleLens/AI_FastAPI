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


def _generate_adaptive_mask(parse_map: np.ndarray, category: str = "top") -> np.ndarray:
    """Adaptive mask: base FASHN mask → Upper Body Rect if coverage is low.

    Improved heuristics based on v6-v7 testing:
    - Mask B (Upper Body Rect) performs better for skin-tight clothing
    - Back view detection (no face) → force Rect
    - Tight-fitting clothes detection → force Rect
    - Coverage threshold raised from 15% to 25%

    Strategy:
        1. Generate base FASHN mask with enhanced dilation
        2. Detect back view (no face) → force Upper Body Rect
        3. Detect tight-fitting clothes (low raw coverage) → force Rect
        4. Check coverage ratio:
           - If coverage >= 25%, use base mask
           - If coverage < 25%, switch to Upper Body Rect

    Args:
        parse_map: FASHN parse map (HxW uint8, class IDs)
        category: "top", "bottom", "dress", etc.

    Returns:
        Binary mask (HxW uint8, 0/255)
    """
    # Step 1: Generate base mask
    base_mask = _generate_agnostic_mask(parse_map, category)
    kernel = np.ones((15, 15), np.uint8)
    base_mask = cv2.dilate(base_mask, kernel, iterations=2)
    base_mask = cv2.GaussianBlur(base_mask, (11, 11), 0)
    _, base_mask = cv2.threshold(base_mask, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Check coverage
    coverage = (base_mask > 0).sum() / base_mask.size

    # Step 3: Back view detection (no face → force Rect)
    has_face = (parse_map == 11).any()
    if not has_face:
        logger.info(f"Adaptive mask: No face detected (back view) → Upper Body Rect")
        return _make_upper_body_rect_mask(parse_map, category)

    # Step 4: Tight-fitting clothes detection
    # Generate raw mask without dilation to check base coverage
    raw_clothes = _generate_agnostic_mask(parse_map, category)
    raw_coverage = (raw_clothes > 0).sum() / raw_clothes.size

    # Very low raw coverage → clothes not detected properly → force Rect
    if raw_coverage < 0.05:
        logger.info(f"Adaptive mask: Very low raw coverage {raw_coverage*100:.1f}% → Upper Body Rect")
        return _make_upper_body_rect_mask(parse_map, category)

    # Step 5: Coverage threshold check (raised from 15% to 25%)
    if coverage >= 0.25:
        logger.debug(f"Adaptive mask: Using base FASHN mask (coverage {coverage*100:.1f}%)")
        return base_mask

    # Step 6: Switch to Mask B (Upper Body Rect)
    logger.info(f"Adaptive mask: Coverage {coverage*100:.1f}% < 25% → Upper Body Rect")
    return _make_upper_body_rect_mask(parse_map, category)


def _make_upper_body_rect_mask(parse_map: np.ndarray, category: str = "top") -> np.ndarray:
    """Upper Body Rect mask: bounding box + shoulder/collarbone expansion.

    This is Mask B from v6 testing — effective for tight-fitting clothing,
    back views, and sitting poses where standard FASHN mask is too narrow.

    Args:
        parse_map: FASHN parse map (HxW uint8, class IDs)
        category: "top", "bottom", "dress", etc.

    Returns:
        Binary mask (HxW uint8, 0/255)
    """
    if category in ("top", "outerwear"):
        clothes_classes = [4]   # upper_clothes
        arm_classes = [14, 15]  # arms
    elif category in ("bottom", "lower"):
        clothes_classes = [6, 5]  # pants, skirt
        arm_classes = [12, 13]    # legs
    elif category == "dress":
        clothes_classes = [7, 4]  # dress + upper
        arm_classes = [14, 15, 12, 13]  # arms + legs
    else:
        clothes_classes = [4]
        arm_classes = [14, 15]

    clothes_mask = np.zeros(parse_map.shape[:2], dtype=np.uint8)
    for cls in clothes_classes:
        clothes_mask[parse_map == cls] = 255

    arms_mask = np.zeros(parse_map.shape[:2], dtype=np.uint8)
    for cls in arm_classes:
        arms_mask[parse_map == cls] = 255

    combined = np.maximum(clothes_mask, arms_mask)

    ys, xs = np.where(combined > 0)
    if len(ys) == 0:
        # No clothes/arms detected → use full image mask
        logger.warning("Upper Body Rect: No clothing detected, using full mask")
        return np.ones(parse_map.shape[:2], dtype=np.uint8) * 255

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Extend upward 30% (shoulders/collarbone region)
    height = y_max - y_min
    y_min = max(0, y_min - int(height * 0.3))

    # Extend left/right 10%
    width = x_max - x_min
    x_min = max(0, x_min - int(width * 0.1))
    x_max = min(parse_map.shape[1] - 1, x_max + int(width * 0.1))

    # Create rectangular mask
    rect_mask = np.zeros(parse_map.shape[:2], dtype=np.uint8)
    rect_mask[y_min:y_max, x_min:x_max] = 255

    # Smooth edges for natural blending
    rect_mask = cv2.GaussianBlur(rect_mask, (21, 21), 0)
    _, rect_mask = cv2.threshold(rect_mask, 127, 255, cv2.THRESH_BINARY)

    rect_coverage = (rect_mask > 0).sum() / rect_mask.size
    logger.debug(f"Upper Body Rect: bbox=[{y_min}:{y_max}, {x_min}:{x_max}], "
                f"coverage {rect_coverage*100:.1f}%")

    return rect_mask


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
            agnostic_mask = _generate_adaptive_mask(
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
