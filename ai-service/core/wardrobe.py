"""
StyleLens V6 — Phase 2: Wardrobe
Clothing analysis with SAM 3 segmentation + FASHN parsing + Gemini analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from core.config import (
    DEVICE, FASHN_CLASSES,
    SAM3_ENABLED, FASHN_PARSER_ENABLED,
    REFERENCE_MODELS,
)
from core.loader import registry
from core.gemini_client import GeminiClient, ClothingAnalysis
from core.gemini_feedback import GeminiFeedbackInspector, InspectionResult
from core.image_preprocess import preprocess_clothing_image
from core.clothing_merger import classify_and_analyze, merge_analyses, AnalyzedView

logger = logging.getLogger("stylelens.wardrobe")


@dataclass
class ClothingItem:
    """Full clothing item data from Phase 2."""
    analysis: ClothingAnalysis = field(default_factory=ClothingAnalysis)
    segmented_image: np.ndarray | None = None
    garment_mask: np.ndarray | None = None
    parse_map: np.ndarray | None = None
    original_images: list[np.ndarray] = field(default_factory=list)
    size_chart: dict = field(default_factory=dict)
    product_info: dict = field(default_factory=dict)
    fitting_model_info: dict = field(default_factory=dict)
    quality_gates: list[InspectionResult] = field(default_factory=list)


def resolve_reference_body(gender: str, body_type: str = "standard") -> dict:
    """Get reference body measurements for a gender/body type."""
    gender_models = REFERENCE_MODELS.get(gender, REFERENCE_MODELS["female"])
    return gender_models.get(body_type, list(gender_models.values())[1])


def _segment_clothing_sam3(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment clothing from image using SAM 3."""
    if not SAM3_ENABLED:
        # Fallback: simple background removal
        mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
        return image_bgr, mask

    predictor = registry.load_sam3()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    # Use center point as prompt (assume clothing is centered)
    h, w = image_bgr.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Pick highest score mask
    best_idx = scores.argmax()
    mask = (masks[best_idx] * 255).astype(np.uint8)

    # Apply mask to image
    segmented = image_bgr.copy()
    segmented[mask == 0] = [255, 255, 255]  # white background

    registry.unload("sam3")
    return segmented, mask


def _parse_fashn(image_bgr: np.ndarray) -> np.ndarray:
    """Parse image into 18-class body parts using FASHN Parser."""
    if not FASHN_PARSER_ENABLED:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    fashn = registry.load_fashn_parser()
    model = fashn["model"]
    processor = fashn["processor"]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # Upsample to original size
    upsampled = torch.nn.functional.interpolate(
        logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
    )
    parse_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    registry.unload("fashn_parser")
    return parse_map


def _extract_garment_mask(parse_map: np.ndarray,
                           category: str = "top") -> np.ndarray:
    """Extract garment mask from FASHN parse map based on category."""
    mask = np.zeros(parse_map.shape[:2], dtype=np.uint8)

    # Map category to FASHN class indices
    if category in ("top", "outerwear"):
        # upper_clothes=4
        mask[parse_map == 4] = 255
    elif category == "bottom":
        # pants=6, skirt=5
        mask[(parse_map == 6) | (parse_map == 5)] = 255
    elif category == "dress":
        # dress=7
        mask[parse_map == 7] = 255
    else:
        # Default: upper_clothes + pants + dress + skirt
        mask[(parse_map == 4) | (parse_map == 5) |
             (parse_map == 6) | (parse_map == 7)] = 255

    return mask


async def analyze_clothing(
    images: list[np.ndarray],
    gemini: GeminiClient,
    size_chart_image: np.ndarray | None = None,
    product_info_images: list[np.ndarray] | None = None,
    fitting_model_image: np.ndarray | None = None,
    reference_body: dict | None = None,
    inspector: GeminiFeedbackInspector | None = None,
) -> ClothingItem:
    """
    Phase 2: Full clothing analysis pipeline.

    Steps:
        1. SAM 3 concept-aware segmentation (clothing isolation)
        1.5. Gemini Gate 2 — body_segmentation
        2. FASHN 18-class parsing → garment mask per class
        3. Gemini multi-image clothing analysis
        3.5. Gemini Gate 4 — clothing_analysis
        4. Size chart / product info extraction (if provided)
    """
    t0 = time.time()
    item = ClothingItem(original_images=images)

    # Preprocess images
    processed = [preprocess_clothing_image(img) for img in images]

    # ── Step 1: SAM 3 Segmentation ─────────────────────────────
    logger.info("Phase 2 Step 1: SAM 3 clothing segmentation")
    primary_img = processed[0]
    segmented, seg_mask = _segment_clothing_sam3(primary_img)
    item.segmented_image = segmented
    item.garment_mask = seg_mask

    # ── Step 1.5: Gemini Gate 2 — Segmentation ─────────────────
    if inspector:
        gate2 = inspector.inspect_body_segmentation(primary_img, seg_mask)
        item.quality_gates.append(gate2)
        if not gate2.pass_check:
            logger.warning(f"Gate 2 failed: {gate2.feedback}")

    # ── Step 2: FASHN 18-class Parsing ─────────────────────────
    logger.info("Phase 2 Step 2: FASHN body parsing")
    parse_map = _parse_fashn(primary_img)
    item.parse_map = parse_map

    # ── Step 3: Gemini Clothing Analysis ───────────────────────
    logger.info("Phase 2 Step 3: Gemini clothing analysis")

    if len(processed) == 1:
        item.analysis = gemini.analyze_detailed_fashion(processed[0])
    else:
        # Multi-image analysis with view classification
        views = classify_and_analyze(processed, gemini)
        item.analysis = merge_analyses(views)

    # ── Step 3.5: Gemini Gate 4 — Clothing Analysis ────────────
    if inspector:
        from dataclasses import asdict
        analysis_dict = asdict(item.analysis)
        gate4 = inspector.inspect_clothing_analysis(primary_img, analysis_dict)
        item.quality_gates.append(gate4)
        if not gate4.pass_check:
            logger.warning(f"Gate 4 failed: {gate4.feedback}")

    # ── Step 4: Size Chart / Product Info ──────────────────────
    if size_chart_image is not None:
        logger.info("Phase 2 Step 4a: Size chart extraction")
        item.size_chart = gemini.analyze_size_chart_image(size_chart_image)

    if product_info_images:
        logger.info("Phase 2 Step 4b: Product info extraction")
        item.product_info = gemini.analyze_product_info_images(product_info_images)

    if fitting_model_image is not None:
        logger.info("Phase 2 Step 4c: Fitting model info extraction")
        item.fitting_model_info = gemini.extract_fitting_model_info(fitting_model_image)

    elapsed = time.time() - t0
    logger.info(f"Phase 2 complete in {elapsed:.1f}s")
    return item
