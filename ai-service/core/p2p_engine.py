"""
StyleLens V6 — Physics-to-Prompt (P2P) Engine
Converts physical measurement deltas (Δ = Garment Size − Body Size)
into visual cue keywords for AI image generation.

Prevents AI from "beautifying" clothing fit by injecting physics-based
descriptions into generation prompts.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from core.config import P2P_BODY_PARTS, P2P_TIGHTNESS_THRESHOLDS

logger = logging.getLogger("stylelens.p2p")


# ── Enums & Dataclasses ──────────────────────────────────────

class TightnessLevel(Enum):
    CRITICAL_TIGHT = "critical_tight"
    TIGHT = "tight"
    OPTIMAL = "optimal"
    LOOSE = "loose"
    VERY_LOOSE = "very_loose"


@dataclass
class BodyPartDelta:
    """Delta analysis for a single body part."""
    body_part: str
    body_cm: float
    garment_cm: float
    delta_cm: float
    tightness: TightnessLevel
    visual_keywords: list[str] = field(default_factory=list)
    prompt_fragment: str = ""


@dataclass
class BodyMeasurements:
    """Standardized body measurements extracted from BodyData or Metadata."""
    shoulder_width_cm: float = 0.0
    chest_cm: float = 0.0
    waist_cm: float = 0.0
    hip_cm: float = 0.0
    sleeve_length_cm: float = 0.0
    inseam_cm: float = 0.0
    torso_length_cm: float = 0.0


@dataclass
class GarmentMeasurements:
    """Standardized garment measurements from ClothingAnalysis."""
    shoulder_cm: float = 0.0
    chest_cm: float = 0.0
    waist_cm: float = 0.0
    hip_cm: float = 0.0
    sleeve_cm: float = 0.0
    length_cm: float = 0.0


@dataclass
class P2PResult:
    """Complete P2P analysis result."""
    deltas: list[BodyPartDelta] = field(default_factory=list)
    overall_tightness: TightnessLevel = TightnessLevel.OPTIMAL
    physics_prompt: str = ""
    mask_expansion_factor: float = 1.0
    confidence: float = 0.0
    method: str = "fallback"  # "measured", "estimated", "fallback"


# ── Visual Keyword Map ────────────────────────────────────────
# 5 body parts × 5 tightness levels = 25 keyword sets

VISUAL_KEYWORD_MAP: dict[str, dict[TightnessLevel, list[str]]] = {
    "shoulder": {
        TightnessLevel.CRITICAL_TIGHT: [
            "seams pulling at shoulder line",
            "restricted movement visible",
            "fabric straining across shoulders",
        ],
        TightnessLevel.TIGHT: [
            "slightly tight at shoulders",
            "minor pull at shoulder seams",
        ],
        TightnessLevel.OPTIMAL: [
            "proper shoulder fit",
            "seams sitting naturally on shoulder bone",
        ],
        TightnessLevel.LOOSE: [
            "dropped shoulders",
            "seam sitting below shoulder bone",
        ],
        TightnessLevel.VERY_LOOSE: [
            "extremely dropped shoulders",
            "excess fabric bunching at shoulder area",
        ],
    },
    "chest": {
        TightnessLevel.CRITICAL_TIGHT: [
            "buttons strained with visible gapping",
            "tension rays between buttons",
            "fabric pulling across chest",
            "gapping at button plackets",
        ],
        TightnessLevel.TIGHT: [
            "slight tension across chest",
            "minor pull at front closure",
        ],
        TightnessLevel.OPTIMAL: [
            "proper chest fit",
            "natural fabric drape across torso",
        ],
        TightnessLevel.LOOSE: [
            "relaxed fit across chest",
            "slight excess fabric at torso",
        ],
        TightnessLevel.VERY_LOOSE: [
            "billowing fabric at chest",
            "excessive material hanging from torso",
        ],
    },
    "waist": {
        TightnessLevel.CRITICAL_TIGHT: [
            "muffin top effect at waistband",
            "fabric digging in at waist",
            "horizontal creasing from tension",
        ],
        TightnessLevel.TIGHT: [
            "snug at waist",
            "slight tension at waistband",
        ],
        TightnessLevel.OPTIMAL: [
            "proper waist fit",
            "natural waist definition",
        ],
        TightnessLevel.LOOSE: [
            "slight bunching at waist",
            "belt gathering visible",
        ],
        TightnessLevel.VERY_LOOSE: [
            "excessive waist fabric",
            "significant bunching and gathering at waist",
        ],
    },
    "hip": {
        TightnessLevel.CRITICAL_TIGHT: [
            "fabric pulling across hips",
            "horizontal stress lines at hip area",
            "pocket flaring open",
        ],
        TightnessLevel.TIGHT: [
            "slightly snug at hips",
            "minor fabric tension at hip",
        ],
        TightnessLevel.OPTIMAL: [
            "proper hip fit",
            "smooth fabric line over hips",
        ],
        TightnessLevel.LOOSE: [
            "relaxed hip fit",
            "slight fabric excess at hip area",
        ],
        TightnessLevel.VERY_LOOSE: [
            "excessive fabric at hips",
            "sagging material below waist",
        ],
    },
    "sleeve": {
        TightnessLevel.CRITICAL_TIGHT: [
            "sleeve riding up arm",
            "bicep constriction visible",
            "fabric tight around upper arm",
        ],
        TightnessLevel.TIGHT: [
            "slightly snug sleeves",
            "minor tightness at upper arm",
        ],
        TightnessLevel.OPTIMAL: [
            "proper sleeve length and fit",
            "natural sleeve drape",
        ],
        TightnessLevel.LOOSE: [
            "slightly long sleeves",
            "minor excess sleeve length",
        ],
        TightnessLevel.VERY_LOOSE: [
            "sleeves hanging past wrists",
            "excess sleeve fabric bunching at cuff",
        ],
    },
}


# ── Anthropometric Estimation Tables ─────────────────────────
# For estimating body measurements from height/weight/gender when
# mesh data or Gemini body analysis is unavailable.

ANTHROPOMETRIC_TABLES = {
    "female": {
        "chest_base": 82.0,     "chest_bmi_coeff": 1.5,
        "waist_base": 64.0,     "waist_bmi_coeff": 2.0,
        "hip_base": 88.0,       "hip_bmi_coeff": 1.3,
        "shoulder_base": 38.0,  "shoulder_height_coeff": 0.04,
        "sleeve_base": 52.0,    "sleeve_height_coeff": 0.10,
    },
    "male": {
        "chest_base": 92.0,     "chest_bmi_coeff": 1.3,
        "waist_base": 76.0,     "waist_bmi_coeff": 2.2,
        "hip_base": 92.0,       "hip_bmi_coeff": 1.0,
        "shoulder_base": 43.0,  "shoulder_height_coeff": 0.05,
        "sleeve_base": 56.0,    "sleeve_height_coeff": 0.12,
    },
}


# ── Garment Size Estimation Tables ───────────────────────────
# For estimating garment measurements when size chart is unavailable.
# Values in cm, indexed by [category][fit_type].

GARMENT_SIZE_ESTIMATES: dict[str, dict[str, dict[str, float]]] = {
    "top": {
        "slim":     {"chest_cm": 88, "waist_cm": 76, "shoulder_cm": 40, "hip_cm": 88, "sleeve_cm": 60},
        "regular":  {"chest_cm": 96, "waist_cm": 84, "shoulder_cm": 43, "hip_cm": 96, "sleeve_cm": 62},
        "relaxed":  {"chest_cm": 104, "waist_cm": 92, "shoulder_cm": 46, "hip_cm": 104, "sleeve_cm": 63},
        "oversized": {"chest_cm": 116, "waist_cm": 104, "shoulder_cm": 50, "hip_cm": 116, "sleeve_cm": 64},
    },
    "bottom": {
        "slim":     {"waist_cm": 72, "hip_cm": 90, "length_cm": 100},
        "regular":  {"waist_cm": 80, "hip_cm": 98, "length_cm": 102},
        "relaxed":  {"waist_cm": 88, "hip_cm": 106, "length_cm": 104},
        "oversized": {"waist_cm": 96, "hip_cm": 114, "length_cm": 106},
    },
    "dress": {
        "slim":     {"chest_cm": 86, "waist_cm": 72, "hip_cm": 90, "shoulder_cm": 38, "length_cm": 100},
        "regular":  {"chest_cm": 94, "waist_cm": 80, "hip_cm": 98, "shoulder_cm": 41, "length_cm": 105},
        "relaxed":  {"chest_cm": 102, "waist_cm": 88, "hip_cm": 106, "shoulder_cm": 44, "length_cm": 108},
        "oversized": {"chest_cm": 114, "waist_cm": 100, "hip_cm": 118, "shoulder_cm": 48, "length_cm": 112},
    },
    "outerwear": {
        "slim":     {"chest_cm": 96, "waist_cm": 84, "shoulder_cm": 43, "hip_cm": 96, "sleeve_cm": 63},
        "regular":  {"chest_cm": 104, "waist_cm": 92, "shoulder_cm": 46, "hip_cm": 104, "sleeve_cm": 65},
        "relaxed":  {"chest_cm": 112, "waist_cm": 100, "shoulder_cm": 49, "hip_cm": 112, "sleeve_cm": 66},
        "oversized": {"chest_cm": 124, "waist_cm": 112, "shoulder_cm": 53, "hip_cm": 124, "sleeve_cm": 68},
    },
}

# ── Elasticity × Tightness → Mask Expansion Lookup ──────────
# Indexed by (elasticity_level, overall_tightness_tendency)
# Returns a multiplier for agnostic mask kernel size.

_ELASTICITY_INDEX = {
    "none": 0, "slight": 1, "moderate": 2, "high": 3,
}
_TIGHTNESS_INDEX = {
    TightnessLevel.CRITICAL_TIGHT: 0,
    TightnessLevel.TIGHT: 1,
    TightnessLevel.OPTIMAL: 2,
    TightnessLevel.LOOSE: 3,
    TightnessLevel.VERY_LOOSE: 4,
}
# [elasticity][tightness] → expansion factor
_MASK_EXPANSION_TABLE = [
    # CT    TIGHT  OPT   LOOSE  VLOOSE
    [0.75,  0.85,  1.00, 1.20,  1.40],  # none
    [0.78,  0.88,  1.00, 1.15,  1.35],  # slight
    [0.82,  0.92,  1.00, 1.10,  1.25],  # moderate
    [0.88,  0.95,  1.00, 1.05,  1.15],  # high
]


# ── Core Functions ────────────────────────────────────────────

def _classify_tightness(delta_cm: float) -> TightnessLevel:
    """
    Classify measurement delta into tightness level.

    Boundaries use half-open intervals:
        Δ < -5      → CRITICAL_TIGHT
        -5 ≤ Δ < -2 → TIGHT
        -2 ≤ Δ < +5 → OPTIMAL
        +5 ≤ Δ < +10 → LOOSE
        Δ ≥ +10     → VERY_LOOSE
    """
    for level_name, (lo, hi) in P2P_TIGHTNESS_THRESHOLDS.items():
        if lo <= delta_cm < hi:
            return TightnessLevel(level_name)
    # Fallback (shouldn't reach here)
    return TightnessLevel.OPTIMAL


def _get_visual_keywords(body_part: str, tightness: TightnessLevel) -> list[str]:
    """Get visual cue keywords for a body part at a given tightness level."""
    part_map = VISUAL_KEYWORD_MAP.get(body_part)
    if part_map is None:
        return []
    return part_map.get(tightness, [])


def extract_body_measurements(
    vertices: np.ndarray | None,
    joints: np.ndarray | None,
    metadata_dict: dict,
    gemini_body_analysis: dict | None = None,
) -> BodyMeasurements:
    """
    Extract body measurements from available data.

    Priority:
        1. Gemini body analysis (if available from video analysis)
        2. Anthropometric estimation from height/weight/gender
    """
    bm = BodyMeasurements()

    # Priority 1: Gemini body analysis
    if gemini_body_analysis:
        bm.shoulder_width_cm = float(gemini_body_analysis.get("shoulder_width_cm", 0))
        bm.chest_cm = float(gemini_body_analysis.get("chest_cm", 0))
        bm.waist_cm = float(gemini_body_analysis.get("waist_cm", 0))
        bm.hip_cm = float(gemini_body_analysis.get("hip_cm", 0))
        if bm.chest_cm > 0 and bm.waist_cm > 0:
            logger.info("P2P: Using Gemini body analysis measurements")
            return bm

    # Priority 2: Metadata measurement fields (if populated)
    for fld in ("shoulder_width_cm", "chest_cm", "waist_cm", "hip_cm"):
        val = metadata_dict.get(fld, 0)
        if val and float(val) > 0:
            setattr(bm, fld, float(val))
    if bm.chest_cm > 0 and bm.waist_cm > 0:
        logger.info("P2P: Using metadata body measurements")
        return bm

    # Priority 3: Anthropometric estimation
    gender = metadata_dict.get("gender", "female")
    height = float(metadata_dict.get("height_cm", 165))
    weight = float(metadata_dict.get("weight_kg", 60))
    bmi = weight / ((height / 100) ** 2) if height > 0 else 22.0

    table = ANTHROPOMETRIC_TABLES.get(gender, ANTHROPOMETRIC_TABLES["female"])

    bm.chest_cm = table["chest_base"] + (bmi - 21.0) * table["chest_bmi_coeff"]
    bm.waist_cm = table["waist_base"] + (bmi - 21.0) * table["waist_bmi_coeff"]
    bm.hip_cm = table["hip_base"] + (bmi - 21.0) * table["hip_bmi_coeff"]
    bm.shoulder_width_cm = table["shoulder_base"] + (height - 165) * table["shoulder_height_coeff"]
    bm.sleeve_length_cm = table["sleeve_base"] + (height - 165) * table["sleeve_height_coeff"]

    logger.info(f"P2P: Estimated body measurements from anthropometric tables "
                f"(height={height}, weight={weight}, bmi={bmi:.1f})")
    return bm


def _is_flat_lay(size_chart: dict) -> bool:
    """
    Detect whether size chart measurements are flat-lay (half-circumference).

    Korean clothing sites typically use flat-lay (단면) measurements where
    chest/waist/hip values are half the full circumference.
    """
    # Check explicit measurement_type field (from Gemini extraction)
    mtype = size_chart.get("measurement_type", "").lower()
    if "flat" in mtype:
        return True
    if "full" in mtype or "circumference" in mtype:
        return False

    # Check notes for flat-lay indicators
    notes = (size_chart.get("notes", "") or "").lower()
    flat_indicators = ["flat", "단면", "가슴단면", "허리단면", "half", "flat width",
                       "flat lay", "반둘레"]
    if any(ind in notes for ind in flat_indicators):
        return True

    # Heuristic: Asian size system + small chest values → likely flat-lay
    size_system = (size_chart.get("size_system", "") or "").lower()
    sizes = size_chart.get("sizes", {})
    for size_data in sizes.values():
        if isinstance(size_data, dict):
            chest = float(size_data.get("chest_cm", 0) or 0)
            if chest > 0:
                # Full circumference chest is typically 76-120cm
                # Flat-lay chest is typically 40-65cm
                if chest < 70 and size_system in ("asian", "kr", "korean", "jp"):
                    return True
                break

    return False


def extract_garment_measurements(
    clothing_analysis: "ClothingAnalysis",
    size_chart: dict,
    fitting_model_info: dict,
    product_info: dict,
) -> GarmentMeasurements:
    """
    Extract garment measurements from available data.

    Priority:
        1. Size chart data (most reliable)
        2. Fitting model info (infer from model body)
        3. Estimation from fit_type + category

    Note: Korean clothing sites use flat-lay (단면) measurements for
    circumference values (chest, waist, hip). These are half the full
    circumference and must be doubled for comparison with body measurements.
    """
    gm = GarmentMeasurements()

    # Priority 1: Size chart
    if size_chart:
        sizes = size_chart.get("sizes", {})
        # Try M first, then any available size
        size_data = sizes.get("M") or sizes.get("m")
        if not size_data:
            for key in sizes:
                size_data = sizes[key]
                break
        if size_data and isinstance(size_data, dict):
            gm.chest_cm = float(size_data.get("chest_cm", 0) or 0)
            gm.waist_cm = float(size_data.get("waist_cm", 0) or 0)
            gm.hip_cm = float(size_data.get("hip_cm", 0) or 0)
            gm.shoulder_cm = float(size_data.get("shoulder_cm", 0) or 0)
            gm.sleeve_cm = float(size_data.get("sleeve_cm", 0) or 0)
            gm.length_cm = float(size_data.get("length_cm", 0) or 0)

            # Detect and convert flat-lay (half-circumference) measurements
            if _is_flat_lay(size_chart):
                logger.info("P2P: Detected flat-lay (단면) measurements — "
                            "doubling circumference values")
                if gm.chest_cm > 0:
                    gm.chest_cm *= 2
                if gm.waist_cm > 0:
                    gm.waist_cm *= 2
                if gm.hip_cm > 0:
                    gm.hip_cm *= 2
                # Note: shoulder, sleeve, and length are NOT circumference
                # measurements — they are linear distances, no doubling needed

            if gm.chest_cm > 0 or gm.waist_cm > 0:
                logger.info(f"P2P: Using size chart garment measurements "
                            f"(chest={gm.chest_cm}, waist={gm.waist_cm}, "
                            f"hip={gm.hip_cm})")
                return gm

    # Priority 2: Fitting model info (may have model_chest_cm directly or under measurements)
    if fitting_model_info:
        # Try direct keys first (model_chest_cm, model_waist_cm, model_hip_cm)
        model_chest = float(fitting_model_info.get("model_chest_cm", 0) or 0)
        model_waist = float(fitting_model_info.get("model_waist_cm", 0) or 0)
        model_hip = float(fitting_model_info.get("model_hip_cm", 0) or 0)

        # Fallback to nested measurements dict
        if model_chest == 0:
            model_meas = fitting_model_info.get("measurements", {})
            if model_meas:
                model_chest = float(model_meas.get("chest_cm", 0) or 0)
                model_waist = float(model_meas.get("waist_cm", 0) or 0)
                model_hip = float(model_meas.get("hip_cm", 0) or 0)

        if model_chest > 0:
            # Garment ≈ model body + fit ease
            fit_on_model = fitting_model_info.get("garment_fit_on_model", "")
            fit_type = fit_on_model or clothing_analysis.fit_type or "regular"
            fit_ease = {"slim": 2, "fitted": 2, "regular": 6, "relaxed": 10, "oversized": 16}
            ease = fit_ease.get(fit_type, 6)
            gm.chest_cm = model_chest + ease
            gm.waist_cm = model_waist + ease if model_waist > 0 else 0
            gm.hip_cm = model_hip + ease if model_hip > 0 else 0
            logger.info(f"P2P: Estimated garment from fitting model body "
                        f"(model_chest={model_chest}, fit={fit_type}, ease={ease}) "
                        f"→ garment_chest={gm.chest_cm}")
            return gm

    # Priority 3: Estimation from fit_type + category
    category = clothing_analysis.category or "top"
    fit_type = clothing_analysis.fit_type or "regular"

    cat_table = GARMENT_SIZE_ESTIMATES.get(category, GARMENT_SIZE_ESTIMATES.get("top", {}))
    fit_data = cat_table.get(fit_type, cat_table.get("regular", {}))

    if fit_data:
        gm.chest_cm = float(fit_data.get("chest_cm", 0))
        gm.waist_cm = float(fit_data.get("waist_cm", 0))
        gm.hip_cm = float(fit_data.get("hip_cm", 0))
        gm.shoulder_cm = float(fit_data.get("shoulder_cm", 0))
        gm.sleeve_cm = float(fit_data.get("sleeve_cm", 0))
        gm.length_cm = float(fit_data.get("length_cm", 0))
        logger.info(f"P2P: Estimated garment measurements from {category}/{fit_type}")

    return gm


def calculate_deltas(
    body: BodyMeasurements,
    garment: GarmentMeasurements,
) -> list[BodyPartDelta]:
    """
    Calculate measurement delta per body part.
    delta = garment_cm - body_cm (positive = garment is larger = loose)
    Skips body parts where either measurement is 0.
    """
    part_mapping = [
        ("shoulder", body.shoulder_width_cm, garment.shoulder_cm),
        ("chest",    body.chest_cm,          garment.chest_cm),
        ("waist",    body.waist_cm,          garment.waist_cm),
        ("hip",      body.hip_cm,            garment.hip_cm),
        ("sleeve",   body.sleeve_length_cm,  garment.sleeve_cm),
    ]

    deltas = []
    for part_name, body_val, garment_val in part_mapping:
        if body_val <= 0 or garment_val <= 0:
            continue

        delta = garment_val - body_val
        tightness = _classify_tightness(delta)
        keywords = _get_visual_keywords(part_name, tightness)

        prompt_fragment = ""
        if keywords:
            prompt_fragment = f"{part_name}: {', '.join(keywords)}"

        deltas.append(BodyPartDelta(
            body_part=part_name,
            body_cm=body_val,
            garment_cm=garment_val,
            delta_cm=round(delta, 1),
            tightness=tightness,
            visual_keywords=keywords,
            prompt_fragment=prompt_fragment,
        ))

    return deltas


def generate_physics_prompt(deltas: list[BodyPartDelta],
                            clothing_desc: str = "") -> str:
    """
    Combine all body part prompt fragments into a coherent physics prompt.

    Returns a formatted string suitable for injection into Gemini prompts:
        "The shirt shows slight tension across the chest with minor pulling.
         The waist area has relaxed fit with slight fabric bunching..."
    """
    if not deltas:
        return ""

    lines = []
    for d in deltas:
        if d.prompt_fragment:
            lines.append(f"- {d.prompt_fragment} (Δ{d.delta_cm:+.1f}cm)")

    if not lines:
        return ""

    header = "Physical fit characteristics based on body-garment measurement comparison:"
    if clothing_desc:
        header = f"Physical fit characteristics for {clothing_desc}:"

    return header + "\n" + "\n".join(lines)


def calculate_mask_expansion(
    deltas: list[BodyPartDelta],
    elasticity: str,
) -> float:
    """
    Calculate agnostic mask expansion factor.

    Returns float multiplier (0.7–1.5) for mask morphological operations:
        < 1.0 → erode mask (tight fit, elastic fabric stretches to cover)
        = 1.0 → no change (optimal fit)
        > 1.0 → dilate mask (loose fit, fabric drapes wider)
    """
    if not deltas:
        return 1.0

    # Determine overall tightness tendency (average delta)
    avg_delta = sum(d.delta_cm for d in deltas) / len(deltas)
    overall_tightness = _classify_tightness(avg_delta)

    # Look up in expansion table
    elast_idx = _ELASTICITY_INDEX.get(elasticity, 0)
    tight_idx = _TIGHTNESS_INDEX.get(overall_tightness, 2)

    return _MASK_EXPANSION_TABLE[elast_idx][tight_idx]


def _determine_overall_tightness(deltas: list[BodyPartDelta]) -> TightnessLevel:
    """Determine overall garment tightness from per-part deltas."""
    if not deltas:
        return TightnessLevel.OPTIMAL

    # Weight chest and waist more heavily
    weights = {"chest": 2.0, "waist": 2.0, "shoulder": 1.5, "hip": 1.5, "sleeve": 1.0}
    weighted_sum = 0.0
    total_weight = 0.0

    for d in deltas:
        w = weights.get(d.body_part, 1.0)
        weighted_sum += d.delta_cm * w
        total_weight += w

    if total_weight == 0:
        return TightnessLevel.OPTIMAL

    avg_delta = weighted_sum / total_weight
    return _classify_tightness(avg_delta)


def run_p2p(
    body_data: "BodyData",
    clothing_item: "ClothingItem",
    gemini_body_analysis: dict | None = None,
) -> P2PResult:
    """
    Top-level P2P engine entry point.

    Orchestrates:
        1. Body measurement extraction
        2. Garment measurement extraction
        3. Delta calculation per body part
        4. Visual keyword generation
        5. Physics prompt assembly
        6. Mask expansion factor calculation
    """
    result = P2PResult()

    try:
        # 1. Extract body measurements
        metadata_dict = {}
        if body_data.metadata:
            metadata_dict = {
                "gender": body_data.metadata.gender,
                "height_cm": body_data.metadata.height_cm,
                "weight_kg": body_data.metadata.weight_kg,
                "body_type": body_data.metadata.body_type,
            }
            # Include explicit measurement fields if present
            for fld in ("shoulder_width_cm", "chest_cm", "waist_cm", "hip_cm"):
                val = getattr(body_data.metadata, fld, 0)
                if val:
                    metadata_dict[fld] = val

        body_meas = extract_body_measurements(
            body_data.vertices, body_data.joints,
            metadata_dict, gemini_body_analysis,
        )

        # 2. Extract garment measurements
        garment_meas = extract_garment_measurements(
            clothing_item.analysis,
            clothing_item.size_chart,
            clothing_item.fitting_model_info,
            clothing_item.product_info,
        )

        # 3. Calculate deltas
        deltas = calculate_deltas(body_meas, garment_meas)
        if not deltas:
            logger.warning("P2P: No measurable deltas (missing measurements)")
            result.method = "fallback"
            return result

        # 4-5. Generate physics prompt
        from core.gemini_client import ClothingAnalysis
        clothing_desc = f"{clothing_item.analysis.category} ({clothing_item.analysis.name})"
        physics_prompt = generate_physics_prompt(deltas, clothing_desc)

        # 6. Mask expansion
        elasticity = clothing_item.analysis.elasticity or "none"
        mask_factor = calculate_mask_expansion(deltas, elasticity)

        # Determine confidence based on measurement source
        has_size_chart = bool(clothing_item.size_chart)
        has_body_analysis = bool(gemini_body_analysis)
        if has_size_chart and has_body_analysis:
            confidence = 0.9
            method = "measured"
        elif has_size_chart or has_body_analysis:
            confidence = 0.7
            method = "measured"
        else:
            confidence = 0.4
            method = "estimated"

        result.deltas = deltas
        result.overall_tightness = _determine_overall_tightness(deltas)
        result.physics_prompt = physics_prompt
        result.mask_expansion_factor = mask_factor
        result.confidence = confidence
        result.method = method

        logger.info(f"P2P: Generated {len(deltas)} body part deltas, "
                    f"overall={result.overall_tightness.value}, "
                    f"mask_factor={mask_factor:.2f}, "
                    f"confidence={confidence:.2f}")

    except Exception as e:
        logger.error(f"P2P engine error: {e}")
        result.method = "fallback"

    return result
