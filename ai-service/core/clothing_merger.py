"""
StyleLens V6 — Clothing Analysis Merger
Multi-image view classification and analysis merging.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.gemini_client import GeminiClient, ClothingAnalysis

logger = logging.getLogger("stylelens.merger")

# View priority for field selection (higher = more authoritative)
VIEW_PRIORITY = {
    "front": 10,
    "flat-lay": 9,
    "45-front-left": 7,
    "45-front-right": 7,
    "left-side": 5,
    "right-side": 5,
    "back": 4,
    "45-back-left": 3,
    "45-back-right": 3,
    "detail-closeup": 2,
}


@dataclass
class AnalyzedView:
    analysis: ClothingAnalysis
    view_angle: str = "front"
    shows_front: bool = True
    shows_back: bool = False
    is_closeup: bool = False
    confidence: float = 0.5
    image_bgr: np.ndarray | None = None


def classify_and_analyze(images_bgr: list[np.ndarray],
                          gemini: GeminiClient) -> list[AnalyzedView]:
    """Classify view angle and analyze each image."""
    views = []

    for img in images_bgr:
        # Classify view
        view_angle = gemini.classify_view(img)

        # Analyze clothing
        analysis = gemini.analyze_detailed_fashion(img)

        view = AnalyzedView(
            analysis=analysis,
            view_angle=view_angle,
            shows_front=view_angle in ("front", "flat-lay", "45-front-left", "45-front-right"),
            shows_back=view_angle in ("back", "45-back-left", "45-back-right"),
            is_closeup=(view_angle == "detail-closeup"),
            confidence=analysis.confidence,
            image_bgr=img,
        )
        views.append(view)

    return views


def merge_analyses(views: list[AnalyzedView]) -> ClothingAnalysis:
    """
    Merge multiple view analyses into a single ClothingAnalysis.
    Primary view provides base fields, detail views contribute design details.
    """
    if not views:
        return ClothingAnalysis()

    if len(views) == 1:
        return views[0].analysis

    # Sort by priority
    sorted_views = sorted(
        views,
        key=lambda v: VIEW_PRIORITY.get(v.view_angle, 0),
        reverse=True,
    )

    # Primary view provides base fields
    primary = sorted_views[0].analysis
    result = ClothingAnalysis(
        name=primary.name,
        category=primary.category,
        color=primary.color,
        color_hex=primary.color_hex,
        fabric=primary.fabric,
        fit_type=primary.fit_type,
        subcategory=primary.subcategory,
        neck_style=primary.neck_style,
        sleeve_type=primary.sleeve_type,
        thickness=primary.thickness,
        elasticity=primary.elasticity,
        fabric_composition=primary.fabric_composition,
        fabric_weight_gsm=primary.fabric_weight_gsm,
        transparency=primary.transparency,
        has_lining=primary.has_lining,
        care_instructions=primary.care_instructions,
        drape_style=primary.drape_style,
        hem_style=primary.hem_style,
        closure_type=primary.closure_type,
        camera_angle=primary.camera_angle,
    )

    # Union design details from all views
    all_buttons = set()
    all_pockets = set()
    all_patterns = set()
    all_style_tags = set()
    all_view_angles = set()

    max_button_count = 0
    max_pocket_count = 0

    for view in sorted_views:
        a = view.analysis
        all_view_angles.add(view.view_angle)

        # Buttons
        if a.button_count > max_button_count:
            max_button_count = a.button_count
            result.button_type = a.button_type
        if a.button_positions and a.button_positions != "N/A":
            all_buttons.add(a.button_positions)

        # Pockets
        if a.pocket_count > max_pocket_count:
            max_pocket_count = a.pocket_count
            result.pocket_type = a.pocket_type
        if a.pocket_positions and a.pocket_positions != "N/A":
            all_pockets.add(a.pocket_positions)

        # Pattern
        if a.pattern_type and a.pattern_type not in ("N/A", "solid"):
            all_patterns.add(a.pattern_type)
        if a.pattern_description:
            result.pattern_description = a.pattern_description  # last wins, ok

        # Logo (any view that sees it)
        if a.logo_text and a.logo_text != "N/A" and not result.logo_text:
            result.logo_text = a.logo_text
            result.logo_position = a.logo_position

        # Zipper (any view)
        if a.zipper_type and a.zipper_type != "N/A" and not result.zipper_type:
            result.zipper_type = a.zipper_type
            result.zipper_position = a.zipper_position

        # Seam details
        if a.seam_details and not result.seam_details:
            result.seam_details = a.seam_details

        # Wrinkles
        if a.wrinkle_intensity and a.wrinkle_intensity != "none":
            result.wrinkle_intensity = a.wrinkle_intensity
            result.wrinkle_locations = a.wrinkle_locations

        # Style tags
        if a.style_tags:
            all_style_tags.update(a.style_tags)

    result.button_count = max_button_count
    result.button_positions = ", ".join(sorted(all_buttons)) if all_buttons else "N/A"
    result.pocket_count = max_pocket_count
    result.pocket_positions = ", ".join(sorted(all_pockets)) if all_pockets else "N/A"
    result.pattern_type = ", ".join(sorted(all_patterns)) if all_patterns else "solid"
    result.style_tags = sorted(all_style_tags)
    result.view_angles_analyzed = sorted(all_view_angles)

    # Merge size info
    result.size_info = _merge_size_info(sorted_views)

    # Confidence: weighted average
    total_weight = 0
    weighted_conf = 0
    for view in sorted_views:
        w = VIEW_PRIORITY.get(view.view_angle, 1)
        weighted_conf += view.analysis.confidence * w
        total_weight += w
    result.confidence = weighted_conf / max(total_weight, 1)

    return result


def _merge_size_info(views: list[AnalyzedView]) -> dict:
    """Merge size info from multiple views with weighted averaging."""
    if not any(v.analysis.size_info for v in views):
        return {}

    merged = {}
    for view in views:
        si = view.analysis.size_info
        if not si:
            continue

        # Front/flat-lay get 2x weight, closeups excluded
        if view.is_closeup:
            continue
        weight = 2 if view.view_angle in ("front", "flat-lay") else 1

        for key, value in si.items():
            if key not in merged:
                merged[key] = {"total": 0, "weight": 0}
            try:
                merged[key]["total"] += float(value) * weight
                merged[key]["weight"] += weight
            except (ValueError, TypeError):
                merged[key] = value  # non-numeric, just keep latest

    result = {}
    for key, data in merged.items():
        if isinstance(data, dict) and "weight" in data and data["weight"] > 0:
            result[key] = round(data["total"] / data["weight"], 1)
        else:
            result[key] = data

    return result
