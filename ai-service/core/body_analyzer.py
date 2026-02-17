"""
StyleLens V6 — Body Shape Analyzer
Extracts body type descriptors from 3D mesh data for SDXL prompt generation.

Uses two complementary approaches:
1. Rendered depth map silhouettes (front 0° + side 90°) for width/depth profiles
2. 3D mesh vertex analysis for volume estimation

Outputs a natural-language body description for SDXL photorealistic generation.
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger("stylelens.body_analyzer")


@dataclass
class BodyMetrics:
    """Measured body metrics from mesh analysis."""
    # Pixel-based measurements from depth maps
    body_height_px: float = 0.0
    shoulder_width_px: float = 0.0
    bust_width_px: float = 0.0
    waist_width_px: float = 0.0
    hip_width_px: float = 0.0
    # Side depth
    bust_depth_px: float = 0.0
    waist_depth_px: float = 0.0
    hip_depth_px: float = 0.0
    # Ratios
    shoulder_waist_ratio: float = 0.0
    bust_waist_ratio: float = 0.0
    hip_waist_ratio: float = 0.0
    waist_hip_ratio: float = 0.0
    # Volume indicators
    torso_cross_section_avg: float = 0.0
    volume_height_ratio: float = 0.0
    # Classification
    build_type: str = ""
    shape_type: str = ""
    waist_definition: str = ""
    depth_profile: str = ""
    # Full description for SDXL prompt
    sdxl_description: str = ""
    # Raw section data
    front_sections: dict = field(default_factory=dict)
    side_sections: dict = field(default_factory=dict)


def _extract_silhouette_profile(depth_map: np.ndarray, view_name: str = "front") -> dict:
    """Extract width profile from a rendered depth map.

    The renderer produces a gray body on bg_color=(200,200,200).
    Body pixels are darker than the background.

    For the front view, uses gap detection to separate arms from torso.
    For the side view, the body is naturally a single connected region.
    """
    gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY) if len(depth_map.shape) == 3 else depth_map
    h, w = gray.shape

    # Body pixels are darker than background (bg=200)
    body_mask = (gray < 185).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.where(body_mask > 0)
    if len(ys) < 100:
        return {}

    y_min_body = ys.min()
    y_max_body = ys.max()
    body_h = y_max_body - y_min_body

    # First pass: find the body centerline X from the full silhouette
    # Use the median X of all body pixels as the centerline
    all_body_xs = xs  # from np.where above
    center_x = float(np.median(all_body_xs))

    # Width at each row
    raw_widths = {}
    for row in range(y_min_body, y_max_body):
        cols = np.where(body_mask[row] > 0)[0]
        if len(cols) < 2:
            continue

        y_frac = (row - y_min_body) / body_h

        if view_name == "front":
            # Symmetry-based torso extraction:
            # 1. Find all segments (gap > 3px)
            # 2. Find the segment containing the centerline
            # 3. That's the torso; everything else is arms/limbs
            diffs = np.diff(cols)
            gaps = np.where(diffs > 3)[0]

            if len(gaps) == 0:
                # Single continuous region — check if arms are merged
                # Use distance from center: torso extends symmetrically
                # Arms are at the extremes
                left_cols = cols[cols <= center_x]
                right_cols = cols[cols >= center_x]
                if len(left_cols) > 0 and len(right_cols) > 0:
                    # For upper body (0.14-0.50), limit to inner region
                    # to exclude merged arms
                    if 0.14 < y_frac < 0.50:
                        # Use density: find where pixel density drops off
                        # (arm junction has lower density)
                        half_w = (cols[-1] - cols[0]) / 2
                        dist_from_center = np.abs(cols - center_x)
                        # Torso core: within 60% of half-width from center
                        core_limit = half_w * 0.70
                        core_cols = cols[dist_from_center <= core_limit]
                        if len(core_cols) >= 2:
                            width = core_cols[-1] - core_cols[0]
                        else:
                            width = cols[-1] - cols[0]
                    else:
                        width = cols[-1] - cols[0]
                else:
                    width = cols[-1] - cols[0]
            else:
                # Multiple segments: find the one containing center_x
                segments = []
                seg_start = cols[0]
                for g in gaps:
                    segments.append((seg_start, cols[g]))
                    seg_start = cols[g + 1]
                segments.append((seg_start, cols[-1]))

                # Find segment containing or closest to center_x
                center_seg = None
                min_dist = float("inf")
                for s_start, s_end in segments:
                    if s_start <= center_x <= s_end:
                        center_seg = (s_start, s_end)
                        break
                    dist = min(abs(s_start - center_x), abs(s_end - center_x))
                    if dist < min_dist:
                        min_dist = dist
                        center_seg = (s_start, s_end)

                width = center_seg[1] - center_seg[0] if center_seg else cols[-1] - cols[0]
        else:
            # Side view: single body region, just use full extent
            width = cols[-1] - cols[0]

        raw_widths[row] = {"width": width, "y_frac": y_frac}

    # Aggregate into body sections
    landmarks = {
        "head": (0.02, 0.10),
        "neck": (0.10, 0.14),
        "shoulder": (0.14, 0.20),
        "upper_chest": (0.20, 0.28),
        "bust": (0.28, 0.36),
        "lower_chest": (0.36, 0.42),
        "waist": (0.42, 0.50),
        "hip": (0.50, 0.58),
        "upper_thigh": (0.58, 0.66),
        "mid_thigh": (0.66, 0.74),
        "knee": (0.74, 0.82),
        "calf": (0.82, 0.90),
        "ankle": (0.90, 0.96),
    }

    sections = {}
    for name, (y_lo, y_hi) in landmarks.items():
        ws = [w["width"] for w in raw_widths.values()
              if y_lo <= w["y_frac"] < y_hi]
        if ws:
            sections[name] = {"median": float(np.median(ws)), "max": float(np.max(ws))}

    sections["_body_height_px"] = body_h
    return sections


def _classify_body(metrics: BodyMetrics) -> BodyMetrics:
    """Classify body type from measured metrics."""
    bh = metrics.body_height_px
    sw = metrics.shoulder_width_px
    bw = metrics.bust_width_px
    ww = metrics.waist_width_px
    hw = metrics.hip_width_px

    # Compute ratios
    if ww > 0:
        metrics.shoulder_waist_ratio = sw / ww
        metrics.bust_waist_ratio = bw / ww
        metrics.hip_waist_ratio = hw / ww
    if hw > 0:
        metrics.waist_hip_ratio = ww / hw

    # Cross-sectional area (bust + waist + hip average)
    areas = []
    for fw, sd in [(bw, metrics.bust_depth_px),
                   (ww, metrics.waist_depth_px),
                   (hw, metrics.hip_depth_px)]:
        if fw > 0 and sd > 0:
            areas.append(np.pi * (fw / 2) * (sd / 2))
    metrics.torso_cross_section_avg = np.mean(areas) if areas else 0

    if bh > 0:
        metrics.volume_height_ratio = metrics.torso_cross_section_avg / (bh ** 2)

    # === Build type (thin/slim/average/full) ===
    # Uses volume ratio: cross-section area relative to height squared
    vr = metrics.volume_height_ratio
    if vr < 0.010:
        metrics.build_type = "very slim"
    elif vr < 0.014:
        metrics.build_type = "slim"
    elif vr < 0.018:
        metrics.build_type = "average"
    elif vr < 0.024:
        metrics.build_type = "slightly full"
    else:
        metrics.build_type = "full-figured"

    # === Shape type (hourglass/pear/inverted triangle/rectangle) ===
    # Use the larger of bust and shoulder as "upper body width"
    upper_w = max(bw, sw) if bw > 0 and sw > 0 else (sw or bw)
    if upper_w > 0 and ww > 0 and hw > 0:
        upper_waist = upper_w / ww
        hip_waist = metrics.hip_waist_ratio
        upper_hip_diff = abs(upper_w - hw) / max(upper_w, hw)

        if hip_waist > 1.10 and upper_waist > 1.06:
            if upper_hip_diff < 0.15:
                metrics.shape_type = "hourglass"
            elif hw > upper_w:
                metrics.shape_type = "pear-shaped"
            else:
                metrics.shape_type = "inverted triangle"
        elif hip_waist > 1.10:
            metrics.shape_type = "pear-shaped"
        elif upper_waist > 1.10:
            metrics.shape_type = "inverted triangle"
        else:
            metrics.shape_type = "straight body"

    # === Waist definition ===
    whr = metrics.waist_hip_ratio
    if whr < 0.72:
        metrics.waist_definition = "very defined waist"
    elif whr < 0.80:
        metrics.waist_definition = "defined waist"
    elif whr < 0.88:
        metrics.waist_definition = "moderate waist"
    else:
        metrics.waist_definition = "straight waist"

    # === Depth profile (side view) ===
    if metrics.bust_depth_px > 0 and bw > 0:
        dr = metrics.bust_depth_px / bw
        if dr > 0.85:
            metrics.depth_profile = "round torso"
        elif dr > 0.60:
            metrics.depth_profile = "moderate depth"
        else:
            metrics.depth_profile = "flat profile"

    # === Compose SDXL description ===
    parts = [metrics.build_type + " build"]
    if metrics.shape_type:
        parts.append(metrics.shape_type + " body shape")
    if metrics.waist_definition and "straight" not in metrics.waist_definition:
        parts.append(metrics.waist_definition)
    if metrics.depth_profile and "moderate" not in metrics.depth_profile:
        parts.append(metrics.depth_profile + " from the side")

    metrics.sdxl_description = ", ".join(parts)
    return metrics


def analyze_body_from_renders(front_render: np.ndarray,
                               side_render: np.ndarray) -> BodyMetrics:
    """Analyze body shape from front (0°) and side (90°) depth map renders.

    Args:
        front_render: BGR image of front-facing depth map render
        side_render: BGR image of side-facing depth map render

    Returns:
        BodyMetrics with measurements and SDXL body description
    """
    front_sections = _extract_silhouette_profile(front_render, "front")
    side_sections = _extract_silhouette_profile(side_render, "side")

    metrics = BodyMetrics()
    metrics.front_sections = front_sections
    metrics.side_sections = side_sections

    # Extract key measurements
    metrics.body_height_px = front_sections.get("_body_height_px", 0)

    # Front widths — use the widest-segment approach
    # For shoulder: it's between arms, so use max to capture full span
    metrics.shoulder_width_px = front_sections.get("shoulder", {}).get("max", 0)
    metrics.bust_width_px = front_sections.get("bust", {}).get("median", 0)
    metrics.waist_width_px = front_sections.get("waist", {}).get("median", 0)
    metrics.hip_width_px = front_sections.get("hip", {}).get("median", 0)

    # Side depths
    metrics.bust_depth_px = side_sections.get("bust", {}).get("median", 0)
    metrics.waist_depth_px = side_sections.get("waist", {}).get("median", 0)
    metrics.hip_depth_px = side_sections.get("hip", {}).get("median", 0)

    # Classify
    metrics = _classify_body(metrics)

    logger.info(
        f"Body analysis: {metrics.build_type} / {metrics.shape_type} / "
        f"WHR={metrics.waist_hip_ratio:.2f} / vol_ratio={metrics.volume_height_ratio:.4f}"
    )
    logger.info(f"SDXL description: {metrics.sdxl_description}")

    return metrics


def analyze_body_from_mesh(vertices: np.ndarray, faces: np.ndarray,
                            resolution: int = 768) -> BodyMetrics:
    """Full body analysis pipeline: render mesh at 0° and 90°, then analyze.

    This is the main entry point for body analysis.
    """
    from core.sw_renderer import render_mesh

    front = render_mesh(vertices, faces, angle_deg=0, resolution=resolution,
                        straighten=True, ground_plane=False)
    side = render_mesh(vertices, faces, angle_deg=90, resolution=resolution,
                       straighten=True, ground_plane=False)

    return analyze_body_from_renders(front, side)


def bust_cup_to_body_scale(
    cup: str,
    height_cm: float,
    weight_kg: float,
    gender: str = "female",
) -> dict:
    """Convert user bust cup size + body metrics to mesh scaling factors.

    This function translates real-world bust measurements into mesh vertex scaling
    factors for sw_renderer._adjust_body_volume(). It accounts for:

    1. Cup size → absolute volume increase (based on Asian bra sizing standards)
    2. Body frame (BMI) → visual prominence adjustment
    3. Band size (estimated from height/weight) → spread vs concentration

    ### Cup Size Volume Reference (per breast, Asian standard):
    - AA cup: ~50cc  (bust-underbust diff: 7.5cm)
    - A cup:  ~100cc (diff: 10cm)
    - B cup:  ~200cc (diff: 12.5cm)
    - C cup:  ~300cc (diff: 15cm)
    - D cup:  ~400cc (diff: 17.5cm)
    - DD/E:   ~500cc (diff: 20cm)
    - F cup:  ~600cc (diff: 22.5cm)
    - G cup:  ~700cc (diff: 25cm)
    - H cup:  ~800cc+ (diff: 27.5cm)

    ### Sister Sizing (same cup volume, different band):
    - 70C = 75B = 80A (same volume)
    - Larger band → breasts spread wider → appear flatter
    - Smaller band → breasts more concentrated → appear more prominent

    ### BMI Effect on Visual Prominence:
    - Low BMI (<18.5): Less surrounding tissue → cup appears more prominent
    - Normal BMI (18.5-24): Standard appearance
    - High BMI (25-30): More surrounding tissue → cup appears less prominent
    - Very High BMI (30+): Significant tissue → cup appears much smaller visually

    Args:
        cup: Cup size string ("AA", "A", "B", "C", "D", "DD", "E", "F", "G", "H")
        height_cm: User's height in centimeters (e.g., 165.0)
        weight_kg: User's weight in kilograms (e.g., 55.0)
        gender: "male" or "female" (males return minimal/no chest scaling)

    Returns:
        Dictionary for sw_renderer._adjust_body_volume(), e.g.:
        {
            'chest': (x_scale, z_scale),
            'shoulder': (x_scale, z_scale)  # only for very large cups
        }

        The 'chest' region in sw_renderer covers body height fraction 0.62-0.76.
        x_scale affects front width, z_scale affects depth/projection.

    Example:
        >>> bust_cup_to_body_scale("C", 165, 55, "female")
        {'chest': (1.084, 1.168)}

        >>> bust_cup_to_body_scale("A", 170, 48, "female")
        {'chest': (1.036, 1.072)}
    """
    # For males, return no scaling
    if gender.lower() == "male":
        logger.info("Male gender: no bust scaling applied")
        return {}

    # Normalize cup size
    cup_upper = cup.upper().strip()

    # Cup to volume multiplier (normalized to B cup = 1.0)
    cup_multipliers = {
        "AA": 0.3,
        "A": 0.6,
        "B": 1.0,
        "C": 1.4,
        "D": 1.8,
        "DD": 2.2,
        "E": 2.2,   # DD and E are equivalent
        "F": 3.0,
        "G": 3.4,
        "H": 3.8,
    }

    if cup_upper not in cup_multipliers:
        logger.warning(f"Unknown cup size '{cup}', defaulting to B cup")
        cup_upper = "B"

    cup_multiplier = cup_multipliers[cup_upper]

    # Calculate BMI
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    # Estimate band size (underbust circumference in cm)
    # Formula: band ≈ height * 0.43 * (BMI/22)^0.3
    # This accounts for both frame size and body fat distribution
    estimated_band_cm = height_cm * 0.43 * ((bmi / 22.0) ** 0.3)

    # Map to standard Asian band sizes: 65, 70, 75, 80, 85, 90, 95
    standard_bands = [65, 70, 75, 80, 85, 90, 95]
    band_cm = min(standard_bands, key=lambda b: abs(b - estimated_band_cm))

    # Band factor: how spread out the volume is
    # Normalize to 75 as reference (typical Asian band size)
    # Larger band → more horizontal spread, less projection
    # Smaller band → more concentrated, more projection
    band_factor = band_cm / 75.0

    # BMI visual adjustment
    # Higher BMI → more surrounding tissue → dampen the visual effect of cup
    # Clamp damping between 0.7 and 1.0
    bmi_offset = max(0, bmi - 22.0)
    bmi_damping = max(0.7, 1.0 - bmi_offset * 0.03)

    # Base scaling calculation
    # x_scale (front width): subtle increase, 6% per cup multiplier unit
    # z_scale (depth/projection): more noticeable, 12% per cup multiplier unit
    base_x_increase = cup_multiplier * 0.06 * bmi_damping
    base_z_increase = cup_multiplier * 0.12 * bmi_damping

    # Adjust by band factor
    # Larger band → increase x slightly more, decrease z
    # Smaller band → decrease x slightly, increase z more
    band_x_adjustment = 1.0 + (band_factor - 1.0) * 0.3  # 30% of band deviation
    band_z_adjustment = 1.0 - (band_factor - 1.0) * 0.4  # 40% inverse of band deviation

    x_scale = 1.0 + base_x_increase * band_x_adjustment
    z_scale = 1.0 + base_z_increase * band_z_adjustment

    # Clamp to reasonable ranges (prevent extreme scaling)
    x_scale = max(0.95, min(1.25, x_scale))
    z_scale = max(0.95, min(1.40, z_scale))

    result = {
        'chest': (x_scale, z_scale)
    }

    # For very large cups (E+), also add slight shoulder scaling
    # This accounts for the fact that larger busts affect upper torso appearance
    if cup_multiplier >= 2.2:
        shoulder_x_scale = 1.0 + (cup_multiplier - 2.2) * 0.02
        shoulder_z_scale = 1.0 + (cup_multiplier - 2.2) * 0.03
        shoulder_x_scale = max(1.0, min(1.08, shoulder_x_scale))
        shoulder_z_scale = max(1.0, min(1.12, shoulder_z_scale))
        result['shoulder'] = (shoulder_x_scale, shoulder_z_scale)

    logger.info(
        f"Bust scaling: cup={cup_upper} height={height_cm}cm weight={weight_kg}kg "
        f"→ BMI={bmi:.1f} band={band_cm}cm → chest=({x_scale:.3f}, {z_scale:.3f})"
    )

    return result


def bust_cup_to_sdxl_description(
    cup: str,
    height_cm: float,
    weight_kg: float,
    gender: str = "female",
) -> str:
    """Generate SDXL-friendly body description incorporating bust size.

    This produces natural language descriptions that help SDXL generate accurate
    body proportions in photorealistic images. The description accounts for both
    the absolute cup size and how it appears in context of the user's overall build.

    ### Description Strategy:
    - Combines overall build (based on BMI) with bust prominence
    - Uses natural, photography-friendly language
    - Avoids technical terms (cup sizes, measurements)
    - Focuses on visual appearance, not numbers

    ### Build Categories (BMI-based):
    - BMI < 18.5: "slim build"
    - BMI 18.5-22: "slender build" or "average build"
    - BMI 22-25: "average build"
    - BMI 25-28: "curvy build"
    - BMI 28+: "full-figured build"

    ### Bust Size Descriptors (context-aware):
    - AA-A: "small bust" or "petite bust"
    - B-C: "moderate bust" or "average bust"
    - D-DD: "full bust" or "ample bust"
    - E-F: "large bust" or "voluptuous bust"
    - G-H: "very large bust" or "generous bust"

    The descriptor is adjusted based on BMI:
    - Low BMI: bust appears more prominent → upgrade descriptor
    - High BMI: bust appears less prominent → downgrade descriptor

    Args:
        cup: Cup size string
        height_cm: User's height in centimeters
        weight_kg: User's weight in kilograms
        gender: "male" or "female"

    Returns:
        Natural language description string for SDXL prompt, e.g.:
        - "slim build with small bust"
        - "average build with moderate bust"
        - "curvy build with full bust"
        - "full-figured with large bust"
        - "average build" (for males)

    Example:
        >>> bust_cup_to_sdxl_description("C", 165, 55, "female")
        "average build with moderate bust"

        >>> bust_cup_to_sdxl_description("F", 160, 70, "female")
        "curvy build with large bust"
    """
    # For males, only return build description
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    # Determine build type from BMI
    if bmi < 18.5:
        build = "slim build"
    elif bmi < 20:
        build = "slender build"
    elif bmi < 22:
        build = "average build"
    elif bmi < 25:
        build = "average build"
    elif bmi < 28:
        build = "curvy build"
    else:
        build = "full-figured build"

    if gender.lower() == "male":
        return build

    # Normalize cup size
    cup_upper = cup.upper().strip()

    # Base bust descriptors by cup size
    bust_descriptors = {
        "AA": "petite bust",
        "A": "small bust",
        "B": "modest bust",
        "C": "moderate bust",
        "D": "full bust",
        "DD": "ample bust",
        "E": "ample bust",
        "F": "large bust",
        "G": "voluptuous bust",
        "H": "generous bust",
    }

    if cup_upper not in bust_descriptors:
        logger.warning(f"Unknown cup size '{cup}' for SDXL description, using 'moderate bust'")
        bust_desc = "moderate bust"
    else:
        bust_desc = bust_descriptors[cup_upper]

    # Adjust bust descriptor based on BMI context
    # Low BMI → bust appears more prominent → upgrade descriptor
    # High BMI → bust appears less prominent → downgrade descriptor

    if bmi < 18.5:
        # Very slim: upgrade bust appearance
        upgrade_map = {
            "petite bust": "small bust",
            "small bust": "modest bust",
            "modest bust": "moderate bust",
            "moderate bust": "full bust",
            "full bust": "ample bust",
            "ample bust": "large bust",
            "large bust": "voluptuous bust",
            "voluptuous bust": "generous bust",
            "generous bust": "very generous bust",
        }
        bust_desc = upgrade_map.get(bust_desc, bust_desc)
    elif bmi > 28:
        # Full-figured: downgrade bust appearance (more tissue around it)
        downgrade_map = {
            "generous bust": "voluptuous bust",
            "voluptuous bust": "large bust",
            "large bust": "ample bust",
            "ample bust": "full bust",
            "full bust": "moderate bust",
            "moderate bust": "modest bust",
            "modest bust": "small bust",
            "small bust": "petite bust",
            "petite bust": "petite bust",  # can't go lower
        }
        bust_desc = downgrade_map.get(bust_desc, bust_desc)

    description = f"{build} with {bust_desc}"

    logger.info(f"SDXL bust description: cup={cup_upper} BMI={bmi:.1f} → '{description}'")

    return description
