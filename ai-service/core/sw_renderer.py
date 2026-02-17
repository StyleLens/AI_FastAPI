"""
StyleLens V6 — Software Mesh Renderer (v2: posture correction + ground plane)
CPU-based mesh renderer using painter's algorithm for depth maps.
"""

import io
import logging

import cv2
import numpy as np

logger = logging.getLogger("stylelens.renderer")


def _straighten_posture(verts: np.ndarray) -> np.ndarray:
    """Correct forward lean, side lean, and turtle neck.

    Two-pass correction:
    1. Global: align feet→head axis to vertical Y
    2. Local: fix neck/head forward protrusion (turtle neck)
    """
    n = len(verts)
    if n < 50:
        return verts

    body_h = np.ptp(verts[:, 1])
    if body_h < 1e-6:
        return verts

    # ── Pass 1: Global spine alignment ──
    sorted_y = np.sort(verts[:, 1])
    k = max(5, n // 20)
    head_mask = verts[:, 1] >= sorted_y[-k]
    feet_mask = verts[:, 1] <= sorted_y[k - 1]

    head_center = verts[head_mask].mean(axis=0)
    feet_center = verts[feet_mask].mean(axis=0)

    spine = head_center - feet_center
    spine_len = np.linalg.norm(spine)
    if spine_len > 1e-6:
        spine_unit = spine / spine_len
        target = np.array([0.0, 1.0, 0.0])
        axis = np.cross(spine_unit, target)
        axis_len = np.linalg.norm(axis)

        if axis_len > 1e-6:
            axis = axis / axis_len
            cos_angle = np.clip(np.dot(spine_unit, target), -1.0, 1.0)
            angle = np.clip(np.arccos(cos_angle), -np.radians(20), np.radians(20))

            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            verts = (verts - feet_center) @ R.T + feet_center

            lean_deg = np.degrees(angle)
            if abs(lean_deg) > 1.0:
                logger.info(f"Global posture correction: {lean_deg:.1f}°")

    # ── Pass 2: Turtle neck correction ──
    # Recalculate height after global correction
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    body_h = y_max - y_min

    # Define body regions by height fraction
    # Reference: normal human posture has head directly above shoulders/chest
    # Use mid-torso (0.65-0.78) as the "spine line" reference
    ref_mask = (verts[:, 1] > y_min + body_h * 0.65) & (verts[:, 1] < y_min + body_h * 0.78)
    upper_mask = verts[:, 1] >= y_min + body_h * 0.78

    if ref_mask.sum() < 10 or upper_mask.sum() < 10:
        return verts

    ref_z = verts[ref_mask, 2].mean()
    ref_x = verts[ref_mask, 0].mean()

    # For everything above 78% height, measure deviation from reference
    upper_z = verts[upper_mask, 2].mean()
    upper_x = verts[upper_mask, 0].mean()

    z_offset = upper_z - ref_z  # offset from torso line
    x_offset = upper_x - ref_x

    # Correct if neck/head deviates >1% of body height from torso line
    z_threshold = body_h * 0.01

    if abs(z_offset) > z_threshold or abs(x_offset) > z_threshold:
        transition_y = y_min + body_h * 0.78

        # Also compute reference Z distribution for compression
        ref_z_min = verts[ref_mask, 2].min()
        ref_z_max = verts[ref_mask, 2].max()
        ref_z_range = ref_z_max - ref_z_min

        for i in range(n):
            if verts[i, 1] > transition_y:
                t = (verts[i, 1] - transition_y) / (y_max - transition_y + 1e-8)
                t = min(1.0, t)
                # Ease-in-out: t^2*(3-2t)
                t_smooth = t * t * (3.0 - 2.0 * t)

                # 1) Shift center to align with torso (120% overcorrection)
                verts[i, 2] -= z_offset * t_smooth * 1.20
                verts[i, 0] -= x_offset * t_smooth * 1.20

                # 2) Compress Z spread of neck/head toward center
                #    This prevents the silhouette from having a protruding chin
                local_z_center = ref_z
                z_dev = verts[i, 2] - local_z_center
                compress = 0.15 * t_smooth  # compress up to 15% at top
                verts[i, 2] -= z_dev * compress

        logger.info(
            f"Turtle neck correction: Z={z_offset:+.4f}, X={x_offset:+.4f} "
            f"(120% shift + 15% Z-compression from H=0.78)"
        )

    return verts


def _close_legs(verts: np.ndarray) -> np.ndarray:
    """Close A-pose legs to natural standing position.

    SMPL-X meshes have legs spread in A-pose. This causes an unnatural
    "mannequin" look in generated images. We rotate each leg inward
    around the hip pivot to bring feet closer together.

    Strategy:
    1. Detect leg gap at each height slice (H=0.04 to H=0.42)
    2. Identify left/right leg vertices via gap center
    3. Rotate each leg inward around hip pivot (H≈0.42)
    """
    n = len(verts)
    if n < 100:
        return verts

    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    body_h = y_max - y_min
    if body_h < 1e-6:
        return verts

    x_center = np.median(verts[:, 0])

    # ── Detect leg gap at multiple heights ──
    leg_zone_top = 0.42  # crotch
    leg_zone_bot = 0.02  # just above feet

    # Verify gap exists (legs are actually spread)
    test_frac = 0.20  # mid-shin
    test_mask = ((verts[:, 1] >= y_min + body_h * (test_frac - 0.02)) &
                 (verts[:, 1] < y_min + body_h * (test_frac + 0.02)))
    if test_mask.sum() < 10:
        return verts

    test_xs = np.sort(verts[test_mask, 0])
    test_diffs = np.diff(test_xs)
    max_gap = test_diffs.max() if len(test_diffs) > 0 else 0
    if max_gap < 0.03:
        logger.info("Legs already close together, skipping leg closing")
        return verts

    # ── Hip pivot points (where legs connect to pelvis) ──
    hip_y = y_min + body_h * leg_zone_top
    # Find hip boundary X positions at H=0.40-0.44
    hip_mask = ((verts[:, 1] >= y_min + body_h * 0.40) &
                (verts[:, 1] < y_min + body_h * 0.44))
    if hip_mask.sum() < 10:
        return verts

    hip_xs = np.sort(verts[hip_mask, 0])
    hip_diffs = np.diff(hip_xs)
    hip_gap_idx = np.argmax(hip_diffs)
    hip_gap_center = (hip_xs[hip_gap_idx] + hip_xs[hip_gap_idx + 1]) / 2

    # Left and right hip pivot X positions (inner edges of hip)
    left_hip_x = np.median(verts[hip_mask & (verts[:, 0] < hip_gap_center), 0])
    right_hip_x = np.median(verts[hip_mask & (verts[:, 0] >= hip_gap_center), 0])

    left_pivot = np.array([left_hip_x, hip_y, np.median(verts[:, 2])])
    right_pivot = np.array([right_hip_x, hip_y, np.median(verts[:, 2])])

    # ── Identify leg vertices using gap detection at each height ──
    # Only classify vertices as "leg" if they are clearly on one side of
    # a detectable gap. This avoids misclassifying crotch vertices.
    is_leg = np.zeros(n, dtype=bool)
    leg_side = np.zeros(n, dtype=int)  # -1=left, 1=right

    for i in range(n):
        y_frac = (verts[i, 1] - y_min) / body_h
        if y_frac > leg_zone_top or y_frac < leg_zone_bot:
            continue

        # Only mark as leg if clearly away from center gap
        # Skip vertices near the gap center (crotch/inner thigh)
        dist_from_gap = abs(verts[i, 0] - hip_gap_center)
        if dist_from_gap < 0.02:
            continue  # too close to center, skip

        is_leg[i] = True
        if verts[i, 0] < hip_gap_center:
            leg_side[i] = -1  # left leg
        else:
            leg_side[i] = 1   # right leg

    leg_count = is_leg.sum()
    if leg_count < 20:
        logger.info("No significant leg vertices found to close")
        return verts

    # ── Calculate closing displacement ──
    # Measure current foot spread
    foot_mask = verts[:, 1] < y_min + body_h * 0.06
    left_foot = foot_mask & (verts[:, 0] < hip_gap_center)
    right_foot = foot_mask & (verts[:, 0] >= hip_gap_center)
    if left_foot.sum() < 3 or right_foot.sum() < 3:
        return verts
    foot_left_x = np.median(verts[left_foot, 0])
    foot_right_x = np.median(verts[right_foot, 0])
    current_spread = foot_right_x - foot_left_x

    # Target: natural standing, feet ~15% body height apart
    target_spread = body_h * 0.15
    if current_spread <= target_spread:
        logger.info(f"Legs already close enough (spread={current_spread:.4f})")
        return verts

    # Per-leg X displacement at foot level
    spread_to_close = (current_spread - target_spread) / 2

    logger.info(f"Leg closing: spread {current_spread:.4f} -> {target_spread:.4f}, "
                f"close_per_side={spread_to_close:.4f}")

    # ── Move leg vertices inward using X-translation with height blend ──
    # At hip (top): zero displacement. At feet (bottom): full displacement.
    # Skip crotch region (H=0.36-0.42) to avoid mesh tearing near inner thigh.
    safe_top = 0.36  # stop applying shift above this to protect crotch
    for i in range(n):
        if not is_leg[i]:
            continue

        side = leg_side[i]
        y_frac = (verts[i, 1] - y_min) / body_h

        # Blend: 0 at safe_top (0.36), 1 at feet (0.02)
        blend = np.clip((safe_top - y_frac) / (safe_top - leg_zone_bot), 0, 1)
        # Smooth ease-in-out for natural transition
        blend = blend * blend * (3.0 - 2.0 * blend)

        # Move inward: left leg moves right (+x), right leg moves left (-x)
        x_shift = spread_to_close * blend * (-side)
        verts[i, 0] += x_shift

    logger.info(f"Legs closed: {leg_count} vertices ({leg_count/n*100:.1f}%)")
    return verts


def _fold_arms_down(verts: np.ndarray) -> np.ndarray:
    """Fold T-pose arms down to sides for natural standing silhouette.

    SMPL-X meshes have arms spread in T/A-pose. This causes massive silhouette
    differences between front (wide) and side (thin) views in depth maps,
    leading SDXL to generate inconsistent body shapes across angles.

    Strategy — two-phase approach:
    Phase A: Scan H=0.46–0.66 where clear arm-torso gaps exist → detect boundaries
    Phase B: Extrapolate boundaries to H=0.66–0.78 (shoulder zone, no gaps)
    Then move all arm vertices inward to hug the torso.
    """
    n = len(verts)
    if n < 100:
        return verts

    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    body_h = y_max - y_min
    if body_h < 1e-6:
        return verts

    x_center = np.median(verts[:, 0])

    # ── Phase A: detect arm boundaries at clear-gap heights ──
    # Scan from H=0.46 to H=0.66 where gaps are obvious
    slice_h = 0.04
    boundary_data = []  # [(frac, left_bound, right_bound)]

    for frac_center in np.arange(0.46, 0.68, slice_h):
        lo = frac_center - slice_h / 2
        hi = frac_center + slice_h / 2
        mask = ((verts[:, 1] >= y_min + body_h * lo) &
                (verts[:, 1] < y_min + body_h * hi))
        if mask.sum() < 20:
            continue

        xs = np.sort(verts[mask, 0])
        diffs = np.diff(xs)
        gap_order = np.argsort(diffs)[::-1]

        left_b = right_b = None
        for gi in gap_order[:10]:
            if diffs[gi] < 0.04:
                break
            gc = (xs[gi] + xs[gi + 1]) / 2
            if gc < x_center and left_b is None:
                left_b = xs[gi]  # inner edge of gap (torso side)
            elif gc > x_center and right_b is None:
                right_b = xs[gi + 1]  # inner edge of gap (torso side)
            if left_b is not None and right_b is not None:
                break

        if left_b is not None and right_b is not None:
            boundary_data.append((frac_center, left_b, right_b))

    if len(boundary_data) < 2:
        logger.info("Could not detect arm-torso boundaries (insufficient gaps)")
        return verts

    # ── Phase B: extrapolate boundaries to shoulder zone ──
    # At detected heights, left boundary gets closer to center going up
    # (arms merge into shoulder). Extrapolate linearly.
    bd = np.array(boundary_data)  # (N, 3): frac, left_b, right_b
    # Sort by height fraction
    bd = bd[bd[:, 0].argsort()]

    # Fit linear trend for boundaries
    fracs = bd[:, 0]
    left_bounds = bd[:, 1]   # negative X values
    right_bounds = bd[:, 2]  # positive X values

    # Linear fit: boundary_x = slope * frac + intercept
    # As frac increases (higher), boundary gets closer to center
    left_slope = np.polyfit(fracs, left_bounds, 1)[0]
    left_at_top = np.polyval(np.polyfit(fracs, left_bounds, 1), 0.78)

    right_slope = np.polyfit(fracs, right_bounds, 1)[0]
    right_at_top = np.polyval(np.polyfit(fracs, right_bounds, 1), 0.78)

    # Shoulder reference: at H=0.80, full width = torso + no arms
    ref_mask = ((verts[:, 1] >= y_min + body_h * 0.78) &
                (verts[:, 1] < y_min + body_h * 0.82))
    if ref_mask.sum() < 5:
        return verts
    shoulder_left = verts[ref_mask, 0].min()
    shoulder_right = verts[ref_mask, 0].max()

    def get_boundary(frac):
        """Get torso boundary at a given height fraction."""
        if frac >= 0.78:
            return shoulder_left, shoulder_right
        elif frac >= fracs[-1]:
            # Interpolate between top detected boundary and shoulder
            t = (frac - fracs[-1]) / (0.78 - fracs[-1])
            lb = left_bounds[-1] * (1 - t) + shoulder_left * t
            rb = right_bounds[-1] * (1 - t) + shoulder_right * t
            return lb, rb
        elif frac <= fracs[0]:
            # Below lowest detection: use lowest detected boundary
            return left_bounds[0], right_bounds[0]
        else:
            # Interpolate within detected range
            lb = np.interp(frac, fracs, left_bounds)
            rb = np.interp(frac, fracs, right_bounds)
            return lb, rb

    # ── Identify arm vertices and rotate them down ──
    # Strategy: find arm vertices at each height, then rotate them around
    # the shoulder pivot so they hang naturally at the sides.

    arm_zone_top = 0.78
    arm_zone_bot = 0.44

    # Shoulder pivot points (where arms connect to torso)
    shoulder_y = y_min + body_h * arm_zone_top
    left_pivot = np.array([shoulder_left, shoulder_y, np.median(verts[:, 2])])
    right_pivot = np.array([shoulder_right, shoulder_y, np.median(verts[:, 2])])

    is_arm = np.zeros(n, dtype=bool)

    # First pass: identify all arm vertices
    for i in range(n):
        y_frac = (verts[i, 1] - y_min) / body_h
        if y_frac > arm_zone_top or y_frac < arm_zone_bot:
            continue
        lb, rb = get_boundary(y_frac)
        if verts[i, 0] < lb:
            is_arm[i] = True
        elif verts[i, 0] > rb:
            is_arm[i] = True

    arm_count = is_arm.sum()
    if arm_count < 20:
        logger.info("No significant arm vertices found to fold")
        return verts

    # Second pass: rotate arm vertices around shoulder pivot
    # T-pose arms are roughly horizontal (0°). Target: ~75° down (nearly vertical).
    # Rotation angle: 75° clockwise in the X-Y plane (around Z axis at pivot).
    target_angle = np.radians(65)  # degrees to rotate arms down

    for i in range(n):
        if not is_arm[i]:
            continue

        # Determine which arm (left or right)
        if verts[i, 0] < x_center:
            pivot = left_pivot
            sign = -1  # left arm: rotate clockwise (arm goes down-left)
        else:
            pivot = right_pivot
            sign = 1   # right arm: rotate clockwise (arm goes down-right)

        # Vector from pivot to vertex (in X-Y plane)
        dx = verts[i, 0] - pivot[0]
        dy = verts[i, 1] - pivot[1]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            continue

        # Current angle from pivot (T-pose: mostly horizontal, angle ≈ 0 or π)
        current_angle = np.arctan2(dy, dx)

        # Blend factor: vertices far from pivot rotate more
        # Near pivot (shoulder area): gentle rotation
        # Far from pivot (hand/forearm): full rotation
        max_arm_len = body_h * 0.38  # arm is roughly 38% of body height
        dist_ratio = np.clip(dist / max_arm_len, 0, 1)
        blend = dist_ratio * dist_ratio  # quadratic ease-in

        # Rotate toward hanging position
        # For right arm: current angle ≈ 0 (horizontal right), target ≈ -75° (down-right)
        # For left arm: current angle ≈ π (horizontal left), target ≈ π+75° (down-left)
        rotation = -target_angle * blend * sign

        new_angle = current_angle + rotation
        verts[i, 0] = pivot[0] + dist * np.cos(new_angle)
        verts[i, 1] = pivot[1] + dist * np.sin(new_angle)

        # Compress Z spread of arms (keep them flat against body)
        z_med = np.median(verts[:, 2])
        z_dev = verts[i, 2] - z_med
        verts[i, 2] -= z_dev * 0.35 * blend

    logger.info(f"Arms rotated down: {arm_count} vertices ({arm_count/n*100:.1f}%), "
                f"target_angle={np.degrees(target_angle):.0f}°")

    return verts


def _adjust_bust_volume(
    verts: np.ndarray,
    cup_scale: float = 1.0,
    band_factor: float = 1.0,
    gender: str = "female",
) -> np.ndarray:
    """Apply anatomically-aware bust volume adjustment to mesh vertices.

    Uses vectorized NumPy operations for performance.
    Targets front hemisphere of bust region with Gaussian weighting.

    Args:
        verts: (N, 3) vertex positions (already straightened/arms-folded)
        cup_scale: Volume multiplier normalized to B-cup=1.0
                   AA=0.3, A=0.6, B=1.0, C=1.4, D=1.8, DD=2.2, F=3.0, G=3.4, H=3.8
        band_factor: Band width relative to reference (75cm=1.0)
        gender: "female" or "male". Males get no adjustment.

    Returns:
        Modified vertex array with bust volume adjusted.
    """
    if gender.lower() == "male":
        logger.info("Bust adjustment skipped (male gender)")
        return verts

    n = len(verts)
    if n < 50:
        return verts

    # Compute body dimensions
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    body_h = y_max - y_min
    if body_h < 1e-6:
        return verts

    x_center = np.median(verts[:, 0])
    z_median = np.median(verts[:, 2])
    z_range = np.ptp(verts[:, 2])
    x_range = np.ptp(verts[:, 0])

    if z_range < 1e-6 or x_range < 1e-6:
        return verts

    # ── Bust region parameters (tightly focused on CHEST ONLY) ──
    bust_h_min = 0.65   # underbust line (well above navel)
    bust_h_max = 0.76   # upper chest (below clavicle)
    bust_peak = 0.71    # nipple line (high enough to avoid belly entirely)

    # ── Compute normalized coordinates for ALL vertices ──
    y_frac = (verts[:, 1] - y_min) / body_h
    z_norm = (verts[:, 2] - z_median) / z_range
    x_dist = np.abs(verts[:, 0] - x_center)
    x_norm = x_dist / (x_range * 0.35)  # tighter width (35% each side, exclude arms)

    # ── Build mask: vertices in bust region ──
    in_height = (y_frac >= bust_h_min) & (y_frac <= bust_h_max)
    in_front = z_norm > 0  # front hemisphere only
    in_width = x_norm <= 1.0  # exclude extreme sides (arms)
    mask = in_height & in_front & in_width

    if not np.any(mask):
        logger.info("Bust adjustment: no vertices in target region")
        return verts

    # ── Height weight: ASYMMETRIC Gaussian bell curve ──
    # Upper bust (above peak): σ = 0.035 (gradual fade to decolletage)
    # Lower bust (below peak): σ = 0.018 (VERY sharp cutoff to avoid belly)
    y_masked = y_frac[mask]
    sigma_upper = 0.035
    sigma_lower = 0.018
    sigma = np.where(y_masked >= bust_peak, sigma_upper, sigma_lower)
    h_weight = np.exp(-0.5 * ((y_masked - bust_peak) / sigma) ** 2)

    # ── Depth weight: stronger for front-facing vertices ──
    d_weight = np.clip(z_norm[mask] * 3.0, 0, 1)  # steeper ramp, faster to full weight

    # ── Width weight: quadratic falloff from center ──
    w_weight = (1.0 - x_norm[mask]) ** 2  # squared for tighter focus on center

    # ── Total blend weight ──
    total_weight = h_weight * d_weight * w_weight

    # ── Scale relative to B-cup baseline ──
    # cup_scale=1.0 (B-cup) should produce NO change
    # cup_scale=0.6 (A-cup) should REDUCE bust (-0.4 relative)
    # cup_scale=2.2 (E-cup) should INCREASE bust (+1.2 relative)
    relative_scale = cup_scale - 1.0  # difference from B-cup

    # ── Z offset (forward projection) — PRIMARY visible change ──
    z_coeff = 0.085  # balanced: visible differences without artifacts
    z_offset = relative_scale * z_coeff * body_h * total_weight
    z_offset /= max(band_factor, 0.8)  # narrower band → more projection

    # ── X offset (lateral spread) — subtle ──
    x_coeff = 0.028  # balanced: visible but not distorting
    x_offset = relative_scale * x_coeff * body_h * total_weight
    x_offset *= min(band_factor, 1.2)  # wider band → more spread

    # ── Apply adjustments ──
    verts_mod = verts.copy()

    # Z: push forward
    verts_mod[mask, 2] += z_offset

    # X: widen symmetrically around center
    x_sign = np.sign(verts_mod[mask, 0] - x_center)
    x_sign[x_sign == 0] = 1.0
    verts_mod[mask, 0] += x_offset * x_sign

    modified_count = int(np.sum(total_weight > 0.01))

    logger.info(
        f"Bust volume adjustment: cup_scale={cup_scale:.2f} (relative={relative_scale:+.2f}), "
        f"band_factor={band_factor:.2f}, modified {modified_count} vertices ({modified_count/n*100:.1f}%), "
        f"z_coeff={z_coeff}, max_z_offset={np.max(np.abs(z_offset)):.4f}"
    )

    return verts_mod


def _adjust_body_volume(verts: np.ndarray, scale_factors: dict | None = None) -> np.ndarray:
    """Adjust mesh body volume by scaling cross-sections at different heights.

    Scales the X and Z axes around the body centerline to make the silhouette
    wider or narrower at specific body regions. This changes the depth map
    silhouette to better match the user's actual body shape.

    Args:
        verts: (N, 3) vertex positions (already straightened)
        scale_factors: dict mapping body region names to (x_scale, z_scale) tuples.
            Regions: 'shoulder', 'chest', 'waist', 'hip', 'thigh'
            Default scales are 1.0 (no change).
            Example: {'waist': (1.15, 1.10), 'hip': (1.20, 1.15)}
    """
    if not scale_factors:
        return verts

    n = len(verts)
    if n < 50:
        return verts

    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    body_h = y_max - y_min
    if body_h < 1e-6:
        return verts

    # Define body region height ranges (fraction of body height)
    region_ranges = {
        'head':     (0.86, 1.00),
        'neck':     (0.82, 0.86),
        'shoulder': (0.76, 0.82),
        'chest':    (0.62, 0.76),
        'waist':    (0.52, 0.62),
        'hip':      (0.42, 0.52),
        'thigh':    (0.28, 0.42),
        'knee':     (0.18, 0.28),
        'calf':     (0.08, 0.18),
    }

    # Build a per-vertex scale array by interpolating between regions
    x_scales = np.ones(n)
    z_scales = np.ones(n)

    # For each specified region, apply scaling with smooth transitions
    for region, (xs, zs) in scale_factors.items():
        if region not in region_ranges:
            continue
        frac_lo, frac_hi = region_ranges[region]
        y_lo = y_min + body_h * frac_lo
        y_hi = y_min + body_h * frac_hi
        y_mid = (y_lo + y_hi) / 2
        y_range = (y_hi - y_lo) / 2

        for i in range(n):
            y_frac = (verts[i, 1] - y_min) / body_h
            if frac_lo - 0.04 <= y_frac <= frac_hi + 0.04:
                # Smooth bell-shaped weight centered on the region
                dist = abs(verts[i, 1] - y_mid) / (y_range + body_h * 0.04)
                weight = max(0.0, 1.0 - dist * dist)  # quadratic falloff
                x_scales[i] = max(x_scales[i], 1.0 + (xs - 1.0) * weight)
                z_scales[i] = max(z_scales[i], 1.0 + (zs - 1.0) * weight)

    # Compute body centerline at each height slice
    x_center = np.median(verts[:, 0])
    z_center = np.median(verts[:, 2])

    # Apply scaling around centerline
    verts[:, 0] = x_center + (verts[:, 0] - x_center) * x_scales
    verts[:, 2] = z_center + (verts[:, 2] - z_center) * z_scales

    applied = {k: v for k, v in scale_factors.items() if k in region_ranges}
    if applied:
        logger.info(f"Body volume adjustment: {applied}")

    return verts


def render_mesh(vertices: np.ndarray, faces: np.ndarray,
                vertex_colors: np.ndarray | None = None,
                angle_deg: float = 0.0,
                resolution: int = 512,
                bg_color: tuple[int, int, int] = (200, 200, 200),
                straighten: bool = True,
                ground_plane: bool = True,
                body_scale: dict | None = None,
                fold_arms: bool = False,
                close_legs: bool = False,
                bust_cup_scale: float | None = None,
                bust_band_factor: float | None = None,
                gender: str = "female") -> np.ndarray:
    """
    Render a 3D mesh to a 2D image using painter's algorithm.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        vertex_colors: (N, 3) RGB colors per vertex, 0-255
        angle_deg: Y-axis rotation angle in degrees
        resolution: Output image resolution (square)
        bg_color: Background color (B, G, R)
        straighten: Apply posture correction (fix forward lean / side lean)
        ground_plane: Draw a ground shadow/plane under the feet
        body_scale: dict mapping body region to (x_scale, z_scale) tuples
                    for volume adjustment. E.g. {'waist': (1.15, 1.10)}
        fold_arms: Fold T-pose arms down to natural position
        close_legs: Close A-pose legs to natural standing position
        bust_cup_scale: Cup size multiplier (AA=0.3, A=0.6, B=1.0, C=1.4, D=1.8, DD=2.2,
                        E=2.6, F=3.0, G=3.4, H=3.8). None = no bust adjustment.
        bust_band_factor: Band width factor (>1.0 = wider band, <1.0 = narrower band).
                          None defaults to 1.0 when bust_cup_scale is set.
        gender: "female" or "male". Bust adjustment only applied to females.

    Returns:
        BGR image (resolution, resolution, 3)
    """
    # Auto-detect upside-down mesh (SMPL convention: head should have higher Y)
    verts = vertices.copy()
    n = len(verts)
    if n >= 20:
        sorted_y = np.sort(verts[:, 1])
        k = max(1, n // 10)
        # Check which end is "narrower" (head-like) by X spread
        top_mask = verts[:, 1] >= sorted_y[-k]
        bot_mask = verts[:, 1] <= sorted_y[k - 1]
        top_x_spread = np.ptp(verts[top_mask, 0]) if top_mask.sum() > 0 else 0
        bot_x_spread = np.ptp(verts[bot_mask, 0]) if bot_mask.sum() > 0 else 0
        # Head is narrower. If narrow end has LOWER Y, mesh is upside-down.
        if top_x_spread > bot_x_spread * 1.2:
            verts[:, 1] = -verts[:, 1]
            logger.info("Mesh Y-axis flipped (head was at bottom)")

    # Posture correction: straighten forward lean and side lean
    if straighten:
        verts = _straighten_posture(verts)

    # Fold T-pose arms to natural position (before volume adjustment)
    if fold_arms:
        verts = _fold_arms_down(verts)

    # Close A-pose legs to natural standing position
    if close_legs:
        verts = _close_legs(verts)

    # Body volume adjustment (scale cross-sections per region)
    if body_scale:
        verts = _adjust_body_volume(verts, body_scale)

    # Bust volume adjustment (cup size specific, anatomically aware)
    if bust_cup_scale is not None and gender.lower() == "female":
        verts = _adjust_bust_volume(
            verts,
            cup_scale=bust_cup_scale,
            band_factor=bust_band_factor if bust_band_factor is not None else 1.0,
            gender=gender
        )

    # Center X/Z on body midline (keep Y as-is for feet-anchored rendering)
    x_center = (verts[:, 0].min() + verts[:, 0].max()) / 2
    z_center = (verts[:, 2].min() + verts[:, 2].max()) / 2
    verts[:, 0] -= x_center
    verts[:, 2] -= z_center

    # Rotate around Y axis
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a],
    ])
    verts = verts @ rot.T

    # Feet-anchored scaling: place feet at bottom of frame, head near top
    y_min = verts[:, 1].min()  # feet
    y_max = verts[:, 1].max()  # head
    y_range = y_max - y_min
    x_range = verts[:, 0].max() - verts[:, 0].min()
    body_extent = max(y_range, x_range)
    if body_extent < 1e-6:
        body_extent = 1.0

    # Normalize so body fits in [-0.5, 0.5] range
    verts_norm = verts.copy()
    verts_norm[:, 0] = verts[:, 0] / body_extent
    verts_norm[:, 1] = (verts[:, 1] - y_min) / body_extent  # feet at 0, head at ~1
    verts_norm[:, 2] = verts[:, 2] / body_extent

    # Layout: top margin (5%), body (80%), ground zone (5%), bottom margin (10%)
    top_margin = 0.05
    body_zone = 0.80
    ground_y_frac = top_margin + body_zone  # 0.85 of resolution from top

    scale = resolution * body_zone
    # px: centered horizontally
    px = (verts_norm[:, 0] * scale + resolution / 2).astype(np.int32)
    # py: feet at ground_y_frac, head near top_margin
    py = (resolution * top_margin + (1.0 - verts_norm[:, 1]) * scale).astype(np.int32)
    pz = verts_norm[:, 2]

    # Default colors if none provided
    if vertex_colors is None:
        vertex_colors = np.full((len(vertices), 3), 180, dtype=np.uint8)

    # Compute face depths for sorting
    face_depths = pz[faces].mean(axis=1)
    order = np.argsort(face_depths)  # painter's: far to near

    # Simple directional lighting
    face_verts_3d = verts_norm[faces]  # (F, 3, 3)
    e1 = face_verts_3d[:, 1] - face_verts_3d[:, 0]
    e2 = face_verts_3d[:, 2] - face_verts_3d[:, 0]
    normals = np.cross(e1, e2)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len = np.maximum(norm_len, 1e-8)
    normals = normals / norm_len

    # Two-point lighting for better depth perception
    light_dir_main = np.array([0.3, 0.5, 0.8])
    light_dir_main = light_dir_main / np.linalg.norm(light_dir_main)
    light_dir_fill = np.array([-0.5, 0.3, 0.4])
    light_dir_fill = light_dir_fill / np.linalg.norm(light_dir_fill)

    main_light = np.clip(normals @ light_dir_main, 0.0, 1.0)
    fill_light = np.clip(normals @ light_dir_fill, 0.0, 1.0) * 0.3
    ambient = 0.25
    brightness = np.clip(main_light * 0.65 + fill_light + ambient, 0.2, 1.0)

    # Render
    image = np.full((resolution, resolution, 3), bg_color, dtype=np.uint8)

    # Draw ground plane / shadow (vectorized)
    if ground_plane:
        ground_py = int(resolution * ground_y_frac)
        feet_x = px[verts_norm[:, 1] < 0.05]
        if len(feet_x) > 2:
            feet_cx = int(np.median(feet_x))
            feet_spread = max(30, int((feet_x.max() - feet_x.min()) * 0.8))
        else:
            feet_cx = resolution // 2
            feet_spread = resolution // 4

        # Elliptical shadow via OpenCV + alpha blend
        shadow_h = max(8, int(resolution * 0.02))
        shadow_mask = np.zeros((resolution, resolution), dtype=np.float32)
        cv2.ellipse(shadow_mask,
                     (feet_cx, ground_py),
                     (feet_spread, shadow_h),
                     0, 0, 360, 0.3, -1, cv2.LINE_AA)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), shadow_h * 0.6)
        shadow_region = shadow_mask > 0.005
        if shadow_region.any():
            dark = np.array(bg_color, dtype=np.float32) * 0.55
            for c in range(3):
                ch = image[:, :, c].astype(np.float32)
                ch[shadow_region] = (
                    ch[shadow_region] * (1 - shadow_mask[shadow_region])
                    + dark[c] * shadow_mask[shadow_region]
                )
                image[:, :, c] = ch.astype(np.uint8)

        # Subtle ground line
        cv2.line(image,
                 (max(0, feet_cx - feet_spread - 20), ground_py),
                 (min(resolution - 1, feet_cx + feet_spread + 20), ground_py),
                 tuple(int(c * 0.75) for c in bg_color), 1, cv2.LINE_AA)

    # Paint faces
    for idx in order:
        f = faces[idx]
        pts = np.array([[px[f[0]], py[f[0]]],
                        [px[f[1]], py[f[1]]],
                        [px[f[2]], py[f[2]]]], dtype=np.int32)

        # Average vertex color for face
        face_color = vertex_colors[f].mean(axis=0) * brightness[idx]
        face_color = np.clip(face_color, 0, 255).astype(np.uint8)
        color = (int(face_color[0]), int(face_color[1]), int(face_color[2]))

        cv2.fillConvexPoly(image, pts, color)

    return image


def _extract_vertex_colors(mesh) -> np.ndarray | None:
    """Extract vertex colors from a trimesh mesh."""
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vc = mesh.visual.vertex_colors
        if vc is not None and len(vc) > 0:
            return np.array(vc[:, :3], dtype=np.uint8)  # RGB only

    # Try texture sampling
    if (hasattr(mesh, "visual") and hasattr(mesh.visual, "uv")
            and mesh.visual.uv is not None):
        try:
            from PIL import Image
            material = mesh.visual.material
            if hasattr(material, "image") and material.image is not None:
                tex = np.array(material.image)
                uv = mesh.visual.uv
                h, w = tex.shape[:2]
                u = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
                v = np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
                return tex[v, u, :3].astype(np.uint8)
        except Exception:
            pass
    return None


def render_mesh_from_glb(glb_bytes: bytes,
                          angle_deg: float = 0.0,
                          resolution: int = 512) -> np.ndarray:
    """Render a GLB model to a 2D image."""
    import trimesh

    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")

    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return np.full((resolution, resolution, 3), 200, dtype=np.uint8)
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene

    vertex_colors = _extract_vertex_colors(combined)
    return render_mesh(
        combined.vertices, combined.faces,
        vertex_colors=vertex_colors,
        angle_deg=angle_deg,
        resolution=resolution,
    )
