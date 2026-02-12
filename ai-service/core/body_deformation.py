"""
StyleLens V6 — Body Deformation
Bust cup and leg BMI vertex deformation for SMPL-like meshes.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("stylelens.deform")

# SMPL joint indices
PELVIS = 0
L_HIP = 1
R_HIP = 2
SPINE1 = 3
L_KNEE = 4
R_KNEE = 5
SPINE2 = 6
SPINE3 = 9
NECK = 12
L_SHOULDER = 15
R_SHOULDER = 16

# Bust deformation: cup → displacement in meters
CUP_DISPLACEMENT_M = {
    "A": 0.005,
    "B": 0.012,
    "C": 0.020,
    "D": 0.030,
}

# BMI → leg thickness scale factor
BMI_LEG_SCALE = [
    (16.0, 0.85),
    (18.5, 0.92),
    (21.0, 1.00),
    (23.5, 1.04),
    (26.0, 1.08),
    (28.5, 1.11),
    (32.0, 1.14),
    (35.0, 1.15),
]


def _compute_vertex_normals(vertices: np.ndarray,
                            faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norm_len = np.maximum(norm_len, 1e-8)
    face_normals = face_normals / norm_len

    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    vn_len = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vn_len = np.maximum(vn_len, 1e-8)
    return vertex_normals / vn_len


def _laplacian_smooth(vertices: np.ndarray, faces: np.ndarray,
                      mask: np.ndarray, iterations: int = 3,
                      weight: float = 0.3) -> np.ndarray:
    """Apply Laplacian smoothing to masked vertices."""
    result = vertices.copy()

    # Build adjacency
    adj = {i: set() for i in range(len(vertices))}
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])

    for _ in range(iterations):
        new_result = result.copy()
        for vi in np.where(mask)[0]:
            neighbors = list(adj.get(vi, []))
            if not neighbors:
                continue
            avg = result[neighbors].mean(axis=0)
            new_result[vi] = result[vi] * (1 - weight) + avg * weight
        result = new_result

    return result


def _apply_bust_deformation(vertices: np.ndarray, faces: np.ndarray,
                            joints: np.ndarray, cup: str) -> np.ndarray:
    """Apply bust cup deformation via vertex normal displacement."""
    displacement = CUP_DISPLACEMENT_M.get(cup.upper(), 0.0)
    if displacement <= 0:
        return vertices

    # Find bust region: vertices near SPINE3 joint, front-facing
    spine3_pos = joints[SPINE3]
    neck_pos = joints[NECK]

    # Vertical range: between SPINE2 and NECK
    y_low = joints[SPINE2][1]
    y_high = neck_pos[1]
    y_range = y_high - y_low
    if y_range < 1e-6:
        return vertices

    # Chest center Y
    chest_y = (y_low + y_high) / 2

    # Select bust vertices: near chest height, forward-facing
    v_y = vertices[:, 1]
    in_range = (v_y > y_low) & (v_y < y_high)

    # Forward facing: z > spine3 z
    forward = vertices[:, 2] > spine3_pos[2]

    mask = in_range & forward

    if mask.sum() == 0:
        return vertices

    # Gaussian falloff from chest center
    y_dist = np.abs(v_y - chest_y) / (y_range / 2)
    falloff = np.exp(-2 * y_dist ** 2)

    # Compute normals for displacement direction
    normals = _compute_vertex_normals(vertices, faces)

    # Apply displacement with falloff
    result = vertices.copy()
    for vi in np.where(mask)[0]:
        offset = normals[vi] * displacement * falloff[vi]
        result[vi] += offset

    # Smooth the deformed region
    result = _laplacian_smooth(result, faces, mask, iterations=2, weight=0.3)

    return result


def _interpolate_bmi_scale(bmi: float) -> float:
    """Interpolate leg thickness scale from BMI table."""
    if bmi <= BMI_LEG_SCALE[0][0]:
        return BMI_LEG_SCALE[0][1]
    if bmi >= BMI_LEG_SCALE[-1][0]:
        return BMI_LEG_SCALE[-1][1]

    for i in range(len(BMI_LEG_SCALE) - 1):
        b0, s0 = BMI_LEG_SCALE[i]
        b1, s1 = BMI_LEG_SCALE[i + 1]
        if b0 <= bmi <= b1:
            t = (bmi - b0) / (b1 - b0)
            return s0 + t * (s1 - s0)

    return 1.0


def _apply_leg_thickness(vertices: np.ndarray, joints: np.ndarray,
                         bmi: float) -> np.ndarray:
    """Apply radial XZ scaling to leg vertices based on BMI."""
    scale = _interpolate_bmi_scale(bmi)
    if abs(scale - 1.0) < 1e-3:
        return vertices

    result = vertices.copy()

    # Leg regions: below hip joints
    hip_y = min(joints[L_HIP][1], joints[R_HIP][1])
    knee_y = min(joints[L_KNEE][1], joints[R_KNEE][1])

    # Process left and right legs
    for hip_idx, knee_idx in [(L_HIP, L_KNEE), (R_HIP, R_KNEE)]:
        hip_pos = joints[hip_idx]
        knee_pos = joints[knee_idx]

        # Find leg vertices
        leg_center_x = (hip_pos[0] + knee_pos[0]) / 2
        leg_mask = (
            (result[:, 1] < hip_pos[1]) &
            (result[:, 1] > knee_pos[1] - 0.1) &
            (np.abs(result[:, 0] - leg_center_x) < 0.15)
        )

        if leg_mask.sum() == 0:
            continue

        # Transition blending at hip
        for vi in np.where(leg_mask)[0]:
            vy = result[vi, 1]

            # Smooth transition from hip (no scaling) to mid-leg (full scaling)
            if vy > knee_y:
                blend = min(1.0, (hip_pos[1] - vy) / max(hip_pos[1] - knee_y, 1e-6))
            else:
                blend = 1.0

            current_scale = 1.0 + (scale - 1.0) * blend

            # Radial XZ scaling around leg center axis
            center_x = hip_pos[0] + (knee_pos[0] - hip_pos[0]) * (
                (hip_pos[1] - vy) / max(hip_pos[1] - knee_pos[1], 1e-6)
            )
            center_z = hip_pos[2] + (knee_pos[2] - hip_pos[2]) * (
                (hip_pos[1] - vy) / max(hip_pos[1] - knee_pos[1], 1e-6)
            )

            result[vi, 0] = center_x + (result[vi, 0] - center_x) * current_scale
            result[vi, 2] = center_z + (result[vi, 2] - center_z) * current_scale

    return result


def apply_body_deformations(vertices: np.ndarray, faces: np.ndarray,
                            joints: np.ndarray,
                            metadata: dict) -> np.ndarray:
    """
    Apply all body deformations based on metadata.

    Args:
        vertices: (N, 3) mesh vertices
        faces: (F, 3) face indices
        joints: (J, 3) joint positions
        metadata: dict with optional keys:
            - bust_cup: "A"/"B"/"C"/"D"
            - bmi: float
            - gender: "male"/"female"

    Returns:
        Deformed vertices (N, 3)
    """
    result = vertices.copy()

    # Bust deformation (female only)
    gender = metadata.get("gender", "female")
    cup = metadata.get("bust_cup", "")
    if gender == "female" and cup and cup.upper() in CUP_DISPLACEMENT_M:
        result = _apply_bust_deformation(result, faces, joints, cup)
        logger.info(f"Applied bust deformation: cup {cup}")

    # Leg thickness from BMI
    bmi = metadata.get("bmi", 0)
    if bmi > 0:
        result = _apply_leg_thickness(result, joints, bmi)
        logger.info(f"Applied leg thickness: BMI {bmi:.1f}")

    return result
