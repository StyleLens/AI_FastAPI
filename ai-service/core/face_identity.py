"""
StyleLens V6 — Face Identity Preservation
InsightFace-based face detection, embedding, and swap for try-on images.
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.config import (
    V5_FACE_SWAP_BLEND_RADIUS,
    V5_FACE_SWAP_SCALE,
    V5_FACE_CROP_SIZE,
)

logger = logging.getLogger("stylelens.face")


@dataclass
class FaceData:
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512))
    landmark_2d_106: np.ndarray | None = None
    landmark_3d_68: np.ndarray | None = None
    bbox: np.ndarray | None = None
    aligned_face: np.ndarray | None = None
    age: int = 0
    gender: int = 0  # 0=female, 1=male
    det_score: float = 0.0
    kps: np.ndarray | None = None  # 5-point keypoints


def extract_face_data(image_bgr: np.ndarray, face_app,
                      crop_size: int = V5_FACE_CROP_SIZE) -> FaceData | None:
    """Extract face data from image using InsightFace."""
    try:
        faces = face_app.get(image_bgr)
        if not faces:
            logger.warning("No face detected in image")
            return None

        # Pick highest score face
        face = max(faces, key=lambda f: f.det_score)

        data = FaceData(
            embedding=face.embedding.copy() if face.embedding is not None else np.zeros(512),
            det_score=float(face.det_score),
            age=int(face.age) if hasattr(face, "age") else 0,
            gender=int(face.gender) if hasattr(face, "gender") else 0,
        )

        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            data.landmark_2d_106 = face.landmark_2d_106.copy()
        if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
            data.landmark_3d_68 = face.landmark_3d_68.copy()
        if hasattr(face, "bbox") and face.bbox is not None:
            data.bbox = face.bbox.copy()
        if hasattr(face, "kps") and face.kps is not None:
            data.kps = face.kps.copy()

        # Align face crop
        if data.kps is not None:
            data.aligned_face = _align_face(image_bgr, data.kps, crop_size)

        return data

    except Exception as e:
        logger.error(f"Face extraction failed: {e}")
        return None


def _align_face(image_bgr: np.ndarray, kps_5: np.ndarray,
                output_size: int = 512) -> np.ndarray:
    """Align face using 5-point landmarks with affine transform."""
    # Standard 5-point reference for 512x512
    ref_pts = np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097607, 0.82469196],
        [0.63151696, 0.82325089],
    ], dtype=np.float32) * output_size

    src_pts = kps_5.astype(np.float32)
    M = cv2.estimateAffinePartial2D(src_pts, ref_pts)[0]
    if M is None:
        # Fallback: center crop
        h, w = image_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        half = output_size // 2
        y1 = max(0, cy - half)
        x1 = max(0, cx - half)
        return cv2.resize(image_bgr[y1:y1+output_size, x1:x1+output_size],
                          (output_size, output_size))

    aligned = cv2.warpAffine(image_bgr, M, (output_size, output_size),
                              borderMode=cv2.BORDER_REPLICATE)
    return aligned


def _create_face_blend_mask(kps: np.ndarray, image_shape: tuple,
                            blur_radius: int = V5_FACE_SWAP_BLEND_RADIUS) -> np.ndarray:
    """Create a soft elliptical mask centered on face keypoints."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    center = kps.mean(axis=0).astype(int)

    # Estimate face size from keypoints
    face_w = int(np.linalg.norm(kps[0] - kps[1]) * 1.8)
    face_h = int(face_w * 1.3)

    cv2.ellipse(mask, tuple(center), (face_w // 2, face_h // 2),
                0, 0, 360, 1.0, -1)

    # Gaussian blur for soft edges
    ksize = blur_radius * 2 + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), blur_radius / 2)
    return mask


def blend_face(target: np.ndarray, source: np.ndarray,
               mask: np.ndarray) -> np.ndarray:
    """Blend source face onto target using mask with color transfer."""
    # Simple histogram-style color transfer in LAB space
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Only transfer in mask region
    mask_3d = mask[:, :, np.newaxis]
    mask_bool = mask > 0.5

    if mask_bool.sum() < 100:
        return target

    # Match mean/std of source to target in mask region
    for c in range(3):
        src_vals = source_lab[:, :, c][mask_bool]
        tgt_vals = target_lab[:, :, c][mask_bool]

        if len(src_vals) == 0 or len(tgt_vals) == 0:
            continue

        src_mean, src_std = src_vals.mean(), max(src_vals.std(), 1e-6)
        tgt_mean, tgt_std = tgt_vals.mean(), max(tgt_vals.std(), 1e-6)

        source_lab[:, :, c] = (source_lab[:, :, c] - src_mean) * (tgt_std / src_std) + tgt_mean

    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    source_corrected = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

    # Alpha blend
    result = target.astype(np.float32) * (1 - mask_3d) + \
             source_corrected.astype(np.float32) * mask_3d
    return np.clip(result, 0, 255).astype(np.uint8)


def swap_face(target_image: np.ndarray, source_face_data: FaceData,
              face_app, blend_radius: int = V5_FACE_SWAP_BLEND_RADIUS,
              scale: float = V5_FACE_SWAP_SCALE) -> np.ndarray:
    """Swap face from source onto target image."""
    # Detect face in target
    target_faces = face_app.get(target_image)
    if not target_faces:
        logger.warning("No face in target image for swap")
        return target_image

    target_face = max(target_faces, key=lambda f: f.det_score)

    if target_face.kps is None or source_face_data.aligned_face is None:
        return target_image

    # Warp aligned source face to target face position
    target_kps = target_face.kps.astype(np.float32)

    # Reference points matching the aligned crop
    ref_size = source_face_data.aligned_face.shape[0]
    ref_pts = np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097607, 0.82469196],
        [0.63151696, 0.82325089],
    ], dtype=np.float32) * ref_size

    M = cv2.estimateAffinePartial2D(ref_pts, target_kps)[0]
    if M is None:
        return target_image

    h, w = target_image.shape[:2]
    warped = cv2.warpAffine(source_face_data.aligned_face, M, (w, h),
                             borderMode=cv2.BORDER_REPLICATE)

    # Create blend mask
    mask = _create_face_blend_mask(target_kps, (h, w), blend_radius)

    return blend_face(target_image, warped, mask)


def apply_face_identity(output_image: np.ndarray, face_data: FaceData,
                        face_app, angle_deg: float = 0.0) -> np.ndarray:
    """Apply face identity preservation based on viewing angle."""
    # Only apply face swap for front/near-front angles
    if angle_deg in (0, 315, 45):
        # Front and near-front: full face swap
        return swap_face(output_image, face_data, face_app)
    elif angle_deg in (270, 90):
        # Side view: partial face swap with lower blend
        result = swap_face(output_image, face_data, face_app,
                          blend_radius=V5_FACE_SWAP_BLEND_RADIUS * 2)
        # Reduce intensity for side views
        alpha = 0.5
        return cv2.addWeighted(output_image, 1 - alpha, result, alpha, 0)
    else:
        # Back views: no face swap needed
        return output_image
