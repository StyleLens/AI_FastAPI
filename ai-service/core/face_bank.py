"""
StyleLens V6 — Face Bank
Multi-reference face identity management.
Manages up to 11 reference images (10 past + 1 current), extracts InsightFace
embeddings, classifies face angles, and selects the best references per target angle.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.config import (
    FACE_BANK_MAX_REFERENCES,
    FACE_BANK_MAX_REFS_PER_ANGLE,
    FACE_BANK_MIN_DET_SCORE,
    FACE_BANK_SIMILARITY_THRESHOLD,
)
from core.multiview import _clean_face_photo

logger = logging.getLogger("stylelens.face_bank")

# ── Angle selection mapping ───────────────────────────────────
# target_body_angle → list of preferred face orientation buckets
_ANGLE_FACE_MAP: dict[int, list[str]] = {
    0:   ["front", "front_left", "front_right"],
    45:  ["front", "front_right", "front_left"],
    90:  ["front_right", "side_right", "front"],
    135: [],  # back angle — no face needed
    180: [],
    225: [],
    270: ["front_left", "side_left", "front"],
    315: ["front", "front_left", "front_right"],
}


@dataclass
class FaceReference:
    """A single face reference image with metadata."""
    image_bgr: np.ndarray
    embedding: np.ndarray
    face_angle: str  # "front", "front_left", "front_right", "side_left", "side_right"
    det_score: float
    landmarks_2d: np.ndarray | None = None
    aligned_face: np.ndarray | None = None
    source_label: str = "unknown"


@dataclass
class FaceBank:
    """Collection of face reference images for one identity."""
    bank_id: str
    references: list[FaceReference] = field(default_factory=list)
    mean_embedding: np.ndarray = field(default_factory=lambda: np.zeros(512))
    gender: str = "unknown"
    created_at: float = 0.0

    def angle_coverage(self) -> dict[str, int]:
        """Count references per face angle bucket."""
        counts: dict[str, int] = {}
        for ref in self.references:
            counts[ref.face_angle] = counts.get(ref.face_angle, 0) + 1
        return counts


class FaceBankBuilder:
    """Builds a FaceBank from multiple reference images."""

    def __init__(self, face_app):
        """
        Args:
            face_app: InsightFace FaceAnalysis instance (from registry.load_insightface())
        """
        self._face_app = face_app
        self._references: list[FaceReference] = []
        self._gender: str = "unknown"

    def add_reference(self, image_bgr: np.ndarray,
                      label: str = "unknown") -> FaceReference | None:
        """Extract face from image, classify angle, and store reference.

        Returns FaceReference on success, None if no face detected or low quality.
        """
        if len(self._references) >= FACE_BANK_MAX_REFERENCES:
            logger.warning(f"Face bank full ({FACE_BANK_MAX_REFERENCES}), skipping {label}")
            return None

        # Clean social media UI artifacts
        cleaned = _clean_face_photo(image_bgr)

        try:
            faces = self._face_app.get(cleaned)
        except Exception as e:
            logger.warning(f"Face detection failed for {label}: {e}")
            return None

        if not faces:
            logger.warning(f"No face detected in {label}")
            return None

        # Pick highest confidence face
        face = max(faces, key=lambda f: f.det_score)

        if face.det_score < FACE_BANK_MIN_DET_SCORE:
            logger.warning(f"Low confidence face in {label}: {face.det_score:.2f}")
            return None

        # Extract embedding
        embedding = face.embedding.copy() if face.embedding is not None else np.zeros(512)

        # Classify face angle
        landmarks_2d = None
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            landmarks_2d = face.landmark_2d_106.copy()

        face_angle = classify_face_angle(landmarks_2d)

        # Aligned face crop
        aligned = None
        if hasattr(face, "kps") and face.kps is not None:
            from core.face_identity import _align_face
            aligned = _align_face(cleaned, face.kps.astype(np.float32), 512)

        # Gender detection
        if hasattr(face, "gender"):
            self._gender = "male" if int(face.gender) == 1 else "female"

        ref = FaceReference(
            image_bgr=cleaned,
            embedding=embedding,
            face_angle=face_angle,
            det_score=float(face.det_score),
            landmarks_2d=landmarks_2d,
            aligned_face=aligned,
            source_label=label,
        )
        self._references.append(ref)
        logger.info(f"Added face reference: {label} → {face_angle} "
                    f"(score={face.det_score:.2f})")
        return ref

    def build(self) -> FaceBank:
        """Finalize face bank: compute mean embedding, validate references.

        Raises:
            ValueError: If no references were added.
        """
        if not self._references:
            raise ValueError("Cannot build FaceBank with no references")

        # Compute mean embedding
        embeddings = np.stack([r.embedding for r in self._references])
        mean_emb = embeddings.mean(axis=0)
        # L2 normalize
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        bank = FaceBank(
            bank_id=str(uuid.uuid4())[:8],
            references=list(self._references),
            mean_embedding=mean_emb,
            gender=self._gender,
            created_at=time.time(),
        )

        logger.info(f"Built FaceBank {bank.bank_id}: "
                    f"{len(bank.references)} refs, "
                    f"coverage={bank.angle_coverage()}")
        return bank


# ── Static Utility Functions ──────────────────────────────────

def classify_face_angle(landmarks_2d: np.ndarray | None) -> str:
    """Classify face orientation from 106-point 2D landmarks.

    Uses the ratio of nose-to-left-ear vs nose-to-right-ear distances
    to determine which direction the face is oriented.

    Returns one of: "front", "front_left", "front_right",
                    "side_left", "side_right"
    """
    if landmarks_2d is None or len(landmarks_2d) < 100:
        return "front"  # Default when landmarks unavailable

    # Key landmark indices in 106-point set:
    # Nose tip ≈ index 86
    # Left ear ≈ index 0  (right side of image when face is frontal)
    # Right ear ≈ index 32 (left side of image when face is frontal)
    nose = landmarks_2d[86]
    left_ear = landmarks_2d[0]
    right_ear = landmarks_2d[32]

    dist_to_left = np.linalg.norm(nose - left_ear)
    dist_to_right = np.linalg.norm(nose - right_ear)

    # Avoid division by zero
    if dist_to_right < 1e-6:
        return "side_left"
    if dist_to_left < 1e-6:
        return "side_right"

    ratio = dist_to_left / dist_to_right

    if 0.75 <= ratio <= 1.35:
        return "front"
    elif ratio < 0.3:
        return "side_right"
    elif ratio < 0.75:
        return "front_right"
    elif ratio > 3.0:
        return "side_left"
    else:  # ratio > 1.35
        return "front_left"


def select_references_for_angle(bank: FaceBank, target_angle_deg: int,
                                 max_refs: int = FACE_BANK_MAX_REFS_PER_ANGLE
                                 ) -> list[FaceReference]:
    """Select best face references for a given body viewing angle.

    For back angles (135°, 180°, 225°) returns empty list (no face needed).
    For front/side angles, selects references matching the preferred face
    orientations, falling back to any available reference.

    Args:
        bank: FaceBank with references
        target_angle_deg: Body viewing angle (0-315, step 45)
        max_refs: Maximum references to return

    Returns:
        List of FaceReference sorted by relevance (best first)
    """
    preferred = _ANGLE_FACE_MAP.get(target_angle_deg, [])
    if not preferred:
        return []  # Back angles — no face needed

    selected: list[FaceReference] = []
    used_labels: set[str] = set()

    # Priority pass: match preferred angle buckets in order
    for pref_angle in preferred:
        for ref in bank.references:
            if ref.source_label in used_labels:
                continue
            if ref.face_angle == pref_angle:
                selected.append(ref)
                used_labels.add(ref.source_label)
                if len(selected) >= max_refs:
                    return selected

    # Fallback: any remaining reference (prefer higher det_score)
    remaining = [r for r in bank.references if r.source_label not in used_labels]
    remaining.sort(key=lambda r: r.det_score, reverse=True)
    for ref in remaining:
        selected.append(ref)
        if len(selected) >= max_refs:
            break

    return selected


def compute_face_similarity(embedding_a: np.ndarray,
                             embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings.

    Returns:
        Similarity score in range [0.0, 1.0] where 1.0 = identical
    """
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    cosine = float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))
    # Clamp to [0, 1] (negative cosine → dissimilar)
    return max(0.0, min(1.0, cosine))
