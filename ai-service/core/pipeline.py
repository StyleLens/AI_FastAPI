"""
StyleLens V6 — Phase 1: Avatar Pipeline
Person detection + 3D body reconstruction → body_data dict + GLB export.
"""

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from core.config import (
    DEVICE, YOLO26_CONF_THRESHOLD,
    YOLO26_ENABLED, SAM3D_BODY_ENABLED,
    FITTING_ANGLES, OUTPUT_DIR,
)
from core.loader import registry
from core.gemini_feedback import GeminiFeedbackInspector, InspectionResult
from core.sw_renderer import render_mesh

logger = logging.getLogger("stylelens.pipeline")


@dataclass
class Metadata:
    """User-provided metadata for avatar generation."""
    gender: str = "female"
    height_cm: float = 170.0
    weight_kg: float = 65.0
    bust_cup: str = ""
    age: int = 25
    body_type: str = "standard"
    # P2P: Body measurements (populated from Gemini video/photo analysis)
    shoulder_width_cm: float = 0.0
    chest_cm: float = 0.0
    waist_cm: float = 0.0
    hip_cm: float = 0.0


@dataclass
class BodyData:
    """Output of Phase 1 avatar pipeline."""
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    joints: np.ndarray | None = None
    betas: np.ndarray | None = None
    gender: str = "female"
    glb_bytes: bytes = b""
    mesh_renders: dict[int, np.ndarray] = field(default_factory=dict)
    metadata: Metadata | None = None
    person_bbox: list[float] = field(default_factory=list)
    person_image: np.ndarray | None = None
    quality_gates: list[InspectionResult] = field(default_factory=list)


def _extract_frames(video_path: str, max_frames: int = 30) -> list[np.ndarray]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _export_glb(vertices: np.ndarray, faces: np.ndarray,
                vertex_colors: np.ndarray | None = None) -> bytes:
    """Export mesh to GLB bytes."""
    import trimesh

    if vertex_colors is not None:
        # Add alpha channel if RGB
        if vertex_colors.shape[1] == 3:
            alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
            vertex_colors = np.hstack([vertex_colors, alpha])
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces,
            vertex_colors=vertex_colors,
        )
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()


async def generate_avatar(
    video_path: str | None = None,
    images: list[np.ndarray] | None = None,
    metadata: Metadata | None = None,
    inspector: GeminiFeedbackInspector | None = None,
) -> BodyData:
    """
    Phase 1: Person detection + 3D body reconstruction.

    Steps:
        1. YOLO26-L person detection
        1.5. Gemini Gate 1 — person_detection
        2. SAM 3D Body DINOv3 → mesh + body params
        2.5. Gemini Gate 3 — body_3d_reconstruction
        3. Export GLB + render previews
    """
    t0 = time.time()
    metadata = metadata or Metadata()
    body_data = BodyData(gender=metadata.gender, metadata=metadata)

    # Prepare input frames
    if video_path:
        frames = _extract_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted from video")
        person_image = frames[len(frames) // 2]  # middle frame
    elif images:
        frames = images
        person_image = images[0]
    else:
        raise ValueError("No video or images provided")

    body_data.person_image = person_image

    # ── Step 1: YOLO26-L Person Detection ──────────────────────
    logger.info("Phase 1 Step 1: Person detection (YOLO26-L)")

    if not YOLO26_ENABLED:
        logger.warning("YOLO26 not available, using full image as person region")
        h, w = person_image.shape[:2]
        body_data.person_bbox = [0, 0, w, h]
    else:
        yolo = registry.load_yolo26()
        results = yolo(person_image, conf=YOLO26_CONF_THRESHOLD, classes=[0],
                       half=False, device=DEVICE)

        if results and len(results[0].boxes) > 0:
            # Pick largest person box
            boxes = results[0].boxes
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
                    (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best_idx = areas.argmax().item()
            bbox = boxes.xyxy[best_idx].cpu().numpy().tolist()
            body_data.person_bbox = bbox
            logger.info(f"Detected person at {bbox}")
        else:
            h, w = person_image.shape[:2]
            body_data.person_bbox = [0, 0, w, h]
            logger.warning("No person detected, using full image")

        registry.unload("yolo26")

    # ── Step 1.5: Gemini Gate 1 — Person Detection ─────────────
    if inspector:
        boxes = [{"bbox": body_data.person_bbox, "class": "person", "conf": 0.9}]
        gate1 = inspector.inspect_person_detection(person_image, boxes)
        body_data.quality_gates.append(gate1)
        if not gate1.pass_check:
            logger.warning(f"Gate 1 failed: {gate1.feedback}")

    # ── Step 2: SAM 3D Body DINOv3 → 3D Mesh ──────────────────
    logger.info("Phase 1 Step 2: 3D body reconstruction (SAM 3D Body)")

    if not SAM3D_BODY_ENABLED:
        logger.warning("SAM 3D Body not available, generating default mesh")
        # Generate a simple default body mesh
        body_data.vertices = _generate_default_body(metadata)
        body_data.faces = _generate_default_faces(len(body_data.vertices))
    else:
        sam3d = registry.load_sam3d_body()
        estimator = sam3d["estimator"]

        # Crop person region
        x1, y1, x2, y2 = [int(v) for v in body_data.person_bbox]
        person_crop = person_image[y1:y2, x1:x2]
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Build bbox array for estimator (relative to crop)
        ch, cw = person_crop.shape[:2]
        bbox_array = np.array([[0, 0, cw, ch]], dtype=np.float32)

        # Run SAM 3D Body estimator
        outputs = estimator.process_one_image(
            person_rgb,
            bboxes=bbox_array,
            inference_type="body",
        )

        if outputs and len(outputs) > 0:
            out = outputs[0]
            # Extract mesh vertices and body params
            if "pred_vertices" in out:
                verts = out["pred_vertices"].copy()
                # SAM 3D Body outputs in camera space where Y,Z are negated
                # (see mhr_head.py line 340: verts[..., [1, 2]] *= -1)
                # Convert to world space (Y-up, Z-forward) for our renderer
                verts[:, 1] *= -1  # Flip Y back to up
                verts[:, 2] *= -1  # Flip Z back to forward
                body_data.vertices = verts
            if "pred_keypoints_3d" in out:
                joints = out["pred_keypoints_3d"].copy()
                joints[:, 1] *= -1
                joints[:, 2] *= -1
                body_data.joints = joints
            if "shape_params" in out:
                body_data.betas = out["shape_params"]

            # Get faces from estimator model
            if hasattr(estimator, "faces") and estimator.faces is not None:
                body_data.faces = estimator.faces
            elif body_data.vertices is not None:
                # Fallback: generate faces via Delaunay or default mesh
                body_data.faces = _generate_default_faces(len(body_data.vertices))

            logger.info(
                f"SAM 3D Body: {body_data.vertices.shape[0] if body_data.vertices is not None else 0} vertices, "
                f"{body_data.faces.shape[0] if body_data.faces is not None else 0} faces"
            )
        else:
            logger.warning("SAM 3D Body returned no results, using default mesh")
            body_data.vertices = _generate_default_body(metadata)
            body_data.faces = _generate_default_faces(len(body_data.vertices))

        registry.unload("sam3d_body")

    # ── Step 2.5: Gemini Gate 3 — 3D Reconstruction ───────────
    if inspector and body_data.vertices is not None:
        mesh_render = render_mesh(
            body_data.vertices, body_data.faces,
            angle_deg=0, resolution=512,
        )
        gate3 = inspector.inspect_body_3d_reconstruction(person_image, mesh_render)
        body_data.quality_gates.append(gate3)
        if not gate3.pass_check:
            logger.warning(f"Gate 3 failed: {gate3.feedback}")

    # ── Step 3: Export GLB + Render Previews ───────────────────
    logger.info("Phase 1 Step 3: GLB export + preview renders")

    if body_data.vertices is not None and body_data.faces is not None:
        body_data.glb_bytes = _export_glb(body_data.vertices, body_data.faces)

        for angle in FITTING_ANGLES:
            body_data.mesh_renders[angle] = render_mesh(
                body_data.vertices, body_data.faces,
                angle_deg=angle, resolution=512,
            )

    elapsed = time.time() - t0
    logger.info(f"Phase 1 complete in {elapsed:.1f}s")
    return body_data


def _generate_default_body(metadata: Metadata) -> np.ndarray:
    """Generate a simple cylindrical body mesh as fallback."""
    import trimesh
    # Create a basic humanoid shape
    body = trimesh.creation.capsule(height=1.7, radius=0.15, count=[16, 8])
    vertices = np.array(body.vertices)
    # Scale based on metadata
    height_scale = metadata.height_cm / 170.0
    vertices *= height_scale
    return vertices


def _generate_default_faces(num_vertices: int) -> np.ndarray:
    """Generate faces for default body. In practice, trimesh handles this."""
    import trimesh
    body = trimesh.creation.capsule(height=1.7, radius=0.15, count=[16, 8])
    return np.array(body.faces)
