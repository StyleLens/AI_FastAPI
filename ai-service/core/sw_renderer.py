"""
StyleLens V6 — Software Mesh Renderer
CPU-based mesh renderer using painter's algorithm for preview images.
"""

import io
import logging

import cv2
import numpy as np

logger = logging.getLogger("stylelens.renderer")


def render_mesh(vertices: np.ndarray, faces: np.ndarray,
                vertex_colors: np.ndarray | None = None,
                angle_deg: float = 0.0,
                resolution: int = 512,
                bg_color: tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
    """
    Render a 3D mesh to a 2D image using painter's algorithm.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        vertex_colors: (N, 3) RGB colors per vertex, 0-255
        angle_deg: Y-axis rotation angle in degrees
        resolution: Output image resolution (square)
        bg_color: Background color (B, G, R)

    Returns:
        BGR image (resolution, resolution, 3)
    """
    # Rotate around Y axis
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a],
    ])
    verts = vertices @ rot.T

    # Center and scale to fit
    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    center = (v_min + v_max) / 2
    extent = (v_max - v_min).max()
    if extent < 1e-6:
        extent = 1.0

    verts = (verts - center) / extent
    margin = 0.05
    scale = resolution * (1 - 2 * margin) / 2
    offset = resolution / 2

    # Project to 2D (orthographic)
    px = (verts[:, 0] * scale + offset).astype(np.int32)
    py = ((-verts[:, 1]) * scale + offset).astype(np.int32)
    pz = verts[:, 2]

    # Default colors if none provided
    if vertex_colors is None:
        vertex_colors = np.full((len(vertices), 3), 180, dtype=np.uint8)

    # Compute face depths for sorting
    face_depths = pz[faces].mean(axis=1)
    order = np.argsort(face_depths)  # painter's: far to near

    # Simple directional lighting
    face_verts = verts[faces]  # (F, 3, 3)
    e1 = face_verts[:, 1] - face_verts[:, 0]
    e2 = face_verts[:, 2] - face_verts[:, 0]
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
        # Combine all meshes
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
