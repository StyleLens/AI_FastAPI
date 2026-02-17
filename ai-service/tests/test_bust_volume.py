#!/usr/bin/env python3
"""
Test script for bust volume adjustment in sw_renderer.

This script demonstrates the new _adjust_bust_volume() feature and how to use
the bust_cup_scale and bust_band_factor parameters in render_mesh().

Usage:
    python test_bust_volume.py <path_to_glb_file>
"""

import sys
import logging
from pathlib import Path

import numpy as np
import trimesh
import cv2

# Add core module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sw_renderer import render_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bust_adjustment(glb_path: str):
    """Test bust volume adjustment with different cup sizes."""

    if not Path(glb_path).exists():
        logger.error(f"GLB file not found: {glb_path}")
        return

    # Load mesh
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            logger.error("No meshes found in scene")
            return
        mesh = trimesh.util.concatenate(meshes)

    vertices = mesh.vertices
    faces = mesh.faces

    logger.info(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Test different cup sizes at front angle (0°)
    cup_sizes = {
        "AA": 0.3,
        "A": 0.6,
        "B": 1.0,
        "C": 1.4,
        "D": 1.8,
        "DD": 2.2,
        "E": 2.6,
    }

    # Test different band factors
    band_factors = {
        "narrow (65)": 0.85,
        "normal (75)": 1.0,
        "wide (85)": 1.15,
    }

    # ── Test 1: Cup size variations (fixed band) ──
    logger.info("\n=== Test 1: Cup Size Variations (band=1.0) ===")
    output_row_1 = []

    for cup_name, cup_scale in cup_sizes.items():
        img = render_mesh(
            vertices, faces,
            angle_deg=0.0,
            resolution=512,
            straighten=True,
            fold_arms=True,
            close_legs=True,
            bust_cup_scale=cup_scale,
            bust_band_factor=1.0,
            gender="female",
        )
        output_row_1.append(img)
        logger.info(f"Rendered cup size {cup_name} (scale={cup_scale:.1f})")

    # Concatenate horizontally
    combined_1 = np.hstack(output_row_1)

    # ── Test 2: Band factor variations (fixed cup) ──
    logger.info("\n=== Test 2: Band Factor Variations (cup=C/1.4) ===")
    output_row_2 = []

    for band_name, band_factor in band_factors.items():
        img = render_mesh(
            vertices, faces,
            angle_deg=0.0,
            resolution=512,
            straighten=True,
            fold_arms=True,
            close_legs=True,
            bust_cup_scale=1.4,  # C cup
            bust_band_factor=band_factor,
            gender="female",
        )
        output_row_2.append(img)
        logger.info(f"Rendered band {band_name} (factor={band_factor:.2f})")

    # Pad with blank images to match row 1 length
    blank = np.full_like(output_row_2[0], 200)
    while len(output_row_2) < len(output_row_1):
        output_row_2.append(blank)

    combined_2 = np.hstack(output_row_2)

    # ── Test 3: Comparison grid (cup × band) ──
    logger.info("\n=== Test 3: Grid Comparison (3 cups × 3 bands) ===")
    grid = []

    test_cups = {"A": 0.6, "C": 1.4, "DD": 2.2}

    for cup_name, cup_scale in test_cups.items():
        row = []
        for band_name, band_factor in band_factors.items():
            img = render_mesh(
                vertices, faces,
                angle_deg=0.0,
                resolution=384,  # smaller for grid
                straighten=True,
                fold_arms=True,
                close_legs=True,
                bust_cup_scale=cup_scale,
                bust_band_factor=band_factor,
                gender="female",
            )
            row.append(img)
            logger.info(f"Grid: cup={cup_name}, band={band_name}")
        grid.append(np.hstack(row))

    combined_grid = np.vstack(grid)

    # ── Test 4: Multi-angle with bust adjustment ──
    logger.info("\n=== Test 4: Multi-Angle View (C cup, 16 angles) ===")
    angles = [i * 22.5 for i in range(16)]
    multi_angle = []

    for angle in angles:
        img = render_mesh(
            vertices, faces,
            angle_deg=angle,
            resolution=256,  # smaller for multi-view
            straighten=True,
            fold_arms=True,
            close_legs=True,
            bust_cup_scale=1.4,  # C cup
            bust_band_factor=1.0,
            gender="female",
        )
        multi_angle.append(img)

    # 4x4 grid
    rows = []
    for i in range(0, 16, 4):
        rows.append(np.hstack(multi_angle[i:i+4]))
    combined_angles = np.vstack(rows)

    # ── Save outputs ──
    output_dir = Path(__file__).parent / "bust_volume_test"
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "test1_cup_sizes.png"), combined_1)
    cv2.imwrite(str(output_dir / "test2_band_factors.png"), combined_2)
    cv2.imwrite(str(output_dir / "test3_grid.png"), combined_grid)
    cv2.imwrite(str(output_dir / "test4_multi_angle.png"), combined_angles)

    logger.info(f"\n=== All tests complete ===")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info("  - test1_cup_sizes.png: Cup size variations (AA to E)")
    logger.info("  - test2_band_factors.png: Band factor variations")
    logger.info("  - test3_grid.png: 3×3 grid (cups × bands)")
    logger.info("  - test4_multi_angle.png: 16-angle rotation (C cup)")


def test_male_vs_female(glb_path: str):
    """Compare male vs female bust adjustment."""

    if not Path(glb_path).exists():
        logger.error(f"GLB file not found: {glb_path}")
        return

    # Load mesh
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if meshes else mesh

    vertices = mesh.vertices
    faces = mesh.faces

    logger.info("\n=== Male vs Female Gender Test ===")

    # Female with bust adjustment
    img_female = render_mesh(
        vertices, faces,
        angle_deg=0.0,
        resolution=512,
        straighten=True,
        fold_arms=True,
        bust_cup_scale=1.4,  # C cup
        gender="female",
    )

    # Male (bust adjustment should be skipped)
    img_male = render_mesh(
        vertices, faces,
        angle_deg=0.0,
        resolution=512,
        straighten=True,
        fold_arms=True,
        bust_cup_scale=1.4,  # same scale, but should be ignored
        gender="male",
    )

    # No bust adjustment baseline
    img_baseline = render_mesh(
        vertices, faces,
        angle_deg=0.0,
        resolution=512,
        straighten=True,
        fold_arms=True,
        gender="female",
    )

    combined = np.hstack([img_baseline, img_female, img_male])

    output_dir = Path(__file__).parent / "bust_volume_test"
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "test5_gender_comparison.png"), combined)

    logger.info("Gender comparison saved to test5_gender_comparison.png")
    logger.info("  [Baseline (no bust)] [Female C cup] [Male (ignored)]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_bust_volume.py <path_to_glb_file>")
        print("\nExample:")
        print("  python test_bust_volume.py /path/to/avatar.glb")
        sys.exit(1)

    glb_path = sys.argv[1]

    # Run all tests
    test_bust_adjustment(glb_path)
    test_male_vs_female(glb_path)

    print("\n✓ All bust volume tests completed successfully!")
