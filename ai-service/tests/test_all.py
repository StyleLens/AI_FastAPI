"""
StyleLens V6 — Unit Tests
Tests for all V6 modules: config, loader, gemini_client, gemini_feedback,
pipeline, wardrobe, catvton_pipeline, fitting, viewer3d, main.py endpoints.
"""

import io
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Config Tests ───────────────────────────────────────────────

class TestConfig:
    def test_base_dir(self):
        from core.config import BASE_DIR
        assert BASE_DIR.exists()

    def test_model_dir(self):
        from core.config import MODEL_DIR
        assert isinstance(MODEL_DIR, Path)

    def test_device(self):
        from core.config import DEVICE
        assert DEVICE in ("mps", "cpu", "cuda")

    def test_fitting_angles(self):
        from core.config import FITTING_ANGLES
        assert FITTING_ANGLES == [0, 45, 90, 135, 180, 225, 270, 315]
        assert len(FITTING_ANGLES) == 8

    def test_stage_thresholds(self):
        from core.config import STAGE_THRESHOLDS
        assert "person_detection" in STAGE_THRESHOLDS
        assert "body_segmentation" in STAGE_THRESHOLDS
        assert "body_3d_reconstruction" in STAGE_THRESHOLDS
        assert "clothing_analysis" in STAGE_THRESHOLDS
        assert "virtual_tryon" in STAGE_THRESHOLDS
        assert "3d_visualization" in STAGE_THRESHOLDS
        assert all(0 < v <= 1 for v in STAGE_THRESHOLDS.values())

    def test_reference_models(self):
        from core.config import REFERENCE_MODELS
        assert "female" in REFERENCE_MODELS
        assert "male" in REFERENCE_MODELS
        assert "standard" in REFERENCE_MODELS["female"]
        assert "height_cm" in REFERENCE_MODELS["female"]["standard"]

    def test_catvton_params(self):
        from core.config import CATVTON_FLUX_STEPS, CATVTON_FLUX_GUIDANCE, CATVTON_FLUX_RESOLUTION
        assert CATVTON_FLUX_STEPS == 30
        assert CATVTON_FLUX_GUIDANCE == 3.5
        assert CATVTON_FLUX_RESOLUTION == 1024

    def test_hunyuan3d_params(self):
        from core.config import HUNYUAN3D_SHAPE_STEPS, HUNYUAN3D_PAINT_STEPS, HUNYUAN3D_TEXTURE_RES
        assert HUNYUAN3D_SHAPE_STEPS == 5  # turbo variant: 4-8 steps
        assert HUNYUAN3D_PAINT_STEPS == 20
        assert HUNYUAN3D_TEXTURE_RES == 4096

    def test_gemini_models(self):
        from core.config import GEMINI_MODEL_NAME, V5_GEMINI_IMAGE_MODEL
        assert GEMINI_MODEL_NAME == "gemini-3-pro-preview"
        assert V5_GEMINI_IMAGE_MODEL == "gemini-3-pro-image-preview"

    def test_fashn_classes(self):
        from core.config import FASHN_CLASSES
        assert len(FASHN_CLASSES) == 18
        assert FASHN_CLASSES[0] == "background"
        assert "upper_clothes" in FASHN_CLASSES

    def test_get_model_status(self):
        from core.config import get_model_status
        status = get_model_status()
        assert isinstance(status, dict)
        assert "yolo26" in status
        assert "hunyuan3d" in status
        assert "gemini" in status

    def test_mps_fallback_device(self):
        from core.config import MPS_FALLBACK_DEVICE, HAS_CUDA
        assert MPS_FALLBACK_DEVICE == "cpu"
        assert isinstance(HAS_CUDA, bool)

    def test_hunyuan3d_shape_only(self):
        from core.config import (
            HUNYUAN3D_ENABLED, HUNYUAN3D_PAINT_ENABLED,
            HUNYUAN3D_SHAPE_ONLY, HAS_CUDA,
        )
        if HUNYUAN3D_ENABLED and not HAS_CUDA:
            assert HUNYUAN3D_SHAPE_ONLY is True
            assert HUNYUAN3D_PAINT_ENABLED is False
        elif HUNYUAN3D_ENABLED and HAS_CUDA:
            assert HUNYUAN3D_SHAPE_ONLY is False
            assert HUNYUAN3D_PAINT_ENABLED is True

    def test_get_device_for_model(self):
        from core.config import get_device_for_model, DEVICE, MPS_FALLBACK_DEVICE
        # CPU fallback models should return fallback on MPS
        if DEVICE == "mps":
            assert get_device_for_model("sam3d_body") == MPS_FALLBACK_DEVICE
            assert get_device_for_model("hunyuan3d_shape") == MPS_FALLBACK_DEVICE
            assert get_device_for_model("hunyuan3d_paint") == MPS_FALLBACK_DEVICE
        # Non-fallback models should return primary device
        assert get_device_for_model("yolo26") == DEVICE
        assert get_device_for_model("fashn_parser") == DEVICE


# ── Loader Tests ───────────────────────────────────────────────

class TestLoader:
    def test_singleton(self):
        from core.loader import ModelRegistry
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2

    def test_is_loaded_false(self):
        from core.loader import registry
        assert not registry.is_loaded("nonexistent_model")

    def test_unload_missing(self):
        from core.loader import registry
        # Should not raise
        registry.unload("nonexistent_model")

    def test_unload_all(self):
        from core.loader import registry
        registry._models["test"] = "dummy"
        registry.unload_all()
        assert len(registry._models) == 0

    def test_unload_except(self):
        from core.loader import registry
        registry._models["keep"] = "a"
        registry._models["remove"] = "b"
        registry.unload_except("keep")
        assert "keep" in registry._models
        assert "remove" not in registry._models
        registry.unload_all()

    def test_status_report(self):
        from core.loader import registry
        report = registry.status_report()
        assert "available" in report
        assert "loaded" in report
        assert "load_times" in report
        assert "device" in report
        assert "has_cuda" in report
        assert "mps_fallback" in report

    def test_paint_loader_guard_no_cuda(self):
        from core.config import HAS_CUDA
        if not HAS_CUDA:
            from core.loader import registry
            with pytest.raises(RuntimeError, match="CUDA"):
                registry.load_hunyuan3d_paint()


# ── Gemini Client Tests ───────────────────────────────────────

class TestGeminiClient:
    def test_clothing_analysis_dataclass(self):
        from core.gemini_client import ClothingAnalysis
        ca = ClothingAnalysis(name="Test Shirt", category="top", color="blue")
        assert ca.name == "Test Shirt"
        assert ca.color_hex == "#000000"
        assert ca.confidence == 0.0

    def test_clothing_analysis_all_fields(self):
        from core.gemini_client import ClothingAnalysis
        ca = ClothingAnalysis()
        d = asdict(ca)
        assert "name" in d
        assert "button_count" in d
        assert "pattern_type" in d
        assert "style_tags" in d
        assert len(d) >= 35

    def test_validate_hex(self):
        from core.gemini_client import _validate_hex
        assert _validate_hex("#FF0000") == "#FF0000"
        assert _validate_hex("FF0000") == "#FF0000"
        assert _validate_hex("invalid") == "#000000"
        assert _validate_hex("") == "#000000"
        assert _validate_hex(None) == "#000000"

    def test_hex_to_rgb(self):
        from core.gemini_client import hex_to_rgb
        assert hex_to_rgb("#FF0000") == (255, 0, 0)
        assert hex_to_rgb("#00FF00") == (0, 255, 0)
        assert hex_to_rgb("#0000FF") == (0, 0, 255)

    def test_angle_to_text(self):
        from core.gemini_client import _angle_to_text
        assert _angle_to_text(0) == "front"
        assert _angle_to_text(180) == "back"
        assert _angle_to_text(90) == "right side"

    def test_fuzzy_match_field(self):
        from core.gemini_client import _fuzzy_match_field
        assert _fuzzy_match_field("name") == "name"
        assert _fuzzy_match_field("garment_name") == "name"
        assert _fuzzy_match_field("color_hex") == "color_hex"
        assert _fuzzy_match_field("fabric") == "fabric"
        assert _fuzzy_match_field("nonexistent_xyz") == "unknown"

    def test_fuzzy_match_normalization(self):
        from core.gemini_client import _fuzzy_match_field
        assert _fuzzy_match_field("Color Hex") == "color_hex"
        assert _fuzzy_match_field("FABRIC") == "fabric"
        assert _fuzzy_match_field("button-count") == "button_count"

    def test_body_analysis_defaults(self):
        from core.gemini_client import GeminiBodyAnalysis
        ba = GeminiBodyAnalysis()
        assert ba.height_cm == 170.0
        assert ba.gender == "female"

    def test_photo_analysis_defaults(self):
        from core.gemini_client import GeminiPhotoAnalysis
        pa = GeminiPhotoAnalysis()
        assert pa.gender == "female"
        assert pa.hair_style == "straight"

    def test_image_to_base64(self):
        from core.gemini_client import _image_to_base64
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        b64 = _image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_gemini_client_init_no_key(self):
        from core.gemini_client import GeminiClient
        with patch("core.gemini_client.GEMINI_ENABLED", False):
            client = GeminiClient()
            assert client._client is None

    def test_parse_json_direct(self):
        from core.gemini_client import GeminiClient
        client = GeminiClient.__new__(GeminiClient)
        client._client = None
        result = client._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_markdown(self):
        from core.gemini_client import GeminiClient
        client = GeminiClient.__new__(GeminiClient)
        client._client = None
        result = client._parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_embedded(self):
        from core.gemini_client import GeminiClient
        client = GeminiClient.__new__(GeminiClient)
        client._client = None
        result = client._parse_json('Some text {"a": 1} more text')
        assert result == {"a": 1}

    def test_parse_json_invalid(self):
        from core.gemini_client import GeminiClient
        client = GeminiClient.__new__(GeminiClient)
        client._client = None
        result = client._parse_json("not json at all")
        assert result == {}


# ── Gemini Feedback Tests ──────────────────────────────────────

class TestGeminiFeedback:
    def test_inspection_result_dataclass(self):
        from core.gemini_feedback import InspectionResult
        r = InspectionResult(stage="test", quality_score=0.8, pass_check=True)
        assert r.stage == "test"
        assert r.quality_score == 0.8

    def test_inspector_init_no_key(self):
        from core.gemini_feedback import GeminiFeedbackInspector
        with patch("core.gemini_feedback.GEMINI_ENABLED", False):
            inspector = GeminiFeedbackInspector()
            assert inspector._client is None

    def test_inspector_summary_empty(self):
        from core.gemini_feedback import GeminiFeedbackInspector
        with patch("core.gemini_feedback.GEMINI_ENABLED", False):
            inspector = GeminiFeedbackInspector()
            summary = inspector.get_summary()
            assert summary["total_inspections"] == 0
            assert summary["overall_score"] == 0.0

    def test_inspector_clear_log(self):
        from core.gemini_feedback import GeminiFeedbackInspector, InspectionResult
        with patch("core.gemini_feedback.GEMINI_ENABLED", False):
            inspector = GeminiFeedbackInspector()
            inspector._inspection_log.append(InspectionResult(stage="test"))
            assert len(inspector.get_inspection_log()) == 1
            inspector.clear_log()
            assert len(inspector.get_inspection_log()) == 0

    def test_parse_response(self):
        from core.gemini_feedback import GeminiFeedbackInspector
        with patch("core.gemini_feedback.GEMINI_ENABLED", False):
            inspector = GeminiFeedbackInspector()
            text = '{"quality_score": 0.85, "feedback": "Good", "issues": [], "retry_suggested": false}'
            result = inspector._parse_response(text, "person_detection")
            assert result.quality_score == 0.85
            assert result.pass_check is True  # 0.85 > 0.70 threshold
            assert result.feedback == "Good"

    def test_parse_response_fail(self):
        from core.gemini_feedback import GeminiFeedbackInspector
        with patch("core.gemini_feedback.GEMINI_ENABLED", False):
            inspector = GeminiFeedbackInspector()
            text = '{"quality_score": 0.5, "feedback": "Poor quality"}'
            result = inspector._parse_response(text, "virtual_tryon")
            assert result.quality_score == 0.5
            assert result.pass_check is False  # 0.5 < 0.80 threshold

    def test_six_gate_stages(self):
        """Verify all 6 gates exist as methods."""
        from core.gemini_feedback import GeminiFeedbackInspector
        inspector = GeminiFeedbackInspector.__new__(GeminiFeedbackInspector)
        assert hasattr(inspector, "inspect_person_detection")
        assert hasattr(inspector, "inspect_body_segmentation")
        assert hasattr(inspector, "inspect_body_3d_reconstruction")
        assert hasattr(inspector, "inspect_clothing_analysis")
        assert hasattr(inspector, "inspect_virtual_tryon")
        assert hasattr(inspector, "inspect_3d_visualization")


# ── Image Preprocess Tests ─────────────────────────────────────

class TestImagePreprocess:
    def test_preprocess_small_image(self):
        from core.image_preprocess import preprocess_clothing_image
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = preprocess_clothing_image(img, min_long_side=400, enhance_detail=False)
        assert max(result.shape[:2]) >= 400

    def test_preprocess_large_image(self):
        from core.image_preprocess import preprocess_clothing_image, MAX_LONG_SIDE
        img = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        result = preprocess_clothing_image(img, enhance_detail=False)
        assert max(result.shape[:2]) <= MAX_LONG_SIDE

    def test_preprocess_with_enhance(self):
        from core.image_preprocess import preprocess_clothing_image
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        result = preprocess_clothing_image(img, min_long_side=500, enhance_detail=True)
        assert result.shape == img.shape

    def test_batch(self):
        from core.image_preprocess import preprocess_batch
        imgs = [np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8) for _ in range(3)]
        results = preprocess_batch(imgs, min_long_side=200, enhance_detail=False)
        assert len(results) == 3

    def test_unsharp_mask(self):
        from core.image_preprocess import _unsharp_mask
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = _unsharp_mask(img)
        assert result.shape == img.shape


# ── SW Renderer Tests ──────────────────────────────────────────

class TestSWRenderer:
    def test_render_mesh(self):
        from core.sw_renderer import render_mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        img = render_mesh(vertices, faces, resolution=256)
        assert img.shape == (256, 256, 3)

    def test_render_with_colors(self):
        from core.sw_renderer import render_mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        img = render_mesh(vertices, faces, vertex_colors=colors, resolution=128)
        assert img.shape == (128, 128, 3)

    def test_render_with_rotation(self):
        from core.sw_renderer import render_mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        img0 = render_mesh(vertices, faces, angle_deg=0, resolution=128)
        img90 = render_mesh(vertices, faces, angle_deg=90, resolution=128)
        # Different angles should produce different images
        assert not np.array_equal(img0, img90)


# ── Face Identity Tests ────────────────────────────────────────

class TestFaceIdentity:
    def test_face_data_defaults(self):
        from core.face_identity import FaceData
        fd = FaceData()
        assert fd.embedding.shape == (512,)
        assert fd.det_score == 0.0

    def test_apply_face_identity_back_angle(self):
        from core.face_identity import apply_face_identity, FaceData
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        fd = FaceData()
        # Back angles should return unchanged image
        result = apply_face_identity(img, fd, None, angle_deg=180)
        assert np.array_equal(result, img)


# ── Body Deformation Tests ─────────────────────────────────────

class TestBodyDeformation:
    def test_cup_displacement(self):
        from core.body_deformation import CUP_DISPLACEMENT_M
        assert CUP_DISPLACEMENT_M["A"] < CUP_DISPLACEMENT_M["D"]
        assert all(v > 0 for v in CUP_DISPLACEMENT_M.values())

    def test_bmi_leg_scale(self):
        from core.body_deformation import _interpolate_bmi_scale
        assert _interpolate_bmi_scale(21.0) == 1.0
        assert _interpolate_bmi_scale(16.0) == 0.85
        assert _interpolate_bmi_scale(35.0) == 1.15
        # Interpolation
        mid = _interpolate_bmi_scale(22.25)
        assert 1.0 < mid < 1.04

    def test_apply_no_deformation(self):
        from core.body_deformation import apply_body_deformations
        verts = np.random.randn(100, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        joints = np.random.randn(24, 3).astype(np.float32)
        result = apply_body_deformations(verts, faces, joints, {})
        assert np.array_equal(result, verts)

    def test_vertex_normals(self):
        from core.body_deformation import _compute_vertex_normals
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        normals = _compute_vertex_normals(verts, faces)
        assert normals.shape == (4, 3)
        # Normals should be unit length
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)


# ── Clothing Merger Tests ──────────────────────────────────────

class TestClothingMerger:
    def test_view_priority(self):
        from core.clothing_merger import VIEW_PRIORITY
        assert VIEW_PRIORITY["front"] > VIEW_PRIORITY["back"]
        assert VIEW_PRIORITY["flat-lay"] > VIEW_PRIORITY["detail-closeup"]

    def test_analyzed_view(self):
        from core.clothing_merger import AnalyzedView
        from core.gemini_client import ClothingAnalysis
        view = AnalyzedView(
            analysis=ClothingAnalysis(name="Test"),
            view_angle="front",
            confidence=0.9,
        )
        assert view.view_angle == "front"
        assert view.shows_front is True

    def test_merge_single(self):
        from core.clothing_merger import merge_analyses, AnalyzedView
        from core.gemini_client import ClothingAnalysis
        views = [AnalyzedView(
            analysis=ClothingAnalysis(name="Shirt", category="top"),
            view_angle="front",
        )]
        result = merge_analyses(views)
        assert result.name == "Shirt"

    def test_merge_empty(self):
        from core.clothing_merger import merge_analyses
        result = merge_analyses([])
        assert result.name == ""

    def test_merge_multiple_views(self):
        from core.clothing_merger import merge_analyses, AnalyzedView
        from core.gemini_client import ClothingAnalysis
        views = [
            AnalyzedView(
                analysis=ClothingAnalysis(name="Shirt", category="top", color="blue",
                                         button_count=3, pocket_count=1),
                view_angle="front", confidence=0.9,
            ),
            AnalyzedView(
                analysis=ClothingAnalysis(name="Shirt", category="top", color="blue",
                                         button_count=3, pocket_count=2,
                                         logo_text="Nike"),
                view_angle="back", confidence=0.8,
            ),
        ]
        result = merge_analyses(views)
        assert result.name == "Shirt"
        assert result.pocket_count == 2  # max from both views
        assert result.logo_text == "Nike"  # from back view


# ── Multiview Tests ────────────────────────────────────────────

class TestMultiview:
    def test_clean_face_photo(self):
        from core.multiview import _clean_face_photo
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        result = _clean_face_photo(img)
        assert result.shape[0] < 500  # top/bottom cropped
        assert result.shape[1] < 400  # sides cropped

    def test_clean_face_tall_image(self):
        from core.multiview import _clean_face_photo
        # Stories format (9:16 aspect)
        img = np.random.randint(0, 255, (1600, 900, 3), dtype=np.uint8)
        result = _clean_face_photo(img)
        assert result.shape[0] < 1600

    def test_clean_face_tiny_image(self):
        from core.multiview import _clean_face_photo
        # Tiny image should return unchanged
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = _clean_face_photo(img)
        assert np.array_equal(result, img)


# ── Pipeline Tests ─────────────────────────────────────────────

class TestPipeline:
    def test_metadata_defaults(self):
        from core.pipeline import Metadata
        m = Metadata()
        assert m.gender == "female"
        assert m.height_cm == 170.0

    def test_body_data_defaults(self):
        from core.pipeline import BodyData
        bd = BodyData()
        assert bd.vertices is None
        assert bd.glb_bytes == b""
        assert bd.gender == "female"


# ── Wardrobe Tests ─────────────────────────────────────────────

class TestWardrobe:
    def test_clothing_item_defaults(self):
        from core.wardrobe import ClothingItem
        ci = ClothingItem()
        assert ci.analysis is not None
        assert ci.segmented_image is None

    def test_resolve_reference_body(self):
        from core.wardrobe import resolve_reference_body
        body = resolve_reference_body("female", "standard")
        assert "height_cm" in body
        assert body["height_cm"] == 165

    def test_resolve_reference_body_male(self):
        from core.wardrobe import resolve_reference_body
        body = resolve_reference_body("male", "bulky")
        assert body["bmi"] == 27.8


# ── CatVTON Pipeline Tests ────────────────────────────────────

class TestCatVTONPipeline:
    def test_pipeline_class_exists(self):
        from core.catvton_pipeline import CatVTONFluxPipeline
        assert CatVTONFluxPipeline is not None

    def test_pipeline_init(self):
        from core.catvton_pipeline import CatVTONFluxPipeline
        pipe = CatVTONFluxPipeline(pipe=MagicMock())
        assert pipe.pipe is not None


# ── Fitting Tests ──────────────────────────────────────────────

class TestFitting:
    def test_fitting_result_defaults(self):
        from core.fitting import FittingResult
        fr = FittingResult()
        assert len(fr.tryon_images) == 0
        assert fr.elapsed_sec == 0.0

    def test_generate_agnostic_mask_top(self):
        from core.fitting import _generate_agnostic_mask
        parse_map = np.zeros((256, 256), dtype=np.uint8)
        parse_map[50:150, 50:150] = 4  # upper_clothes
        mask = _generate_agnostic_mask(parse_map, "top")
        assert mask.max() == 255

    def test_generate_agnostic_mask_bottom(self):
        from core.fitting import _generate_agnostic_mask
        parse_map = np.zeros((256, 256), dtype=np.uint8)
        parse_map[150:250, 50:150] = 6  # pants
        mask = _generate_agnostic_mask(parse_map, "bottom")
        assert mask.max() == 255

    def test_generate_agnostic_mask_dress(self):
        from core.fitting import _generate_agnostic_mask
        parse_map = np.zeros((256, 256), dtype=np.uint8)
        parse_map[50:200, 50:150] = 7  # dress
        mask = _generate_agnostic_mask(parse_map, "dress")
        assert mask.max() == 255


# ── Viewer3D Tests ─────────────────────────────────────────────

class TestViewer3D:
    def test_viewer3d_result_defaults(self):
        from core.viewer3d import Viewer3DResult
        vr = Viewer3DResult()
        assert vr.glb_bytes == b""
        assert vr.glb_id == ""

    def test_select_best_front(self):
        from core.viewer3d import _select_best_front
        images = {
            0: np.zeros((10, 10, 3), np.uint8),
            180: np.ones((10, 10, 3), np.uint8),
        }
        result = _select_best_front(images)
        assert np.array_equal(result, images[0])

    def test_select_best_front_no_zero(self):
        from core.viewer3d import _select_best_front
        images = {
            315: np.ones((10, 10, 3), np.uint8) * 100,
            180: np.ones((10, 10, 3), np.uint8) * 200,
        }
        result = _select_best_front(images)
        assert result[0, 0, 0] == 100  # 315 is second priority

    def test_generate_placeholder_glb(self):
        from core.viewer3d import _generate_placeholder_glb
        glb = _generate_placeholder_glb()
        assert isinstance(glb, bytes)
        assert len(glb) > 0
        # GLB magic bytes
        assert glb[:4] == b"glTF"

    def test_get_glb_path_missing(self):
        from core.viewer3d import get_glb_path
        result = get_glb_path("nonexistent_id")
        assert result is None


# ── Main.py / FastAPI Tests ────────────────────────────────────

class TestMainAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "service" in data
        assert data["version"] == "6.0.0"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"

    def test_avatar_no_input(self, client):
        r = client.post("/avatar/generate")
        assert r.status_code in (400, 422)

    def test_wardrobe_no_gemini(self, client):
        # Without Gemini, should return 503
        import main
        if main.gemini is None:
            r = client.post("/wardrobe/add-image",
                           files={"image": ("test.jpg", b"\xff\xd8\xff", "image/jpeg")})
            assert r.status_code == 503

    def test_fitting_no_body(self, client):
        r = client.post("/fitting/try-on")
        assert r.status_code == 400

    def test_viewer3d_no_fitting(self, client):
        r = client.post("/viewer3d/generate")
        assert r.status_code == 400

    def test_glb_not_found(self, client):
        r = client.get("/viewer3d/model/nonexistent")
        assert r.status_code == 404

    def test_quality_report(self, client):
        r = client.get("/quality/report")
        assert r.status_code == 200


# ── P2P Engine Tests ──────────────────────────────────────────

class TestP2PEngine:
    """Tests for core/p2p_engine.py — Physics-to-Prompt delta engine."""

    def test_tightness_classification_all_levels(self):
        from core.p2p_engine import _classify_tightness, TightnessLevel
        assert _classify_tightness(-8.0) == TightnessLevel.CRITICAL_TIGHT
        assert _classify_tightness(-3.0) == TightnessLevel.TIGHT
        assert _classify_tightness(0.0) == TightnessLevel.OPTIMAL
        assert _classify_tightness(7.0) == TightnessLevel.LOOSE
        assert _classify_tightness(15.0) == TightnessLevel.VERY_LOOSE

    def test_tightness_boundary_values(self):
        from core.p2p_engine import _classify_tightness, TightnessLevel
        # At boundary -5: should be TIGHT (half-open: [-5, -2))
        assert _classify_tightness(-5.0) == TightnessLevel.TIGHT
        # At boundary -2: should be OPTIMAL (half-open: [-2, +5))
        assert _classify_tightness(-2.0) == TightnessLevel.OPTIMAL
        # At boundary +5: should be LOOSE (half-open: [+5, +10))
        assert _classify_tightness(5.0) == TightnessLevel.LOOSE
        # At boundary +10: should be VERY_LOOSE (half-open: [+10, +999))
        assert _classify_tightness(10.0) == TightnessLevel.VERY_LOOSE

    def test_visual_keywords_per_body_part(self):
        from core.p2p_engine import _get_visual_keywords, TightnessLevel
        for part in ("shoulder", "chest", "waist", "hip", "sleeve"):
            kw = _get_visual_keywords(part, TightnessLevel.CRITICAL_TIGHT)
            assert isinstance(kw, list)
            assert len(kw) >= 2, f"{part} CRITICAL_TIGHT should have ≥2 keywords"
            kw_opt = _get_visual_keywords(part, TightnessLevel.OPTIMAL)
            assert len(kw_opt) >= 1

    def test_visual_keywords_unknown_part(self):
        from core.p2p_engine import _get_visual_keywords, TightnessLevel
        kw = _get_visual_keywords("nonexistent_part", TightnessLevel.TIGHT)
        assert kw == []

    def test_calculate_deltas_mixed(self):
        from core.p2p_engine import (
            calculate_deltas, BodyMeasurements, GarmentMeasurements, TightnessLevel,
        )
        body = BodyMeasurements(shoulder_width_cm=42, chest_cm=96, waist_cm=80, hip_cm=98)
        garment = GarmentMeasurements(shoulder_cm=40, chest_cm=88, waist_cm=84, hip_cm=108)

        deltas = calculate_deltas(body, garment)
        assert len(deltas) >= 4
        part_map = {d.body_part: d for d in deltas}

        # shoulder: 40 - 42 = -2 → OPTIMAL (boundary)
        assert part_map["shoulder"].delta_cm == -2.0
        assert part_map["shoulder"].tightness == TightnessLevel.OPTIMAL

        # chest: 88 - 96 = -8 → CRITICAL_TIGHT
        assert part_map["chest"].delta_cm == -8.0
        assert part_map["chest"].tightness == TightnessLevel.CRITICAL_TIGHT

        # waist: 84 - 80 = +4 → OPTIMAL
        assert part_map["waist"].delta_cm == 4.0
        assert part_map["waist"].tightness == TightnessLevel.OPTIMAL

        # hip: 108 - 98 = +10 → VERY_LOOSE
        assert part_map["hip"].delta_cm == 10.0
        assert part_map["hip"].tightness == TightnessLevel.VERY_LOOSE

    def test_calculate_deltas_skip_zero(self):
        from core.p2p_engine import calculate_deltas, BodyMeasurements, GarmentMeasurements
        body = BodyMeasurements(chest_cm=90, waist_cm=0)
        garment = GarmentMeasurements(chest_cm=96, waist_cm=84)
        deltas = calculate_deltas(body, garment)
        parts = [d.body_part for d in deltas]
        assert "chest" in parts
        assert "waist" not in parts  # body.waist_cm == 0 → skip

    def test_generate_physics_prompt_content(self):
        from core.p2p_engine import (
            generate_physics_prompt, BodyPartDelta, TightnessLevel,
        )
        deltas = [
            BodyPartDelta(
                body_part="chest", body_cm=96, garment_cm=88,
                delta_cm=-8.0, tightness=TightnessLevel.CRITICAL_TIGHT,
                visual_keywords=["buttons strained with visible gapping"],
                prompt_fragment="chest: buttons strained with visible gapping",
            ),
            BodyPartDelta(
                body_part="waist", body_cm=80, garment_cm=84,
                delta_cm=4.0, tightness=TightnessLevel.OPTIMAL,
                visual_keywords=["proper waist fit"],
                prompt_fragment="waist: proper waist fit",
            ),
        ]
        prompt = generate_physics_prompt(deltas, "shirt")
        assert "chest" in prompt
        assert "waist" in prompt
        assert "shirt" in prompt
        assert "Δ-8.0cm" in prompt
        assert "Δ+4.0cm" in prompt

    def test_generate_physics_prompt_empty(self):
        from core.p2p_engine import generate_physics_prompt
        assert generate_physics_prompt([]) == ""

    def test_body_measurements_dataclass(self):
        from core.p2p_engine import BodyMeasurements
        bm = BodyMeasurements()
        assert bm.shoulder_width_cm == 0.0
        assert bm.chest_cm == 0.0
        assert bm.sleeve_length_cm == 0.0

    def test_garment_measurements_dataclass(self):
        from core.p2p_engine import GarmentMeasurements
        gm = GarmentMeasurements()
        assert gm.shoulder_cm == 0.0
        assert gm.chest_cm == 0.0
        assert gm.length_cm == 0.0

    def test_p2p_result_dataclass(self):
        from core.p2p_engine import P2PResult, TightnessLevel
        r = P2PResult()
        assert r.deltas == []
        assert r.overall_tightness == TightnessLevel.OPTIMAL
        assert r.physics_prompt == ""
        assert r.mask_expansion_factor == 1.0
        assert r.confidence == 0.0
        assert r.method == "fallback"

    def test_mask_expansion_elastic_tight(self):
        from core.p2p_engine import (
            calculate_mask_expansion, BodyPartDelta, TightnessLevel,
        )
        # Tight fit + high elasticity → factor < 1.0 (erode mask)
        deltas = [
            BodyPartDelta(
                body_part="chest", body_cm=96, garment_cm=88,
                delta_cm=-8.0, tightness=TightnessLevel.CRITICAL_TIGHT,
            ),
        ]
        factor = calculate_mask_expansion(deltas, "high")
        assert factor < 1.0, f"Expected <1.0 for high-elast tight, got {factor}"

    def test_mask_expansion_loose_inelastic(self):
        from core.p2p_engine import (
            calculate_mask_expansion, BodyPartDelta, TightnessLevel,
        )
        # Loose fit + no elasticity → factor > 1.0 (dilate mask)
        deltas = [
            BodyPartDelta(
                body_part="hip", body_cm=90, garment_cm=110,
                delta_cm=20.0, tightness=TightnessLevel.VERY_LOOSE,
            ),
        ]
        factor = calculate_mask_expansion(deltas, "none")
        assert factor > 1.0, f"Expected >1.0 for inelastic loose, got {factor}"

    def test_mask_expansion_empty(self):
        from core.p2p_engine import calculate_mask_expansion
        assert calculate_mask_expansion([], "none") == 1.0

    def test_extract_body_from_metadata(self):
        from core.p2p_engine import extract_body_measurements
        metadata = {"gender": "female", "height_cm": 165, "weight_kg": 55}
        bm = extract_body_measurements(None, None, metadata, None)
        assert bm.chest_cm > 0
        assert bm.waist_cm > 0
        assert bm.hip_cm > 0
        assert bm.shoulder_width_cm > 0

    def test_extract_body_from_gemini_analysis(self):
        from core.p2p_engine import extract_body_measurements
        gemini = {"shoulder_width_cm": 40, "chest_cm": 88, "waist_cm": 70, "hip_cm": 94}
        bm = extract_body_measurements(None, None, {}, gemini)
        assert bm.chest_cm == 88
        assert bm.waist_cm == 70

    def test_extract_garment_from_size_chart(self):
        from core.p2p_engine import extract_garment_measurements
        from core.gemini_client import ClothingAnalysis
        ca = ClothingAnalysis(category="top", fit_type="regular")
        size_chart = {
            "sizes": {"M": {"chest_cm": 100, "waist_cm": 88, "hip_cm": 102, "shoulder_cm": 44}}
        }
        gm = extract_garment_measurements(ca, size_chart, {}, {})
        assert gm.chest_cm == 100
        assert gm.waist_cm == 88

    def test_extract_garment_from_estimation(self):
        from core.p2p_engine import extract_garment_measurements
        from core.gemini_client import ClothingAnalysis
        ca = ClothingAnalysis(category="top", fit_type="slim")
        gm = extract_garment_measurements(ca, {}, {}, {})
        assert gm.chest_cm == 88  # from GARMENT_SIZE_ESTIMATES["top"]["slim"]
        assert gm.waist_cm == 76

    def test_run_p2p_end_to_end(self):
        from core.p2p_engine import run_p2p, TightnessLevel
        from core.pipeline import BodyData, Metadata
        from core.wardrobe import ClothingItem
        from core.gemini_client import ClothingAnalysis

        bd = BodyData()
        bd.metadata = Metadata(gender="female", height_cm=165, weight_kg=55)

        ci = ClothingItem()
        ci.analysis = ClothingAnalysis(category="top", fit_type="regular", elasticity="moderate")

        result = run_p2p(bd, ci)
        assert len(result.deltas) > 0
        assert isinstance(result.overall_tightness, TightnessLevel)
        assert isinstance(result.physics_prompt, str)
        assert result.mask_expansion_factor > 0
        assert result.method in ("measured", "estimated", "fallback")

    def test_determine_overall_tightness(self):
        from core.p2p_engine import _determine_overall_tightness, BodyPartDelta, TightnessLevel
        # All tight → overall tight
        deltas = [
            BodyPartDelta(body_part="chest", body_cm=96, garment_cm=88,
                          delta_cm=-8.0, tightness=TightnessLevel.CRITICAL_TIGHT),
            BodyPartDelta(body_part="waist", body_cm=80, garment_cm=76,
                          delta_cm=-4.0, tightness=TightnessLevel.TIGHT),
        ]
        overall = _determine_overall_tightness(deltas)
        assert overall in (TightnessLevel.CRITICAL_TIGHT, TightnessLevel.TIGHT)

    def test_visual_keyword_map_completeness(self):
        from core.p2p_engine import VISUAL_KEYWORD_MAP, TightnessLevel
        for part in ("shoulder", "chest", "waist", "hip", "sleeve"):
            assert part in VISUAL_KEYWORD_MAP
            for level in TightnessLevel:
                assert level in VISUAL_KEYWORD_MAP[part], \
                    f"Missing {part}/{level.value} in VISUAL_KEYWORD_MAP"


# ── P2P Ensemble Tests ───────────────────────────────────────

class TestP2PEnsemble:
    """Tests for core/p2p_ensemble.py — Multi-agent Gemini ensemble."""

    def test_ensemble_result_dataclass(self):
        from core.p2p_ensemble import P2PEnsembleResult
        er = P2PEnsembleResult()
        assert er.agent_a_output == {}
        assert er.agent_b_output == {}
        assert er.agent_c_output == {}
        assert er.ensemble_confidence == 0.0
        assert er.method == "ensemble"

    def test_build_fallback_result(self):
        from core.p2p_ensemble import _build_fallback_result
        from core.p2p_engine import BodyMeasurements, GarmentMeasurements
        body = BodyMeasurements(chest_cm=90, waist_cm=75, hip_cm=95, shoulder_width_cm=40)
        garment = GarmentMeasurements(chest_cm=96, waist_cm=84, hip_cm=98, shoulder_cm=42)
        result = _build_fallback_result(body, garment, "test shirt")
        assert result.method == "fallback"
        assert result.p2p_result.method == "fallback"
        assert len(result.p2p_result.deltas) > 0
        assert result.p2p_result.physics_prompt != ""

    def test_measurements_to_dict(self):
        from core.p2p_ensemble import _measurements_to_dict
        from core.p2p_engine import BodyMeasurements
        bm = BodyMeasurements(chest_cm=90, waist_cm=75, hip_cm=0)
        d = _measurements_to_dict(bm)
        assert d["chest_cm"] == 90
        assert d["waist_cm"] == 75
        assert "hip_cm" not in d  # 0 values filtered out

    @pytest.mark.asyncio
    async def test_ensemble_timeout_fallback(self):
        from core.p2p_ensemble import run_p2p_ensemble
        from core.p2p_engine import BodyMeasurements, GarmentMeasurements

        # Mock Gemini that always times out
        mock_gemini = MagicMock()
        mock_gemini._call_text = MagicMock(side_effect=lambda *a, **kw: (_ for _ in ()).throw(
            TimeoutError("Simulated timeout")
        ))
        mock_gemini._parse_json = MagicMock(return_value={})

        body = BodyMeasurements(chest_cm=90, waist_cm=75, shoulder_width_cm=40)
        garment = GarmentMeasurements(chest_cm=96, waist_cm=84, shoulder_cm=42)

        result = await run_p2p_ensemble(
            mock_gemini, body, garment, "test shirt", timeout_sec=0.1,
        )
        # Should fall back to deterministic engine
        assert result.method == "fallback"
        assert len(result.p2p_result.deltas) > 0

    def test_agent_prompts_contain_measurements(self):
        from core.p2p_ensemble import _AGENT_A_PROMPT, _AGENT_B_PROMPT, _AGENT_C_PROMPT
        assert "{body_json}" in _AGENT_A_PROMPT
        assert "{garment_json}" in _AGENT_A_PROMPT
        assert "{clothing_desc}" in _AGENT_A_PROMPT
        assert "{agent_a_json}" in _AGENT_B_PROMPT
        assert "{agent_a_json}" in _AGENT_C_PROMPT
        assert "{agent_b_json}" in _AGENT_C_PROMPT

    def test_build_ensemble_result(self):
        from core.p2p_ensemble import _build_ensemble_result
        from core.p2p_engine import BodyMeasurements, GarmentMeasurements
        agent_a = {"deltas": [{"body_part": "chest", "delta_cm": -3}]}
        agent_b = {"validation_passed": True, "corrections": []}
        agent_c = {
            "final_physics_prompt": "The shirt shows slight tension at chest",
            "validated_deltas": [
                {"body_part": "chest", "delta_cm": -3.0, "tightness": "tight",
                 "description": "slight tension at chest"},
            ],
            "overall_tightness": "tight",
            "confidence": 0.8,
        }
        body = BodyMeasurements(chest_cm=90, waist_cm=75, shoulder_width_cm=40)
        garment = GarmentMeasurements(chest_cm=87, waist_cm=84, shoulder_cm=42)
        result = _build_ensemble_result(agent_a, agent_b, agent_c, body, garment)
        assert result.method == "ensemble"
        assert result.p2p_result.method == "ensemble"
        assert "tension" in result.p2p_result.physics_prompt


# ── P2P Integration Tests ────────────────────────────────────

class TestP2PIntegration:
    """Tests verifying P2P integration into config, multiview, fitting, feedback."""

    def test_p2p_config_exists(self):
        from core.config import P2P_ENABLED, P2P_ENSEMBLE_ENABLED, P2P_BODY_PARTS
        assert isinstance(P2P_ENABLED, bool)
        assert isinstance(P2P_ENSEMBLE_ENABLED, bool)
        assert len(P2P_BODY_PARTS) == 5
        assert "shoulder" in P2P_BODY_PARTS
        assert "chest" in P2P_BODY_PARTS

    def test_p2p_tightness_thresholds(self):
        from core.config import P2P_TIGHTNESS_THRESHOLDS
        assert len(P2P_TIGHTNESS_THRESHOLDS) == 5
        for name, (lo, hi) in P2P_TIGHTNESS_THRESHOLDS.items():
            assert lo < hi, f"Threshold {name}: lo ({lo}) should be < hi ({hi})"

    def test_multiview_accepts_physics_prompt(self):
        """Verify generate_front_view_gemini accepts physics_prompt parameter."""
        import inspect
        from core.multiview import generate_front_view_gemini, generate_angle_with_reference
        sig1 = inspect.signature(generate_front_view_gemini)
        assert "physics_prompt" in sig1.parameters
        sig2 = inspect.signature(generate_angle_with_reference)
        assert "physics_prompt" in sig2.parameters

    def test_gate5_accepts_physics_prompt(self):
        """Verify inspect_virtual_tryon accepts physics_prompt parameter."""
        import inspect
        from core.gemini_feedback import GeminiFeedbackInspector
        sig = inspect.signature(GeminiFeedbackInspector.inspect_virtual_tryon)
        assert "physics_prompt" in sig.parameters

    def test_fitting_result_has_p2p(self):
        """Verify FittingResult has p2p_result field."""
        from core.fitting import FittingResult
        fr = FittingResult()
        assert hasattr(fr, "p2p_result")
        assert fr.p2p_result is None

    def test_pipeline_metadata_has_measurements(self):
        """Verify Metadata has P2P body measurement fields."""
        from core.pipeline import Metadata
        m = Metadata()
        assert hasattr(m, "shoulder_width_cm")
        assert hasattr(m, "chest_cm")
        assert hasattr(m, "waist_cm")
        assert hasattr(m, "hip_cm")
        assert m.shoulder_width_cm == 0.0

    def test_p2p_analyze_endpoint_no_body(self, client=None):
        """Verify /p2p/analyze returns 400 without Phase 1 data."""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        r = client.post("/p2p/analyze")
        assert r.status_code == 400

    def test_fitting_mask_expansion(self):
        """Verify _apply_p2p_mask_expansion function exists and works."""
        from core.fitting import _apply_p2p_mask_expansion
        import cv2
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[80:180, 80:180] = 255

        # Expansion > 1.0 → dilated (more white pixels)
        expanded = _apply_p2p_mask_expansion(mask, 1.3)
        assert expanded.sum() > mask.sum()

        # Contraction < 1.0 → eroded (fewer white pixels)
        contracted = _apply_p2p_mask_expansion(mask, 0.7)
        assert contracted.sum() < mask.sum()

        # Factor ≈ 1.0 → no change
        unchanged = _apply_p2p_mask_expansion(mask, 1.02)
        assert np.array_equal(unchanged, mask)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. Face Bank
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFaceBank:
    """Face Bank multi-reference identity management."""

    def test_face_reference_defaults(self):
        """FaceReference dataclass has expected defaults."""
        from core.face_bank import FaceReference
        ref = FaceReference(
            image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            embedding=np.zeros(512),
            face_angle="front",
            det_score=0.95,
        )
        assert ref.face_angle == "front"
        assert ref.det_score == 0.95
        assert ref.source_label == "unknown"
        assert ref.landmarks_2d is None
        assert ref.aligned_face is None

    def test_face_bank_defaults(self):
        """FaceBank dataclass has expected defaults."""
        from core.face_bank import FaceBank
        bank = FaceBank(bank_id="test_001")
        assert bank.bank_id == "test_001"
        assert len(bank.references) == 0
        assert bank.gender == "unknown"
        assert bank.mean_embedding.shape == (512,)

    def test_face_bank_angle_coverage(self):
        """FaceBank.angle_coverage counts references per angle bucket."""
        from core.face_bank import FaceBank, FaceReference
        refs = [
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512), "front", 0.9),
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512), "front", 0.8),
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512), "front_left", 0.85),
        ]
        bank = FaceBank(bank_id="t", references=refs)
        coverage = bank.angle_coverage()
        assert coverage["front"] == 2
        assert coverage["front_left"] == 1

    def test_face_bank_max_references_config(self):
        """Config has FACE_BANK_MAX_REFERENCES constant."""
        from core.config import FACE_BANK_MAX_REFERENCES
        assert FACE_BANK_MAX_REFERENCES == 11

    def test_face_bank_similarity_threshold_config(self):
        """Config has FACE_BANK_SIMILARITY_THRESHOLD constant."""
        from core.config import FACE_BANK_SIMILARITY_THRESHOLD
        assert 0.0 < FACE_BANK_SIMILARITY_THRESHOLD < 1.0

    def test_face_bank_in_model_status(self):
        """get_model_status includes face_bank key."""
        from core.config import get_model_status
        status = get_model_status()
        assert "face_bank" in status

    def test_stage_thresholds_has_face_consistency(self):
        """STAGE_THRESHOLDS includes face_consistency gate."""
        from core.config import STAGE_THRESHOLDS
        assert "face_consistency" in STAGE_THRESHOLDS
        assert STAGE_THRESHOLDS["face_consistency"] == 0.75

    def test_classify_face_angle_front(self):
        """classify_face_angle returns 'front' for balanced landmarks."""
        from core.face_bank import classify_face_angle
        # Create 106+ landmarks with nose equidistant from both ears
        landmarks = np.zeros((107, 2))
        landmarks[0] = [10, 50]    # left ear
        landmarks[32] = [90, 50]   # right ear
        landmarks[86] = [50, 50]   # nose (equidistant)
        result = classify_face_angle(landmarks)
        assert result == "front"

    def test_classify_face_angle_right(self):
        """classify_face_angle detects non-front angle when nose closer to right ear.

        When nose is close to right ear: dist_to_left is large, dist_to_right is small,
        ratio = dist_to_left / dist_to_right is high → front_left or side_left
        (face is turned so we see the left side).
        """
        from core.face_bank import classify_face_angle
        landmarks = np.zeros((107, 2))
        landmarks[0] = [10, 50]    # left ear
        landmarks[32] = [90, 50]   # right ear
        landmarks[86] = [75, 50]   # nose (closer to right → seeing left side)
        result = classify_face_angle(landmarks)
        assert result in ("front_left", "side_left")

    def test_classify_face_angle_left(self):
        """classify_face_angle detects non-front angle when nose closer to left ear.

        When nose is close to left ear: dist_to_left is small, dist_to_right is large,
        ratio = dist_to_left / dist_to_right is low → front_right or side_right
        (face is turned so we see the right side).
        """
        from core.face_bank import classify_face_angle
        landmarks = np.zeros((107, 2))
        landmarks[0] = [10, 50]    # left ear
        landmarks[32] = [90, 50]   # right ear
        landmarks[86] = [25, 50]   # nose (closer to left → seeing right side)
        result = classify_face_angle(landmarks)
        assert result in ("front_right", "side_right")

    def test_classify_face_angle_none_landmarks(self):
        """classify_face_angle defaults to 'front' when landmarks are None."""
        from core.face_bank import classify_face_angle
        assert classify_face_angle(None) == "front"

    def test_classify_face_angle_insufficient_landmarks(self):
        """classify_face_angle defaults to 'front' with too few landmarks."""
        from core.face_bank import classify_face_angle
        short = np.zeros((50, 2))
        assert classify_face_angle(short) == "front"

    def test_select_references_front_angle(self):
        """select_references_for_angle prefers 'front' refs for 0° angle."""
        from core.face_bank import FaceBank, FaceReference, select_references_for_angle
        refs = [
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512),
                          "front", 0.9, source_label="a"),
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512),
                          "front_right", 0.85, source_label="b"),
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512),
                          "side_left", 0.8, source_label="c"),
        ]
        bank = FaceBank(bank_id="t", references=refs)
        selected = select_references_for_angle(bank, 0, max_refs=2)
        assert len(selected) == 2
        # First should be the front-facing one
        assert selected[0].face_angle == "front"

    def test_select_references_back_angle_empty(self):
        """select_references_for_angle returns empty for back angles (180°)."""
        from core.face_bank import FaceBank, FaceReference, select_references_for_angle
        refs = [
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512),
                          "front", 0.9, source_label="a"),
        ]
        bank = FaceBank(bank_id="t", references=refs)
        selected = select_references_for_angle(bank, 180)
        assert selected == []

    def test_select_references_side_angle(self):
        """select_references_for_angle prefers side refs for 90° angle."""
        from core.face_bank import FaceBank, FaceReference, select_references_for_angle
        refs = [
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512),
                          "front", 0.9, source_label="a"),
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512),
                          "front_right", 0.85, source_label="b"),
            FaceReference(np.zeros((50, 50, 3), np.uint8), np.zeros(512),
                          "side_right", 0.8, source_label="c"),
        ]
        bank = FaceBank(bank_id="t", references=refs)
        selected = select_references_for_angle(bank, 90, max_refs=2)
        assert len(selected) == 2
        # front_right should come first for 90° angle
        assert selected[0].face_angle == "front_right"

    def test_compute_face_similarity_identical(self):
        """Identical embeddings have similarity 1.0."""
        from core.face_bank import compute_face_similarity
        emb = np.random.randn(512).astype(np.float32)
        sim = compute_face_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_compute_face_similarity_orthogonal(self):
        """Orthogonal embeddings have similarity near 0.0."""
        from core.face_bank import compute_face_similarity
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = compute_face_similarity(a, b)
        assert sim == 0.0

    def test_compute_face_similarity_zero_embedding(self):
        """Zero embeddings return 0.0 similarity."""
        from core.face_bank import compute_face_similarity
        zero = np.zeros(512, dtype=np.float32)
        nonzero = np.random.randn(512).astype(np.float32)
        sim = compute_face_similarity(zero, nonzero)
        assert sim == 0.0

    def test_mean_embedding_is_normalized(self):
        """FaceBankBuilder.build() produces L2-normalized mean embedding."""
        from core.face_bank import FaceBank, FaceReference
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        refs = [
            FaceReference(np.zeros((50, 50, 3), np.uint8), emb1, "front", 0.9),
            FaceReference(np.zeros((50, 50, 3), np.uint8), emb2, "front_left", 0.85),
        ]
        # Manually compute mean embedding
        mean = (emb1 + emb2) / 2
        mean = mean / np.linalg.norm(mean)

        bank = FaceBank(bank_id="t", references=refs, mean_embedding=mean)
        norm = np.linalg.norm(bank.mean_embedding)
        assert abs(norm - 1.0) < 1e-5, f"Mean embedding norm: {norm}"

    def test_multiview_accepts_face_references(self):
        """generate_front_view_gemini accepts face_references parameter."""
        import inspect
        from core.multiview import generate_front_view_gemini
        sig = inspect.signature(generate_front_view_gemini)
        assert "face_references" in sig.parameters

    def test_multiview_angle_accepts_face_references(self):
        """generate_angle_with_reference accepts face_references parameter."""
        import inspect
        from core.multiview import generate_angle_with_reference
        sig = inspect.signature(generate_angle_with_reference)
        assert "face_references" in sig.parameters

    def test_fitting_accepts_face_bank(self):
        """generate_fitting accepts face_bank parameter."""
        import inspect
        from core.fitting import generate_fitting
        sig = inspect.signature(generate_fitting)
        assert "face_bank" in sig.parameters

    def test_gemini_feedback_has_face_consistency_gate(self):
        """GeminiFeedbackInspector has inspect_face_consistency method."""
        from core.gemini_feedback import GeminiFeedbackInspector
        assert hasattr(GeminiFeedbackInspector, "inspect_face_consistency")

    def test_session_has_face_bank_field(self):
        """PipelineSession has face_bank attribute."""
        from orchestrator.session import PipelineSession
        session = PipelineSession(session_id="test", created_at=0.0)
        assert hasattr(session, "face_bank")
        assert session.face_bank is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
