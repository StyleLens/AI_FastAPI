import cv2
import numpy as np
import pytest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.pipeline import BodyData
from core.p2p_engine import (
    BodyMeasurements,
    BodyPartDelta,
    GarmentMeasurements,
    TightnessLevel,
    classify_body_shape,
    generate_fit_report,
)
from orchestrator.routes.avatar import router as avatar_router
from orchestrator.session import SessionManager
from orchestrator.worker_client import WorkerClient


def _create_test_app() -> FastAPI:
    app = FastAPI()
    app.state.session_manager = SessionManager(max_sessions=5, ttl_sec=300)
    app.state.inspector = None
    app.state.worker_client = WorkerClient()
    app.include_router(avatar_router)
    return app


@pytest.fixture
def test_app():
    return _create_test_app()


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


@pytest.fixture
def session_manager(test_app):
    return test_app.state.session_manager


def _jpeg_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _delta(part: str, delta_cm: float, tightness: TightnessLevel) -> BodyPartDelta:
    return BodyPartDelta(
        body_part=part,
        body_cm=80.0,
        garment_cm=80.0 + delta_cm,
        delta_cm=delta_cm,
        tightness=tightness,
    )


def _report(deltas, shape_key="rectangle", shape_ko="직사각형형"):
    return generate_fit_report(
        BodyMeasurements(chest_cm=88, waist_cm=70, hip_cm=92),
        GarmentMeasurements(chest_cm=90, waist_cm=74, hip_cm=96),
        deltas,
        shape_key,
        shape_ko,
        category="top",
        fit_type="regular",
    )


class TestClassifyBodyShape:
    # 남성·여성 체형 분류 테스트.
    @pytest.mark.parametrize(
        ("gender", "measures", "shape_key", "name_ko"),
        [
            (
                "female",
                {"shoulder_width_cm": 38.0, "chest_cm": 90.0, "waist_cm": 70.0, "hip_cm": 94.0},
                "hourglass",
                "모래시계형 (Hourglass)",
            ),
            (
                "female",
                {"shoulder_width_cm": 38.0, "chest_cm": 96.0, "waist_cm": 90.0, "hip_cm": 98.0},
                "apple",
                "원형/사과형 체형 (Apple/Round)",
            ),
            (
                "female",
                {"shoulder_width_cm": 36.0, "chest_cm": 84.0, "waist_cm": 66.0, "hip_cm": 98.0},
                "pear",
                "배형/하체 발달형 (Pear/Triangle)",
            ),
            (
                "female",
                {"shoulder_width_cm": 40.0, "chest_cm": 96.0, "waist_cm": 74.0, "hip_cm": 88.0},
                "inverted_triangle",
                "역삼각형형 (Inverted Triangle)",
            ),
            (
                "female",
                {"shoulder_width_cm": 38.0, "chest_cm": 88.0, "waist_cm": 76.0, "hip_cm": 92.0},
                "rectangle",
                "직사각형/슬림 스트레이트형 (Rectangle/Column)",
            ),
            (
                "male",
                {"shoulder_width_cm": 44.0, "chest_cm": 104.0, "waist_cm": 100.0, "hip_cm": 102.0},
                "apple",
                "복부 집중형 (Apple/Oval)",
            ),
            (
                "male",
                {"shoulder_width_cm": 48.0, "chest_cm": 106.0, "waist_cm": 86.0, "hip_cm": 96.0},
                "inverted_triangle",
                "역삼각형 체형 (Athletic V-Shape)",
            ),
            (
                "male",
                {"shoulder_width_cm": 42.0, "chest_cm": 94.0, "waist_cm": 76.0, "hip_cm": 92.0},
                "rectangle",
                "슬림 직사각형형 (Rectangle/Lean)",
            ),
            (
                "male",
                {"shoulder_width_cm": 40.0, "chest_cm": 90.0, "waist_cm": 80.0, "hip_cm": 102.0},
                "pear",
                "하체 발달형 (Pear/Trapezoid)",
            ),
            (
                "male",
                {"shoulder_width_cm": 43.0, "chest_cm": 98.0, "waist_cm": 86.0, "hip_cm": 92.5},
                "rectangle",
                "표준 직사각형형 (Standard Rectangle)",
            ),
            (
                "MALE",
                {"shoulder_width_cm": 44.0, "chest_cm": 104.0, "waist_cm": 100.0, "hip_cm": 102.0},
                "apple",
                "복부 집중형 (Apple/Oval)",
            ),
        ],
    )
    def test_shape_key_and_name(self, gender, measures, shape_key, name_ko):
        body = BodyMeasurements(**measures)
        key, name, metrics = classify_body_shape(body, gender)

        # 분류 키와 한글 체형명 검증.
        assert key == shape_key
        assert name == name_ko

        # 비율 지표 반올림 검증.
        assert metrics["whr"] == round(measures["waist_cm"] / measures["hip_cm"], 2)
        assert metrics["chest_waist_ratio"] == round(measures["chest_cm"] / measures["waist_cm"], 2)
        assert metrics["shoulder_cm"] == round(measures["shoulder_width_cm"], 1)
        assert metrics["chest_cm"] == measures["chest_cm"]
        assert metrics["waist_cm"] == measures["waist_cm"]
        assert metrics["hip_cm"] == measures["hip_cm"]


class TestGenerateFitReport:
    def test_part_diagnosis_and_tight_guide(self):
        # 부위별 진단과 타이트 가이드 테스트.
        report = _report([
            _delta("shoulder", -8.0, TightnessLevel.CRITICAL_TIGHT),
            _delta("chest", -3.0, TightnessLevel.TIGHT),
            _delta("waist", 1.0, TightnessLevel.OPTIMAL),
            _delta("hip", 6.0, TightnessLevel.LOOSE),
            _delta("sleeve", 12.0, TightnessLevel.VERY_LOOSE),
        ])
        parts = report["part_evaluations"]

        # 부위별 진단 문구 검증.
        assert parts["shoulder"]["name_ko"] == "어깨"
        assert parts["shoulder"]["tightness"] == "critical_tight"
        assert parts["shoulder"]["delta_cm"] == -8.0
        assert "강한 당김" in parts["shoulder"]["comment"]

        assert parts["chest"]["name_ko"] == "가슴둘레"
        assert parts["chest"]["tightness"] == "tight"
        assert "밀착" in parts["chest"]["comment"]

        assert parts["waist"]["name_ko"] == "허리둘레"
        assert parts["waist"]["tightness"] == "optimal"
        assert "이상적인" in parts["waist"]["comment"]

        assert parts["hip"]["name_ko"] == "엉덩이둘레"
        assert parts["hip"]["tightness"] == "loose"
        assert "여유로운" in parts["hip"]["comment"]

        assert parts["sleeve"]["name_ko"] == "소매길이"
        assert parts["sleeve"]["tightness"] == "very_loose"
        assert "오버사이즈" in parts["sleeve"]["comment"]

        # 타이트 가이드 검증.
        assert "대비형" in report["fit_summary"]
        assert report["fit_grade"] == "타이트 (Size-Up 권장)"
        assert "한 사이즈 크게" in report["recommended_size_guidance"]

    def test_loose_guide(self):
        # 루즈 가이드 테스트.
        report = _report([
            _delta("shoulder", 6.0, TightnessLevel.LOOSE),
            _delta("chest", 7.0, TightnessLevel.LOOSE),
            _delta("waist", 12.0, TightnessLevel.VERY_LOOSE),
        ])

        # 루즈 가이드 검증.
        assert "릴랙스드" in report["fit_summary"]
        assert report["fit_grade"] == "루즈핏 (정사이즈/Size-Down 가능)"
        assert "사이즈 다운" in report["recommended_size_guidance"]

    def test_true_to_size_guide(self):
        # 정사이즈 가이드 테스트.
        report = _report([
            _delta("chest", 1.0, TightnessLevel.OPTIMAL),
            _delta("waist", 0.0, TightnessLevel.OPTIMAL),
            _delta("hip", 3.0, TightnessLevel.OPTIMAL),
        ])

        # 정사이즈 가이드 검증.
        assert "스탠다드" in report["fit_summary"]
        assert report["fit_grade"] == "정사이즈 적합 (Best Fit)"
        assert "정사이즈" in report["recommended_size_guidance"]
        assert report["part_evaluations"]["waist"]["delta_cm"] == 0.0

    @pytest.mark.parametrize(
        ("shape_key", "fragment"),
        [
            ("hourglass", "모래시계"),
            ("pear", "골반"),
            ("apple", "드레이프"),
            ("inverted_triangle", "와이드"),
            ("rectangle", "레이어드"),
        ],
    )
    def test_styling_tips(self, shape_key, fragment):
        # 체형별 스타일링 팁 테스트.
        report = _report(
            [_delta("chest", 1.0, TightnessLevel.OPTIMAL)],
            shape_key=shape_key,
            shape_ko="테스트 체형",
        )

        # 스타일링 팁 검증.
        assert report["body_shape_type"] == shape_key
        assert report["body_shape_name"] == "테스트 체형"
        assert report["styling_tips"]
        assert fragment in report["styling_tips"][0]


class TestAvatarGenerateBodyMeasurements:
    def test_measurements_land_in_session_metadata(self, client, session_manager, test_app):
        # 치수 폼 수신과 세션 메타데이터 반영 테스트.
        sid = session_manager.create()
        captured = {}

        async def _fake_local(rid, video_bytes, image_img, metadata, inspector):
            captured["metadata"] = metadata
            captured["has_image"] = image_img is not None
            return BodyData(gender=metadata.gender, metadata=metadata, glb_bytes=b"")

        worker = test_app.state.worker_client
        with patch.object(worker, "is_configured", return_value=False), patch(
            "orchestrator.routes.avatar._local_generate_from_data",
            _fake_local,
        ):
            resp = client.post(
                "/avatar/generate",
                params={"session_id": sid},
                data={
                    "gender": "female",
                    "shoulder_width_cm": "38.5",
                    "chest_cm": "86",
                    "waist_cm": "68",
                    "hip_cm": "94",
                },
                files={"image": ("person.jpg", _jpeg_bytes(), "image/jpeg")},
            )

        # 정상 수신 검증.
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert body["gender"] == "female"
        assert captured["has_image"] is True

        # 세션 body_data.metadata 반영 검증.
        meta = session_manager.get(sid).body_data.metadata
        assert meta is captured["metadata"]
        assert meta.shoulder_width_cm == 38.5
        assert meta.chest_cm == 86.0
        assert meta.waist_cm == 68.0
        assert meta.hip_cm == 94.0

