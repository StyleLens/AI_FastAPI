# StyleLens V6 — AI Pipeline & API Specification

> Version: 1.0 | Updated: 2026-02-12

---

## 1. System Architecture Overview

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│  Spring Boot │────▶│  FastAPI           │────▶│  Modal GPU Worker    │
│  (Tier 2)    │ HTTP│  Orchestrator      │ HTTP│  (Tier 4)            │
│  Proxy       │◀────│  (Tier 3)          │◀────│  NVIDIA H100         │
└──────────────┘     └───────────────────┘     └──────────────────────┘
                          │
                          ▼
                     Local Models (MPS/CPU)
                     + Gemini API
```

### Tier 구성

| Tier | 역할 | 기술 스택 | 위치 |
|------|------|----------|------|
| Tier 1 | Frontend (React) | Next.js | Client |
| Tier 2 | Backend Proxy | Spring Boot 3.2, Java 21 | Server |
| Tier 3 | AI Orchestrator | FastAPI, Python 3.12 | Server (local/cloud) |
| Tier 4 | GPU Worker | Modal, NVIDIA H100 | Cloud (serverless) |

### 실행 모드

- **Local Mode** (`WORKER_URL` 미설정): 모든 모델을 로컬 MPS/CPU에서 실행
- **Distributed Mode** (`WORKER_URL` 설정): GPU 필요 작업을 Modal Worker로 위임

---

## 2. 파이프라인 흐름

```
Video/Images + Metadata
       │
       ▼
[Phase 1: Avatar]  ─── YOLO26-L → SAM3D Body → SMPL Mesh → GLB
       │                 Quality Gate 1 (person detection)
       │                 Quality Gate 3 (3D reconstruction)
       ▼
[Phase 2: Wardrobe] ── SAM 3 → FASHN → Gemini Analysis
       │                 Quality Gate 2 (segmentation)
       │                 Quality Gate 4 (clothing analysis)
       ▼
[Phase 3: Fitting]  ── CatVTON-FLUX × 8 angles (GPU)
       │               OR Gemini image gen (API fallback)
       │               + P2P Physics → mask expansion
       │               + Face Bank → face consistency
       │                 Quality Gate 5 (virtual try-on)
       │                 Quality Gate 5.5 (face consistency)
       ▼
[Phase 4: 3D Gen]  ─── Hunyuan3D Shape + Paint → GLB
                         Quality Gate 6 (3D visualization)
```

---

## 3. Phase 상세

### Phase 1: Avatar Generation

**목적**: 사용자 영상/이미지에서 3D 아바타 메시(GLB) 생성

**파일**: `core/pipeline.py`

**처리 흐름**:
1. 영상 프레임 추출 (최대 30프레임)
2. YOLO26-L: 사람 감지 (NMS-free, MPS)
3. SAM3D Body DINOv3: 3D 메시 + 관절 + body shape params (β)
4. SMPL 메시 구축 (A-pose, shoulder 65°)
5. GLB 내보내기 (trimesh)
6. 8개 각도 렌더링 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)

**입력**:
```python
Metadata:
  gender: str          # "female" | "male"
  height_cm: float     # 키 (cm)
  weight_kg: float     # 몸무게 (kg)
  bust_cup: str        # 컵 사이즈 (여성)
  body_type: str       # "slim" | "standard" | "average"
```

**출력**:
```python
BodyData:
  vertices: np.ndarray       # (N, 3) 정점 좌표
  faces: np.ndarray           # (F, 3) 면 인덱스
  joints: np.ndarray          # (J, 3) 관절 좌표
  betas: np.ndarray           # (10,) shape params
  gender: str
  glb_bytes: bytes            # GLB 바이너리
  mesh_renders: dict[int, np.ndarray]  # angle → 렌더 이미지
  quality_gates: list[InspectionResult]
```

**모델**:
| 모델 | 용도 | VRAM | Device |
|------|------|------|--------|
| YOLO26-L | 사람 감지 | ~200MB | MPS/CUDA |
| SAM3D Body DINOv3 | 3D 복원 | ~1.2GB | MPS/CUDA |

**성능**: ~20s (M4 Mac, MPS)

---

### Phase 2: Wardrobe Analysis

**목적**: 의류 이미지 분석 — 세그멘테이션, 파싱, AI 분석

**파일**: `core/wardrobe.py`

**처리 흐름**:
1. SAM 3: 의류 세그멘테이션 (center-point prompt)
2. FASHN SegFormer-B4: 18-class 패션 파싱
3. Gemini: 멀티이미지 분석 (view classification, field merge)
4. 사이즈 차트 / 상품 정보 / 피팅 모델 추출 (옵션)

**입력**: 의류 이미지 1~10장 + 옵션(사이즈 차트, 상품 정보, 피팅 모델)

**출력**:
```python
ClothingItem:
  analysis: ClothingAnalysis   # Gemini 분석 결과
  segmented_image: np.ndarray  # 세그멘테이션 이미지
  garment_mask: np.ndarray     # 의류 마스크
  parse_map: np.ndarray        # 18-class 파싱맵
  size_chart: dict             # 사이즈 차트 데이터
  product_info: dict           # 상품 정보
  quality_gates: list
```

**FASHN 18 Classes**:
```
0: background, 1: hat, 2: hair, 3: sunglasses,
4: upper_clothes, 5: skirt, 6: pants, 7: dress,
8: belt, 9: left_shoe, 10: right_shoe, 11: face,
12: left_leg, 13: right_leg, 14: left_arm, 15: right_arm,
16: bag, 17: scarf
```

**모델**:
| 모델 | 용도 | VRAM | Device |
|------|------|------|--------|
| SAM 3 | 세그멘테이션 | ~2.4GB | MPS/CUDA |
| FASHN SegFormer-B4 | 패션 파싱 | ~250MB | MPS/CUDA |
| Gemini API | 분석/분류 | - | API |

**성능**: ~30-50s (멀티이미지)

---

### Phase 3: Virtual Try-On (Fitting)

**목적**: 8개 각도에서 가상 피팅 이미지 생성

**파일**: `core/fitting.py`

**처리 흐름**:
1. 메시 렌더 → 기본 인물 이미지
2. FASHN 파싱 → 어그노스틱 마스크 생성
3. P2P 물리 분석 (옵션) → 마스크 확장 계수
4. 각도별 피팅 이미지 생성:
   - **Primary**: CatVTON-FLUX (CUDA GPU 필요, ~27GB VRAM)
   - **Fallback**: Gemini 이미지 생성 (API)
5. Face Bank → 얼굴 일관성 검증 (옵션)

**입력**: BodyData + ClothingItem + face_photo (옵션)

**출력**:
```python
FittingResult:
  tryon_images: dict[int, np.ndarray]  # angle → 피팅 이미지
  method_used: dict[int, str]          # angle → 생성 방법
  quality_gates: list
  p2p_result: P2PResult | None
  elapsed_sec: float
```

**생성 방법 우선순위**:
1. `catvton-flux` — CatVTON-FLUX (GPU, 최고 품질)
2. `gemini-front` — Gemini 정면 생성 (API)
3. `gemini-angle` — Gemini 측면 생성 (API, 정면 참조)
4. `mesh-render` — 메시 렌더 폴백

**모델**:
| 모델 | 용도 | VRAM | Device |
|------|------|------|--------|
| CatVTON-FLUX | 가상 피팅 | ~27GB | **CUDA only** |
| FASHN | 마스크 생성 | ~250MB | MPS/CUDA |
| InsightFace buffalo_l | 얼굴 감지 | ~500MB | CPU |
| Gemini API | 이미지 생성 | - | API |

**P2P Engine**:
- 신체 치수 vs 의류 치수 비교
- 타이트니스 분류: critical_tight / tight / optimal / loose / very_loose
- 마스크 확장 계수 자동 계산

**성능**: ~226s (Gemini 8각도, M4 Mac) | TBD (CatVTON, H100)

---

### Phase 4: 3D Model Generation

**목적**: 피팅 이미지 → 텍스처 3D GLB 모델

**파일**: `core/viewer3d.py`

**처리 흐름**:
1. 최적 정면 이미지 선택 (0° > 315° > 45°)
2. rembg 배경 제거 (U2Net)
3. Hunyuan3D Shape: 3D 형상 생성 (50 steps)
4. Hunyuan3D Paint: 텍스처 페인팅 (20 steps, 4K) — **CUDA only**
5. GLB 내보내기 + 프리뷰 렌더 (4각도)

**입력**: 8각도 피팅 이미지 dict

**출력**:
```python
Viewer3DResult:
  glb_bytes: bytes
  glb_id: str
  glb_path: str
  preview_renders: dict[int, np.ndarray]  # 0°, 90°, 180°, 270°
  quality_gates: list
  elapsed_sec: float
```

**모델**:
| 모델 | 용도 | VRAM | Device |
|------|------|------|--------|
| rembg (U2Net) | 배경 제거 | ~200MB | CPU |
| Hunyuan3D Shape | 3D 형상 | ~10GB | MPS/CUDA |
| Hunyuan3D Paint | 텍스처 | ~10GB | **CUDA only** |

**Fallback**: CUDA 미사용 시 Shape-only 모드 (텍스처 없음)

**성능**: ~30-45s (Shape+Paint, H100)

---

## 4. Quality Gate 체계

모든 Phase에 Gemini 기반 품질 검증 게이트 적용:

| Gate | Stage | 임계값 | 검증 내용 |
|------|-------|--------|----------|
| 1 | person_detection | 0.70 | 사람 감지 정확도 |
| 2 | body_segmentation | 0.75 | 세그멘테이션 품질 |
| 3 | body_3d_reconstruction | 0.70 | 3D 복원 품질 |
| 4 | clothing_analysis | 0.75 | 의류 분석 정확도 |
| 5 | virtual_tryon | 0.80 | 피팅 이미지 품질 |
| 5.5 | face_consistency | 0.75 | 얼굴 일관성 |
| 6 | 3d_visualization | 0.80 | 3D 모델 품질 |

---

## 5. Model Registry

**파일**: `core/loader.py`

싱글톤 패턴의 모델 레지스트리. 순차 로드/언로드 + MPS 메모리 관리.

| 모델명 | 로드 함수 | VRAM | 비고 |
|--------|----------|------|------|
| yolo26 | `load_yolo26()` | ~200MB | YOLO26-L |
| sam3 | `load_sam3()` | ~2.4GB | SAM 3 |
| sam3d_body | `load_sam3d_body()` | ~1.2GB | SAM3D DINOv3 |
| fashn_parser | `load_fashn_parser()` | ~250MB | SegFormer-B4 |
| catvton_flux | `load_catvton_flux()` | ~27GB | CUDA only |
| hunyuan3d_shape | `load_hunyuan3d_shape()` | ~10GB | Shape pipeline |
| hunyuan3d_paint | `load_hunyuan3d_paint()` | ~10GB | CUDA only |
| insightface | `load_insightface()` | ~500MB | buffalo_l |

**메모리 관리**:
```python
registry.unload_except("yolo26")   # yolo26만 남기고 언로드
registry.unload_all()               # 전체 언로드
torch.mps.empty_cache()             # MPS 캐시 정리
```

---

## 6. Modal GPU Worker (Tier 4)

**파일**: `worker/modal_app.py`

3개 컨테이너로 분리, NVIDIA H100 80GB.

### 6.1 LightModelWorker
- **모델**: SAM 3 + SAM3D Body + FASHN (~4GB)
- **GPU**: H100, Memory 16GB
- **Scaledown**: 120초

### 6.2 CatVTONWorker
- **모델**: FLUX.1-dev GGUF + CatVTON LoRA (~27GB)
- **GPU**: H100, Memory 32GB
- **Scaledown**: 300초, Timeout 600초

### 6.3 Hunyuan3DWorker
- **모델**: Shape (~10GB) → 언로드 → Paint (~10GB)
- **GPU**: H100, Memory 32GB
- **Scaledown**: 180초, Timeout 600초

### Worker HTTP Endpoints

| Method | Path | Worker | Timeout |
|--------|------|--------|---------|
| GET | `/health` | - | 10s |
| POST | `/reconstruct-3d` | Light | 300s |
| POST | `/segment-sam3` | Light | 300s |
| POST | `/parse-fashn` | Light | 300s |
| POST | `/tryon-catvton-batch` | CatVTON | 600s |
| POST | `/generate-3d-full` | Hunyuan3D | 600s |

### 직렬화 포맷
- 이미지: Base64 JPEG
- NumPy 배열: Base64 + pickle
- Parse Map: Base64 PNG (uint8)
- GLB: Base64 bytes

---

## 7. Orchestrator API Endpoints

**Base URL**: `http://localhost:8000`

### 7.1 시스템

| Method | Path | 설명 |
|--------|------|------|
| GET | `/` | 서비스 정보 및 상태 |
| GET | `/health` | 상세 헬스체크 (Worker 포함) |
| GET | `/sessions` | 활성 세션 목록 |
| GET | `/ui` | 테스트 콘솔 UI |
| GET | `/test-data/list` | 테스트 데이터 목록 |
| GET | `/test-data/{category}/{filename}` | 테스트 데이터 파일 |

### 7.2 Phase 1: Avatar

#### `POST /avatar/generate`
3D 아바타 생성 (영상/이미지 → GLB)

**Request** (multipart/form-data):
| Field | Type | Required | 설명 |
|-------|------|----------|------|
| video | File | 택1 | 영상 파일 |
| image | File | 택1 | 이미지 파일 |
| gender | string | O | "female" / "male" |
| height_cm | float | O | 키 (cm) |
| weight_kg | float | O | 몸무게 (kg) |
| bust_cup | string | X | 컵 사이즈 |
| body_type | string | X | "slim" / "standard" / "average" |
| session_id | string (query) | X | 세션 ID (default: "default") |

**Response** (200):
```json
{
  "request_id": "uuid",
  "session_id": "default",
  "gender": "female",
  "has_mesh": true,
  "vertex_count": 18439,
  "glb_size_bytes": 664000,
  "renders": {
    "0": "base64_jpeg...",
    "45": "base64_jpeg...",
    ...
  },
  "quality_gates": [
    {"stage": "person_detection", "score": 0.95, "pass": true, "feedback": "..."}
  ]
}
```

#### `GET /avatar/glb`
현재 아바타 GLB 다운로드

**Query**: `session_id` (default: "default")
**Response**: Binary GLB (Content-Type: model/gltf-binary)

---

### 7.3 Phase 2: Wardrobe

#### `POST /wardrobe/add-image`
단일 의류 이미지 분석

**Request** (multipart/form-data):
| Field | Type | Required |
|-------|------|----------|
| image | File | O |
| session_id | string (query) | X |

**Response** (200):
```json
{
  "request_id": "uuid",
  "session_id": "default",
  "analysis": {
    "category": "top",
    "subcategory": "t-shirt",
    "color_hex": "#2B4A7F",
    "pattern": "solid",
    "material": "cotton",
    "fit_type": "regular",
    "collar_type": "crew_neck",
    "sleeve_length": "short",
    "garment_measurements": {"chest_cm": 52, "length_cm": 68}
  },
  "quality_gates": [...]
}
```

#### `POST /wardrobe/add-images`
멀티이미지 + 사이즈 차트 + 상품 정보

**Request** (multipart/form-data):
| Field | Type | Required |
|-------|------|----------|
| images | File[] | O (1~10장) |
| size_chart | File | X |
| product_info_1 | File | X |
| product_info_2 | File | X |
| fitting_model | File | X |
| session_id | string (query) | X |

**Response** (200):
```json
{
  "request_id": "uuid",
  "session_id": "default",
  "analysis": {...},
  "size_chart": {"S": {...}, "M": {...}, "L": {...}},
  "product_info": {"brand": "...", "price": "..."},
  "fitting_model_info": {"height_cm": 170, "size": "M"},
  "quality_gates": [...]
}
```

#### `POST /wardrobe/add-url`
URL에서 의류 이미지 분석

**Request** (form): `url` (string), `session_id` (query)

#### `POST /wardrobe/extract-model-info`
피팅 모델 사진에서 정보 추출 (Gemini text-only)

**Request** (multipart/form-data): `image` (File), `session_id` (query)

---

### 7.4 Phase 3: Fitting

#### `POST /fitting/try-on`
CatVTON-FLUX 8각도 가상 피팅

**전제 조건**: Phase 1 + Phase 2 완료 필수

**Request** (multipart/form-data):
| Field | Type | Required |
|-------|------|----------|
| face_photo | File | X |
| session_id | string (query) | X |

**Response** (200):
```json
{
  "request_id": "uuid",
  "session_id": "default",
  "images": {
    "0": "base64_jpeg...",
    "45": "base64_jpeg...",
    ...
  },
  "methods": {
    "0": "gemini-front",
    "45": "gemini-angle",
    ...
  },
  "elapsed_sec": 226.5,
  "quality_gates": [...],
  "p2p": {
    "physics_prompt": "slightly loose fit around chest...",
    "overall_tightness": "optimal",
    "mask_expansion_factor": 1.05,
    "confidence": 0.85,
    "deltas": [
      {"body_part": "chest", "delta_cm": 2.5, "tightness": "optimal", "visual_keywords": ["relaxed"]}
    ]
  },
  "face_bank": {
    "bank_id": "uuid",
    "total_references": 3,
    "angle_coverage": 0.75
  }
}
```

---

### 7.5 Phase 4: 3D Viewer

#### `POST /viewer3d/generate`
Hunyuan3D 3D 모델 생성

**전제 조건**: Phase 3 완료 필수

**Request**: `session_id` (query)

**Response** (200):
```json
{
  "request_id": "uuid",
  "session_id": "default",
  "glb_id": "uuid",
  "glb_size_bytes": 8500000,
  "glb_url": "/viewer3d/model/uuid",
  "previews": {
    "0": "base64_jpeg...",
    "90": "base64_jpeg...",
    "180": "base64_jpeg...",
    "270": "base64_jpeg..."
  },
  "elapsed_sec": 35.2,
  "quality_gates": [...]
}
```

#### `GET /viewer3d/model/{glb_id}`
생성된 GLB 모델 다운로드

**Response**: Binary GLB (Content-Type: model/gltf-binary)

---

### 7.6 보조 엔드포인트

#### `POST /p2p/analyze`
P2P 물리 분석 (CPU, 결정론적)

**전제 조건**: Phase 1 + Phase 2 완료 필수

**Response** (200):
```json
{
  "physics_prompt": "...",
  "overall_tightness": "optimal",
  "deltas": [
    {
      "body_part": "chest",
      "body_cm": 88.0,
      "garment_cm": 52.0,
      "delta_cm": 2.5,
      "tightness": "optimal",
      "visual_keywords": ["relaxed"],
      "prompt_fragment": "..."
    }
  ],
  "mask_expansion_factor": 1.05,
  "confidence": 0.85,
  "method": "physics_direct"
}
```

#### `POST /face-bank/upload`
Face Bank에 얼굴 참조 업로드

**Request** (multipart/form-data):
| Field | Type | Required |
|-------|------|----------|
| current_photo | File | O |
| past_photos | File[] | X (최대 10장) |
| session_id | string (query) | X |

**Response** (200):
```json
{
  "bank_id": "uuid",
  "total_references": 5,
  "gender": "female",
  "angle_coverage": 0.75,
  "references": [
    {"label": "current", "face_angle": 0.0, "det_score": 0.99}
  ]
}
```

#### `GET /face-bank/{session_id}/status`
Face Bank 상태 조회

#### `GET /quality/report`
품질 게이트 종합 보고서

---

## 8. Gemini 모델 사용 규칙

| 모델 ID | 용도 | 이미지 생성 |
|---------|------|------------|
| `gemini-3-pro-image-preview` | 이미지 생성 전용 | O |
| `gemini-3-pro-preview` | 텍스트/분석 전용 | X |
| `gemini-2.5-flash-image` | 이미지 생성 fallback | O |
| `gemini-3-flash-preview` | 텍스트 전용 (빠름) | X |

**Fallback 체인**: `gemini-3-pro-image-preview` → `gemini-2.5-flash-image` → `gemini-3-flash-preview`

**필수 설정**: `response_modalities=["IMAGE", "TEXT"]` (이미지 생성 시)

**금지**: `gemini-3-pro-image` (404 에러), `gemini-3-flash-preview`로 이미지 생성 시도

---

## 9. 디렉토리 구조

```
test-body-recon/
├── core/                      # 핵심 AI 로직
│   ├── config.py              # 전체 설정 (모델 경로, 디바이스, 임계값)
│   ├── pipeline.py            # Phase 1: Avatar 파이프라인
│   ├── wardrobe.py            # Phase 2: 의류 분석
│   ├── fitting.py             # Phase 3: 가상 피팅
│   ├── viewer3d.py            # Phase 4: 3D 생성
│   ├── loader.py              # 모델 레지스트리 (싱글톤)
│   ├── sw_renderer.py         # 소프트웨어 메시 렌더러
│   ├── gemini_client.py       # Gemini API 클라이언트
│   ├── gemini_feedback.py     # 품질 게이트 (Gemini Inspector)
│   ├── p2p_engine.py          # Physics-to-Prompt 엔진
│   ├── p2p_ensemble.py        # P2P 앙상블 (Gemini 보강)
│   ├── face_bank.py           # Face Bank 빌더
│   ├── face_identity.py       # 얼굴 감지/임베딩 (InsightFace)
│   ├── multiview.py           # 멀티뷰 Gemini 생성
│   ├── body_deformation.py    # 체형 변형 (bust cup, BMI)
│   ├── clothing_merger.py     # 멀티이미지 의류 필드 병합
│   ├── image_preprocess.py    # 이미지 전처리
│   └── catvton_pipeline.py    # CatVTON-FLUX 래퍼
├── orchestrator/              # Tier 3 오케스트레이터
│   ├── main.py                # FastAPI 앱 + 라이프사이클
│   ├── config.py              # 환경 설정
│   ├── worker_client.py       # Tier 4 HTTP 클라이언트
│   └── routes/                # API 라우트
│       ├── avatar.py          # /avatar/*
│       ├── wardrobe.py        # /wardrobe/*
│       ├── fitting.py         # /fitting/*
│       ├── viewer3d.py        # /viewer3d/*
│       ├── p2p.py             # /p2p/*
│       ├── quality.py         # /quality/*
│       └── face_bank.py       # /face-bank/*
├── worker/                    # Tier 4 GPU Worker
│   └── modal_app.py           # Modal serverless (H100)
├── model/                     # 모델 가중치 (gitignore)
├── output/                    # 생성 결과물
├── docs/                      # 문서
└── tests/                     # 테스트
```

---

## 10. 환경 설정

### .env
```env
GEMINI_API_KEY=your_key_here

# Modal GPU Worker (Tier 4)
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret
WORKER_URL=https://your-username--stylelens-v6-worker-health.modal.run
```

### 주요 설정 (core/config.py)
```python
FITTING_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
A_POSE_SHOULDER_ANGLE = 65
FACE_BANK_MAX_REFERENCES = 11
SESSION_MAX = 10
SESSION_TTL_SEC = 3600
```

---

## 11. 에러 코드

| 코드 | 상태 | 설명 |
|------|------|------|
| AVATAR_NOT_READY | 400 | Phase 1 미완료 |
| WARDROBE_EMPTY | 400 | Phase 2 미완료 |
| FITTING_NOT_READY | 400 | Phase 3 미완료 |
| WORKER_UNAVAILABLE | 503 | GPU Worker 연결 불가 |
| WORKER_TIMEOUT | 504 | GPU Worker 타임아웃 |
| GEMINI_ERROR | 502 | Gemini API 에러 |
| MODEL_LOAD_ERROR | 500 | 모델 로드 실패 |
| SESSION_NOT_FOUND | 404 | 세션 없음 |
| QUALITY_GATE_FAIL | 422 | 품질 게이트 미통과 |
