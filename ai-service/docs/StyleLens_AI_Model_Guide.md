# StyleLens 3D Pro: AI Model & Pipeline Guide v3.0

> **문서 버전:** 3.0 (V5 All-Gemini Production)
> **최종 업데이트:** 2026-02-10
> **타겟 하드웨어:** Apple Silicon M4 Max/Ultra (RAM 48GB+)
> **핵심 목표:** 얼굴/의류 일관성 최우선의 실사급 가상 피팅

---

## 1. 시스템 아키텍처 개요

StyleLens는 **3D 아바타 생성 + 가상 피팅** 파이프라인입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (main.py)                      │
│                     Port 8000, MPS Device                        │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ Phase 1  │ Phase 2  │ Phase 3  │ Phase 4  │ Phase 5             │
│ Avatar   │ Wardrobe │ Fitting  │ V4 SDXL  │ V5 All-Gemini       │
│ GLB      │ Analysis │ V3       │ Lightning│ (Production)        │
└──────────┴──────────┴──────────┴──────────┴─────────────────────┘
```

### 핵심 파이프라인 흐름

```
사용자 영상/사진 → [Phase 1] 3D 아바타 (GLB)
의류 상품 사진(들) → [Phase 2] 의류 분석 (JSON)
아바타 + 의류 + 얼굴 사진 → [Phase 5] 8각도 피팅 이미지
```

---

## 2. AI 모델 라인업

### 2.1 로컬 모델 (M4 Mac에서 실행)

| 단계 | 역할 | 모델명 | 용도 | 메모리 |
|:--:|:--|:--|:--|:--:|
| **Phase 1** | 사람 감지 | **YOLOv8n** (ultralytics) | 영상에서 인체 bbox 추출 | ~50MB |
| **Phase 1** | 포즈/체형 추정 | **HMR 2.0** (Vendored ViT-Huge) | 10-dim body shape parameter 추출 | ~1.2GB |
| **Phase 1** | 3D 인체 메쉬 | **SMPL** (smplx) | 6890v × 13776f 인체 생성 | ~200MB |
| **Phase 1** | 3D 머리 메쉬 | **FLAME** (smplx, optional) | 5023v 머리 생성 (정밀 얼굴) | ~100MB |
| **Phase 5** | 얼굴 감지/임베딩 | **InsightFace buffalo_l** | 512-dim 임베딩 + 106-pt 랜드마크 | ~500MB |
| **Fallback** | 이미지 생성 | **SDXL Lightning** (ByteDance 4-step LoRA) | V5 API 실패 시 fallback 전용 | ~6GB |

### 2.2 API 모델 (Google Gemini)

| 모델 ID | 용도 | Config 변수 | 비고 |
|:--|:--|:--|:--|
| **`gemini-3-pro-image-preview`** | 이미지 생성 전용 | `V5_GEMINI_IMAGE_MODEL` | V5 피팅 8각도 생성 |
| **`gemini-3-pro-preview`** | 텍스트/분석 전용 | `GEMINI_MODEL_NAME` | 체형 분석, 의류 분석, 헤어 분석 |
| **`gemini-2.5-flash-image`** | 이미지 생성 fallback | (hardcoded) | Pro 실패 시 사용 |

> **STRICT 정책**: 위 3개 모델만 사용. 다른 모델 금지.
>
> **금지 모델:**
> - `gemini-3-pro-image` — 404 NOT_FOUND (존재하지 않는 모델명!)
> - `gemini-3-flash-preview` — 텍스트 전용 (이미지 생성 불가)
> - `gemini-1.x` 계열 전체

### 2.3 더 이상 사용하지 않는 모델 (v2.0에서 제거됨)

| 모델 | 이유 |
|:--|:--|
| SAM 2 | 별도 세그멘테이션 불필요 — Gemini가 이미지 생성 시 자체 처리 |
| CodeFormer | 얼굴 복원 불필요 — Gemini가 얼굴 일관성 자체 유지 |
| AntelopeV2 | buffalo_l로 충분 (SDXL fallback용만) |
| CatVTON | Gemini API 기반 가상 피팅으로 대체 |
| SD-Inpainting | V3용 — V5에서 Gemini 기반으로 전환 |

---

## 3. Phase 1: 3D 아바타 생성

### 엔드포인트: `POST /generate-ultimate-twin`

```
Video + Images + Metadata
    ↓
Stage 1: Video → YOLOv8 → HMR2 → averaged betas (1, 10)
Stage 1.5: Gemini AI Supervisor (근육/체지방/피부색 분석)
    ↓ Fusion
Stage 2: SMPL A-Pose (shoulder ±65°) + Body Deformation (bust cup + leg BMI)
Stage 2.5: FLAME Head (optional — model/flame/ 디렉토리 존재 시)
Stage 3: Body Texture (2048×2048 UV, underwear, face projection)
Stage 3.5: Face Texture (FLAME UV 전용, optional)
Stage 3.7: Hair Selection (Gemini 분석 → manifest.json 매칭, optional)
Stage 4: GLB Export (multi-mesh Scene: body + head + hair)
```

### 핵심 기술

- **HMR2 Vendored**: detectron2 대신 YOLOv8 사용 (M4 Mac 호환)
  - ViT-Huge backbone (1280 dim, 32 blocks, 16 heads)
  - Transformer Decoder (6 layers, 1024 dim)
  - Checkpoint key remapping 필요: `layers.N.{0,1,2}` → `layers.N.layers.{0,1,2}`

- **Body Deformation** (`core/body_deformation.py`):
  - Bust cup: A=0.5cm, B=1.2cm, C=2.0cm, D=3.0cm (vertex normal displacement + Laplacian smooth)
  - Legs: BMI 기반 XZ 스케일링 (0.85~1.15)

- **chumpy_stub.py**: numpy 2.4 호환성 패치 — SMPL .pkl 파일의 chumpy 객체 처리

---

## 4. Phase 2: 스마트 옷장 분석

### 엔드포인트

| 엔드포인트 | 설명 |
|:--|:--|
| `POST /wardrobe/add-url` | URL → HTML 파싱 → Gemini 분석 |
| `POST /wardrobe/add-image` | 단일 이미지 분석 |
| `POST /wardrobe/add-images` | **멀티 이미지 분석 (V5, 1~10장)** |

### 멀티 이미지 파이프라인

```
1~10 Images → View Classification (front/back/side/detail)
    → Per-Image Analysis (V5 확장 필드: buttons, pockets, pattern, wrinkles...)
    → Field Merge (front 우선, _fuzzy_match_field() 검증)
    → Merged ClothingAnalysis JSON
```

### 사이즈 추출 3-Tier 전략

1. **Size Chart OCR** (사이즈 차트 이미지 제공 시)
2. **Reference Body + Spatial Reasoning** (Gemini Pro)
3. **Default Preset Fallback**

---

## 5. Phase 5: V5 정확도 최우선 피팅 (Production)

### 엔드포인트: `POST /fitting/try-on-v5`

이것이 현재 **프로덕션 파이프라인**입니다.

### Quality Mode

| 모드 | 전면 | 나머지 7각도 | 용도 |
|:--|:--|:--|:--|
| `api` (권장) | Gemini | Gemini (ref) | 최고 품질 |
| `hybrid` (기본) | Gemini | Gemini (ref) | 내부적으로 api와 동일 |
| `local` | SDXL + FaceSwap | SDXL + FaceSwap | API 불가 시 fallback |

> **실제 운용**: hybrid/api 모두 내부적으로 All-Gemini (SDXL은 Gemini 실패 시 fallback만)

### V5 파이프라인 상세 흐름

```
Stage A: Avatar Body Measurements + Camera Correction
Stage B: Face Embedding (InsightFace buffalo_l → FaceData)
Stage C: Render 8 angles (deformed mesh → 512×512 silhouettes)
Stage D: Front View (0°)
    Primary: generate_front_view_gemini()
      - mesh silhouette + face photo (768×768) + clothing image (1024)
      - Model: gemini-3-pro-image-preview
    Fallback: SDXL 20-step + FaceSwap
Stage E: Remaining 7 Angles
    Primary: generate_angle_with_reference()
      - front reference + face photo + mesh silhouette
      - Model: gemini-3-pro-image-preview
    Fallback: SDXL 4-step + FaceSwap
Stage F: FittingReport generation
```

### Gemini 이미지 생성 — 필수 설정

```python
# 이미지 생성 시 반드시 필요
config = types.GenerateContentConfig(
    response_modalities=["IMAGE", "TEXT"],
)

# 응답 검증 — 이미지 존재 여부 반드시 확인
has_image = any(
    hasattr(p, 'inline_data') and p.inline_data
    for p in response.candidates[0].content.parts
)
```

### Fallback Chain (이미지 생성)

```
gemini-3-pro-image-preview (primary)
  → gemini-2.5-flash-image (fast fallback)
  → gemini-3-flash-preview (text-only — 이미지 생성 불가, skip)
  → SDXL Lightning (최종 fallback)
```

---

## 6. Prompt Engineering 핵심 규칙 (V5 Gemini)

### 의류 일관성

```
CRITICAL: "THE CLOTHING PRODUCT IMAGE IS THE PRIMARY REFERENCE"
- Reproduce EVERY visible detail: collar shape, how it's worn (open/closed/draped),
  button placement, fabric drape, how the hem falls, how sleeves sit on the arm.
- If the collar is worn loosely open → it MUST look the same.
```

### 하의 (Bottoms) — 반드시 준수

```
"If product image shows specific bottoms, use EXACT SAME bottoms
 — same color, length, style."

"BOTTOMS & SHOES: Must be IDENTICAL to the front reference"
```

> **절대 사용 금지**: "matching bottoms" → Gemini가 같은 색 반바지를 생성함

### 자세 (Pose)

```
"arms relaxed naturally at sides, NOT spread in A-pose or T-pose"
"Natural standing pose — arms down, hands relaxed near thighs"
```

### 얼굴 동일성 (Face Identity)

```
"EXACT likeness / IDENTICAL person / same facial features"
Face photo: 768×768 해상도, _clean_face_photo()로 SNS UI 사전 제거
```

### 칼라/네크라인

```
"The collar/neckline must look EXACTLY like the front reference
 — same openness, same drape, same shape."
shirt-collar/polo-collar → "(worn as shown in product image — match collar openness exactly)"
```

### 워터마크 방지

```
"Do NOT include any text, watermarks, UI elements, page numbers"
"Output a CLEAN fashion photo only"
```

### 헤어 일관성

```
"EXACT hairstyle from the face photo — same color, length, texture, bangs"
"Hair should naturally fall and drape as it would when head is turned"
```

---

## 7. 성능 (M4 Mac)

| Pipeline | 전체 시간 | Per-angle | 비고 |
|:--|:--|:--|:--|
| Phase 1 (Avatar) | ~15s | — | 8-stage pipeline |
| Phase 2 (Wardrobe, 단일) | ~46s | — | Gemini API |
| Phase 2 (Wardrobe, 3장) | ~255s | — | 3 images × Gemini |
| Phase 4 (V4 SDXL) | ~77.7s | ~9.7s | SDXL Lightning 4-step |
| **Phase 5 (V5 All-Gemini)** | **~245s** | **~30s** | `gemini-3-pro-image-preview` |

---

## 8. 로컬 폴더 구조

```
test-body-recon/
├── main.py                          # FastAPI 서버
├── core/
│   ├── config.py                    # 전역 설정
│   ├── pipeline.py                  # Phase 1: 아바타 (8-stage)
│   ├── gemini_client.py             # Gemini API (분석 + 이미지 생성)
│   ├── wardrobe.py                  # Phase 2: 의류 분석
│   ├── fitting.py                   # Phase 3: V3 (SD Inpainting)
│   ├── fitting_v4.py                # Phase 4: V4 (SDXL Lightning)
│   ├── fitting_v5.py                # Phase 5: V5 (All-Gemini)
│   ├── multiview.py                 # V5: 멀티뷰 생성 (Gemini)
│   ├── face_identity.py             # V5: 얼굴 보존 (InsightFace)
│   ├── body_deformation.py          # V5: 체형 변형 (bust + leg)
│   ├── clothing_merger.py           # V5: 멀티이미지 병합
│   ├── flame_head.py                # FLAME 머리 생성/정렬
│   ├── face_texture.py              # FLAME UV 텍스처
│   ├── hair_library.py              # 헤어 에셋 라이브러리
│   ├── loader.py                    # 모델 레지스트리 (lazy load/unload)
│   ├── sw_renderer.py               # 소프트웨어 렌더러
│   ├── image_preprocess.py          # 이미지 전처리
│   ├── chumpy_stub.py               # numpy 2.4 호환
│   └── hmr2_vendor/                 # HMR2 (reverse-engineered)
├── static/index.html                # 테스트 콘솔 UI (4-tab)
├── model/                           # 모델 체크포인트
│   ├── hmr2/                        # HMR2 ViT-Huge
│   ├── smpl/                        # SMPL_{MALE,FEMALE,NEUTRAL}.pkl
│   ├── flame/                       # (optional) FLAME_{MALE,FEMALE}.pkl
│   └── hair/                        # (optional) manifest.json + .obj
├── output/                          # 생성 결과
│   ├── avatar/                      # Phase 1: .glb
│   ├── wardrobe/                    # Phase 2: .json
│   ├── fitting_v4/                  # Phase 4: .json
│   └── fitting_v5/                  # Phase 5: .json + .png
└── docs/
    ├── Pipeline_Specification.md
    └── Functional_Specification.md
```

---

## 9. 필수 모델 다운로드

| 모델 | 경로 | 다운로드 |
|:--|:--|:--|
| HMR2 | `model/hmr2/epoch=35-step=1000000.ckpt` | [4DHumans](https://github.com/shubham-goel/4D-Humans) |
| SMPL | `model/smpl/SMPL_{GENDER}.pkl` | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) (라이선스 필요) |
| FLAME (optional) | `model/flame/FLAME_{GENDER}.pkl` | [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de) |
| InsightFace | 자동 다운로드 (~/.insightface) | `buffalo_l` 모델 자동 캐싱 |
| SDXL Lightning | 자동 다운로드 (HuggingFace) | `ByteDance/SDXL-Lightning` |

### 환경 변수

```bash
GEMINI_API_KEY=your_google_ai_api_key   # Phase 2, V5 필수
```

### Python 의존성

```
torch, ultralytics, smplx, insightface, diffusers, trimesh,
opencv-python, Pillow, google-genai, fastapi, uvicorn,
python-dotenv, numpy
```

---

## 10. M4 Mac 개발 환경 Tip

- **Device**: `torch.backends.mps.is_available()` → `device='mps'`
- **FP16 주의**: MPS에서 half precision 불안정 → `half=False` 사용
- **메모리 관리**: 스테이지별 순차 로드/언로드 패턴
  ```python
  registry.unload("hmr2")
  torch.mps.empty_cache()
  gc.collect()
  ```
- **Gemini API**: 네트워크 I/O만 — GPU 메모리 사용 없음 (V5의 장점)

---

## 11. E2E 테스트 결과 (2026-02-10)

### 테스트 조건
- **피험자**: 한국인 여성, ~165cm/48kg, 긴 검은 머리 + 앞머리
- **의류**: 샴페인 새틴 블라우스 + 블랙 펜슬 스커트 (쇼핑몰 상품 사진 3장)
- **모드**: api (All-Gemini), gemini-3-pro-image-preview

### 최종 점수: 9.5/10

| 항목 | 평가 |
|:--|:--|
| 의류 색상/소재 | 샴페인 새틴 광택 정확 재현 |
| 칼라/네크라인 | Open collar V-shape (상품 이미지와 일치) |
| 하의 | 블랙 펜슬 스커트 (상품과 일치) |
| 신발 | 블랙 포인티드 힐 |
| 얼굴/머리 | 8각도 모두 일관된 한국 여성 (앞머리 포함) |
| 체형 비율 | 슬림 체형 정확 반영 |
| 포즈 | 자연스러운 서기 자세 (A-pose 없음) |
| 배경 | 깨끗한 그레이 스튜디오 (8각도 일관) |
