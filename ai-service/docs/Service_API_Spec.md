# StyleLens V6 — 서비스 API & 기능 명세서

> 대상: Frontend (React/Next.js), Backend (Spring Boot), DB (Supabase)
> Version: 1.0 | Updated: 2026-02-12

---

## 1. 서비스 개요

StyleLens는 사용자 사진/영상에서 3D 아바타를 생성하고, 의류를 분석하여 가상 피팅 이미지와 3D 모델을 제공하는 AI 가상 피팅 서비스입니다.

### 핵심 기능

| 기능 | 설명 | 주요 Output |
|------|------|------------|
| 3D Avatar 생성 | 사용자 영상/사진 → 3D 아바타 | GLB 파일, 8각도 렌더링 |
| 의류 분석 | 의류 사진 → AI 분석 | 카테고리, 색상, 핏, 치수 |
| 가상 피팅 | 아바타 + 의류 → 피팅 이미지 | 8각도 피팅 이미지 |
| 3D 뷰어 | 피팅 이미지 → 3D 모델 | GLB 파일, 4방향 프리뷰 |

### 파이프라인 흐름

```
[Phase 1: Avatar] → [Phase 2: Wardrobe] → [Phase 3: Fitting] → [Phase 4: 3D Viewer]
    영상/사진          의류 이미지            가상 피팅             3D 모델
```

**순서 의존성**: Phase 1 → Phase 2 완료 후 Phase 3 가능, Phase 3 완료 후 Phase 4 가능

---

## 2. 시스템 아키텍처

```
┌───────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Frontend  │────▶│ Spring Boot  │────▶│ AI Orchestrator │────▶│ Modal GPU    │
│ (Next.js) │◀────│ (Tier 2)     │◀────│ (Tier 3)        │◀────│ (Tier 4)     │
└───────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                       │                                            │
                       ▼                                            ▼
                  ┌──────────┐                              ┌──────────────┐
                  │ Supabase │                              │ NVIDIA H100  │
                  │ (DB/Auth)│                              │ 80GB VRAM    │
                  └──────────┘                              └──────────────┘
```

### 통신 방식

| 구간 | 프로토콜 | 인증 |
|------|---------|------|
| Frontend ↔ Spring Boot | HTTPS + WebSocket | JWT (Google OAuth2) |
| Spring Boot ↔ AI Orchestrator | HTTP (내부 네트워크) | 없음 (네트워크 격리) |
| AI Orchestrator ↔ Modal Worker | HTTPS | Modal Token |
| Frontend ↔ Supabase | HTTPS | Supabase anon key |

---

## 3. 인증 & 권한

### Google OAuth2 Flow

```
Frontend → Google OAuth2 → Callback → Spring Boot → JWT 발급
```

1. Frontend: Google 로그인 버튼 클릭
2. Google OAuth2 인증 → Authorization Code
3. Spring Boot: Code → Access Token 교환
4. 사용자 정보 조회 → DB 저장/조회
5. JWT 토큰 발급 (Access 1시간 + Refresh 30일)

### JWT 구조

```json
{
  "sub": "user_uuid",
  "email": "user@example.com",
  "role": "USER",
  "iat": 1707700000,
  "exp": 1707703600
}
```

### 역할 (Role)

| Role | 설명 | 제한 |
|------|------|------|
| USER | 일반 사용자 | 기본 쿼터 |
| ADMIN | 관리자 | 관리 페이지 접근 |
| SUPER_ADMIN | 최고 관리자 | 전체 권한 (aisamdasu1204@gmail.com 고정) |

### Rate Limiting (Spring Boot)

| 기능 | 제한 |
|------|------|
| Avatar 생성 | 5회/시간 |
| Fitting 생성 | 10회/시간 |
| 3D 생성 | 3회/시간 |
| 전체 API | 100회/시간 |

---

## 4. API 명세 (Spring Boot ↔ Frontend)

Spring Boot는 AI Orchestrator의 프록시 역할. 아래 API는 Frontend에서 호출하는 최종 스펙.

### 4.1 인증

#### `POST /api/auth/google`
Google OAuth2 로그인

**Request**:
```json
{
  "code": "google_authorization_code"
}
```

**Response** (200):
```json
{
  "accessToken": "jwt_token",
  "refreshToken": "refresh_token",
  "expiresIn": 3600,
  "user": {
    "id": "uuid",
    "email": "user@gmail.com",
    "name": "홍길동",
    "picture": "https://...",
    "role": "USER"
  }
}
```

#### `POST /api/auth/refresh`
토큰 갱신

**Request**: `Authorization: Bearer {refreshToken}`

**Response** (200):
```json
{
  "accessToken": "new_jwt_token",
  "expiresIn": 3600
}
```

---

### 4.2 Phase 1: Avatar

#### `POST /api/avatar/generate`
3D 아바타 생성

**Request** (multipart/form-data):
| Field | Type | Required | 설명 |
|-------|------|----------|------|
| video | File | 택1 | 영상 파일 (MP4/MOV, 최대 100MB) |
| image | File | 택1 | 이미지 파일 (JPG/PNG, 최대 10MB) |
| gender | string | O | "female" / "male" |
| height_cm | number | O | 키 (cm), 범위: 140~200 |
| weight_kg | number | O | 몸무게 (kg), 범위: 30~150 |
| bust_cup | string | X | "A" / "B" / "C" / "D" (여성) |
| body_type | string | X | "slim" / "standard" / "average" |

**Response** (200):
```json
{
  "requestId": "uuid",
  "sessionId": "uuid",
  "gender": "female",
  "hasMesh": true,
  "vertexCount": 18439,
  "glbSizeBytes": 664000,
  "glbUrl": "/api/avatar/glb?sessionId=uuid",
  "renders": {
    "0": "base64_jpeg",
    "45": "base64_jpeg",
    "90": "base64_jpeg",
    "135": "base64_jpeg",
    "180": "base64_jpeg",
    "225": "base64_jpeg",
    "270": "base64_jpeg",
    "315": "base64_jpeg"
  },
  "qualityGates": [
    {
      "stage": "person_detection",
      "score": 0.95,
      "pass": true,
      "feedback": "Person clearly detected"
    }
  ]
}
```

**Error Cases**:
| Status | Code | 설명 |
|--------|------|------|
| 400 | INVALID_INPUT | 영상/이미지 없음 또는 형식 오류 |
| 413 | FILE_TOO_LARGE | 파일 크기 초과 |
| 429 | RATE_LIMITED | 시간당 제한 초과 |
| 500 | MODEL_ERROR | AI 모델 오류 |

#### `GET /api/avatar/glb`
GLB 파일 다운로드

**Query**: `sessionId` (required)
**Response**: Binary GLB (Content-Type: `model/gltf-binary`)

---

### 4.3 Phase 2: Wardrobe

#### `POST /api/wardrobe/add-image`
단일 의류 이미지 분석

**Request** (multipart/form-data):
| Field | Type | Required |
|-------|------|----------|
| image | File | O (JPG/PNG, 최대 10MB) |

**Response** (200):
```json
{
  "requestId": "uuid",
  "sessionId": "uuid",
  "analysis": {
    "category": "top",
    "subcategory": "t-shirt",
    "colorHex": "#2B4A7F",
    "colorName": "navy blue",
    "pattern": "solid",
    "material": "cotton",
    "fitType": "regular",
    "collarType": "crew_neck",
    "sleeveLength": "short",
    "garmentMeasurements": {
      "chestCm": 52,
      "lengthCm": 68,
      "shoulderCm": 44
    }
  },
  "qualityGates": [...]
}
```

#### `POST /api/wardrobe/add-images`
멀티이미지 의류 분석 (최대 10장)

**Request** (multipart/form-data):
| Field | Type | Required | 설명 |
|-------|------|----------|------|
| images | File[] | O | 의류 이미지 1~10장 |
| sizeChart | File | X | 사이즈 차트 이미지 |
| productInfo1 | File | X | 상품 상세 이미지 1 |
| productInfo2 | File | X | 상품 상세 이미지 2 |
| fittingModel | File | X | 피팅 모델 사진 |

**Response** (200):
```json
{
  "requestId": "uuid",
  "analysis": {...},
  "sizeChart": {
    "S": {"chest": 48, "length": 65},
    "M": {"chest": 52, "length": 68},
    "L": {"chest": 56, "length": 71}
  },
  "productInfo": {
    "brand": "ZARA",
    "price": "49,900원",
    "material": "cotton 100%"
  },
  "fittingModelInfo": {
    "heightCm": 170,
    "size": "M",
    "notes": "Regular fit"
  }
}
```

#### `POST /api/wardrobe/add-url`
URL에서 의류 분석

**Request** (application/json):
```json
{
  "url": "https://example.com/clothing.jpg"
}
```

#### `POST /api/wardrobe/extract-model-info`
피팅 모델 정보 추출

**Request** (multipart/form-data): `image` (File)

---

### 4.4 Phase 3: Fitting

#### `POST /api/fitting/try-on`
8각도 가상 피팅 이미지 생성

**전제 조건**: Phase 1 + Phase 2 완료 필수

**Request** (multipart/form-data):
| Field | Type | Required | 설명 |
|-------|------|----------|------|
| facePhoto | File | X | 얼굴 사진 (피팅 정확도 향상) |

**Response** (200):
```json
{
  "requestId": "uuid",
  "sessionId": "uuid",
  "images": {
    "0": "base64_jpeg",
    "45": "base64_jpeg",
    "90": "base64_jpeg",
    "135": "base64_jpeg",
    "180": "base64_jpeg",
    "225": "base64_jpeg",
    "270": "base64_jpeg",
    "315": "base64_jpeg"
  },
  "methods": {
    "0": "gemini-front",
    "45": "gemini-angle"
  },
  "elapsedSec": 226.5,
  "p2p": {
    "physicsPrompt": "slightly loose fit around chest...",
    "overallTightness": "optimal",
    "deltas": [
      {
        "bodyPart": "chest",
        "deltaCm": 2.5,
        "tightness": "optimal",
        "visualKeywords": ["relaxed", "comfortable"]
      }
    ]
  },
  "faceBank": {
    "bankId": "uuid",
    "totalReferences": 3,
    "angleCoverage": 0.75
  },
  "qualityGates": [...]
}
```

**Error Cases**:
| Status | Code | 설명 |
|--------|------|------|
| 400 | AVATAR_NOT_READY | Phase 1 미완료 |
| 400 | WARDROBE_EMPTY | Phase 2 미완료 |
| 503 | WORKER_UNAVAILABLE | GPU Worker 연결 불가 |
| 504 | WORKER_TIMEOUT | GPU Worker 타임아웃 |

**소요 시간**: 약 30~240초 (방법에 따라 상이)

---

### 4.5 Phase 4: 3D Viewer

#### `POST /api/viewer3d/generate`
피팅 결과 → 3D GLB 모델 생성

**전제 조건**: Phase 3 완료 필수

**Response** (200):
```json
{
  "requestId": "uuid",
  "sessionId": "uuid",
  "glbId": "uuid",
  "glbSizeBytes": 8500000,
  "glbUrl": "/api/viewer3d/model/uuid",
  "previews": {
    "0": "base64_jpeg",
    "90": "base64_jpeg",
    "180": "base64_jpeg",
    "270": "base64_jpeg"
  },
  "elapsedSec": 35.2,
  "qualityGates": [...]
}
```

#### `GET /api/viewer3d/model/{glbId}`
3D GLB 모델 다운로드

**Response**: Binary GLB (Content-Type: `model/gltf-binary`)

---

### 4.6 보조 API

#### `POST /api/p2p/analyze`
P2P 물리 분석 (의류 핏 예측)

**Response** (200):
```json
{
  "physicsPrompt": "...",
  "overallTightness": "optimal",
  "deltas": [
    {
      "bodyPart": "chest",
      "bodyCm": 88.0,
      "garmentCm": 52.0,
      "deltaCm": 2.5,
      "tightness": "optimal",
      "visualKeywords": ["relaxed"]
    }
  ],
  "maskExpansionFactor": 1.05,
  "confidence": 0.85
}
```

**Tightness 분류**:
| 값 | 설명 | delta 범위 |
|---|------|-----------|
| critical_tight | 매우 타이트 | < -3cm |
| tight | 타이트 | -3 ~ -1cm |
| optimal | 적정 | -1 ~ +3cm |
| loose | 루즈 | +3 ~ +6cm |
| very_loose | 매우 루즈 | > +6cm |

#### `POST /api/face-bank/upload`
Face Bank 얼굴 참조 업로드

**Request** (multipart/form-data):
| Field | Type | Required | 설명 |
|-------|------|----------|------|
| currentPhoto | File | O | 현재 얼굴 사진 |
| pastPhotos | File[] | X | 과거 참조 사진 (최대 10장) |

**Response** (200):
```json
{
  "bankId": "uuid",
  "totalReferences": 5,
  "gender": "female",
  "angleCoverage": 0.75,
  "references": [
    {"label": "current", "faceAngle": 0.0, "detScore": 0.99}
  ]
}
```

#### `GET /api/face-bank/status`
Face Bank 상태 조회

#### `GET /api/quality/report`
품질 게이트 종합 보고서

---

## 5. WebSocket 실시간 진행상황

### 연결

```
WS /ws/progress/{sessionId}
Authorization: Bearer {jwt_token}
```

### 이벤트 타입

| 이벤트 | 설명 | 주요 필드 |
|--------|------|----------|
| `phase_start` | Phase 시작 | phaseId, name |
| `progress` | 진행률 업데이트 | progress (0.0~1.0), message |
| `intermediate_result` | 중간 결과 프리뷰 | type, thumbnailBase64 |
| `quality_gate` | 품질 검증 결과 | stage, score, pass, feedback |
| `phase_complete` | Phase 완료 | elapsedSec, resultUrl |
| `error` | 에러 발생 | code, message, fallbackAvailable |

### 이벤트 예시

```json
{
  "type": "progress",
  "phaseId": "avatar",
  "progress": 0.65,
  "message": "3D 메시 생성 중..."
}
```

```json
{
  "type": "quality_gate",
  "phaseId": "fitting",
  "stage": "virtual_tryon",
  "score": 0.92,
  "pass": true,
  "feedback": "Clothing accurately rendered"
}
```

### 재연결 전략

- 최대 5회 재시도
- Exponential backoff: 1s → 2s → 4s → 8s → 16s (최대 30s)
- 재연결 시 마지막 이벤트 이후 누락분 수신

---

## 6. DB 스키마 (Supabase)

### users

| Column | Type | 설명 |
|--------|------|------|
| id | uuid (PK) | 사용자 ID |
| email | varchar(255) | 이메일 (unique) |
| google_id | varchar(255) | Google OAuth2 ID |
| name | varchar(100) | 이름 |
| picture | text | 프로필 사진 URL |
| role | enum | USER / ADMIN / SUPER_ADMIN |
| storage_quota_bytes | bigint | 스토리지 쿼터 |
| storage_used_bytes | bigint | 사용량 |
| created_at | timestamp | 생성일 |
| updated_at | timestamp | 수정일 |

### avatars

| Column | Type | 설명 |
|--------|------|------|
| id | uuid (PK) | 아바타 ID |
| user_id | uuid (FK) | 사용자 |
| gender | varchar(10) | 성별 |
| height_cm | float | 키 |
| weight_kg | float | 몸무게 |
| bust_cup | varchar(5) | 컵 사이즈 |
| body_type | varchar(20) | 체형 |
| vertex_count | int | 정점 수 |
| glb_path | text | S3 GLB 경로 |
| glb_size_bytes | int | GLB 크기 |
| is_active | boolean | 현재 활성 아바타 |
| created_at | timestamp | 생성일 |

### wardrobe_items

| Column | Type | 설명 |
|--------|------|------|
| id | uuid (PK) | 의류 ID |
| user_id | uuid (FK) | 사용자 |
| category | varchar(20) | 카테고리 (top/bottom/dress/outerwear) |
| subcategory | varchar(30) | 세부 카테고리 |
| color_hex | varchar(7) | 대표 색상 (#RRGGBB) |
| color_name | varchar(30) | 색상명 |
| pattern | varchar(20) | 패턴 |
| material | varchar(30) | 소재 |
| fit_type | varchar(20) | 핏 타입 |
| collar_type | varchar(30) | 칼라 타입 |
| sleeve_length | varchar(20) | 소매 길이 |
| measurements | jsonb | 의류 치수 (chest_cm, length_cm 등) |
| size_chart | jsonb | 사이즈 차트 |
| product_info | jsonb | 상품 정보 |
| image_urls | text[] | 이미지 URL 목록 |
| thumbnail_url | text | 썸네일 URL |
| created_at | timestamp | 생성일 |

### fitting_results

| Column | Type | 설명 |
|--------|------|------|
| id | uuid (PK) | 피팅 결과 ID |
| user_id | uuid (FK) | 사용자 |
| avatar_id | uuid (FK) | 아바타 |
| wardrobe_item_id | uuid (FK) | 의류 |
| method | varchar(20) | 생성 방법 (catvton-flux/gemini) |
| elapsed_sec | float | 소요 시간 |
| images | jsonb | {angle: image_url} |
| methods_per_angle | jsonb | {angle: method_name} |
| p2p_result | jsonb | P2P 분석 결과 |
| quality_score | float | 종합 품질 점수 |
| created_at | timestamp | 생성일 |

### viewer3d_models

| Column | Type | 설명 |
|--------|------|------|
| id | uuid (PK) | 3D 모델 ID |
| user_id | uuid (FK) | 사용자 |
| fitting_result_id | uuid (FK) | 피팅 결과 |
| glb_path | text | S3 GLB 경로 |
| glb_size_bytes | int | GLB 크기 |
| preview_urls | jsonb | {angle: preview_url} |
| elapsed_sec | float | 소요 시간 |
| quality_score | float | 품질 점수 |
| created_at | timestamp | 생성일 |

### face_banks

| Column | Type | 설명 |
|--------|------|------|
| id | uuid (PK) | Face Bank ID |
| user_id | uuid (FK) | 사용자 |
| gender | varchar(10) | 성별 |
| total_references | int | 참조 이미지 수 |
| angle_coverage | float | 각도 커버리지 (0~1) |
| reference_data | jsonb | 참조 이미지 메타 |
| created_at | timestamp | 생성일 |

### Indexes

```sql
CREATE INDEX idx_avatars_user ON avatars(user_id);
CREATE INDEX idx_wardrobe_user ON wardrobe_items(user_id);
CREATE INDEX idx_fitting_user ON fitting_results(user_id);
CREATE INDEX idx_fitting_avatar ON fitting_results(avatar_id);
CREATE INDEX idx_viewer3d_fitting ON viewer3d_models(fitting_result_id);
CREATE UNIQUE INDEX idx_users_email ON users(email);
CREATE UNIQUE INDEX idx_users_google ON users(google_id);
```

---

## 7. 파일 저장소 (S3/MinIO)

### Bucket 구조

```
stylelens-storage/
├── avatars/{userId}/{avatarId}/
│   └── avatar.glb
├── photos/{userId}/{photoId}/
│   └── face_768.webp
├── wardrobe/{userId}/{itemId}/
│   ├── original_01.webp
│   ├── original_02.webp
│   └── thumbnail.webp
├── fitting/{userId}/{fittingId}/
│   ├── angle_000.webp
│   ├── angle_045.webp
│   └── ...
├── viewer3d/{userId}/{modelId}/
│   ├── model.glb
│   └── preview_000.webp
└── temp/{uploadId}/
    └── upload.tmp
```

### 파일 제한

| 타입 | 최대 크기 | 형식 |
|------|----------|------|
| 이미지 | 10MB | JPG, PNG, WebP |
| 영상 | 100MB | MP4, MOV |
| 배치 업로드 | 50MB (총합) | 이미지 |
| GLB | 제한 없음 | glTF Binary |

### 저장 쿼터

| 등급 | 쿼터 |
|------|------|
| Free | 1GB |
| Standard | 5GB |
| Premium | 20GB |

### CDN (CloudFront)

| 콘텐츠 | Cache TTL |
|--------|-----------|
| Avatar GLB | 1일 |
| Wardrobe 이미지 | 7일 |
| Fitting 결과 | 7일 |
| 3D 모델 | 7일 |
| Temp | 캐시 안 함 |

### 수명 정책

| 경로 | TTL |
|------|-----|
| temp/ | 1일 |
| fitting/ | 90일 후 아카이브 |
| avatars/ | 영구 |
| wardrobe/ | 영구 |

---

## 8. Spring Boot 프록시 구현 가이드

### 프록시 테이블

| Frontend Path | AI Orchestrator Path | Method | Timeout |
|---------------|---------------------|--------|---------|
| /api/avatar/generate | /avatar/generate | POST (multipart) | LONG (120s) |
| /api/avatar/glb | /avatar/glb | GET (binary) | SHORT (10s) |
| /api/wardrobe/add-image | /wardrobe/add-image | POST (multipart) | MEDIUM (60s) |
| /api/wardrobe/add-images | /wardrobe/add-images | POST (multipart) | MEDIUM (60s) |
| /api/wardrobe/add-url | /wardrobe/add-url | POST (form) | MEDIUM (60s) |
| /api/wardrobe/extract-model-info | /wardrobe/extract-model-info | POST (multipart) | MEDIUM (60s) |
| /api/fitting/try-on | /fitting/try-on | POST (multipart) | LONG (300s) |
| /api/viewer3d/generate | /viewer3d/generate | POST (json) | LONG (300s) |
| /api/viewer3d/model/{id} | /viewer3d/model/{id} | GET (binary) | SHORT (10s) |
| /api/p2p/analyze | /p2p/analyze | POST (json) | MEDIUM (30s) |
| /api/face-bank/upload | /face-bank/upload | POST (multipart) | MEDIUM (30s) |
| /api/face-bank/status | /face-bank/{id}/status | GET (json) | SHORT (5s) |
| /api/quality/report | /quality/report | GET (json) | SHORT (5s) |

### Timeout 카테고리

| 카테고리 | 시간 | 용도 |
|---------|------|------|
| SHORT | 5~10s | 조회, 다운로드 |
| MEDIUM | 30~60s | 분석, 전처리 |
| LONG | 120~300s | GPU 연산 (피팅, 3D) |

### 503 재시도

Worker cold start 시 503 반환. Spring Boot에서 재시도:
- 대기: 30초
- 최대 재시도: 1회
- Backoff: Exponential

### 세션 매핑

```
Spring HttpSession (userId) → AI Orchestrator sessionId
```

- 사용자별 1개 AI 세션 (동시 요청 방지)
- AI 세션 TTL: 1시간
- 최대 동시 세션: 10개

---

## 9. 에러 코드 통합

### HTTP Status → 에러 코드 매핑

| AI 에러 | Spring 변환 | Frontend 처리 |
|---------|------------|--------------|
| 400 INVALID_INPUT | 400 Bad Request | 입력 폼 검증 표시 |
| 400 AVATAR_NOT_READY | 400 Bad Request | Phase 1 먼저 완료 안내 |
| 400 WARDROBE_EMPTY | 400 Bad Request | Phase 2 먼저 완료 안내 |
| 400 FITTING_NOT_READY | 400 Bad Request | Phase 3 먼저 완료 안내 |
| 404 SESSION_NOT_FOUND | 404 Not Found | 세션 만료 안내 |
| 413 FILE_TOO_LARGE | 413 Payload Too Large | 파일 크기 안내 |
| 422 QUALITY_GATE_FAIL | 422 Unprocessable | 품질 미달 → 재시도 안내 |
| 429 RATE_LIMITED | 429 Too Many Requests | 제한 초과 안내 + 대기시간 |
| 500 MODEL_ERROR | 502 Bad Gateway | 서버 오류 안내 |
| 502 GEMINI_ERROR | 502 Bad Gateway | AI 서비스 일시 오류 |
| 503 WORKER_UNAVAILABLE | 503 Service Unavailable | Worker 시작 중 → 재시도 |
| 504 WORKER_TIMEOUT | 504 Gateway Timeout | 처리 시간 초과 → 재시도 |

### 에러 응답 형식

```json
{
  "error": {
    "code": "AVATAR_NOT_READY",
    "message": "Phase 1(Avatar)을 먼저 완료해주세요.",
    "details": {
      "requiredPhase": "avatar",
      "currentPhase": null
    }
  },
  "timestamp": "2026-02-12T15:30:00Z"
}
```

---

## 10. 의류 분석 데이터 스키마

### ClothingAnalysis (Gemini 분석 결과)

```typescript
interface ClothingAnalysis {
  category: "top" | "bottom" | "dress" | "outerwear" | "accessory";
  subcategory: string;        // "t-shirt", "jeans", "mini-skirt", etc.
  colorHex: string;           // "#RRGGBB"
  colorName: string;          // "navy blue"
  pattern: string;            // "solid", "stripe", "plaid", "floral"
  material: string;           // "cotton", "polyester", "denim"
  fitType: string;            // "tight", "slim", "regular", "loose", "oversized"
  collarType?: string;        // "crew_neck", "v_neck", "collar", "hood"
  sleeveLength?: string;      // "sleeveless", "short", "3/4", "long"
  hemLength?: string;         // "crop", "regular", "long", "maxi"
  garmentMeasurements: {
    chestCm?: number;
    waistCm?: number;
    hipCm?: number;
    lengthCm?: number;
    shoulderCm?: number;
    sleevesCm?: number;
    inseamCm?: number;
  };
  features?: string[];        // ["buttons", "pockets", "zipper", "logo"]
  season?: string;            // "spring", "summer", "fall", "winter", "all"
  style?: string;             // "casual", "formal", "sport", "street"
}
```

---

## 11. Async Job 패턴 (Long-Running Operations)

Phase 3 (Fitting)과 Phase 4 (3D 생성)은 장시간 소요. Spring Boot에서 비동기 처리:

### 흐름

```
1. Frontend: POST /api/fitting/try-on
2. Spring Boot: 즉시 202 Accepted + jobId 반환
3. Spring Boot: 백그라운드에서 AI Orchestrator 호출
4. Frontend: WebSocket으로 진행상황 수신
5. 완료 시: WebSocket phase_complete 이벤트 + 결과 URL
6. Frontend: GET 결과 조회
```

### 대안: 동기 호출

WebSocket 미구현 시, 긴 타임아웃(300s)으로 동기 호출 가능.
단, 프론트엔드에서 로딩 UI 표시 필수.
