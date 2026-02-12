# StyleLens V6 Frontend API Contract

> **Version**: 1.0.0
> **Last Updated**: 2026-02-11
> **Target Audience**: React 19 + TypeScript Frontend Team
> **Communication Language**: Korean + English (기술 용어는 English, 설명은 Korean)

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Base URL & Authentication](#2-base-url--authentication)
3. [Error Handling](#3-error-handling)
4. [Session Management](#4-session-management)
5. [Phase 1: Avatar (3D Body)](#5-phase-1-avatar-3d-body)
6. [Phase 2: Wardrobe](#6-phase-2-wardrobe)
7. [Phase 3: Virtual Try-On](#7-phase-3-virtual-try-on)
8. [Phase 4: 3D Viewer](#8-phase-4-3d-viewer)
9. [P2P Physics Analysis](#9-p2p-physics-analysis)
10. [Quality & Health](#10-quality--health)
11. [TypeScript Interfaces](#11-typescript-interfaces)
12. [Image Handling (Base64 in React)](#12-image-handling-base64-in-react)
13. [Three.js GLB Integration](#13-threejs-glb-integration)
14. [Recommended React Component Structure](#14-recommended-react-component-structure)
15. [Quality Gate Display Pattern](#15-quality-gate-display-pattern)
16. [P2P Tightness Visualization](#16-p2p-tightness-visualization)
17. [Timing Expectations](#17-timing-expectations)

---

## 1. Overview & Architecture

StyleLens V6는 4-Tier 분산 가상 피팅 시스템입니다. Frontend는 **오직 Backend (Spring Boot)** 와만 통신하며, AI Orchestrator나 Model Worker를 직접 호출하지 않습니다.

```
Architecture Diagram
====================

 [Tier 1]              [Tier 2]              [Tier 3]              [Tier 4]
 Frontend              Backend               AI Orchestrator       Model Worker
 React 19              Spring Boot           FastAPI               PyTorch
 Vercel                Lightsail             Lightsail (No GPU)    Modal L40S GPU

 +------------+        +---------------+     +----------------+    +----------------+
 |            |  HTTPS |               | HTTP|                |gRPC|                |
 | React 19   |------->| Spring Boot   |---->| FastAPI        |--->| PyTorch        |
 | TypeScript  |<-------| JWT Auth      |<----| Orchestrator   |<---| SMPL/HMR2      |
 | Vercel CDN |        | Session Mgmt  |     | Task Queue     |    | CatVTON-Flux   |
 |            |        | Rate Limiting  |     | Gemini Client  |    | Texture Gen    |
 +------------+        +---------------+     +----------------+    +----------------+
       |                      |
       |                      |
  User Browser           AWS Lightsail         AWS Lightsail        Modal Cloud
                                                                   (On-demand GPU)
```

**Request Flow (요청 흐름)**:
1. Frontend --> Backend: REST API call (JWT 포함)
2. Backend --> AI Orchestrator: Internal proxy (session 검증 후 전달)
3. AI Orchestrator --> Model Worker: GPU 작업 위임 (gRPC/HTTP)
4. Response는 역순으로 전달

**핵심 원칙**:
- Frontend는 `{BACKEND_URL}/api/v6/...` 만 호출
- 모든 인증은 JWT 기반 (Spring Boot 발급)
- Session은 Backend가 자동 관리
- Binary 파일 (GLB)은 별도 GET endpoint로 다운로드
- 이미지 데이터는 base64-encoded JPEG string으로 전달

---

## 2. Base URL & Authentication

### Base URL

| Environment | URL |
|---|---|
| Local Development | `http://localhost:8080/api/v6` |
| Staging | `https://staging-api.stylelens.app/api/v6` |
| Production | `https://api.stylelens.app/api/v6` |

### Authentication (JWT)

Spring Boot가 발급하는 JWT token을 모든 요청의 `Authorization` header에 포함해야 합니다.

```typescript
// axios instance 설정 예시
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL + '/api/v6',
  timeout: 180_000, // 3분 — 긴 AI 작업 고려
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('stylelens_jwt');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

**인증 흐름**:
1. Google OAuth2 로그인 --> Spring Boot가 JWT 발급
2. JWT를 `localStorage` 또는 secure cookie에 저장
3. 매 요청마다 `Authorization: Bearer <token>` header 포함
4. Token 만료 시 refresh token으로 갱신

**JWT Payload 구조** (참고용):
```json
{
  "sub": "user@gmail.com",
  "iat": 1739260800,
  "exp": 1739347200,
  "roles": ["USER"],
  "session_id": "optional-current-session"
}
```

---

## 3. Error Handling

### HTTP Status Codes

| Status | 의미 | Frontend 대응 |
|---|---|---|
| `200` | 성공 | 정상 처리 |
| `201` | 리소스 생성 완료 | 성공 처리 (POST 응답) |
| `400` | 잘못된 요청 (validation 실패) | 입력값 검증 메시지 표시 |
| `401` | 인증 실패 / Token 만료 | 로그인 페이지로 redirect |
| `403` | 권한 없음 | 접근 불가 메시지 표시 |
| `404` | 리소스 없음 | Session/데이터 없음 안내 |
| `409` | 충돌 (중복 요청 등) | 이전 요청 완료 대기 안내 |
| `413` | 파일 크기 초과 | 파일 크기 제한 안내 |
| `422` | 처리 불가 (AI 분석 실패 등) | quality gate 실패 상세 표시 |
| `429` | Rate limit 초과 | 재시도 대기 안내 (Retry-After header 확인) |
| `500` | 서버 내부 오류 | 일반 오류 메시지 + 재시도 버튼 |
| `502` | AI Worker 응답 없음 | GPU 서버 상태 확인 중 메시지 |
| `503` | 서비스 점검 / 과부하 | 잠시 후 재시도 안내 |
| `504` | AI 작업 timeout | 긴 작업 재시도 또는 quality 하향 안내 |

### Error Response Format

모든 에러 응답은 아래 형식을 따릅니다:

```json
{
  "error": {
    "code": "AVATAR_PERSON_NOT_DETECTED",
    "message": "영상에서 사람을 감지할 수 없습니다. 전신이 보이는 영상을 사용해주세요.",
    "message_en": "Could not detect a person in the video. Please use a video showing the full body.",
    "details": {
      "stage": "person_detection",
      "score": 0.12,
      "threshold": 0.5
    },
    "retry_allowed": true,
    "suggestion": "다시 촬영하거나 조명이 밝은 환경에서 재시도해주세요."
  },
  "request_id": "abc12345",
  "session_id": "uuid-string",
  "timestamp": "2026-02-11T12:00:00Z"
}
```

### Error Code 목록 (주요 코드)

| Code | Phase | 설명 |
|---|---|---|
| `AUTH_TOKEN_EXPIRED` | - | JWT 만료 |
| `AUTH_TOKEN_INVALID` | - | JWT 형식 오류 |
| `SESSION_NOT_FOUND` | - | 유효하지 않은 session_id |
| `SESSION_EXPIRED` | - | Session 만료 (기본 24시간) |
| `AVATAR_PERSON_NOT_DETECTED` | 1 | 사람 감지 실패 |
| `AVATAR_MULTI_PERSON` | 1 | 복수 인물 감지 |
| `AVATAR_LOW_QUALITY` | 1 | 입력 영상/이미지 품질 부족 |
| `AVATAR_BODY_OCCLUDED` | 1 | 신체 가림 (일부만 보임) |
| `WARDROBE_NOT_CLOTHING` | 2 | 의류가 아닌 이미지 |
| `WARDROBE_ANALYSIS_FAILED` | 2 | 의류 분석 실패 |
| `WARDROBE_IMAGE_TOO_SMALL` | 2 | 이미지 해상도 부족 |
| `FITTING_PHASE1_REQUIRED` | 3 | Avatar 미생성 |
| `FITTING_PHASE2_REQUIRED` | 3 | Wardrobe 미등록 |
| `FITTING_GENERATION_FAILED` | 3 | Try-on 이미지 생성 실패 |
| `FITTING_GPU_UNAVAILABLE` | 3 | GPU Worker 없음 |
| `VIEWER_PHASE3_REQUIRED` | 4 | Try-on 미완료 |
| `VIEWER_GLB_NOT_FOUND` | 4 | GLB 파일 없음 |
| `RATE_LIMIT_EXCEEDED` | - | 요청 횟수 초과 |
| `FILE_TOO_LARGE` | - | 파일 크기 초과 |
| `WORKER_TIMEOUT` | 3,4 | GPU Worker timeout |

### React Error Handler 예시

```typescript
import { AxiosError } from 'axios';

interface ApiError {
  error: {
    code: string;
    message: string;
    message_en: string;
    details?: Record<string, unknown>;
    retry_allowed: boolean;
    suggestion?: string;
  };
  request_id: string;
  session_id?: string;
  timestamp: string;
}

function handleApiError(err: AxiosError<ApiError>) {
  const status = err.response?.status;
  const body = err.response?.data;

  if (status === 401) {
    // Token 만료 — 로그인으로 redirect
    window.location.href = '/login';
    return;
  }

  if (status === 429) {
    const retryAfter = err.response?.headers['retry-after'];
    toast.warn(`요청이 너무 많습니다. ${retryAfter}초 후 재시도해주세요.`);
    return;
  }

  if (body?.error) {
    toast.error(body.error.message);
    if (body.error.suggestion) {
      toast.info(body.error.suggestion);
    }
  } else {
    toast.error('알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
  }
}
```

---

## 4. Session Management

Session은 Backend가 자동 관리합니다. Frontend는 `session_id`를 응답에서 받아 이후 요청에 사용합니다.

### Session 생명주기

```
User Login --> 첫 API 요청 --> Backend가 session_id 할당
                                     |
                            session_id를 state에 저장
                                     |
                            이후 요청에 session_id 포함
                                     |
                            24시간 후 자동 만료
```

### Session 사용 패턴

```typescript
// React Context로 session 관리
const [sessionId, setSessionId] = useState<string | null>(null);

// Phase 1 응답에서 session_id 획득
const result = await api.post('/avatar/generate', formData);
setSessionId(result.data.session_id);

// 이후 Phase에서 session_id 사용 (query param 또는 body에 포함)
const glb = await api.get(`/avatar/glb?session_id=${sessionId}`, {
  responseType: 'arraybuffer',
});
```

**주의사항**:
- `session_id`는 Phase 1 응답에서 처음 발급됩니다.
- 같은 session 내에서 Phase 1~4가 순차적으로 진행됩니다.
- Session이 만료되면 Phase 1부터 다시 시작해야 합니다.
- 복수 session 동시 진행 가능 (각각 독립적).

---

## 5. Phase 1: Avatar (3D Body)

사용자의 동영상 또는 이미지에서 3D body mesh를 생성합니다.

### POST /api/v6/avatar/generate

**Description**: 동영상(mp4) 또는 이미지(jpg/png)를 업로드하여 3D avatar를 생성합니다. 내부적으로 YOLOv8 person detection, HMR2 body reconstruction, SMPL mesh 생성, FLAME head 합성, Hair asset 부착이 순차적으로 실행됩니다.

**Content-Type**: `multipart/form-data`

**Request Fields**:

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | File (mp4) | video 또는 image 중 하나 필수 | 전신이 보이는 동영상 (3-15초 권장) |
| `image` | File (jpg/png) | video 또는 image 중 하나 필수 | 전신이 보이는 이미지 |
| `gender` | string | **Required** | `"male"` 또는 `"female"` |
| `height_cm` | float | **Required** | 키 (cm). 예: `165.5` |
| `weight_kg` | float | **Required** | 체중 (kg). 예: `55.0` |
| `bust_cup` | string | Optional | 여성만. `"A"`, `"B"`, `"C"`, `"D"` |
| `body_type` | string | **Required** | `"standard"`, `"athletic"`, `"slim"`, `"plus"` |

**Request 예시 (TypeScript)**:

```typescript
async function generateAvatar(params: {
  file: File;
  fileType: 'video' | 'image';
  gender: 'male' | 'female';
  heightCm: number;
  weightKg: number;
  bustCup?: string;
  bodyType: 'standard' | 'athletic' | 'slim' | 'plus';
}): Promise<AvatarResponse> {
  const formData = new FormData();

  // video 또는 image 필드로 파일 추가
  formData.append(params.fileType, params.file);
  formData.append('gender', params.gender);
  formData.append('height_cm', params.heightCm.toString());
  formData.append('weight_kg', params.weightKg.toString());
  formData.append('body_type', params.bodyType);

  if (params.bustCup) {
    formData.append('bust_cup', params.bustCup);
  }

  const response = await api.post<AvatarResponse>(
    '/avatar/generate',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120_000, // 최대 2분 대기
    }
  );

  return response.data;
}
```

**Response** (`200 OK`):

```json
{
  "request_id": "abc12345",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "gender": "female",
  "has_mesh": true,
  "vertex_count": 6890,
  "glb_size_bytes": 524288,
  "renders": {
    "0": "base64-jpeg-string...",
    "45": "base64-jpeg-string...",
    "90": "base64-jpeg-string...",
    "135": "base64-jpeg-string...",
    "180": "base64-jpeg-string...",
    "225": "base64-jpeg-string...",
    "270": "base64-jpeg-string...",
    "315": "base64-jpeg-string..."
  },
  "quality_gates": [
    {
      "stage": "person_detection",
      "score": 0.95,
      "pass": true,
      "feedback": "인물이 명확하게 감지되었습니다."
    },
    {
      "stage": "body_reconstruction",
      "score": 0.88,
      "pass": true,
      "feedback": "신체 구조가 정상적으로 복원되었습니다."
    },
    {
      "stage": "mesh_quality",
      "score": 0.82,
      "pass": true,
      "feedback": "메쉬 품질이 양호합니다."
    }
  ]
}
```

**Timing**: 15~60초 (입력 타입, 서버 부하에 따라 변동)

**파일 크기 제한**:
- Video: 최대 50MB
- Image: 최대 10MB

---

### GET /api/v6/avatar/glb

**Description**: 생성된 3D avatar의 GLB 파일을 다운로드합니다. Three.js viewer에서 사용합니다.

**Query Parameters**:

| Param | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | **Required** | Phase 1 응답의 session_id |

**Response**: Binary GLB file

| Header | Value |
|---|---|
| `Content-Type` | `model/gltf-binary` |
| `Content-Disposition` | `attachment; filename="avatar.glb"` |
| `Content-Length` | file size in bytes |

**Request 예시**:

```typescript
async function downloadGlb(sessionId: string): Promise<ArrayBuffer> {
  const response = await api.get('/avatar/glb', {
    params: { session_id: sessionId },
    responseType: 'arraybuffer',
  });
  return response.data;
}
```

---

## 6. Phase 2: Wardrobe

의류 이미지를 분석하여 40개 이상의 속성을 추출합니다.

### POST /api/v6/wardrobe/add-image

**Description**: 단일 의류 이미지를 업로드하고 분석합니다.

**Content-Type**: `multipart/form-data`

**Request Fields**:

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File (jpg/png/webp) | **Required** | 의류 이미지 (최대 10MB) |

**Response** (`200 OK`):

```json
{
  "request_id": "def67890",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "analysis": {
    "name": "리넨 오버핏 셔츠",
    "category": "top",
    "sub_category": "shirt",
    "color_hex": "#E8DCC8",
    "color_name": "베이지",
    "secondary_color_hex": null,
    "pattern": "solid",
    "pattern_detail": null,
    "fabric": "linen",
    "fabric_weight": "light",
    "transparency": "opaque",
    "fit_style": "overfit",
    "silhouette": "boxy",
    "length": "regular",
    "sleeve_length": "long",
    "sleeve_style": "straight",
    "neckline": "collar",
    "collar_type": "button_down",
    "closure": "button_front",
    "closure_detail": "full_button",
    "buttons": true,
    "button_count": 7,
    "button_style": "shell",
    "pockets": true,
    "pocket_count": 1,
    "pocket_type": "chest_patch",
    "hem_style": "curved",
    "cuff_style": "button",
    "darts": false,
    "pleats": false,
    "elastic": false,
    "drawstring": false,
    "belt_loops": false,
    "lining": false,
    "hood": false,
    "zipper": false,
    "embellishment": null,
    "print_graphic": null,
    "brand_visible": false,
    "season": "spring_summer",
    "occasion": "casual",
    "gender_target": "unisex",
    "size_visible": null,
    "care_label_visible": false
  },
  "quality_gates": [
    {
      "stage": "clothing_detection",
      "score": 0.97,
      "pass": true,
      "feedback": "의류가 명확하게 감지되었습니다."
    },
    {
      "stage": "attribute_extraction",
      "score": 0.91,
      "pass": true,
      "feedback": "속성이 정상적으로 추출되었습니다."
    }
  ]
}
```

---

### POST /api/v6/wardrobe/add-images

**Description**: 복수 의류 이미지를 업로드하고 통합 분석합니다. 여러 각도(정면, 후면, 디테일 등)의 이미지를 함께 분석하여 더 정확한 결과를 제공합니다.

**Content-Type**: `multipart/form-data`

**Request Fields**:

| Field | Type | Required | Description |
|---|---|---|---|
| `images[]` | File[] (1-10개) | **Required** | 의류 이미지 배열 |
| `size_chart` | File (jpg/png) | Optional | 사이즈 차트 이미지 |
| `product_info_1` | File (jpg/png) | Optional | 제품 정보 이미지 1 (소재, 세탁법 등) |
| `product_info_2` | File (jpg/png) | Optional | 제품 정보 이미지 2 |
| `fitting_model` | File (jpg/png) | Optional | 피팅 모델 이미지 (착용 사진) |

**Request 예시**:

```typescript
async function addMultipleImages(params: {
  images: File[];
  sizeChart?: File;
  productInfo1?: File;
  productInfo2?: File;
  fittingModel?: File;
}): Promise<WardrobeMultiResponse> {
  const formData = new FormData();

  // 복수 이미지: 동일한 key "images[]"로 append
  params.images.forEach((img) => {
    formData.append('images[]', img);
  });

  if (params.sizeChart) formData.append('size_chart', params.sizeChart);
  if (params.productInfo1) formData.append('product_info_1', params.productInfo1);
  if (params.productInfo2) formData.append('product_info_2', params.productInfo2);
  if (params.fittingModel) formData.append('fitting_model', params.fittingModel);

  const response = await api.post<WardrobeMultiResponse>(
    '/wardrobe/add-images',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60_000,
    }
  );

  return response.data;
}
```

**Response** (`200 OK`):

```json
{
  "request_id": "ghi11223",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "analysis": {
    "name": "와이드 데님 팬츠",
    "category": "bottom",
    "sub_category": "jeans",
    "color_hex": "#4A6A8A",
    "pattern": "solid",
    "fabric": "denim",
    "fit_style": "wide",
    "length": "full",
    "closure": "zipper_button",
    "pockets": true,
    "pocket_count": 4,
    "belt_loops": true
  },
  "size_chart": {
    "sizes": ["S", "M", "L", "XL"],
    "measurements": {
      "waist_cm": [66, 70, 74, 78],
      "hip_cm": [92, 96, 100, 104],
      "inseam_cm": [78, 79, 80, 81],
      "thigh_cm": [58, 60, 62, 64]
    }
  },
  "product_info": {
    "material_composition": "cotton 98%, elastane 2%",
    "care_instructions": ["machine_wash_cold", "tumble_dry_low"],
    "country_of_origin": "Bangladesh",
    "price": null,
    "brand": null
  },
  "fitting_model_info": {
    "height_cm": 170.0,
    "weight_kg": null,
    "size_worn": "M",
    "body_measurements": {
      "waist_cm": 68,
      "hip_cm": 94
    }
  },
  "quality_gates": [
    {
      "stage": "multi_view_merge",
      "score": 0.93,
      "pass": true,
      "feedback": "4개 이미지에서 속성이 통합되었습니다."
    }
  ]
}
```

---

### POST /api/v6/wardrobe/add-url

**Description**: URL에서 의류 이미지를 가져와 분석합니다. 쇼핑몰 상품 페이지 URL을 지원합니다.

**Content-Type**: `application/x-www-form-urlencoded`

**Request Fields**:

| Field | Type | Required | Description |
|---|---|---|---|
| `url` | string | **Required** | 의류 이미지 URL 또는 상품 페이지 URL |

**Request 예시**:

```typescript
async function addByUrl(url: string): Promise<WardrobeResponse> {
  const response = await api.post<WardrobeResponse>(
    '/wardrobe/add-url',
    new URLSearchParams({ url }),
    {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      timeout: 30_000,
    }
  );
  return response.data;
}
```

**Response**: `add-image`과 동일한 형식.

---

### POST /api/v6/wardrobe/extract-model-info

**Description**: 피팅 모델 이미지에서 신체 치수 정보를 추출합니다.

**Content-Type**: `multipart/form-data`

**Request Fields**:

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | File (jpg/png) | **Required** | 피팅 모델 이미지 |

**Response** (`200 OK`):

```json
{
  "height_cm": 172.0,
  "weight_kg": 58.0,
  "body_measurements": {
    "shoulder_cm": 42.5,
    "chest_cm": 88.0,
    "waist_cm": 68.0,
    "hip_cm": 95.0,
    "arm_length_cm": 58.0,
    "inseam_cm": 78.0,
    "thigh_cm": 54.0
  }
}
```

---

## 7. Phase 3: Virtual Try-On

Avatar와 Wardrobe가 모두 준비된 상태에서 가상 피팅 이미지를 생성합니다.

### POST /api/v6/fitting/try-on

**Description**: 8각도 가상 피팅 이미지를 생성합니다. Phase 1 (Avatar) + Phase 2 (Wardrobe)가 완료된 session이 필요합니다.

**Content-Type**: `multipart/form-data`

**Request Fields**:

| Field | Type | Required | Description |
|---|---|---|---|
| `face_photo` | File (jpg/png) | Optional | 얼굴 사진 (face identity 유지용, 768x768 권장) |

> **Note**: `session_id`는 이미 JWT 또는 header를 통해 Backend가 자동으로 식별합니다. 별도로 전송할 필요 없습니다.

**Request 예시**:

```typescript
async function tryOn(facePhoto?: File): Promise<FittingResponse> {
  const formData = new FormData();

  if (facePhoto) {
    formData.append('face_photo', facePhoto);
  }

  const response = await api.post<FittingResponse>(
    '/fitting/try-on',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 180_000, // 최대 3분 — GPU 작업 포함
    }
  );

  return response.data;
}
```

**Response** (`200 OK`):

```json
{
  "request_id": "jkl44556",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "images": {
    "0": "base64-jpeg-front...",
    "45": "base64-jpeg...",
    "90": "base64-jpeg...",
    "135": "base64-jpeg...",
    "180": "base64-jpeg...",
    "225": "base64-jpeg...",
    "270": "base64-jpeg...",
    "315": "base64-jpeg..."
  },
  "methods": {
    "0": "catvton_flux_worker",
    "45": "catvton_flux_worker",
    "90": "catvton_flux_worker",
    "135": "catvton_flux_worker",
    "180": "catvton_flux_worker",
    "225": "catvton_flux_worker",
    "270": "catvton_flux_worker",
    "315": "catvton_flux_worker"
  },
  "elapsed_sec": 45.2,
  "quality_gates": [
    {
      "stage": "face_identity",
      "score": 0.89,
      "pass": true,
      "feedback": "얼굴 유사도가 양호합니다."
    },
    {
      "stage": "clothing_fit",
      "score": 0.92,
      "pass": true,
      "feedback": "의류 피팅이 자연스럽습니다."
    }
  ],
  "p2p": {
    "physics_prompt": "fitted torso, natural drape at waist, slight stretch across bust",
    "overall_tightness": "fitted",
    "mask_expansion_factor": 1.05,
    "confidence": 0.87,
    "method": "ensemble_v1",
    "deltas": [
      {
        "body_part": "chest",
        "delta_cm": -3.2,
        "tightness": "fitted",
        "visual_keywords": ["smooth", "contoured"]
      },
      {
        "body_part": "waist",
        "delta_cm": 2.1,
        "tightness": "regular",
        "visual_keywords": ["natural_drape", "gentle_fold"]
      },
      {
        "body_part": "hip",
        "delta_cm": -1.8,
        "tightness": "semi_fitted",
        "visual_keywords": ["slight_pull"]
      },
      {
        "body_part": "shoulder",
        "delta_cm": 4.5,
        "tightness": "loose",
        "visual_keywords": ["dropped_shoulder", "relaxed"]
      }
    ]
  }
}
```

**`methods` 필드 설명**: 각 각도별 이미지 생성에 사용된 AI 모델/방법. 디버깅 및 품질 추적용.

| Method | 설명 |
|---|---|
| `catvton_flux_worker` | CatVTON-Flux GPU worker (기본, 최고 품질) |
| `gemini_image_gen` | Gemini 이미지 생성 (fallback) |
| `sdxl_lightning` | SDXL Lightning (legacy fallback) |

**Timing**: 30~120초 (GPU Worker 가용성에 따라 변동)

---

## 8. Phase 4: 3D Viewer

피팅 결과를 3D GLB 모델로 변환합니다. Three.js로 360도 회전 뷰를 제공할 수 있습니다.

### POST /api/v6/viewer3d/generate

**Description**: Phase 3의 피팅 결과를 3D GLB 모델로 변환합니다.

**Content-Type**: `application/json` (body 없음, session 기반)

**Request 예시**:

```typescript
async function generate3DViewer(): Promise<Viewer3DResponse> {
  const response = await api.post<Viewer3DResponse>(
    '/viewer3d/generate',
    {},
    { timeout: 300_000 } // 최대 5분
  );
  return response.data;
}
```

**Response** (`200 OK`):

```json
{
  "request_id": "mno77889",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "glb_id": "abc12345",
  "glb_size_bytes": 1048576,
  "glb_url": "/viewer3d/model/abc12345",
  "previews": {
    "0": "base64-jpeg-preview...",
    "45": "base64-jpeg-preview...",
    "90": "base64-jpeg-preview...",
    "135": "base64-jpeg-preview...",
    "180": "base64-jpeg-preview...",
    "225": "base64-jpeg-preview...",
    "270": "base64-jpeg-preview...",
    "315": "base64-jpeg-preview..."
  },
  "elapsed_sec": 120.5,
  "quality_gates": [
    {
      "stage": "texture_mapping",
      "score": 0.85,
      "pass": true,
      "feedback": "텍스처 매핑이 완료되었습니다."
    },
    {
      "stage": "mesh_export",
      "score": 0.95,
      "pass": true,
      "feedback": "GLB 파일이 정상적으로 생성되었습니다."
    }
  ]
}
```

---

### GET /api/v6/viewer3d/model/{glb_id}

**Description**: 생성된 3D fitting GLB 파일을 다운로드합니다.

**Path Parameters**:

| Param | Type | Description |
|---|---|---|
| `glb_id` | string | `viewer3d/generate` 응답의 `glb_id` |

**Response**: Binary GLB file

| Header | Value |
|---|---|
| `Content-Type` | `model/gltf-binary` |
| `Content-Length` | file size in bytes |
| `Cache-Control` | `public, max-age=86400` |

**Request 예시**:

```typescript
async function download3DModel(glbId: string): Promise<ArrayBuffer> {
  const response = await api.get(`/viewer3d/model/${glbId}`, {
    responseType: 'arraybuffer',
  });
  return response.data;
}
```

---

## 9. P2P Physics Analysis

Body-to-clothing physics 분석을 독립적으로 수행합니다. Phase 1 + Phase 2가 완료된 session이 필요합니다.

### POST /api/v6/p2p/analyze

**Description**: 사용자 신체와 의류 간의 물리적 핏 관계를 분석합니다. 각 신체 부위별 여유/타이트함, 시각적 키워드를 반환합니다.

**Content-Type**: `application/json` (body 없음, session 기반)

**Response** (`200 OK`):

```json
{
  "physics_prompt": "fitted torso with slight stretch across bust, natural drape at waist, relaxed through hip",
  "overall_tightness": "fitted",
  "mask_expansion_factor": 1.05,
  "confidence": 0.87,
  "method": "ensemble_v1",
  "ensemble": {
    "rule_based": {
      "overall_tightness": "fitted",
      "confidence": 0.82
    },
    "ml_based": {
      "overall_tightness": "semi_fitted",
      "confidence": 0.91
    },
    "final_method": "weighted_average"
  },
  "deltas": [
    {
      "body_part": "chest",
      "delta_cm": -3.2,
      "tightness": "fitted",
      "visual_keywords": ["smooth", "contoured", "slight_stretch"]
    },
    {
      "body_part": "waist",
      "delta_cm": 2.1,
      "tightness": "regular",
      "visual_keywords": ["natural_drape", "gentle_fold"]
    },
    {
      "body_part": "hip",
      "delta_cm": -1.8,
      "tightness": "semi_fitted",
      "visual_keywords": ["slight_pull", "smooth_line"]
    },
    {
      "body_part": "shoulder",
      "delta_cm": 4.5,
      "tightness": "loose",
      "visual_keywords": ["dropped_shoulder", "relaxed", "excess_fabric"]
    },
    {
      "body_part": "thigh",
      "delta_cm": 6.0,
      "tightness": "loose",
      "visual_keywords": ["flowing", "space_between"]
    },
    {
      "body_part": "arm",
      "delta_cm": 3.8,
      "tightness": "regular",
      "visual_keywords": ["comfortable", "natural_hang"]
    }
  ]
}
```

**Tightness 레벨 (순서)**:

| Level | delta_cm 범위 (대략) | 사용자 표시 (Korean) | Color (권장) |
|---|---|---|---|
| `tight` | < -5 | 타이트 | `#EF4444` (red) |
| `fitted` | -5 ~ -1 | 핏 | `#F97316` (orange) |
| `semi_fitted` | -1 ~ 2 | 세미핏 | `#EAB308` (yellow) |
| `regular` | 2 ~ 5 | 레귤러 | `#22C55E` (green) |
| `loose` | 5 ~ 10 | 루즈 | `#3B82F6` (blue) |
| `oversize` | > 10 | 오버사이즈 | `#8B5CF6` (purple) |

---

## 10. Quality & Health

### GET /api/v6/quality/report

**Description**: 현재 session의 전체 quality gate 요약을 반환합니다.

**Query Parameters**:

| Param | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | Optional | 특정 session. 미지정 시 현재 session |

**Response** (`200 OK`):

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "phases": {
    "avatar": {
      "completed": true,
      "overall_score": 0.88,
      "gates": [
        {"stage": "person_detection", "score": 0.95, "pass": true},
        {"stage": "body_reconstruction", "score": 0.88, "pass": true},
        {"stage": "mesh_quality", "score": 0.82, "pass": true}
      ]
    },
    "wardrobe": {
      "completed": true,
      "overall_score": 0.94,
      "gates": [
        {"stage": "clothing_detection", "score": 0.97, "pass": true},
        {"stage": "attribute_extraction", "score": 0.91, "pass": true}
      ]
    },
    "fitting": {
      "completed": true,
      "overall_score": 0.90,
      "gates": [
        {"stage": "face_identity", "score": 0.89, "pass": true},
        {"stage": "clothing_fit", "score": 0.92, "pass": true}
      ]
    },
    "viewer3d": {
      "completed": false,
      "overall_score": null,
      "gates": []
    }
  }
}
```

---

### GET /api/v6/health

**Description**: 시스템 상태를 확인합니다. 인증 불필요.

**Response** (`200 OK`):

```json
{
  "status": "healthy",
  "version": "6.0.0",
  "components": {
    "backend": {"status": "healthy", "latency_ms": 2},
    "orchestrator": {"status": "healthy", "latency_ms": 45},
    "gpu_worker": {"status": "healthy", "available_gpus": 2, "queue_length": 3}
  },
  "timestamp": "2026-02-11T12:00:00Z"
}
```

**Health Status 값**:
- `healthy` : 정상
- `degraded` : 일부 기능 제한 (예: GPU 1대만 가동)
- `unhealthy` : 서비스 불가

---

### GET /api/v6/sessions

**Description**: 현재 사용자의 활성 session 목록을 반환합니다.

**Response** (`200 OK`):

```json
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "created_at": "2026-02-11T10:00:00Z",
      "last_activity": "2026-02-11T11:30:00Z",
      "phases_completed": ["avatar", "wardrobe", "fitting"],
      "expires_at": "2026-02-12T10:00:00Z"
    }
  ]
}
```

---

## 11. TypeScript Interfaces

아래 interface 정의를 프로젝트에서 사용하세요. `types/api.ts` 등의 파일에 저장하는 것을 권장합니다.

```typescript
// ============================================================
// types/api.ts — StyleLens V6 API Type Definitions
// ============================================================

// ---- Common Types ----

export interface QualityGate {
  stage: string;
  score: number;
  pass: boolean;
  feedback: string;
}

export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
    message_en: string;
    details?: Record<string, unknown>;
    retry_allowed: boolean;
    suggestion?: string;
  };
  request_id: string;
  session_id?: string;
  timestamp: string;
}

/** 8-angle render map. Key는 각도 (0, 45, 90, ..., 315), value는 base64 JPEG */
export type AngleRenderMap = Record<
  '0' | '45' | '90' | '135' | '180' | '225' | '270' | '315',
  string
>;

/** 8-angle method map. 각 각도별 이미지 생성 방법 */
export type AngleMethodMap = Record<
  '0' | '45' | '90' | '135' | '180' | '225' | '270' | '315',
  string
>;

// ---- Phase 1: Avatar ----

export type Gender = 'male' | 'female';
export type BodyType = 'standard' | 'athletic' | 'slim' | 'plus';
export type BustCup = 'A' | 'B' | 'C' | 'D';

export interface AvatarGenerateRequest {
  file: File;
  fileType: 'video' | 'image';
  gender: Gender;
  heightCm: number;
  weightKg: number;
  bustCup?: BustCup;
  bodyType: BodyType;
}

export interface AvatarResponse {
  request_id: string;
  session_id: string;
  gender: Gender;
  has_mesh: boolean;
  vertex_count: number;
  glb_size_bytes: number;
  renders: AngleRenderMap;
  quality_gates: QualityGate[];
}

// ---- Phase 2: Wardrobe ----

export type ClothingCategory = 'top' | 'bottom' | 'dress' | 'outerwear';

export interface ClothingAnalysis {
  name: string;
  category: ClothingCategory;
  sub_category: string;
  color_hex: string;
  color_name?: string;
  secondary_color_hex?: string | null;
  pattern: string;
  pattern_detail?: string | null;
  fabric: string;
  fabric_weight?: string;
  transparency?: string;
  fit_style: string;
  silhouette?: string;
  length?: string;
  sleeve_length?: string;
  sleeve_style?: string;
  neckline?: string;
  collar_type?: string;
  closure?: string;
  closure_detail?: string;
  buttons?: boolean;
  button_count?: number;
  button_style?: string;
  pockets?: boolean;
  pocket_count?: number;
  pocket_type?: string;
  hem_style?: string;
  cuff_style?: string;
  darts?: boolean;
  pleats?: boolean;
  elastic?: boolean;
  drawstring?: boolean;
  belt_loops?: boolean;
  lining?: boolean;
  hood?: boolean;
  zipper?: boolean;
  embellishment?: string | null;
  print_graphic?: string | null;
  brand_visible?: boolean;
  season?: string;
  occasion?: string;
  gender_target?: string;
  size_visible?: string | null;
  care_label_visible?: boolean;
}

export interface WardrobeResponse {
  request_id: string;
  session_id: string;
  analysis: ClothingAnalysis;
  quality_gates: QualityGate[];
}

export interface SizeChart {
  sizes: string[];
  measurements: Record<string, number[]>;
}

export interface ProductInfo {
  material_composition?: string;
  care_instructions?: string[];
  country_of_origin?: string;
  price?: number | null;
  brand?: string | null;
}

export interface FittingModelInfo {
  height_cm?: number;
  weight_kg?: number | null;
  size_worn?: string;
  body_measurements?: Record<string, number>;
}

export interface WardrobeMultiResponse {
  request_id: string;
  session_id: string;
  analysis: ClothingAnalysis;
  size_chart?: SizeChart;
  product_info?: ProductInfo;
  fitting_model_info?: FittingModelInfo;
  quality_gates: QualityGate[];
}

export interface ExtractModelInfoResponse {
  height_cm: number;
  weight_kg: number;
  body_measurements: Record<string, number>;
}

// ---- Phase 3: Fitting ----

export type TightnessLevel =
  | 'tight'
  | 'fitted'
  | 'semi_fitted'
  | 'regular'
  | 'loose'
  | 'oversize';

export interface P2PDelta {
  body_part: string;
  delta_cm: number;
  tightness: TightnessLevel;
  visual_keywords: string[];
}

export interface P2PAnalysis {
  physics_prompt: string;
  overall_tightness: TightnessLevel;
  mask_expansion_factor: number;
  confidence: number;
  method: string;
  deltas: P2PDelta[];
}

export interface P2PFullAnalysis extends P2PAnalysis {
  ensemble?: {
    rule_based: { overall_tightness: TightnessLevel; confidence: number };
    ml_based: { overall_tightness: TightnessLevel; confidence: number };
    final_method: string;
  };
}

export interface FittingResponse {
  request_id: string;
  session_id: string;
  images: AngleRenderMap;
  methods: AngleMethodMap;
  elapsed_sec: number;
  quality_gates: QualityGate[];
  p2p: P2PAnalysis;
}

// ---- Phase 4: 3D Viewer ----

export interface Viewer3DResponse {
  request_id: string;
  session_id: string;
  glb_id: string;
  glb_size_bytes: number;
  glb_url: string;
  previews: AngleRenderMap;
  elapsed_sec: number;
  quality_gates: QualityGate[];
}

// ---- Quality & Health ----

export interface PhaseQuality {
  completed: boolean;
  overall_score: number | null;
  gates: QualityGate[];
}

export interface QualityReport {
  session_id: string;
  phases: {
    avatar: PhaseQuality;
    wardrobe: PhaseQuality;
    fitting: PhaseQuality;
    viewer3d: PhaseQuality;
  };
}

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

export interface HealthResponse {
  status: HealthStatus;
  version: string;
  components: {
    backend: { status: HealthStatus; latency_ms: number };
    orchestrator: { status: HealthStatus; latency_ms: number };
    gpu_worker: {
      status: HealthStatus;
      available_gpus: number;
      queue_length: number;
    };
  };
  timestamp: string;
}

export interface SessionInfo {
  session_id: string;
  created_at: string;
  last_activity: string;
  phases_completed: string[];
  expires_at: string;
}

export interface SessionsResponse {
  sessions: SessionInfo[];
}
```

---

## 12. Image Handling (Base64 in React)

API 응답의 이미지 데이터는 base64-encoded JPEG string입니다. React에서 표시하는 방법:

### Base64 이미지 표시

```tsx
// 기본 표시 방법
function AvatarRender({ base64Data }: { base64Data: string }) {
  // 응답 데이터에 data:image/jpeg;base64, prefix가 포함되어 있지 않으므로 직접 추가
  const src = `data:image/jpeg;base64,${base64Data}`;

  return (
    <img
      src={src}
      alt="Avatar render"
      loading="lazy"
      style={{ maxWidth: '100%', height: 'auto' }}
    />
  );
}
```

### 8-Angle Carousel Component

```tsx
import { useState, useMemo } from 'react';
import type { AngleRenderMap } from '@/types/api';

const ANGLES = ['0', '45', '90', '135', '180', '225', '270', '315'] as const;

const ANGLE_LABELS: Record<string, string> = {
  '0': '정면',
  '45': '우측 45도',
  '90': '우측',
  '135': '우측 후면',
  '180': '후면',
  '225': '좌측 후면',
  '270': '좌측',
  '315': '좌측 45도',
};

interface AngleCarouselProps {
  renders: AngleRenderMap;
  className?: string;
}

function AngleCarousel({ renders, className }: AngleCarouselProps) {
  const [currentAngle, setCurrentAngle] = useState<(typeof ANGLES)[number]>('0');

  const imageSrc = useMemo(
    () => `data:image/jpeg;base64,${renders[currentAngle]}`,
    [renders, currentAngle]
  );

  return (
    <div className={className}>
      {/* Main image */}
      <div className="relative aspect-[3/4] bg-gray-100 rounded-lg overflow-hidden">
        <img
          src={imageSrc}
          alt={`${ANGLE_LABELS[currentAngle]} 뷰`}
          className="w-full h-full object-contain"
        />
        <span className="absolute bottom-2 left-2 bg-black/60 text-white text-sm px-2 py-1 rounded">
          {ANGLE_LABELS[currentAngle]}
        </span>
      </div>

      {/* Angle selector thumbnails */}
      <div className="flex gap-1 mt-2 overflow-x-auto">
        {ANGLES.map((angle) => (
          <button
            key={angle}
            onClick={() => setCurrentAngle(angle)}
            className={`flex-shrink-0 w-16 h-20 rounded border-2 overflow-hidden transition-colors ${
              currentAngle === angle
                ? 'border-blue-500'
                : 'border-transparent hover:border-gray-300'
            }`}
          >
            <img
              src={`data:image/jpeg;base64,${renders[angle]}`}
              alt={ANGLE_LABELS[angle]}
              className="w-full h-full object-cover"
            />
          </button>
        ))}
      </div>
    </div>
  );
}
```

### Base64 이미지 메모리 최적화

base64 이미지는 메모리를 많이 사용합니다. 아래 패턴으로 최적화하세요:

```typescript
/**
 * base64 string을 Blob URL로 변환.
 * Blob URL은 메모리 효율적이며, 사용 후 revoke 가능.
 */
function base64ToBlobUrl(base64: string, mimeType = 'image/jpeg'): string {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: mimeType });
  return URL.createObjectURL(blob);
}

/**
 * Blob URL 정리 (컴포넌트 unmount 시 호출)
 */
function revokeBlobUrl(url: string): void {
  URL.revokeObjectURL(url);
}

// React Hook 사용 예시
import { useState, useEffect, useMemo } from 'react';

function useBase64AsBlob(base64: string | null): string | null {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!base64) {
      setBlobUrl(null);
      return;
    }

    const url = base64ToBlobUrl(base64);
    setBlobUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [base64]);

  return blobUrl;
}
```

---

## 13. Three.js GLB Integration

Avatar (Phase 1) 및 3D Viewer (Phase 4) GLB 파일을 Three.js로 표시하는 방법입니다.

### 필요 패키지

```bash
npm install three @react-three/fiber @react-three/drei
npm install -D @types/three
```

### GLB Viewer Component

```tsx
import { Suspense, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

interface GLBModelProps {
  url: string;      // GLB URL (api endpoint 또는 blob URL)
  autoRotate?: boolean;
}

function GLBModel({ url, autoRotate = false }: GLBModelProps) {
  const { scene } = useGLTF(url);
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    // 모델 중심 맞추기
    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3());
    scene.position.sub(center);

    // 크기 정규화 (높이 2 기준)
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 2 / maxDim;
    scene.scale.setScalar(scale);
  }, [scene]);

  useFrame((_, delta) => {
    if (autoRotate && groupRef.current) {
      groupRef.current.rotation.y += delta * 0.5;
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={scene} />
    </group>
  );
}

interface ThreeViewerProps {
  glbUrl: string;
  width?: string | number;
  height?: string | number;
  autoRotate?: boolean;
}

export function ThreeViewer({
  glbUrl,
  width = '100%',
  height = 500,
  autoRotate = false,
}: ThreeViewerProps) {
  return (
    <div style={{ width, height }}>
      <Canvas
        camera={{ position: [0, 0, 4], fov: 45 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />

        <Suspense fallback={null}>
          <GLBModel url={glbUrl} autoRotate={autoRotate} />
          <Environment preset="studio" />
          <ContactShadows
            position={[0, -1, 0]}
            opacity={0.4}
            blur={2}
            far={4}
          />
        </Suspense>

        <OrbitControls
          enablePan={false}
          minDistance={2}
          maxDistance={8}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 1.5}
        />
      </Canvas>
    </div>
  );
}
```

### GLB URL 생성 패턴

```typescript
/**
 * Backend endpoint에서 GLB를 로드하기 위한 URL 생성.
 * Three.js useGLTF는 URL을 직접 fetch하므로, JWT를 URL에 포함하거나
 * arraybuffer로 먼저 다운로드 후 blob URL을 사용.
 */
async function loadGlbAsBlobUrl(
  endpoint: string,
  params?: Record<string, string>
): Promise<string> {
  const response = await api.get(endpoint, {
    params,
    responseType: 'arraybuffer',
  });

  const blob = new Blob([response.data], { type: 'model/gltf-binary' });
  return URL.createObjectURL(blob);
}

// 사용 예시 — Phase 1 Avatar GLB
const avatarGlbUrl = await loadGlbAsBlobUrl('/avatar/glb', {
  session_id: sessionId,
});

// 사용 예시 — Phase 4 Viewer GLB
const viewerGlbUrl = await loadGlbAsBlobUrl(`/viewer3d/model/${glbId}`);

// 컴포넌트에서 사용
<ThreeViewer glbUrl={avatarGlbUrl} autoRotate />

// cleanup (컴포넌트 unmount 시)
URL.revokeObjectURL(avatarGlbUrl);
```

> **Important**: Three.js의 `useGLTF`는 내부적으로 `fetch`를 호출하므로, JWT header를 자동으로 포함하지 않습니다. 반드시 위 패턴처럼 `arraybuffer`로 다운로드 후 blob URL을 전달하거나, Backend에서 GLB endpoint에 token-based query parameter 인증을 지원해야 합니다.

---

## 14. Recommended React Component Structure

```
src/
  api/
    client.ts                  # axios instance, interceptors
    avatar.ts                  # Phase 1 API 호출 함수
    wardrobe.ts                # Phase 2 API 호출 함수
    fitting.ts                 # Phase 3 API 호출 함수
    viewer3d.ts                # Phase 4 API 호출 함수
    p2p.ts                     # P2P analysis API
    health.ts                  # Health/Quality API
  types/
    api.ts                     # 모든 TypeScript interface (Section 11)
  hooks/
    useSession.ts              # session_id 관리
    useAvatar.ts               # Phase 1 상태 + mutation
    useWardrobe.ts             # Phase 2 상태 + mutation
    useFitting.ts              # Phase 3 상태 + mutation
    useViewer3D.ts             # Phase 4 상태 + mutation
    useQualityReport.ts        # Quality gate polling
    useHealth.ts               # Health check polling
  components/
    common/
      AngleCarousel.tsx         # 8-angle 이미지 carousel
      QualityGateDisplay.tsx    # Quality gate 시각화
      LoadingOverlay.tsx        # 긴 작업 로딩 UI
      ErrorBoundary.tsx         # API error 처리
    phase1/
      AvatarUploadForm.tsx      # 동영상/이미지 업로드 폼
      AvatarPreview.tsx         # 8-angle avatar preview
      ThreeViewer.tsx           # Three.js GLB viewer
    phase2/
      WardrobeUploadForm.tsx    # 단일/복수 이미지 업로드
      WardrobeUrlForm.tsx       # URL 입력 폼
      ClothingAnalysisCard.tsx  # 의류 분석 결과 카드
      SizeChartDisplay.tsx      # 사이즈 차트 표시
    phase3/
      FittingTrigger.tsx        # Try-on 시작 버튼 + face photo 업로드
      FittingResult.tsx         # 피팅 결과 + 8-angle carousel
      P2PTightnessChart.tsx     # 부위별 타이트함 bar chart
    phase4/
      Viewer3DTrigger.tsx       # 3D 생성 시작 버튼
      Viewer3DResult.tsx        # 3D viewer + previews
  pages/
    HomePage.tsx                # Landing page
    DashboardPage.tsx           # Session 관리 + Phase 진행 현황
    AvatarPage.tsx              # Phase 1 전용 페이지
    WardrobePage.tsx            # Phase 2 전용 페이지
    FittingPage.tsx             # Phase 3 전용 페이지
    ViewerPage.tsx              # Phase 4 전용 페이지
  contexts/
    SessionContext.tsx           # session_id 전역 상태
    AuthContext.tsx              # JWT 인증 상태
```

### Custom Hook 패턴 (React 19 + TanStack Query 권장)

```typescript
// hooks/useAvatar.ts
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { generateAvatar, downloadAvatarGlb } from '@/api/avatar';
import { useSession } from './useSession';
import type { AvatarGenerateRequest, AvatarResponse } from '@/types/api';

export function useAvatarGenerate() {
  const queryClient = useQueryClient();
  const { setSessionId } = useSession();

  return useMutation({
    mutationFn: (params: AvatarGenerateRequest) => generateAvatar(params),
    onSuccess: (data: AvatarResponse) => {
      // session_id를 전역 상태에 저장
      setSessionId(data.session_id);
      // quality report 캐시 무효화
      queryClient.invalidateQueries({ queryKey: ['qualityReport'] });
    },
  });
}

// hooks/useSession.ts
import { createContext, useContext, useState, type ReactNode } from 'react';

interface SessionContextType {
  sessionId: string | null;
  setSessionId: (id: string) => void;
  clearSession: () => void;
}

const SessionContext = createContext<SessionContextType | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(
    () => sessionStorage.getItem('stylelens_session_id')
  );

  const handleSetSessionId = (id: string) => {
    setSessionId(id);
    sessionStorage.setItem('stylelens_session_id', id);
  };

  const clearSession = () => {
    setSessionId(null);
    sessionStorage.removeItem('stylelens_session_id');
  };

  return (
    <SessionContext value={{ sessionId, setSessionId: handleSetSessionId, clearSession }}>
      {children}
    </SessionContext>
  );
}

export function useSession(): SessionContextType {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}
```

---

## 15. Quality Gate Display Pattern

모든 Phase 응답에 포함되는 `quality_gates` 배열을 시각화하는 컴포넌트 패턴입니다.

```tsx
import type { QualityGate } from '@/types/api';

interface QualityGateDisplayProps {
  gates: QualityGate[];
  showDetails?: boolean;
}

function getScoreColor(score: number): string {
  if (score >= 0.9) return 'text-green-600';
  if (score >= 0.7) return 'text-yellow-600';
  if (score >= 0.5) return 'text-orange-600';
  return 'text-red-600';
}

function getScoreBgColor(score: number): string {
  if (score >= 0.9) return 'bg-green-100';
  if (score >= 0.7) return 'bg-yellow-100';
  if (score >= 0.5) return 'bg-orange-100';
  return 'bg-red-100';
}

function getStatusIcon(pass: boolean): string {
  return pass ? 'check_circle' : 'error';
}

export function QualityGateDisplay({
  gates,
  showDetails = true,
}: QualityGateDisplayProps) {
  const allPassed = gates.every((g) => g.pass);
  const overallScore =
    gates.length > 0
      ? gates.reduce((sum, g) => sum + g.score, 0) / gates.length
      : 0;

  return (
    <div className="border rounded-lg p-4">
      {/* Overall status */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-sm text-gray-700">Quality Check</h3>
        <div className={`flex items-center gap-1 text-sm font-medium ${
          allPassed ? 'text-green-600' : 'text-red-600'
        }`}>
          <span className="material-icons text-base">
            {allPassed ? 'verified' : 'warning'}
          </span>
          {allPassed ? 'All Passed' : 'Issues Found'}
        </div>
      </div>

      {/* Overall score bar */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Overall Score</span>
          <span className={getScoreColor(overallScore)}>
            {(overallScore * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              overallScore >= 0.9
                ? 'bg-green-500'
                : overallScore >= 0.7
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
            }`}
            style={{ width: `${overallScore * 100}%` }}
          />
        </div>
      </div>

      {/* Individual gates */}
      {showDetails && (
        <div className="space-y-2">
          {gates.map((gate) => (
            <div
              key={gate.stage}
              className={`flex items-center gap-3 p-2 rounded ${getScoreBgColor(gate.score)}`}
            >
              <span className={`material-icons text-lg ${
                gate.pass ? 'text-green-600' : 'text-red-600'
              }`}>
                {getStatusIcon(gate.pass)}
              </span>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-800">
                    {formatStageName(gate.stage)}
                  </span>
                  <span className={`text-sm font-mono ${getScoreColor(gate.score)}`}>
                    {(gate.score * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-xs text-gray-600 truncate">{gate.feedback}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** stage 이름을 사용자 친화적으로 변환 */
function formatStageName(stage: string): string {
  const map: Record<string, string> = {
    person_detection: '인물 감지',
    body_reconstruction: '신체 복원',
    mesh_quality: '메쉬 품질',
    clothing_detection: '의류 감지',
    attribute_extraction: '속성 추출',
    multi_view_merge: '다중 이미지 통합',
    face_identity: '얼굴 일치도',
    clothing_fit: '의류 피팅',
    texture_mapping: '텍스처 매핑',
    mesh_export: '메쉬 생성',
  };
  return map[stage] ?? stage.replace(/_/g, ' ');
}
```

---

## 16. P2P Tightness Visualization

신체 부위별 타이트함/여유를 시각화하는 bar chart 컴포넌트입니다.

```tsx
import type { P2PDelta, TightnessLevel } from '@/types/api';

const TIGHTNESS_CONFIG: Record<
  TightnessLevel,
  { label: string; color: string; bgColor: string; position: number }
> = {
  tight:      { label: '타이트',     color: '#EF4444', bgColor: '#FEE2E2', position: 0 },
  fitted:     { label: '핏',        color: '#F97316', bgColor: '#FFF7ED', position: 1 },
  semi_fitted:{ label: '세미핏',    color: '#EAB308', bgColor: '#FEFCE8', position: 2 },
  regular:    { label: '레귤러',    color: '#22C55E', bgColor: '#F0FDF4', position: 3 },
  loose:      { label: '루즈',      color: '#3B82F6', bgColor: '#EFF6FF', position: 4 },
  oversize:   { label: '오버사이즈', color: '#8B5CF6', bgColor: '#F5F3FF', position: 5 },
};

const BODY_PART_LABELS: Record<string, string> = {
  chest: '가슴',
  waist: '허리',
  hip: '엉덩이',
  shoulder: '어깨',
  thigh: '허벅지',
  arm: '팔',
  calf: '종아리',
};

interface P2PTightnessChartProps {
  deltas: P2PDelta[];
  overallTightness: TightnessLevel;
  confidence: number;
}

export function P2PTightnessChart({
  deltas,
  overallTightness,
  confidence,
}: P2PTightnessChartProps) {
  const overallConfig = TIGHTNESS_CONFIG[overallTightness];

  return (
    <div className="border rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-800">핏 분석 (P2P Physics)</h3>
        <div className="flex items-center gap-2">
          <span
            className="px-2 py-1 rounded text-sm font-medium"
            style={{
              color: overallConfig.color,
              backgroundColor: overallConfig.bgColor,
            }}
          >
            {overallConfig.label}
          </span>
          <span className="text-xs text-gray-500">
            신뢰도 {(confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Tightness Scale Legend */}
      <div className="flex gap-1">
        {Object.entries(TIGHTNESS_CONFIG).map(([key, config]) => (
          <div
            key={key}
            className="flex-1 text-center py-1 rounded text-xs font-medium"
            style={{
              backgroundColor: config.bgColor,
              color: config.color,
              border: key === overallTightness ? `2px solid ${config.color}` : 'none',
            }}
          >
            {config.label}
          </div>
        ))}
      </div>

      {/* Body Part Bars */}
      <div className="space-y-3">
        {deltas.map((delta) => {
          const config = TIGHTNESS_CONFIG[delta.tightness];
          // bar 위치: delta_cm 기준 (중앙 = 0, 좌 = tight, 우 = loose)
          const barPosition = Math.max(0, Math.min(100, 50 + delta.delta_cm * 4));

          return (
            <div key={delta.body_part} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-gray-700">
                  {BODY_PART_LABELS[delta.body_part] ?? delta.body_part}
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className="text-xs font-medium"
                    style={{ color: config.color }}
                  >
                    {config.label}
                  </span>
                  <span className="text-xs text-gray-500">
                    {delta.delta_cm > 0 ? '+' : ''}{delta.delta_cm.toFixed(1)}cm
                  </span>
                </div>
              </div>

              {/* Bar visualization */}
              <div className="relative h-4 bg-gray-100 rounded-full overflow-hidden">
                {/* Center line (0 delta) */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300 z-10" />

                {/* Delta bar */}
                <div
                  className="absolute top-0 bottom-0 rounded-full transition-all duration-300"
                  style={{
                    backgroundColor: config.color,
                    left: delta.delta_cm < 0
                      ? `${barPosition}%`
                      : '50%',
                    width: `${Math.abs(delta.delta_cm) * 4}%`,
                    maxWidth: '50%',
                    opacity: 0.7,
                  }}
                />
              </div>

              {/* Visual keywords */}
              <div className="flex gap-1 flex-wrap">
                {delta.visual_keywords.map((keyword) => (
                  <span
                    key={keyword}
                    className="text-xs px-1.5 py-0.5 rounded"
                    style={{
                      backgroundColor: config.bgColor,
                      color: config.color,
                    }}
                  >
                    {keyword.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Delta 해석 안내 */}
      <div className="text-xs text-gray-400 border-t pt-2">
        delta_cm: 양수 = 의류가 신체보다 넉넉 / 음수 = 의류가 신체보다 타이트
      </div>
    </div>
  );
}
```

---

## 17. Timing Expectations

각 Phase별 예상 소요 시간입니다. Frontend에서 적절한 loading UI와 timeout을 설정하는 데 참고하세요.

| Phase | Endpoint | 평균 소요 | 최소 | 최대 | 권장 Timeout |
|---|---|---|---|---|---|
| **Phase 1** | `POST /avatar/generate` (video) | 30s | 15s | 60s | 120s |
| **Phase 1** | `POST /avatar/generate` (image) | 15s | 8s | 30s | 60s |
| **Phase 1** | `GET /avatar/glb` | < 1s | - | 3s | 10s |
| **Phase 2** | `POST /wardrobe/add-image` | 5s | 2s | 15s | 30s |
| **Phase 2** | `POST /wardrobe/add-images` | 12s | 5s | 30s | 60s |
| **Phase 2** | `POST /wardrobe/add-url` | 8s | 3s | 20s | 30s |
| **Phase 2** | `POST /wardrobe/extract-model-info` | 5s | 2s | 15s | 30s |
| **Phase 3** | `POST /fitting/try-on` | 60s | 30s | 120s | 180s |
| **Phase 4** | `POST /viewer3d/generate` | 120s | 60s | 240s | 300s |
| **Phase 4** | `GET /viewer3d/model/{id}` | < 2s | - | 5s | 10s |
| **P2P** | `POST /p2p/analyze` | 3s | 1s | 10s | 15s |
| **Health** | `GET /health` | < 1s | - | 2s | 5s |
| **Quality** | `GET /quality/report` | < 1s | - | 3s | 10s |

### Loading UI 권장 패턴

```tsx
import { useState, useEffect } from 'react';

interface PhaseTimingConfig {
  avgSec: number;
  maxSec: number;
  stages: string[];
}

const PHASE_TIMING: Record<string, PhaseTimingConfig> = {
  avatar: {
    avgSec: 30,
    maxSec: 60,
    stages: [
      '입력 데이터 분석 중...',
      '인물 감지 중...',
      '신체 구조 복원 중...',
      '3D 메쉬 생성 중...',
      '텍스처 매핑 중...',
      '최종 검증 중...',
    ],
  },
  fitting: {
    avgSec: 60,
    maxSec: 120,
    stages: [
      '피팅 데이터 준비 중...',
      'GPU Worker에 작업 전송 중...',
      '정면 이미지 생성 중...',
      '측면 이미지 생성 중...',
      '후면 이미지 생성 중...',
      '품질 검증 중...',
      '결과 취합 중...',
    ],
  },
  viewer3d: {
    avgSec: 120,
    maxSec: 240,
    stages: [
      '3D 모델 준비 중...',
      '텍스처 베이킹 중...',
      '의류 메쉬 합성 중...',
      'GLB 파일 생성 중...',
      '최종 검증 중...',
    ],
  },
};

function PhasedLoadingIndicator({
  phase,
  elapsedSec,
}: {
  phase: keyof typeof PHASE_TIMING;
  elapsedSec: number;
}) {
  const config = PHASE_TIMING[phase];
  const progress = Math.min(elapsedSec / config.avgSec, 0.95); // 95%까지만 표시
  const stageIndex = Math.min(
    Math.floor(progress * config.stages.length),
    config.stages.length - 1
  );

  return (
    <div className="space-y-3 p-6">
      {/* Progress bar */}
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full transition-all duration-1000 ease-out"
          style={{ width: `${progress * 100}%` }}
        />
      </div>

      {/* Stage label */}
      <p className="text-sm text-gray-600 text-center animate-pulse">
        {config.stages[stageIndex]}
      </p>

      {/* Time info */}
      <p className="text-xs text-gray-400 text-center">
        경과: {elapsedSec}초 / 예상: ~{config.avgSec}초
      </p>
    </div>
  );
}
```

---

## Appendix A: Request/Response Quick Reference

| Endpoint | Method | Content-Type | Auth | Phase |
|---|---|---|---|---|
| `/avatar/generate` | POST | `multipart/form-data` | JWT | 1 |
| `/avatar/glb` | GET | - | JWT | 1 |
| `/wardrobe/add-image` | POST | `multipart/form-data` | JWT | 2 |
| `/wardrobe/add-images` | POST | `multipart/form-data` | JWT | 2 |
| `/wardrobe/add-url` | POST | `application/x-www-form-urlencoded` | JWT | 2 |
| `/wardrobe/extract-model-info` | POST | `multipart/form-data` | JWT | 2 |
| `/fitting/try-on` | POST | `multipart/form-data` | JWT | 3 |
| `/viewer3d/generate` | POST | `application/json` | JWT | 4 |
| `/viewer3d/model/{glb_id}` | GET | - | JWT | 4 |
| `/p2p/analyze` | POST | `application/json` | JWT | - |
| `/quality/report` | GET | - | JWT | - |
| `/health` | GET | - | None | - |
| `/sessions` | GET | - | JWT | - |

## Appendix B: File Size Limits

| Type | Max Size | Formats |
|---|---|---|
| Video | 50 MB | mp4 |
| Image (clothing) | 10 MB | jpg, png, webp |
| Image (face photo) | 10 MB | jpg, png |
| Image (size chart) | 10 MB | jpg, png |
| Multiple images total | 100 MB | jpg, png, webp |

## Appendix C: Rate Limits

| Endpoint Category | Limit | Window |
|---|---|---|
| Avatar generate | 5 requests | per hour |
| Wardrobe add | 30 requests | per hour |
| Fitting try-on | 10 requests | per hour |
| Viewer3D generate | 5 requests | per hour |
| P2P analyze | 30 requests | per hour |
| Health check | 60 requests | per minute |

Rate limit 초과 시 `429 Too Many Requests` 응답과 함께 `Retry-After` header가 포함됩니다.

---

> **Document maintained by**: StyleLens Backend/AI Team
> **Frontend 문의**: Backend API 관련 질문은 Slack `#stylelens-api` 채널로 문의해주세요.
