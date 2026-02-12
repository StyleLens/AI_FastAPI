# StyleLens V6 WebSocket Progress Events

> **Version**: 1.0
> **Last Updated**: 2026-02-11
> **Audience**: Frontend Team (React/TypeScript)

AI Orchestrator가 처리 진행 상황을 실시간으로 전송하는 WebSocket event 명세입니다.
Spring Boot backend가 AI server의 event를 proxy하여 frontend에 전달합니다.

---

## Table of Contents

1. [WebSocket Connection](#1-websocket-connection)
2. [Event Types](#2-event-types)
3. [TypeScript Interfaces](#3-typescript-interfaces)
4. [Connection Management](#4-connection-management)
5. [React Hook: useSessionProgress](#5-react-hook-usesessionprogress)
6. [Progress Bar Component](#6-progress-bar-component)
7. [Phase Timeline Visualization](#7-phase-timeline-visualization)
8. [Error Handling & Fallback Notification](#8-error-handling--fallback-notification)
9. [Spring Boot WebSocket Proxy Configuration](#9-spring-boot-websocket-proxy-configuration)

---

## 1. WebSocket Connection

### Endpoint

```
WS {BACKEND_URL}/ws/progress/{session_id}
```

- `session_id`: 서버에서 발급한 고유 session identifier (UUID v4 형식)
- 인증: 초기 HTTP handshake 시 `Authorization` header 또는 query parameter로 JWT token 전달

### Connection Example

```typescript
const ws = new WebSocket(
  `wss://api.stylelens.com/ws/progress/${sessionId}`,
  // subprotocol로 token 전달 (선택)
);
```

### Query Parameter 방식 인증

WebSocket은 custom header를 지원하지 않는 브라우저 환경이 있으므로,
query parameter 방식도 지원합니다:

```
WS {BACKEND_URL}/ws/progress/{session_id}?token={jwt_token}
```

---

## 2. Event Types

모든 event는 JSON 형식이며, `event` field로 type을 구분합니다.
`timestamp`는 ISO 8601 UTC 형식입니다.

### 2.1 `phase_start`

Phase가 시작될 때 발생합니다. UI에서 현재 진행 단계를 표시하는 데 사용합니다.

```json
{
  "event": "phase_start",
  "phase": "phase1",
  "phase_name": "Avatar Generation",
  "timestamp": "2026-02-11T10:00:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event` | `string` | 항상 `"phase_start"` |
| `phase` | `string` | Phase identifier (`"phase1"`, `"phase2"`, `"phase3"`) |
| `phase_name` | `string` | 사용자에게 표시할 phase 이름 (한글/영문) |
| `timestamp` | `string` | Event 발생 시각 (ISO 8601 UTC) |

**Phase 목록:**

| Phase ID | Phase Name | Description |
|----------|-----------|-------------|
| `phase1` | Avatar Generation | Video에서 3D avatar 생성 (YOLOv8 + HMR2 + SMPL) |
| `phase2` | Wardrobe Registration | 의류 이미지 분석 및 등록 |
| `phase3` | Virtual Try-On | Avatar에 의류 fitting 및 multi-angle 렌더링 |

---

### 2.2 `progress`

Phase 내 세부 진행률을 전달합니다. Progress bar 업데이트에 사용합니다.

```json
{
  "event": "progress",
  "phase": "phase1",
  "progress": 0.35,
  "message": "Extracting frames from video...",
  "timestamp": "2026-02-11T10:00:05Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event` | `string` | 항상 `"progress"` |
| `phase` | `string` | 현재 phase identifier |
| `progress` | `number` | 진행률 `0.0` ~ `1.0` (소수점 2자리) |
| `message` | `string` | 현재 작업 설명 (사용자에게 표시 가능) |
| `timestamp` | `string` | Event 발생 시각 |

**Phase별 주요 progress message 예시:**

| Phase | Progress | Message |
|-------|----------|---------|
| phase1 | 0.05 | `"Uploading video..."` |
| phase1 | 0.15 | `"Extracting frames from video..."` |
| phase1 | 0.30 | `"Detecting person with YOLOv8..."` |
| phase1 | 0.45 | `"Running HMR2 body estimation..."` |
| phase1 | 0.60 | `"Generating SMPL mesh..."` |
| phase1 | 0.70 | `"Generating FLAME head mesh..."` |
| phase1 | 0.80 | `"Applying body texture..."` |
| phase1 | 0.90 | `"Assembling GLB (body + head + hair)..."` |
| phase1 | 1.00 | `"Avatar generation complete"` |
| phase2 | 0.10 | `"Analyzing clothing images..."` |
| phase2 | 0.50 | `"Classifying clothing views..."` |
| phase2 | 0.80 | `"Merging clothing fields..."` |
| phase2 | 1.00 | `"Wardrobe registration complete"` |
| phase3 | 0.05 | `"Loading avatar and clothing data..."` |
| phase3 | 0.15 | `"Applying body deformation..."` |
| phase3 | 0.25 | `"Rendering front view with Gemini..."` |
| phase3 | 0.35~0.85 | `"Generating angle {N}/8..."` |
| phase3 | 0.90 | `"Compositing final results..."` |
| phase3 | 1.00 | `"Virtual try-on complete"` |

---

### 2.3 `intermediate_result`

처리 중간 결과물을 전달합니다. Preview 이미지나 detection 결과를 UI에 즉시 표시할 수 있습니다.

```json
{
  "event": "intermediate_result",
  "phase": "phase1",
  "type": "person_detection",
  "data": {
    "bbox": [120, 50, 380, 500],
    "confidence": 0.97,
    "preview_b64": "base64-jpeg-thumbnail"
  },
  "timestamp": "2026-02-11T10:00:08Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event` | `string` | 항상 `"intermediate_result"` |
| `phase` | `string` | 현재 phase identifier |
| `type` | `string` | 중간 결과물 type (아래 표 참조) |
| `data` | `object` | Type에 따른 결과 데이터 |
| `timestamp` | `string` | Event 발생 시각 |

**Intermediate Result Types:**

| Type | Phase | Data Fields | Description |
|------|-------|-------------|-------------|
| `person_detection` | phase1 | `bbox`, `confidence`, `preview_b64` | YOLOv8 인물 감지 결과. `bbox`는 `[x, y, w, h]` 형식 |
| `mesh_preview` | phase1 | `preview_b64`, `vertex_count` | SMPL mesh 생성 preview (wireframe thumbnail) |
| `segmentation_preview` | phase2 | `preview_b64`, `categories` | 의류 segmentation 결과 (color-coded mask) |
| `parse_map_preview` | phase2 | `preview_b64`, `regions` | Human parsing map (body part regions) |
| `tryon_single_angle` | phase3 | `angle_index`, `angle_deg`, `preview_b64` | 각 angle별 try-on 결과 (8장 중 1장씩 전송) |
| `3d_shape_preview` | phase1 | `preview_b64`, `format` | 3D shape preview (GLB thumbnail render) |

> **Note**: `preview_b64`는 JPEG thumbnail (최대 512px)이며, base64 encoded string입니다.
> Data URI로 직접 사용: `data:image/jpeg;base64,${preview_b64}`

---

### 2.4 `quality_gate`

각 처리 단계의 품질 검증 결과입니다. `pass: false`인 경우 사용자에게 경고를 표시하거나
재시도를 안내할 수 있습니다.

```json
{
  "event": "quality_gate",
  "phase": "phase1",
  "gate": {
    "stage": "person_detection",
    "score": 0.95,
    "pass": true,
    "feedback": "Clear person detected"
  },
  "timestamp": "2026-02-11T10:00:10Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event` | `string` | 항상 `"quality_gate"` |
| `phase` | `string` | 현재 phase identifier |
| `gate.stage` | `string` | 검증 단계 이름 |
| `gate.score` | `number` | 품질 점수 `0.0` ~ `1.0` |
| `gate.pass` | `boolean` | 통과 여부 |
| `gate.feedback` | `string` | 사용자에게 표시할 피드백 메시지 |
| `timestamp` | `string` | Event 발생 시각 |

**Quality Gate Stages:**

| Stage | Phase | Threshold | Description |
|-------|-------|-----------|-------------|
| `person_detection` | phase1 | 0.85 | 인물 감지 confidence |
| `pose_estimation` | phase1 | 0.80 | Pose estimation 품질 |
| `mesh_quality` | phase1 | 0.75 | 생성된 mesh의 품질 점수 |
| `gemini_supervisor` | phase1 | 0.70 | Gemini supervisor 종합 판단 |
| `clothing_analysis` | phase2 | 0.80 | 의류 분석 신뢰도 |
| `face_identity` | phase3 | 0.75 | Face identity 유지 점수 |
| `tryon_quality` | phase3 | 0.70 | Try-on 결과 종합 품질 |

---

### 2.5 `phase_complete`

Phase가 완료되었을 때 발생합니다. 소요 시간과 성공 여부를 포함합니다.

```json
{
  "event": "phase_complete",
  "phase": "phase1",
  "elapsed_sec": 23.5,
  "success": true,
  "result_url": "https://cdn.stylelens.com/avatars/abc123.glb",
  "timestamp": "2026-02-11T10:00:24Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event` | `string` | 항상 `"phase_complete"` |
| `phase` | `string` | 완료된 phase identifier |
| `elapsed_sec` | `number` | Phase 총 소요 시간 (초) |
| `success` | `boolean` | 성공 여부 |
| `result_url` | `string?` | 결과물 URL (있는 경우에만) |
| `timestamp` | `string` | Event 발생 시각 |

---

### 2.6 `error`

처리 중 오류가 발생했을 때 전송됩니다. `fallback: true`인 경우 자동 대체 처리가 진행 중임을 의미합니다.

```json
{
  "event": "error",
  "phase": "phase1",
  "error": "Worker timeout",
  "code": "WORKER_TIMEOUT",
  "fallback": true,
  "message": "Falling back to local processing...",
  "timestamp": "2026-02-11T10:00:15Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event` | `string` | 항상 `"error"` |
| `phase` | `string` | 오류 발생 phase |
| `error` | `string` | 오류 원인 (내부용) |
| `code` | `string?` | 오류 코드 (있는 경우) |
| `fallback` | `boolean` | 자동 fallback 진행 여부 |
| `message` | `string` | 사용자에게 표시할 메시지 |
| `timestamp` | `string` | Event 발생 시각 |

**Error Codes:**

| Code | Description | Fallback |
|------|-------------|----------|
| `WORKER_TIMEOUT` | AI worker 응답 시간 초과 | local processing 전환 |
| `GPU_OOM` | GPU 메모리 부족 | CPU fallback 또는 queue 재시도 |
| `MODEL_LOAD_FAIL` | 모델 로딩 실패 | 대체 모델 사용 |
| `GEMINI_RATE_LIMIT` | Gemini API rate limit 도달 | flash model fallback |
| `GEMINI_CONTENT_BLOCK` | Gemini 콘텐츠 정책 차단 | 프롬프트 조정 후 재시도 |
| `FACE_NOT_DETECTED` | 얼굴 감지 실패 | face identity 없이 진행 |
| `INVALID_INPUT` | 입력 데이터 오류 | fallback 없음, 사용자 재시도 필요 |
| `SESSION_EXPIRED` | Session 만료 | 재연결 필요 |

---

## 3. TypeScript Interfaces

```typescript
// ============================================================
// WebSocket Event Type Definitions
// StyleLens V6 Frontend
// ============================================================

/** Phase identifier */
type PhaseId = "phase1" | "phase2" | "phase3";

/** Intermediate result type */
type IntermediateResultType =
  | "person_detection"
  | "mesh_preview"
  | "segmentation_preview"
  | "parse_map_preview"
  | "tryon_single_angle"
  | "3d_shape_preview";

/** Quality gate stage */
type QualityGateStage =
  | "person_detection"
  | "pose_estimation"
  | "mesh_quality"
  | "gemini_supervisor"
  | "clothing_analysis"
  | "face_identity"
  | "tryon_quality";

/** Error code */
type ErrorCode =
  | "WORKER_TIMEOUT"
  | "GPU_OOM"
  | "MODEL_LOAD_FAIL"
  | "GEMINI_RATE_LIMIT"
  | "GEMINI_CONTENT_BLOCK"
  | "FACE_NOT_DETECTED"
  | "INVALID_INPUT"
  | "SESSION_EXPIRED";

// ------------------------------------------------------------
// Base Event
// ------------------------------------------------------------
interface BaseEvent {
  event: string;
  timestamp: string; // ISO 8601 UTC
}

// ------------------------------------------------------------
// Individual Event Types
// ------------------------------------------------------------
interface PhaseStartEvent extends BaseEvent {
  event: "phase_start";
  phase: PhaseId;
  phase_name: string;
}

interface ProgressEvent extends BaseEvent {
  event: "progress";
  phase: PhaseId;
  progress: number; // 0.0 ~ 1.0
  message: string;
}

interface IntermediateResultEvent extends BaseEvent {
  event: "intermediate_result";
  phase: PhaseId;
  type: IntermediateResultType;
  data: PersonDetectionData | MeshPreviewData | SegmentationData
    | ParseMapData | TryOnAngleData | ShapePreviewData;
}

interface QualityGateEvent extends BaseEvent {
  event: "quality_gate";
  phase: PhaseId;
  gate: {
    stage: QualityGateStage;
    score: number;  // 0.0 ~ 1.0
    pass: boolean;
    feedback: string;
  };
}

interface PhaseCompleteEvent extends BaseEvent {
  event: "phase_complete";
  phase: PhaseId;
  elapsed_sec: number;
  success: boolean;
  result_url?: string;
}

interface ErrorEvent extends BaseEvent {
  event: "error";
  phase: PhaseId;
  error: string;
  code?: ErrorCode;
  fallback: boolean;
  message: string;
}

// ------------------------------------------------------------
// Intermediate Result Data Types
// ------------------------------------------------------------
interface PersonDetectionData {
  bbox: [number, number, number, number]; // [x, y, w, h]
  confidence: number;
  preview_b64: string;
}

interface MeshPreviewData {
  preview_b64: string;
  vertex_count: number;
}

interface SegmentationData {
  preview_b64: string;
  categories: string[];
}

interface ParseMapData {
  preview_b64: string;
  regions: Record<string, number>; // region_name -> pixel_count
}

interface TryOnAngleData {
  angle_index: number;  // 0~7
  angle_deg: number;    // 0, 45, 90, ...
  preview_b64: string;
}

interface ShapePreviewData {
  preview_b64: string;
  format: "glb" | "obj";
}

// ------------------------------------------------------------
// Union Type (모든 event를 하나의 type으로)
// ------------------------------------------------------------
type SessionProgressEvent =
  | PhaseStartEvent
  | ProgressEvent
  | IntermediateResultEvent
  | QualityGateEvent
  | PhaseCompleteEvent
  | ErrorEvent;
```

---

## 4. Connection Management

### Auto-Reconnect Pattern

WebSocket 연결이 끊어질 경우 자동으로 재연결하는 패턴입니다.
Exponential backoff를 적용하여 서버 부하를 방지합니다.

```typescript
interface ReconnectConfig {
  maxRetries: number;       // 최대 재시도 횟수 (default: 5)
  baseDelay: number;        // 초기 대기 시간 ms (default: 1000)
  maxDelay: number;         // 최대 대기 시간 ms (default: 30000)
  backoffMultiplier: number; // 지수 증가 배수 (default: 2)
}

const DEFAULT_RECONNECT_CONFIG: ReconnectConfig = {
  maxRetries: 5,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
};

class SessionWebSocket {
  private ws: WebSocket | null = null;
  private retryCount = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isManuallyClosed = false;

  constructor(
    private sessionId: string,
    private backendUrl: string,
    private token: string,
    private onEvent: (event: SessionProgressEvent) => void,
    private onConnectionChange: (connected: boolean) => void,
    private config: ReconnectConfig = DEFAULT_RECONNECT_CONFIG,
  ) {}

  connect(): void {
    this.isManuallyClosed = false;
    this.createConnection();
  }

  disconnect(): void {
    this.isManuallyClosed = true;
    this.clearReconnectTimer();
    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
      this.ws = null;
    }
  }

  private createConnection(): void {
    const protocol = this.backendUrl.startsWith("https") ? "wss" : "ws";
    const host = this.backendUrl.replace(/^https?:\/\//, "");
    const url = `${protocol}://${host}/ws/progress/${this.sessionId}?token=${this.token}`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log(`[WS] Connected to session ${this.sessionId}`);
      this.retryCount = 0; // 연결 성공 시 retry count 초기화
      this.onConnectionChange(true);
    };

    this.ws.onmessage = (msg: MessageEvent) => {
      try {
        const event = JSON.parse(msg.data) as SessionProgressEvent;
        this.onEvent(event);
      } catch (e) {
        console.error("[WS] Failed to parse event:", e);
      }
    };

    this.ws.onclose = (e: CloseEvent) => {
      console.log(`[WS] Closed: code=${e.code}, reason=${e.reason}`);
      this.onConnectionChange(false);

      // 정상 종료이거나 수동 종료인 경우 재연결하지 않음
      if (this.isManuallyClosed || e.code === 1000) return;

      this.scheduleReconnect();
    };

    this.ws.onerror = (e: Event) => {
      console.error("[WS] Error:", e);
      // onclose가 이후에 호출되므로 여기서는 reconnect하지 않음
    };
  }

  private scheduleReconnect(): void {
    if (this.retryCount >= this.config.maxRetries) {
      console.error(
        `[WS] Max retries (${this.config.maxRetries}) reached. Giving up.`
      );
      return;
    }

    const delay = Math.min(
      this.config.baseDelay * Math.pow(this.config.backoffMultiplier, this.retryCount),
      this.config.maxDelay,
    );

    console.log(
      `[WS] Reconnecting in ${delay}ms (attempt ${this.retryCount + 1}/${this.config.maxRetries})`
    );

    this.reconnectTimer = setTimeout(() => {
      this.retryCount++;
      this.createConnection();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
```

### Connection Lifecycle

```
[connect()] ──> [WebSocket OPEN] ──> [receiving events...]
                     │                       │
                     │                  [onclose]
                     │                       │
                     │              ┌─ manual close? ──> STOP
                     │              │
                     │              └─ abnormal close
                     │                       │
                     │              [exponential backoff wait]
                     │                       │
                     │              ┌─ max retries? ──> STOP (notify user)
                     │              │
                     └──────────────└─ [reconnect]
```

---

## 5. React Hook: `useSessionProgress`

Session progress를 구독하는 React custom hook입니다.

```typescript
import { useEffect, useRef, useState, useCallback } from "react";

// -----------------------------------------------------------
// Hook Return Type
// -----------------------------------------------------------
interface UseSessionProgressReturn {
  /** 현재 phase */
  currentPhase: PhaseId | null;
  /** 현재 phase 이름 (표시용) */
  currentPhaseName: string;
  /** 현재 progress (0.0 ~ 1.0) */
  progress: number;
  /** 현재 진행 메시지 */
  message: string;
  /** 중간 결과물 목록 */
  intermediateResults: IntermediateResultEvent[];
  /** Quality gate 결과 목록 */
  qualityGates: QualityGateEvent[];
  /** 완료된 phase 목록 */
  completedPhases: PhaseCompleteEvent[];
  /** 최근 error (있는 경우) */
  lastError: ErrorEvent | null;
  /** WebSocket 연결 상태 */
  isConnected: boolean;
  /** 전체 pipeline 완료 여부 */
  isComplete: boolean;
}

// -----------------------------------------------------------
// Hook Implementation
// -----------------------------------------------------------
function useSessionProgress(
  sessionId: string | null,
  backendUrl: string,
  token: string,
): UseSessionProgressReturn {
  const [currentPhase, setCurrentPhase] = useState<PhaseId | null>(null);
  const [currentPhaseName, setCurrentPhaseName] = useState("");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [intermediateResults, setIntermediateResults] = useState<
    IntermediateResultEvent[]
  >([]);
  const [qualityGates, setQualityGates] = useState<QualityGateEvent[]>([]);
  const [completedPhases, setCompletedPhases] = useState<
    PhaseCompleteEvent[]
  >([]);
  const [lastError, setLastError] = useState<ErrorEvent | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const wsRef = useRef<SessionWebSocket | null>(null);

  const handleEvent = useCallback((event: SessionProgressEvent) => {
    switch (event.event) {
      case "phase_start":
        setCurrentPhase(event.phase);
        setCurrentPhaseName(event.phase_name);
        setProgress(0);
        setMessage("");
        setLastError(null);
        break;

      case "progress":
        setProgress(event.progress);
        setMessage(event.message);
        break;

      case "intermediate_result":
        setIntermediateResults((prev) => [...prev, event]);
        break;

      case "quality_gate":
        setQualityGates((prev) => [...prev, event]);
        break;

      case "phase_complete":
        setCompletedPhases((prev) => [...prev, event]);
        setProgress(1.0);
        break;

      case "error":
        setLastError(event);
        break;
    }
  }, []);

  useEffect(() => {
    if (!sessionId) return;

    const ws = new SessionWebSocket(
      sessionId,
      backendUrl,
      token,
      handleEvent,
      setIsConnected,
    );

    ws.connect();
    wsRef.current = ws;

    return () => {
      ws.disconnect();
      wsRef.current = null;
    };
  }, [sessionId, backendUrl, token, handleEvent]);

  const isComplete =
    completedPhases.length === 3 &&
    completedPhases.every((p) => p.success);

  return {
    currentPhase,
    currentPhaseName,
    progress,
    message,
    intermediateResults,
    qualityGates,
    completedPhases,
    lastError,
    isConnected,
    isComplete,
  };
}

export { useSessionProgress };
export type { UseSessionProgressReturn };
```

### Usage Example

```tsx
function AvatarGenerationPage({ sessionId }: { sessionId: string }) {
  const {
    currentPhase,
    currentPhaseName,
    progress,
    message,
    intermediateResults,
    qualityGates,
    lastError,
    isConnected,
    isComplete,
  } = useSessionProgress(sessionId, BACKEND_URL, authToken);

  return (
    <div>
      <ConnectionStatus connected={isConnected} />
      <PhaseTimeline
        currentPhase={currentPhase}
        completedPhases={completedPhases}
      />
      <ProgressBar
        progress={progress}
        message={message}
        phaseName={currentPhaseName}
      />
      <IntermediatePreview results={intermediateResults} />
      <QualityGateList gates={qualityGates} />
      {lastError && <ErrorNotification error={lastError} />}
      {isComplete && <CompletionBanner />}
    </div>
  );
}
```

---

## 6. Progress Bar Component

Phase별 progress를 시각적으로 표시하는 component입니다.

```tsx
import React from "react";

// -----------------------------------------------------------
// Phase Progress Bar
// -----------------------------------------------------------
interface ProgressBarProps {
  progress: number;   // 0.0 ~ 1.0
  message: string;
  phaseName: string;
}

function ProgressBar({ progress, message, phaseName }: ProgressBarProps) {
  const percent = Math.round(progress * 100);

  return (
    <div className="progress-container">
      {/* Phase 이름 */}
      <div className="progress-header">
        <span className="phase-name">{phaseName}</span>
        <span className="progress-percent">{percent}%</span>
      </div>

      {/* Progress bar */}
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${percent}%` }}
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>

      {/* 현재 작업 메시지 */}
      {message && (
        <p className="progress-message">{message}</p>
      )}
    </div>
  );
}

// -----------------------------------------------------------
// Overall Pipeline Progress (3 phases 합산)
// -----------------------------------------------------------
interface OverallProgressProps {
  currentPhase: PhaseId | null;
  progress: number;
  completedPhases: PhaseCompleteEvent[];
}

function OverallProgress({
  currentPhase,
  progress,
  completedPhases,
}: OverallProgressProps) {
  // Phase별 weight (총합 1.0)
  const PHASE_WEIGHTS: Record<PhaseId, { start: number; weight: number }> = {
    phase1: { start: 0.0, weight: 0.50 },  // Avatar: 50%
    phase2: { start: 0.50, weight: 0.15 }, // Wardrobe: 15%
    phase3: { start: 0.65, weight: 0.35 }, // Try-on: 35%
  };

  let overallProgress = 0;

  // 완료된 phase 합산
  for (const completed of completedPhases) {
    const config = PHASE_WEIGHTS[completed.phase];
    if (config && completed.success) {
      overallProgress += config.weight;
    }
  }

  // 현재 진행 중인 phase 반영
  if (currentPhase) {
    const config = PHASE_WEIGHTS[currentPhase];
    if (config) {
      overallProgress += config.weight * progress;
    }
  }

  const percent = Math.round(overallProgress * 100);

  return (
    <div className="overall-progress">
      <div className="overall-track">
        <div
          className="overall-fill"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="overall-label">
        Overall: {percent}%
      </span>
    </div>
  );
}
```

### CSS Example

```css
.progress-container {
  margin: 16px 0;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
}

.phase-name {
  font-weight: 600;
  color: #1a1a2e;
}

.progress-percent {
  color: #6c63ff;
  font-weight: 500;
}

.progress-track {
  width: 100%;
  height: 8px;
  background: #e8e8f0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #6c63ff, #a855f7);
  border-radius: 4px;
  transition: width 0.3s ease-out;
}

.progress-message {
  margin-top: 8px;
  font-size: 13px;
  color: #666;
  animation: fadeIn 0.3s ease;
}

/* Overall progress */
.overall-progress {
  margin: 24px 0;
}

.overall-track {
  width: 100%;
  height: 12px;
  background: #e8e8f0;
  border-radius: 6px;
  overflow: hidden;
}

.overall-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #6c63ff, #a855f7);
  border-radius: 6px;
  transition: width 0.5s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}
```

---

## 7. Phase Timeline Visualization

3개 phase의 진행 상태를 timeline으로 표시하는 component입니다.

```tsx
import React from "react";

// -----------------------------------------------------------
// Phase 설정
// -----------------------------------------------------------
const PHASES = [
  { id: "phase1" as PhaseId, name: "Avatar Generation", icon: "🧍" },
  { id: "phase2" as PhaseId, name: "Wardrobe Registration", icon: "👔" },
  { id: "phase3" as PhaseId, name: "Virtual Try-On", icon: "✨" },
];

type PhaseStatus = "pending" | "active" | "completed" | "failed";

// -----------------------------------------------------------
// Phase Timeline Component
// -----------------------------------------------------------
interface PhaseTimelineProps {
  currentPhase: PhaseId | null;
  completedPhases: PhaseCompleteEvent[];
  lastError: ErrorEvent | null;
}

function PhaseTimeline({
  currentPhase,
  completedPhases,
  lastError,
}: PhaseTimelineProps) {
  const getPhaseStatus = (phaseId: PhaseId): PhaseStatus => {
    const completed = completedPhases.find((p) => p.phase === phaseId);
    if (completed) return completed.success ? "completed" : "failed";
    if (currentPhase === phaseId) return "active";
    return "pending";
  };

  const getElapsedTime = (phaseId: PhaseId): string | null => {
    const completed = completedPhases.find((p) => p.phase === phaseId);
    if (!completed) return null;
    return `${completed.elapsed_sec.toFixed(1)}s`;
  };

  return (
    <div className="phase-timeline">
      {PHASES.map((phase, index) => {
        const status = getPhaseStatus(phase.id);
        const elapsed = getElapsedTime(phase.id);

        return (
          <React.Fragment key={phase.id}>
            {/* Phase node */}
            <div className={`timeline-node timeline-${status}`}>
              <div className="timeline-icon">
                {status === "completed" && <CheckIcon />}
                {status === "failed" && <XIcon />}
                {status === "active" && <SpinnerIcon />}
                {status === "pending" && <span>{index + 1}</span>}
              </div>
              <div className="timeline-label">
                <span className="timeline-name">{phase.name}</span>
                {elapsed && (
                  <span className="timeline-elapsed">{elapsed}</span>
                )}
              </div>
            </div>

            {/* Connector (마지막 phase 뒤에는 표시하지 않음) */}
            {index < PHASES.length - 1 && (
              <div
                className={`timeline-connector ${
                  status === "completed" ? "connector-done" : ""
                }`}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
```

### CSS Example

```css
.phase-timeline {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 24px 16px;
}

.timeline-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  min-width: 120px;
}

.timeline-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 16px;
}

/* Status별 스타일 */
.timeline-pending .timeline-icon {
  background: #e8e8f0;
  color: #999;
}

.timeline-active .timeline-icon {
  background: #6c63ff;
  color: white;
  box-shadow: 0 0 0 4px rgba(108, 99, 255, 0.2);
}

.timeline-completed .timeline-icon {
  background: #10b981;
  color: white;
}

.timeline-failed .timeline-icon {
  background: #ef4444;
  color: white;
}

.timeline-connector {
  flex: 1;
  height: 2px;
  background: #e8e8f0;
  min-width: 40px;
}

.connector-done {
  background: #10b981;
}

.timeline-name {
  font-size: 13px;
  font-weight: 500;
  color: #333;
  text-align: center;
}

.timeline-elapsed {
  font-size: 11px;
  color: #999;
}

/* Active phase 애니메이션 */
.timeline-active .timeline-icon {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 4px rgba(108, 99, 255, 0.2); }
  50% { box-shadow: 0 0 0 8px rgba(108, 99, 255, 0.1); }
}
```

---

## 8. Error Handling & Fallback Notification

### Error Notification Component

Error event 수신 시 사용자에게 알림을 표시합니다.
`fallback: true`인 경우 자동 대체 처리 중임을 안내하고,
`fallback: false`인 경우 사용자 action이 필요함을 표시합니다.

```tsx
interface ErrorNotificationProps {
  error: ErrorEvent;
  onRetry?: () => void;
  onDismiss?: () => void;
}

function ErrorNotification({ error, onRetry, onDismiss }: ErrorNotificationProps) {
  const isFallback = error.fallback;

  return (
    <div
      className={`error-notification ${isFallback ? "error-warning" : "error-critical"}`}
      role="alert"
    >
      <div className="error-icon">
        {isFallback ? <WarningIcon /> : <ErrorIcon />}
      </div>

      <div className="error-content">
        <p className="error-title">
          {isFallback
            ? "Processing with alternative method"
            : "An error occurred"}
        </p>
        <p className="error-message">{error.message}</p>

        {error.code && (
          <span className="error-code">Code: {error.code}</span>
        )}
      </div>

      <div className="error-actions">
        {!isFallback && onRetry && (
          <button className="btn-retry" onClick={onRetry}>
            Retry
          </button>
        )}
        {onDismiss && (
          <button className="btn-dismiss" onClick={onDismiss}>
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
```

### Error Handling Strategy

```typescript
function handleError(error: ErrorEvent): void {
  switch (error.code) {
    case "WORKER_TIMEOUT":
    case "GPU_OOM":
    case "MODEL_LOAD_FAIL":
      // Fallback이 자동으로 진행되는 경우
      // 사용자에게 대기 안내만 표시
      if (error.fallback) {
        showToast("info", error.message);
      }
      break;

    case "GEMINI_RATE_LIMIT":
      // Rate limit의 경우 잠시 후 자동 재시도
      showToast("warning", "API 사용량 한도에 도달했습니다. 잠시 후 자동으로 재시도합니다.");
      break;

    case "GEMINI_CONTENT_BLOCK":
      // 콘텐츠 정책 차단 - 입력 이미지 교체 안내
      showToast("warning", "입력 이미지가 정책에 의해 차단되었습니다. 다른 이미지를 사용해 주세요.");
      break;

    case "FACE_NOT_DETECTED":
      // 얼굴 미감지 - face identity 없이 진행
      showToast("info", "얼굴이 감지되지 않아 기본 얼굴로 진행합니다.");
      break;

    case "INVALID_INPUT":
      // 사용자 재시도 필요
      showModal("error", "입력 데이터에 문제가 있습니다.", error.message);
      break;

    case "SESSION_EXPIRED":
      // WebSocket 재연결 필요
      showModal("error", "세션이 만료되었습니다.", "페이지를 새로고침해 주세요.");
      break;

    default:
      // 알 수 없는 오류
      if (error.fallback) {
        showToast("info", error.message);
      } else {
        showToast("error", error.message || "알 수 없는 오류가 발생했습니다.");
      }
  }
}
```

---

## 9. Spring Boot WebSocket Proxy Configuration

Spring Boot backend가 AI Orchestrator의 WebSocket event를 frontend로 proxy하는 구성입니다.

### 9.1 Dependencies (build.gradle)

```groovy
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-websocket'
}
```

### 9.2 WebSocket Configuration

```java
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final ProgressWebSocketHandler progressHandler;
    private final JwtHandshakeInterceptor jwtInterceptor;

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry
            .addHandler(progressHandler, "/ws/progress/{sessionId}")
            .addInterceptors(jwtInterceptor)
            .setAllowedOrigins(
                "http://localhost:3000",
                "https://stylelens.com"
            );
    }
}
```

### 9.3 JWT Handshake Interceptor

WebSocket handshake 단계에서 JWT token을 검증합니다.

```java
@Component
public class JwtHandshakeInterceptor implements HandshakeInterceptor {

    private final JwtTokenProvider jwtProvider;

    @Override
    public boolean beforeHandshake(
        ServerHttpRequest request,
        ServerHttpResponse response,
        WebSocketHandler wsHandler,
        Map<String, Object> attributes
    ) {
        // Query parameter에서 token 추출
        String token = extractTokenFromQuery(request.getURI());
        if (token == null || !jwtProvider.validateToken(token)) {
            return false; // Handshake 거부
        }

        // Session에 user 정보 저장
        String userId = jwtProvider.getUserId(token);
        attributes.put("userId", userId);

        // URI에서 sessionId 추출
        String path = request.getURI().getPath();
        String sessionId = path.substring(path.lastIndexOf('/') + 1);
        attributes.put("sessionId", sessionId);

        return true;
    }

    @Override
    public void afterHandshake(
        ServerHttpRequest request,
        ServerHttpResponse response,
        WebSocketHandler wsHandler,
        Exception exception
    ) {
        // no-op
    }

    private String extractTokenFromQuery(URI uri) {
        String query = uri.getQuery();
        if (query == null) return null;
        return Arrays.stream(query.split("&"))
            .filter(p -> p.startsWith("token="))
            .map(p -> p.substring(6))
            .findFirst()
            .orElse(null);
    }
}
```

### 9.4 Progress WebSocket Handler

AI Orchestrator에서 수신한 event를 해당 session의 WebSocket client에게 전달합니다.

```java
@Component
public class ProgressWebSocketHandler extends TextWebSocketHandler {

    // sessionId -> WebSocket sessions (1:N, 같은 session을 여러 tab에서 볼 수 있음)
    private final Map<String, Set<WebSocketSession>> sessionMap =
        new ConcurrentHashMap<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        String sessionId = (String) session.getAttributes().get("sessionId");
        sessionMap
            .computeIfAbsent(sessionId, k -> ConcurrentHashMap.newKeySet())
            .add(session);

        log.info("WS connected: sessionId={}, wsId={}",
            sessionId, session.getId());
    }

    @Override
    public void afterConnectionClosed(
        WebSocketSession session,
        CloseStatus status
    ) {
        String sessionId = (String) session.getAttributes().get("sessionId");
        Set<WebSocketSession> sessions = sessionMap.get(sessionId);
        if (sessions != null) {
            sessions.remove(session);
            if (sessions.isEmpty()) {
                sessionMap.remove(sessionId);
            }
        }

        log.info("WS closed: sessionId={}, status={}",
            sessionId, status);
    }

    /**
     * AI Orchestrator에서 호출하는 method.
     * 해당 sessionId에 연결된 모든 WebSocket client에게 event를 broadcast합니다.
     */
    public void broadcastEvent(String sessionId, String eventJson) {
        Set<WebSocketSession> sessions = sessionMap.get(sessionId);
        if (sessions == null || sessions.isEmpty()) return;

        TextMessage message = new TextMessage(eventJson);

        for (WebSocketSession ws : sessions) {
            try {
                if (ws.isOpen()) {
                    ws.sendMessage(message);
                }
            } catch (IOException e) {
                log.warn("Failed to send WS message: sessionId={}, wsId={}",
                    sessionId, ws.getId(), e);
            }
        }
    }
}
```

### 9.5 AI Orchestrator Event Listener

AI server(FastAPI)에서 SSE 또는 HTTP callback으로 전달되는 event를 수신하여
WebSocket으로 relay합니다.

```java
@Service
@RequiredArgsConstructor
public class AiProgressListener {

    private final ProgressWebSocketHandler wsHandler;
    private final WebClient webClient;

    /**
     * AI Orchestrator의 SSE stream을 구독합니다.
     * Session 시작 시 호출됩니다.
     */
    public void subscribeToProgress(String sessionId) {
        String aiUrl = String.format("%s/progress/stream/%s",
            aiOrchestratorUrl, sessionId);

        webClient.get()
            .uri(aiUrl)
            .accept(MediaType.TEXT_EVENT_STREAM)
            .retrieve()
            .bodyToFlux(String.class)
            .doOnNext(eventJson -> {
                wsHandler.broadcastEvent(sessionId, eventJson);
            })
            .doOnError(e -> {
                log.error("SSE stream error: sessionId={}", sessionId, e);
                String errorEvent = buildErrorEvent(sessionId, e.getMessage());
                wsHandler.broadcastEvent(sessionId, errorEvent);
            })
            .doOnComplete(() -> {
                log.info("SSE stream completed: sessionId={}", sessionId);
            })
            .subscribe();
    }

    private String buildErrorEvent(String sessionId, String errorMsg) {
        return String.format("""
            {
              "event": "error",
              "phase": "unknown",
              "error": "%s",
              "code": "PROXY_ERROR",
              "fallback": false,
              "message": "서버 연결에 문제가 발생했습니다.",
              "timestamp": "%s"
            }
            """, errorMsg, Instant.now().toString());
    }
}
```

### 9.6 Architecture Diagram

```
┌─────────────┐     SSE Stream      ┌──────────────────┐    WebSocket     ┌──────────┐
│ AI Server   │ ──────────────────── │ Spring Boot      │ ────────────── │ Frontend │
│ (FastAPI)   │  /progress/stream/   │ Backend          │  /ws/progress/  │ (React)  │
│             │  {session_id}        │                  │  {session_id}   │          │
│ - YOLOv8    │                      │ - JWT 검증        │                 │ - Hook   │
│ - HMR2      │                      │ - SSE → WS 변환   │                 │ - UI     │
│ - Gemini    │                      │ - Session 관리     │                 │          │
└─────────────┘                      └──────────────────┘                 └──────────┘
```

### 9.7 Configuration Properties

```yaml
# application.yml
stylelens:
  ai:
    orchestrator-url: http://localhost:8000  # AI FastAPI server
    progress:
      sse-timeout: 300s         # SSE stream timeout (phase 최대 소요 시간)
      ws-idle-timeout: 600s     # WebSocket idle timeout
      max-sessions-per-user: 3  # 사용자당 동시 session 수 제한

spring:
  websocket:
    max-text-message-buffer-size: 65536  # 64KB (preview image 포함 고려)
    max-binary-message-buffer-size: 65536
```

---

## Appendix: Quick Reference

### Event Flow (정상 처리)

```
phase_start (phase1)
  ├── progress (0.15, "Extracting frames...")
  ├── progress (0.30, "Detecting person...")
  ├── intermediate_result (person_detection)
  ├── quality_gate (person_detection, pass=true)
  ├── progress (0.45, "Running HMR2...")
  ├── progress (0.60, "Generating SMPL mesh...")
  ├── intermediate_result (mesh_preview)
  ├── quality_gate (mesh_quality, pass=true)
  ├── progress (0.90, "Assembling GLB...")
  └── progress (1.00, "Avatar generation complete")
phase_complete (phase1, success=true, elapsed=23.5s)

phase_start (phase2)
  ├── progress (0.10, "Analyzing clothing images...")
  ├── intermediate_result (segmentation_preview)
  ├── quality_gate (clothing_analysis, pass=true)
  └── progress (1.00, "Wardrobe registration complete")
phase_complete (phase2, success=true, elapsed=8.2s)

phase_start (phase3)
  ├── progress (0.15, "Applying body deformation...")
  ├── progress (0.25, "Rendering front view...")
  ├── intermediate_result (tryon_single_angle, angle=0)
  ├── progress (0.35~0.85, "Generating angle N/8...")
  ├── intermediate_result (tryon_single_angle, angle=45..315)
  ├── quality_gate (face_identity, pass=true)
  ├── quality_gate (tryon_quality, pass=true)
  └── progress (1.00, "Virtual try-on complete")
phase_complete (phase3, success=true, elapsed=226.0s)
```

### Event Flow (Error + Fallback)

```
phase_start (phase1)
  ├── progress (0.30, "Detecting person...")
  ├── error (GEMINI_RATE_LIMIT, fallback=true, "Falling back to flash model...")
  ├── progress (0.35, "Retrying with fallback model...")
  ├── intermediate_result (person_detection)
  └── progress (1.00, "Avatar generation complete")
phase_complete (phase1, success=true)
```
