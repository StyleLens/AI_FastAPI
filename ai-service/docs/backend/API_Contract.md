# StyleLens V6 Backend API Contract

> Spring Boot (Tier 2) - AI Orchestrator Proxy Specification
> Version: 1.0 | Last Updated: 2026-02-11

---

## 1. Architecture Overview (아키텍처 개요)

```
[Frontend (React)]  <-->  [Spring Boot (Tier 2)]  <-->  [AI Orchestrator (Tier 3)]
     Tier 1                   Proxy + Auth                  FastAPI + GPU
     :3000                    :8080                         :8000
```

Spring Boot는 Frontend와 AI Orchestrator 사이의 **Proxy 역할**을 수행한다.
모든 AI 관련 요청은 Spring Boot를 경유하여 AI Orchestrator로 전달되며,
인증/인가, Rate Limiting, Logging, Error Translation 등의 Cross-cutting Concern을 처리한다.

**핵심 원칙:**
- Frontend는 AI Orchestrator에 직접 접근하지 않는다
- Spring Boot는 AI 비즈니스 로직을 포함하지 않는다 (순수 Proxy)
- 모든 요청에 `session_id`를 주입하여 AI Orchestrator의 세션과 매핑한다

---

## 2. Orchestrator Endpoint Proxy Table (프록시 엔드포인트 전체 목록)

### 2.1 Endpoint 매핑표

| Frontend Path | Method | Orchestrator Path | Timeout Category | Content-Type | Description |
|---|---|---|---|---|---|
| `/api/v6/avatar/generate` | POST | `/avatar/generate` | MEDIUM (120s) | `multipart/form-data` | 비디오/이미지 → 3D Avatar 생성 |
| `/api/v6/avatar/glb` | GET | `/avatar/glb` | SHORT (30s) | - | Binary GLB 파일 다운로드 |
| `/api/v6/wardrobe/add-image` | POST | `/wardrobe/add-image` | MEDIUM (60s) | `multipart/form-data` | 의류 이미지 단건 등록 |
| `/api/v6/wardrobe/add-images` | POST | `/wardrobe/add-images` | MEDIUM (120s) | `multipart/form-data` | 의류 이미지 다건 등록 (최대 10장) |
| `/api/v6/wardrobe/add-url` | POST | `/wardrobe/add-url` | MEDIUM (60s) | `application/x-www-form-urlencoded` | URL 기반 의류 등록 |
| `/api/v6/wardrobe/extract-model-info` | POST | `/wardrobe/extract-model-info` | SHORT (30s) | `application/json` | Gemini 모델 정보 추출 |
| `/api/v6/fitting/try-on` | POST | `/fitting/try-on` | LONG (300s) | `multipart/form-data` | Virtual Try-On (GPU 집중) |
| `/api/v6/viewer3d/generate` | POST | `/viewer3d/generate` | LONG (300s) | `multipart/form-data` | Hunyuan3D 3D 모델 생성 |
| `/api/v6/viewer3d/model/{glb_id}` | GET | `/viewer3d/model/{glb_id}` | SHORT (30s) | - | 3D GLB 모델 다운로드 |
| `/api/v6/p2p/analyze` | POST | `/p2p/analyze` | MEDIUM (60s) | `application/json` | P2P Physics 분석 (CPU) |
| `/api/v6/quality/report` | GET | `/quality/report` | SHORT (10s) | - | 품질 리포트 조회 |
| `/api/v6/health` | GET | `/health` | SHORT (10s) | - | Health Check |
| `/api/v6/sessions` | GET | `/sessions` | SHORT (10s) | - | 세션 목록 조회 |

### 2.2 Timeout Category 정의

| Category | Duration | 대상 | Retry |
|---|---|---|---|
| SHORT | 10-30s | GLB 다운로드, Health Check, 리포트 | No |
| MEDIUM | 60-120s | 이미지 업로드, Avatar 생성, P2P 분석 | Yes (503 only) |
| LONG | 300s | Try-On, 3D Generation (GPU 작업) | No (Async Queue) |

---

## 3. Proxy Implementation Pattern (프록시 구현 패턴)

### 3.1 WebClient 기반 구현 (권장)

Spring WebFlux의 `WebClient`를 사용한다. `RestTemplate`은 Blocking이므로 Long-running 요청에 부적합하다.

```java
@Configuration
public class WebClientConfig {

    @Value("${stylelens.ai-orchestrator.url:http://localhost:8000}")
    private String orchestratorUrl;

    @Bean
    public WebClient aiOrchestratorClient() {
        HttpClient httpClient = HttpClient.create()
            .responseTimeout(Duration.ofSeconds(30))  // default, 엔드포인트별 override
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 5000);

        return WebClient.builder()
            .baseUrl(orchestratorUrl)
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            .codecs(configurer -> configurer
                .defaultCodecs()
                .maxInMemorySize(100 * 1024 * 1024))  // 100MB for video
            .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
            .filter(logRequest())
            .filter(logResponse())
            .build();
    }

    private ExchangeFilterFunction logRequest() {
        return ExchangeFilterFunction.ofRequestProcessor(request -> {
            log.info("[AI-PROXY] >>> {} {}", request.method(), request.url());
            return Mono.just(request);
        });
    }

    private ExchangeFilterFunction logResponse() {
        return ExchangeFilterFunction.ofResponseProcessor(response -> {
            log.info("[AI-PROXY] <<< Status: {}", response.statusCode());
            return Mono.just(response);
        });
    }
}
```

### 3.2 Generic Proxy Service

```java
@Service
@RequiredArgsConstructor
public class AiOrchestratorProxyService {

    private final WebClient aiOrchestratorClient;
    private final SessionMappingService sessionMappingService;

    /**
     * JSON 요청 프록시 (GET/POST with JSON body)
     */
    public <T> Mono<ResponseEntity<T>> proxyJsonRequest(
            HttpMethod method,
            String orchestratorPath,
            @Nullable Object requestBody,
            Class<T> responseType,
            Duration timeout,
            String springSessionId
    ) {
        String aiSessionId = sessionMappingService
            .getOrCreateAiSessionId(springSessionId);

        WebClient.RequestBodySpec spec = aiOrchestratorClient
            .method(method)
            .uri(uriBuilder -> uriBuilder
                .path(orchestratorPath)
                .queryParam("session_id", aiSessionId)
                .build());

        if (requestBody != null) {
            spec.bodyValue(requestBody);
        }

        return spec.retrieve()
            .toEntity(responseType)
            .timeout(timeout)
            .onErrorResume(WebClientResponseException.class,
                ex -> handleOrchestratorError(ex))
            .onErrorResume(TimeoutException.class,
                ex -> handleTimeoutError(ex, orchestratorPath));
    }

    /**
     * Multipart 요청 프록시 (파일 업로드)
     */
    public Mono<ResponseEntity<String>> proxyMultipartRequest(
            String orchestratorPath,
            MultiValueMap<String, HttpEntity<?>> multipartData,
            Duration timeout,
            String springSessionId
    ) {
        String aiSessionId = sessionMappingService
            .getOrCreateAiSessionId(springSessionId);

        return aiOrchestratorClient.post()
            .uri(uriBuilder -> uriBuilder
                .path(orchestratorPath)
                .queryParam("session_id", aiSessionId)
                .build())
            .contentType(MediaType.MULTIPART_FORM_DATA)
            .body(BodyInserters.fromMultipartData(multipartData))
            .retrieve()
            .toEntity(String.class)
            .timeout(timeout);
    }

    /**
     * Binary 응답 프록시 (GLB 파일 등)
     */
    public Mono<ResponseEntity<byte[]>> proxyBinaryRequest(
            String orchestratorPath,
            Duration timeout,
            String springSessionId
    ) {
        String aiSessionId = sessionMappingService
            .getOrCreateAiSessionId(springSessionId);

        return aiOrchestratorClient.get()
            .uri(uriBuilder -> uriBuilder
                .path(orchestratorPath)
                .queryParam("session_id", aiSessionId)
                .build())
            .accept(MediaType.APPLICATION_OCTET_STREAM)
            .retrieve()
            .toEntity(byte[].class)
            .timeout(timeout);
    }
}
```

### 3.3 Controller Example (Avatar)

```java
@RestController
@RequestMapping("/api/v6/avatar")
@RequiredArgsConstructor
public class AvatarProxyController {

    private final AiOrchestratorProxyService proxyService;

    private static final Duration GENERATE_TIMEOUT = Duration.ofSeconds(120);
    private static final Duration GLB_TIMEOUT = Duration.ofSeconds(30);

    @PostMapping("/generate")
    public Mono<ResponseEntity<String>> generateAvatar(
            @RequestParam("video") MultipartFile video,
            @RequestParam(value = "images", required = false) List<MultipartFile> images,
            HttpSession session
    ) {
        // File size validation
        if (video.getSize() > 100 * 1024 * 1024) {
            throw new FileSizeLimitExceededException("Video", video.getSize(), 100 * 1024 * 1024);
        }

        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("video", video.getResource());

        if (images != null) {
            for (MultipartFile image : images) {
                validateImageSize(image);
                builder.part("images", image.getResource());
            }
        }

        return proxyService.proxyMultipartRequest(
            "/avatar/generate",
            builder.build(),
            GENERATE_TIMEOUT,
            session.getId()
        );
    }

    @GetMapping("/glb")
    public Mono<ResponseEntity<byte[]>> getGlb(HttpSession session) {
        return proxyService.proxyBinaryRequest(
            "/avatar/glb",
            GLB_TIMEOUT,
            session.getId()
        ).map(response -> ResponseEntity.ok()
            .contentType(MediaType.parseMediaType("model/gltf-binary"))
            .header(HttpHeaders.CONTENT_DISPOSITION,
                "attachment; filename=\"avatar.glb\"")
            .body(response.getBody()));
    }

    private void validateImageSize(MultipartFile image) {
        if (image.getSize() > 10 * 1024 * 1024) {
            throw new FileSizeLimitExceededException(
                "Image", image.getSize(), 10 * 1024 * 1024);
        }
    }
}
```

---

## 4. Session Management (세션 관리)

### 4.1 Session Mapping 구조

```
Spring HttpSession (JSESSIONID)
        │
        ▼
┌─────────────────────────┐
│   SessionMappingService │
│                         │
│  springSessionId ──────>│──── aiOrchestratorSessionId (UUID)
│  userId ────────────────│──── springSessionId
│  createdAt ─────────────│──── lastAccessedAt
└─────────────────────────┘
        │
        ▼
AI Orchestrator session_id (query param)
```

### 4.2 Session Mapping Service

```java
@Service
public class SessionMappingService {

    // Spring Session ID → AI Orchestrator Session ID
    private final ConcurrentHashMap<String, AiSession> sessionMap
        = new ConcurrentHashMap<>();

    @Data
    @AllArgsConstructor
    static class AiSession {
        private String aiSessionId;
        private Long userId;
        private Instant createdAt;
        private Instant lastAccessedAt;
    }

    /**
     * Spring Session에 매핑된 AI Session ID를 반환한다.
     * 없으면 새로 생성한다.
     */
    public String getOrCreateAiSessionId(String springSessionId) {
        AiSession session = sessionMap.computeIfAbsent(
            springSessionId,
            key -> new AiSession(
                UUID.randomUUID().toString(),
                null,
                Instant.now(),
                Instant.now()
            )
        );
        session.setLastAccessedAt(Instant.now());
        return session.getAiSessionId();
    }

    /**
     * 사용자 ID를 세션에 바인딩한다. (로그인 시 호출)
     */
    public void bindUser(String springSessionId, Long userId) {
        AiSession session = sessionMap.get(springSessionId);
        if (session != null) {
            session.setUserId(userId);
        }
    }

    /**
     * 만료된 세션을 정리한다. (1시간 이상 미접근)
     */
    @Scheduled(fixedRate = 600_000)  // 10분마다
    public void cleanupExpiredSessions() {
        Instant cutoff = Instant.now().minus(Duration.ofHours(1));
        sessionMap.entrySet().removeIf(
            entry -> entry.getValue().getLastAccessedAt().isBefore(cutoff)
        );
    }
}
```

### 4.3 Session ID 주입 방식

AI Orchestrator에 `session_id`를 전달하는 방법:

```
# Query Parameter 방식 (기본)
POST http://localhost:8000/avatar/generate?session_id=abc-123-def

# Header 방식 (대안)
X-AI-Session-Id: abc-123-def
```

Query Parameter 방식을 기본으로 사용한다. AI Orchestrator(FastAPI)가 Query Parameter로
세션을 관리하고 있기 때문이다.

---

## 5. Error Handling (에러 처리)

### 5.1 AI Orchestrator Error → Frontend Error 변환

AI Orchestrator가 반환하는 에러 형식:

```json
{
    "detail": "No SMPL model found for session abc-123",
    "status_code": 400
}
```

Spring Boot가 Frontend에 반환하는 표준 에러 형식:

```json
{
    "error": {
        "code": "AVATAR_NOT_FOUND",
        "message": "아바타가 아직 생성되지 않았습니다. 먼저 아바타를 생성해주세요.",
        "detail": "No SMPL model found for session abc-123",
        "timestamp": "2026-02-11T10:30:00Z",
        "traceId": "trace-xyz-789"
    }
}
```

### 5.2 Error Code Mapping

| Orchestrator Status | Orchestrator Detail (contains) | Frontend Error Code | HTTP Status | 사용자 메시지 |
|---|---|---|---|---|
| 400 | "No SMPL model" | `AVATAR_NOT_FOUND` | 400 | 아바타를 먼저 생성해주세요 |
| 400 | "No video" | `INVALID_INPUT` | 400 | 비디오 파일이 필요합니다 |
| 400 | "No clothing" | `WARDROBE_EMPTY` | 400 | 의류를 먼저 등록해주세요 |
| 404 | "session not found" | `SESSION_EXPIRED` | 404 | 세션이 만료되었습니다 |
| 413 | - | `FILE_TOO_LARGE` | 413 | 파일 크기 제한을 초과했습니다 |
| 422 | - | `VALIDATION_ERROR` | 422 | 입력 데이터를 확인해주세요 |
| 500 | - | `AI_INTERNAL_ERROR` | 502 | AI 서버 내부 오류가 발생했습니다 |
| 503 | - | `AI_UNAVAILABLE` | 503 | AI 서버가 준비 중입니다 |
| Timeout | - | `AI_TIMEOUT` | 504 | 처리 시간이 초과되었습니다 |
| Connection refused | - | `AI_UNREACHABLE` | 503 | AI 서버에 연결할 수 없습니다 |

### 5.3 Global Exception Handler

```java
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(WebClientResponseException.class)
    public ResponseEntity<ErrorResponse> handleOrchestratorError(
            WebClientResponseException ex,
            HttpServletRequest request
    ) {
        String traceId = UUID.randomUUID().toString().substring(0, 8);
        String detail = extractDetail(ex.getResponseBodyAsString());

        ErrorCode errorCode = ErrorCodeMapper.fromOrchestratorError(
            ex.getStatusCode().value(), detail);

        log.error("[{}] Orchestrator error: {} {} - {}",
            traceId, ex.getStatusCode(), ex.getStatusText(), detail);

        return ResponseEntity
            .status(errorCode.getHttpStatus())
            .body(ErrorResponse.of(errorCode, detail, traceId));
    }

    @ExceptionHandler(TimeoutException.class)
    public ResponseEntity<ErrorResponse> handleTimeout(TimeoutException ex) {
        String traceId = UUID.randomUUID().toString().substring(0, 8);

        log.error("[{}] Orchestrator timeout: {}", traceId, ex.getMessage());

        return ResponseEntity
            .status(HttpStatus.GATEWAY_TIMEOUT)
            .body(ErrorResponse.of(
                ErrorCode.AI_TIMEOUT,
                "Processing exceeded time limit",
                traceId));
    }

    @ExceptionHandler(FileSizeLimitExceededException.class)
    public ResponseEntity<ErrorResponse> handleFileSizeExceeded(
            FileSizeLimitExceededException ex
    ) {
        return ResponseEntity
            .status(HttpStatus.PAYLOAD_TOO_LARGE)
            .body(ErrorResponse.of(ErrorCode.FILE_TOO_LARGE, ex.getMessage(), null));
    }
}
```

---

## 6. Retry Pattern (재시도 패턴)

### 6.1 503 Retry (Worker Cold Start 대응)

AI Orchestrator가 GPU Worker를 Cold Start할 때 503을 반환할 수 있다.
이 경우 **1회만** 30초 대기 후 재시도한다.

```java
@Component
public class OrchestratorRetryPolicy {

    /**
     * 503 응답 시 1회 재시도 (30초 대기)
     * LONG timeout 카테고리(fitting, 3D gen)는 Async Queue로 처리하므로 제외
     */
    public <T> Mono<T> withRetry(Mono<T> request, String endpoint) {
        return request
            .retryWhen(Retry.fixedDelay(1, Duration.ofSeconds(30))
                .filter(throwable -> {
                    if (throwable instanceof WebClientResponseException ex) {
                        boolean is503 = ex.getStatusCode() == HttpStatus.SERVICE_UNAVAILABLE;
                        if (is503) {
                            log.warn("[RETRY] 503 from {} - retrying in 30s", endpoint);
                        }
                        return is503;
                    }
                    return false;
                })
                .onRetryExhaustedThrow((spec, signal) ->
                    signal.failure())
            );
    }
}
```

### 6.2 Retry 적용 대상

| Endpoint Category | Retry on 503 | 이유 |
|---|---|---|
| SHORT (health, report, GLB) | Yes, 1회 | 빠른 응답 기대, cold start 가능 |
| MEDIUM (upload, avatar) | Yes, 1회 | Worker 준비 대기 |
| LONG (fitting, 3D gen) | **No** | Async Queue 패턴 사용 |

---

## 7. Request/Response Logging (요청/응답 로깅)

### 7.1 Logging 구조

```java
@Component
public class ProxyLoggingFilter implements WebFilter {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        long startTime = System.currentTimeMillis();
        String requestId = UUID.randomUUID().toString().substring(0, 8);

        ServerHttpRequest request = exchange.getRequest();

        // 요청 로그 (body 제외 — 파일 데이터 로깅 방지)
        log.info("[{}] >>> {} {} | Content-Type: {} | Content-Length: {}",
            requestId,
            request.getMethod(),
            request.getURI().getPath(),
            request.getHeaders().getContentType(),
            request.getHeaders().getContentLength());

        return chain.filter(exchange)
            .doFinally(signalType -> {
                long duration = System.currentTimeMillis() - startTime;
                HttpStatusCode status = exchange.getResponse().getStatusCode();

                log.info("[{}] <<< {} | {}ms | {}",
                    requestId,
                    status,
                    duration,
                    exchange.getRequest().getURI().getPath());

                // 느린 요청 경고 (30초 이상)
                if (duration > 30_000) {
                    log.warn("[{}] SLOW REQUEST: {}ms for {}",
                        requestId, duration,
                        exchange.getRequest().getURI().getPath());
                }
            });
    }
}
```

### 7.2 로그 레벨 가이드

| Level | 대상 | 예시 |
|---|---|---|
| INFO | 정상 요청/응답 | 모든 프록시 호출 |
| WARN | 느린 요청, 재시도 | 30s+ 응답, 503 retry |
| ERROR | 실패 응답, 타임아웃 | 500, timeout, connection refused |
| DEBUG | Request/Response Header 상세 | 개발 환경 전용 |

**주의**: multipart body (이미지/비디오)는 절대 로깅하지 않는다.
Content-Length와 Content-Type만 기록한다.

---

## 8. Rate Limiting (요청 제한)

### 8.1 Per-User Rate Limit 권장값

| Endpoint Group | Limit | Window | 이유 |
|---|---|---|---|
| `/api/v6/avatar/generate` | 5회 | 1시간 | GPU 집약적 |
| `/api/v6/wardrobe/*` | 30회 | 1시간 | 이미지 업로드 |
| `/api/v6/fitting/try-on` | 10회 | 1시간 | GPU 집약적 + Gemini API 비용 |
| `/api/v6/viewer3d/generate` | 3회 | 1시간 | Hunyuan3D 매우 무거움 |
| `/api/v6/p2p/*` | 20회 | 1시간 | CPU only |
| `/api/v6/health`, `/sessions` | Unlimited | - | 관리용 |
| 전체 | 100회 | 1시간 | 글로벌 상한 |

### 8.2 Bucket4j 기반 구현

```java
@Configuration
public class RateLimitConfig {

    @Bean
    public FilterRegistrationBean<RateLimitFilter> rateLimitFilter() {
        FilterRegistrationBean<RateLimitFilter> registration
            = new FilterRegistrationBean<>();
        registration.setFilter(new RateLimitFilter());
        registration.addUrlPatterns("/api/v6/*");
        registration.setOrder(1);
        return registration;
    }
}

@Component
public class RateLimitFilter extends OncePerRequestFilter {

    // userId → Bucket
    private final ConcurrentHashMap<Long, Bucket> buckets
        = new ConcurrentHashMap<>();

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {

        Long userId = extractUserId(request);
        if (userId == null) {
            filterChain.doFilter(request, response);
            return;
        }

        Bucket bucket = buckets.computeIfAbsent(userId, this::createBucket);

        if (bucket.tryConsume(1)) {
            filterChain.doFilter(request, response);
        } else {
            response.setStatus(HttpStatus.TOO_MANY_REQUESTS.value());
            response.setContentType(MediaType.APPLICATION_JSON_VALUE);
            response.getWriter().write(
                """
                {
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
                    }
                }
                """);
        }
    }

    private Bucket createBucket(Long userId) {
        return Bucket.builder()
            .addLimit(Bandwidth.classic(
                100, Refill.intervally(100, Duration.ofHours(1))))
            .build();
    }
}
```

### 8.3 Endpoint-Specific Rate Limit

GPU 집약적 엔드포인트는 별도의 Bucket을 사용한다.

```java
// endpoint별 세분화 예시
private Bucket createGpuBucket(Long userId) {
    return Bucket.builder()
        .addLimit(Bandwidth.classic(
            5, Refill.intervally(5, Duration.ofHours(1))))  // avatar: 5/hr
        .build();
}

private Bucket createFittingBucket(Long userId) {
    return Bucket.builder()
        .addLimit(Bandwidth.classic(
            10, Refill.intervally(10, Duration.ofHours(1))))  // fitting: 10/hr
        .build();
}
```

---

## 9. File Size Limits (파일 크기 제한)

### 9.1 제한 정책

| File Type | Max Size | Validation Point | 비고 |
|---|---|---|---|
| 이미지 (JPG, PNG, WebP) | 10 MB | Spring Boot Controller | per file |
| 비디오 (MP4, MOV) | 100 MB | Spring Boot Controller | per file |
| GLB 모델 (다운로드) | 50 MB | - | 응답 크기, 제한 없음 |
| 다건 이미지 업로드 합계 | 50 MB | Spring Boot Controller | 10장 합산 |

### 9.2 Spring Boot 설정

```yaml
# application.yml
spring:
  servlet:
    multipart:
      max-file-size: 100MB      # 단일 파일 최대
      max-request-size: 150MB   # 전체 요청 최대 (video + images)
      file-size-threshold: 5MB  # 이 크기 이상이면 디스크에 임시 저장
      location: /tmp/stylelens-upload  # 임시 파일 경로
```

### 9.3 Validation 구현

```java
@Component
public class FileValidationService {

    private static final long MAX_IMAGE_SIZE = 10 * 1024 * 1024;     // 10MB
    private static final long MAX_VIDEO_SIZE = 100 * 1024 * 1024;    // 100MB
    private static final long MAX_BATCH_SIZE = 50 * 1024 * 1024;     // 50MB
    private static final int MAX_BATCH_COUNT = 10;

    private static final Set<String> ALLOWED_IMAGE_TYPES = Set.of(
        "image/jpeg", "image/png", "image/webp"
    );
    private static final Set<String> ALLOWED_VIDEO_TYPES = Set.of(
        "video/mp4", "video/quicktime"
    );

    public void validateImage(MultipartFile file) {
        if (file.getSize() > MAX_IMAGE_SIZE) {
            throw new FileSizeLimitExceededException(
                "이미지 파일은 10MB 이하여야 합니다.",
                file.getSize(), MAX_IMAGE_SIZE);
        }
        if (!ALLOWED_IMAGE_TYPES.contains(file.getContentType())) {
            throw new InvalidFileTypeException(
                "JPG, PNG, WebP 형식만 지원합니다.");
        }
    }

    public void validateVideo(MultipartFile file) {
        if (file.getSize() > MAX_VIDEO_SIZE) {
            throw new FileSizeLimitExceededException(
                "비디오 파일은 100MB 이하여야 합니다.",
                file.getSize(), MAX_VIDEO_SIZE);
        }
        if (!ALLOWED_VIDEO_TYPES.contains(file.getContentType())) {
            throw new InvalidFileTypeException(
                "MP4, MOV 형식만 지원합니다.");
        }
    }

    public void validateImageBatch(List<MultipartFile> files) {
        if (files.size() > MAX_BATCH_COUNT) {
            throw new IllegalArgumentException(
                "이미지는 최대 " + MAX_BATCH_COUNT + "장까지 업로드 가능합니다.");
        }
        long totalSize = files.stream()
            .mapToLong(MultipartFile::getSize).sum();
        if (totalSize > MAX_BATCH_SIZE) {
            throw new FileSizeLimitExceededException(
                "이미지 합계는 50MB 이하여야 합니다.",
                totalSize, MAX_BATCH_SIZE);
        }
        files.forEach(this::validateImage);
    }
}
```

---

## 10. Spring Boot Configuration (application.yml)

```yaml
server:
  port: 8080
  servlet:
    session:
      timeout: 1h
      cookie:
        name: STYLELENS_SESSION
        http-only: true
        secure: true       # production에서 true
        same-site: lax

spring:
  application:
    name: stylelens-backend
  profiles:
    active: ${SPRING_PROFILES_ACTIVE:dev}

  # Multipart 설정
  servlet:
    multipart:
      max-file-size: 100MB
      max-request-size: 150MB
      file-size-threshold: 5MB
      location: /tmp/stylelens-upload

  # Security (see Auth_Flow.md)
  security:
    oauth2:
      client:
        registration:
          google:
            client-id: ${GOOGLE_CLIENT_ID}
            client-secret: ${GOOGLE_CLIENT_SECRET}
            scope: openid, profile, email

# === StyleLens Custom Configuration ===
stylelens:
  ai-orchestrator:
    url: ${AI_ORCHESTRATOR_URL:http://localhost:8000}
    timeout:
      short: 30s
      medium: 120s
      long: 300s
    retry:
      max-attempts: 1
      delay: 30s               # 503 cold start 대기

  rate-limit:
    global-per-hour: 100
    avatar-per-hour: 5
    fitting-per-hour: 10
    viewer3d-per-hour: 3

  file:
    max-image-size: 10MB
    max-video-size: 100MB
    max-batch-size: 50MB
    max-batch-count: 10
    allowed-image-types: image/jpeg, image/png, image/webp
    allowed-video-types: video/mp4, video/quicktime

  storage:
    type: ${STORAGE_TYPE:local}         # local | s3
    s3:
      bucket: ${S3_BUCKET:stylelens-assets}
      region: ${AWS_REGION:ap-northeast-2}
    local:
      base-path: /data/stylelens

  async:
    core-pool-size: 4
    max-pool-size: 8
    queue-capacity: 50

# Actuator (운영 모니터링)
management:
  endpoints:
    web:
      exposure:
        include: health, info, metrics, prometheus
  endpoint:
    health:
      show-details: when_authorized

# Logging
logging:
  level:
    com.stylelens: INFO
    com.stylelens.proxy: DEBUG           # 개발 환경
    org.springframework.web: WARN
  pattern:
    console: "%d{HH:mm:ss.SSS} [%thread] %-5level [%X{traceId}] %logger{36} - %msg%n"
```

---

## 11. Async Job Queue Pattern (비동기 작업 큐)

### 11.1 Long-Running Operation 처리

Fitting Try-On (300s)과 3D Generation (300s)은 동기 Proxy가 부적합하다.
**Async Job Queue** 패턴을 사용하여 Frontend가 Polling으로 상태를 확인한다.

```
1. Frontend: POST /api/v6/fitting/try-on  →  { "job_id": "job-123" }  (즉시 응답)
2. Spring Boot: AI Orchestrator에 비동기 요청 전달
3. Frontend: GET  /api/v6/jobs/job-123    →  { "status": "PROCESSING", "progress": 45 }
4. Frontend: GET  /api/v6/jobs/job-123    →  { "status": "COMPLETED", "result": { ... } }
```

### 11.2 Job Entity

```java
@Entity
@Table(name = "async_jobs")
@Getter @Setter
public class AsyncJob {

    @Id
    private String jobId;

    @Enumerated(EnumType.STRING)
    private JobType jobType;          // FITTING, VIEWER3D

    @Enumerated(EnumType.STRING)
    private JobStatus status;         // QUEUED, PROCESSING, COMPLETED, FAILED

    private Long userId;
    private String springSessionId;
    private String aiSessionId;

    @Column(columnDefinition = "TEXT")
    private String requestPayload;    // 원본 요청 JSON

    @Column(columnDefinition = "TEXT")
    private String resultPayload;     // 결과 JSON (완료 시)

    private String errorMessage;      // 에러 메시지 (실패 시)
    private Integer progress;         // 0-100 진행률

    private Instant createdAt;
    private Instant startedAt;
    private Instant completedAt;

    public enum JobType { FITTING, VIEWER3D }
    public enum JobStatus { QUEUED, PROCESSING, COMPLETED, FAILED }
}
```

### 11.3 Async Job Service

```java
@Service
@RequiredArgsConstructor
public class AsyncJobService {

    private final AsyncJobRepository jobRepository;
    private final WebClient aiOrchestratorClient;
    private final SessionMappingService sessionMappingService;

    @Async("aiTaskExecutor")
    public void submitFittingJob(String jobId, String payload,
                                  String springSessionId) {
        AsyncJob job = jobRepository.findById(jobId).orElseThrow();
        job.setStatus(AsyncJob.JobStatus.PROCESSING);
        job.setStartedAt(Instant.now());
        jobRepository.save(job);

        String aiSessionId = sessionMappingService
            .getOrCreateAiSessionId(springSessionId);

        try {
            String result = aiOrchestratorClient.post()
                .uri(uriBuilder -> uriBuilder
                    .path("/fitting/try-on")
                    .queryParam("session_id", aiSessionId)
                    .build())
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(payload)
                .retrieve()
                .bodyToMono(String.class)
                .block(Duration.ofSeconds(300));

            job.setStatus(AsyncJob.JobStatus.COMPLETED);
            job.setResultPayload(result);
            job.setProgress(100);
            job.setCompletedAt(Instant.now());

        } catch (Exception ex) {
            job.setStatus(AsyncJob.JobStatus.FAILED);
            job.setErrorMessage(ex.getMessage());
            job.setCompletedAt(Instant.now());
            log.error("[JOB-{}] Fitting failed: {}", jobId, ex.getMessage());
        }

        jobRepository.save(job);
    }

    public AsyncJob getJobStatus(String jobId) {
        return jobRepository.findById(jobId)
            .orElseThrow(() -> new JobNotFoundException(jobId));
    }
}
```

### 11.4 Job Status Controller

```java
@RestController
@RequestMapping("/api/v6/jobs")
@RequiredArgsConstructor
public class JobController {

    private final AsyncJobService asyncJobService;

    @GetMapping("/{jobId}")
    public ResponseEntity<JobStatusResponse> getJobStatus(
            @PathVariable String jobId,
            @AuthenticationPrincipal UserPrincipal user
    ) {
        AsyncJob job = asyncJobService.getJobStatus(jobId);

        // 본인의 Job만 조회 가능
        if (!job.getUserId().equals(user.getId())) {
            throw new AccessDeniedException("이 작업에 접근 권한이 없습니다.");
        }

        return ResponseEntity.ok(JobStatusResponse.from(job));
    }
}

@Data
public class JobStatusResponse {
    private String jobId;
    private String status;        // QUEUED | PROCESSING | COMPLETED | FAILED
    private Integer progress;     // 0-100
    private Object result;        // COMPLETED일 때만
    private String error;         // FAILED일 때만
    private Long elapsedSeconds;

    public static JobStatusResponse from(AsyncJob job) {
        JobStatusResponse res = new JobStatusResponse();
        res.setJobId(job.getJobId());
        res.setStatus(job.getStatus().name());
        res.setProgress(job.getProgress());

        if (job.getStatus() == AsyncJob.JobStatus.COMPLETED) {
            res.setResult(JsonParser.parse(job.getResultPayload()));
        }
        if (job.getStatus() == AsyncJob.JobStatus.FAILED) {
            res.setError(job.getErrorMessage());
        }
        if (job.getStartedAt() != null) {
            Instant end = job.getCompletedAt() != null
                ? job.getCompletedAt() : Instant.now();
            res.setElapsedSeconds(
                Duration.between(job.getStartedAt(), end).getSeconds());
        }

        return res;
    }
}
```

### 11.5 Async Executor Configuration

```java
@Configuration
@EnableAsync
public class AsyncConfig {

    @Bean("aiTaskExecutor")
    public Executor aiTaskExecutor(
            @Value("${stylelens.async.core-pool-size:4}") int corePoolSize,
            @Value("${stylelens.async.max-pool-size:8}") int maxPoolSize,
            @Value("${stylelens.async.queue-capacity:50}") int queueCapacity
    ) {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(corePoolSize);
        executor.setMaxPoolSize(maxPoolSize);
        executor.setQueueCapacity(queueCapacity);
        executor.setThreadNamePrefix("ai-task-");
        executor.setRejectedExecutionHandler(
            new ThreadPoolExecutor.CallerRunsPolicy());
        executor.initialize();
        return executor;
    }
}
```

---

## 12. Frontend Polling Pattern (참고: Frontend 팀 전달용)

Async Job을 사용하는 경우 Frontend의 Polling 패턴:

```typescript
// Frontend 참고 코드 (React)
async function waitForJob(jobId: string): Promise<JobResult> {
    const POLL_INTERVAL = 3000; // 3초
    const MAX_WAIT = 360_000;   // 6분
    const startTime = Date.now();

    while (Date.now() - startTime < MAX_WAIT) {
        const res = await fetch(`/api/v6/jobs/${jobId}`);
        const data = await res.json();

        if (data.status === 'COMPLETED') return data.result;
        if (data.status === 'FAILED') throw new Error(data.error);

        // Progress UI 업데이트
        updateProgress(data.progress);

        await sleep(POLL_INTERVAL);
    }

    throw new Error('Job timeout');
}
```

---

## Appendix A: Proxy Endpoint Quick Reference (빠른 참조)

```java
// Avatar
POST  /api/v6/avatar/generate    → /avatar/generate       [MEDIUM, multipart]
GET   /api/v6/avatar/glb          → /avatar/glb            [SHORT, binary]

// Wardrobe
POST  /api/v6/wardrobe/add-image        → /wardrobe/add-image        [MEDIUM, multipart]
POST  /api/v6/wardrobe/add-images       → /wardrobe/add-images       [MEDIUM, multipart]
POST  /api/v6/wardrobe/add-url          → /wardrobe/add-url          [MEDIUM, form]
POST  /api/v6/wardrobe/extract-model-info → /wardrobe/extract-model-info [SHORT, json]

// Fitting (Async Job)
POST  /api/v6/fitting/try-on     → /fitting/try-on        [LONG → ASYNC]

// 3D Viewer (Async Job)
POST  /api/v6/viewer3d/generate  → /viewer3d/generate     [LONG → ASYNC]
GET   /api/v6/viewer3d/model/*   → /viewer3d/model/*      [SHORT, binary]

// Analysis
POST  /api/v6/p2p/analyze        → /p2p/analyze           [MEDIUM, json]

// Admin
GET   /api/v6/quality/report     → /quality/report        [SHORT]
GET   /api/v6/health             → /health                [SHORT]
GET   /api/v6/sessions           → /sessions              [SHORT]

// Async Job Polling (Spring Boot 자체)
GET   /api/v6/jobs/{jobId}       → DB 조회                [LOCAL]
```
