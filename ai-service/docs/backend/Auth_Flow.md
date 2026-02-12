# StyleLens V6 Authentication & Authorization Flow

> Google OAuth2 + JWT + Spring Security Specification
> Version: 1.0 | Last Updated: 2026-02-11

---

## 1. Architecture Overview (인증 아키텍처 개요)

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│   React      │     │  Spring Boot     │     │  Google OAuth2   │     │ AI Orchestr. │
│   Frontend   │────>│  (Tier 2)        │────>│  Server          │     │ (Tier 3)     │
│              │<────│                  │<────│                  │     │              │
│   :3000      │     │  :8080           │     │                  │     │  :8000       │
└──────────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘
       │                     │                                                │
       │   JWT Bearer        │   OAuth2 Code Flow                             │
       │   (모든 요청)       │   (로그인 시만)                                │
       │                     │                                                │
       │                     │────── Internal Network (No Auth) ──────────────│
       │                     │   AI Orchestrator는 인증 없음                  │
       │                     │   (VPC/Docker Network로 보호)                  │
```

**핵심 원칙:**
- Frontend ↔ Spring Boot: **JWT Bearer Token** (모든 API 요청)
- Spring Boot → Google: **OAuth2 Authorization Code Flow** (로그인 시)
- Spring Boot → AI Orchestrator: **인증 없음** (Internal Network 격리)
- AI Orchestrator는 외부에서 직접 접근 불가 (Spring Boot만 접근)

---

## 2. Google OAuth2 Login Flow (로그인 흐름)

### 2.1 전체 시퀀스

```
Step 1: 사용자가 "Google로 로그인" 버튼 클릭
┌──────────┐                  ┌──────────────┐                 ┌──────────────┐
│ Frontend │ ── GET ──────>   │ Spring Boot  │                 │   Google     │
│          │ /api/v6/auth/    │              │                 │              │
│          │ google/login     │              │                 │              │
└──────────┘                  └──────────────┘                 └──────────────┘
                                     │
Step 2: Spring Boot가 Google OAuth URL로 Redirect
                                     │
                                     ├── 302 Redirect ─────────────────────────>
                                     │   https://accounts.google.com/o/oauth2/v2/auth
                                     │   ?client_id=...
                                     │   &redirect_uri=.../api/v6/auth/google/callback
                                     │   &scope=openid+profile+email
                                     │   &response_type=code
                                     │   &state={csrf_token}
                                     │
Step 3: 사용자가 Google에서 로그인 + 동의
                                                               │
Step 4: Google이 Authorization Code와 함께 Redirect            │
                              <──────────── 302 Redirect ──────┘
                              │   /api/v6/auth/google/callback
                              │   ?code={auth_code}
                              │   &state={csrf_token}
                              │
Step 5: Spring Boot가 Code로 Token 교환
                              │
                              ├── POST ────────────────────────>
                              │   https://oauth2.googleapis.com/token
                              │   (code → access_token + id_token)
                              │
                              <── { access_token, id_token } ──┘
                              │
Step 6: id_token에서 사용자 정보 추출
                              │   email, name, picture
                              │
Step 7: 사용자 DB 조회/생성 + JWT 발급
                              │
Step 8: Frontend로 JWT 전달
┌──────────┐                  │
│ Frontend │ <── 302 Redirect ┘
│          │   /?token={jwt_access_token}
│          │   &refresh_token={jwt_refresh_token}
└──────────┘
                              │
Step 9: 이후 모든 API 요청에 JWT 포함
                              │
                    Authorization: Bearer {jwt_access_token}
```

### 2.2 Login Endpoint

```java
@RestController
@RequestMapping("/api/v6/auth")
@RequiredArgsConstructor
public class AuthController {

    private final GoogleOAuth2Service googleOAuth2Service;
    private final JwtTokenProvider jwtTokenProvider;
    private final UserService userService;

    @Value("${stylelens.frontend.url:http://localhost:3000}")
    private String frontendUrl;

    /**
     * Step 1-2: Google OAuth2 로그인 시작
     * Frontend에서 이 URL을 직접 호출하거나 새 창으로 연다.
     */
    @GetMapping("/google/login")
    public ResponseEntity<Void> initiateGoogleLogin(HttpSession session) {
        String state = UUID.randomUUID().toString();
        session.setAttribute("oauth2_state", state);

        String authUrl = googleOAuth2Service.buildAuthorizationUrl(state);

        return ResponseEntity
            .status(HttpStatus.FOUND)
            .location(URI.create(authUrl))
            .build();
    }

    /**
     * Step 4-8: Google Callback 처리
     */
    @GetMapping("/google/callback")
    public ResponseEntity<Void> handleGoogleCallback(
            @RequestParam("code") String authCode,
            @RequestParam("state") String state,
            HttpSession session
    ) {
        // CSRF 검증
        String savedState = (String) session.getAttribute("oauth2_state");
        if (savedState == null || !savedState.equals(state)) {
            throw new OAuth2AuthenticationException("Invalid state parameter");
        }
        session.removeAttribute("oauth2_state");

        // Step 5: Code → Token 교환
        GoogleTokenResponse tokenResponse =
            googleOAuth2Service.exchangeCodeForToken(authCode);

        // Step 6: 사용자 정보 추출
        GoogleUserInfo userInfo =
            googleOAuth2Service.extractUserInfo(tokenResponse.getIdToken());

        // Step 7: DB 조회/생성 + Role 할당
        User user = userService.findOrCreateUser(userInfo);

        // Step 7: JWT 발급
        String accessToken = jwtTokenProvider.createAccessToken(user);
        String refreshToken = jwtTokenProvider.createRefreshToken(user);

        // Step 8: Frontend로 Redirect
        String redirectUrl = UriComponentsBuilder
            .fromUriString(frontendUrl)
            .path("/auth/callback")
            .queryParam("token", accessToken)
            .queryParam("refresh_token", refreshToken)
            .build().toUriString();

        return ResponseEntity
            .status(HttpStatus.FOUND)
            .location(URI.create(redirectUrl))
            .build();
    }

    /**
     * JWT Refresh
     */
    @PostMapping("/refresh")
    public ResponseEntity<TokenResponse> refreshToken(
            @RequestBody RefreshTokenRequest request
    ) {
        String newAccessToken = jwtTokenProvider
            .refreshAccessToken(request.getRefreshToken());

        return ResponseEntity.ok(new TokenResponse(newAccessToken));
    }

    /**
     * 로그아웃 (JWT 무효화)
     */
    @PostMapping("/logout")
    public ResponseEntity<Void> logout(
            @AuthenticationPrincipal UserPrincipal user
    ) {
        jwtTokenProvider.revokeRefreshToken(user.getId());
        return ResponseEntity.noContent().build();
    }
}
```

---

## 3. JWT Token Structure (JWT 토큰 구조)

### 3.1 Access Token

```json
{
    "header": {
        "alg": "HS512",
        "typ": "JWT"
    },
    "payload": {
        "sub": "12345",                           // userId (DB PK)
        "email": "user@example.com",
        "name": "홍길동",
        "picture": "https://lh3.google...",       // Google profile image
        "role": "USER",                            // USER | ADMIN | SUPER_ADMIN
        "iat": 1739260200,                         // issued at
        "exp": 1739263800                          // expires (1시간)
    },
    "signature": "..."
}
```

### 3.2 Refresh Token

```json
{
    "header": {
        "alg": "HS512",
        "typ": "JWT"
    },
    "payload": {
        "sub": "12345",
        "type": "refresh",
        "jti": "unique-token-id",                  // 토큰 고유 ID (무효화용)
        "iat": 1739260200,
        "exp": 1741852200                           // 30일
    },
    "signature": "..."
}
```

### 3.3 Token 수명

| Token Type | Lifetime | 갱신 방법 |
|---|---|---|
| Access Token | **1시간** | Refresh Token으로 갱신 |
| Refresh Token | **30일** | 재로그인 |

### 3.4 JWT Token Provider

```java
@Component
public class JwtTokenProvider {

    @Value("${stylelens.jwt.secret}")
    private String jwtSecret;

    @Value("${stylelens.jwt.access-token-expiry:3600000}")  // 1시간
    private long accessTokenExpiry;

    @Value("${stylelens.jwt.refresh-token-expiry:2592000000}")  // 30일
    private long refreshTokenExpiry;

    private final RefreshTokenRepository refreshTokenRepository;

    public String createAccessToken(User user) {
        Date now = new Date();
        Date expiry = new Date(now.getTime() + accessTokenExpiry);

        return Jwts.builder()
            .setSubject(String.valueOf(user.getId()))
            .claim("email", user.getEmail())
            .claim("name", user.getName())
            .claim("picture", user.getPictureUrl())
            .claim("role", user.getRole().name())
            .setIssuedAt(now)
            .setExpiration(expiry)
            .signWith(SignatureAlgorithm.HS512, jwtSecret)
            .compact();
    }

    public String createRefreshToken(User user) {
        String tokenId = UUID.randomUUID().toString();
        Date now = new Date();
        Date expiry = new Date(now.getTime() + refreshTokenExpiry);

        // DB에 저장 (무효화를 위해)
        RefreshTokenEntity entity = new RefreshTokenEntity();
        entity.setTokenId(tokenId);
        entity.setUserId(user.getId());
        entity.setExpiresAt(expiry.toInstant());
        entity.setRevoked(false);
        refreshTokenRepository.save(entity);

        return Jwts.builder()
            .setSubject(String.valueOf(user.getId()))
            .claim("type", "refresh")
            .setId(tokenId)
            .setIssuedAt(now)
            .setExpiration(expiry)
            .signWith(SignatureAlgorithm.HS512, jwtSecret)
            .compact();
    }

    public UserPrincipal validateAccessToken(String token) {
        Claims claims = Jwts.parser()
            .setSigningKey(jwtSecret)
            .parseClaimsJws(token)
            .getBody();

        return UserPrincipal.builder()
            .id(Long.parseLong(claims.getSubject()))
            .email(claims.get("email", String.class))
            .name(claims.get("name", String.class))
            .role(UserRole.valueOf(claims.get("role", String.class)))
            .build();
    }

    public String refreshAccessToken(String refreshToken) {
        Claims claims = Jwts.parser()
            .setSigningKey(jwtSecret)
            .parseClaimsJws(refreshToken)
            .getBody();

        // DB에서 토큰 유효성 확인
        String tokenId = claims.getId();
        RefreshTokenEntity entity = refreshTokenRepository
            .findByTokenId(tokenId)
            .orElseThrow(() -> new InvalidTokenException("Refresh token not found"));

        if (entity.isRevoked()) {
            throw new InvalidTokenException("Refresh token has been revoked");
        }

        Long userId = Long.parseLong(claims.getSubject());
        User user = userService.findById(userId);

        return createAccessToken(user);
    }

    public void revokeRefreshToken(Long userId) {
        refreshTokenRepository.revokeAllByUserId(userId);
    }
}
```

---

## 4. Spring Security Configuration

### 4.1 Security Filter Chain

```java
@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true)
@RequiredArgsConstructor
public class SecurityConfig {

    private final JwtAuthenticationFilter jwtAuthFilter;
    private final JwtAuthenticationEntryPoint jwtEntryPoint;

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http)
            throws Exception {
        http
            // CSRF 비활성화 (JWT 사용, SPA 아키텍처)
            .csrf(csrf -> csrf.disable())

            // CORS 설정
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))

            // Session 사용 안 함 (JWT Stateless)
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))

            // 인증 예외 핸들러
            .exceptionHandling(ex -> ex
                .authenticationEntryPoint(jwtEntryPoint))

            // URL별 접근 권한
            .authorizeHttpRequests(auth -> auth
                // 인증 없이 접근 가능
                .requestMatchers(
                    "/api/v6/auth/**",           // OAuth2 로그인
                    "/api/v6/health",            // Health Check
                    "/actuator/health"           // Actuator Health
                ).permitAll()

                // ADMIN 이상만 접근 가능
                .requestMatchers(
                    "/api/v6/sessions",
                    "/api/v6/quality/**",
                    "/actuator/**"
                ).hasAnyRole("ADMIN", "SUPER_ADMIN")

                // SUPER_ADMIN만 접근 가능
                .requestMatchers(
                    "/api/v6/admin/**"
                ).hasRole("SUPER_ADMIN")

                // 나머지는 인증 필요
                .anyRequest().authenticated()
            )

            // JWT Filter를 UsernamePasswordAuthentication 앞에 배치
            .addFilterBefore(jwtAuthFilter,
                UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowedOrigins(List.of(
            "http://localhost:3000",               // 개발
            "https://stylelens.app"                // 프로덕션
        ));
        config.setAllowedMethods(List.of(
            "GET", "POST", "PUT", "DELETE", "OPTIONS"));
        config.setAllowedHeaders(List.of("*"));
        config.setAllowCredentials(true);
        config.setMaxAge(3600L);

        UrlBasedCorsConfigurationSource source
            = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/api/**", config);
        return source;
    }
}
```

### 4.2 JWT Authentication Filter

```java
@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtTokenProvider jwtTokenProvider;

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {

        String token = extractToken(request);

        if (token != null) {
            try {
                UserPrincipal principal =
                    jwtTokenProvider.validateAccessToken(token);

                UsernamePasswordAuthenticationToken authentication =
                    new UsernamePasswordAuthenticationToken(
                        principal,
                        null,
                        principal.getAuthorities()  // ROLE_USER, ROLE_ADMIN 등
                    );

                authentication.setDetails(
                    new WebAuthenticationDetailsSource()
                        .buildDetails(request));

                SecurityContextHolder.getContext()
                    .setAuthentication(authentication);

            } catch (ExpiredJwtException ex) {
                // Token 만료 — 401 반환 (Frontend가 Refresh 시도)
                log.debug("JWT expired for request: {}",
                    request.getRequestURI());
            } catch (JwtException ex) {
                log.warn("Invalid JWT: {}", ex.getMessage());
            }
        }

        filterChain.doFilter(request, response);
    }

    private String extractToken(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (bearerToken != null && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getRequestURI();
        return path.startsWith("/api/v6/auth/")
            || path.equals("/api/v6/health")
            || path.startsWith("/actuator/");
    }
}
```

### 4.3 JWT Authentication Entry Point

```java
@Component
public class JwtAuthenticationEntryPoint implements AuthenticationEntryPoint {

    @Override
    public void commence(
            HttpServletRequest request,
            HttpServletResponse response,
            AuthenticationException authException
    ) throws IOException {
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");

        response.getWriter().write("""
            {
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "인증이 필요합니다. 로그인해주세요."
                }
            }
            """);
    }
}
```

---

## 5. Internal API Authentication (내부 API 인증)

### 5.1 Spring Boot → AI Orchestrator

Spring Boot와 AI Orchestrator 사이에는 **별도 인증이 없다.**
네트워크 수준에서 격리하여 보안을 확보한다.

```
[Internet]
    │
    ▼
[Load Balancer / Reverse Proxy]  ← 외부 접근 차단: AI Orchestrator
    │
    ▼
[Spring Boot :8080]  ── Internal Network ──  [AI Orchestrator :8000]
    │                                              │
    │     Docker Compose / K8s / VPC               │
    │     동일 네트워크에서만 통신                   │
```

### 5.2 네트워크 격리 방법

**Docker Compose (개발/스테이징):**
```yaml
# docker-compose.yml
services:
  spring-boot:
    ports:
      - "8080:8080"       # 외부 노출
    networks:
      - frontend
      - backend

  ai-orchestrator:
    # ports 미노출 — 외부 접근 불가
    networks:
      - backend           # Spring Boot와만 통신

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true        # 외부 접근 차단
```

**Kubernetes (프로덕션):**
```yaml
# ai-orchestrator Service — ClusterIP (내부 전용)
apiVersion: v1
kind: Service
metadata:
  name: ai-orchestrator
spec:
  type: ClusterIP         # 외부 접근 불가
  ports:
    - port: 8000
  selector:
    app: ai-orchestrator
```

### 5.3 Internal Header (선택사항)

추가 보안이 필요한 경우, 공유 비밀키 Header를 사용할 수 있다.

```java
// Spring Boot → AI Orchestrator 요청 시
@Bean
public WebClient aiOrchestratorClient(
        @Value("${stylelens.ai-orchestrator.internal-key:}") String internalKey
) {
    WebClient.Builder builder = WebClient.builder()
        .baseUrl(orchestratorUrl);

    if (!internalKey.isBlank()) {
        builder.defaultHeader("X-Internal-Key", internalKey);
    }

    return builder.build();
}
```

```python
# AI Orchestrator (FastAPI) 검증
@app.middleware("http")
async def verify_internal_key(request: Request, call_next):
    expected = os.getenv("INTERNAL_KEY", "")
    if expected and request.headers.get("X-Internal-Key") != expected:
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)
```

---

## 6. User Roles (사용자 역할)

### 6.1 Role 정의

| Role | 코드값 | 설명 | 접근 범위 |
|---|---|---|---|
| `USER` | `ROLE_USER` | 일반 사용자 | Avatar, Wardrobe, Fitting, 3D Viewer |
| `ADMIN` | `ROLE_ADMIN` | 관리자 | USER + Session 관리, Quality Report |
| `SUPER_ADMIN` | `ROLE_SUPER_ADMIN` | 최고 관리자 | ADMIN + 전체 시스템 관리 |

### 6.2 Super Admin 설정

Super Admin은 **하드코딩**으로 설정한다.
추가 Super Admin이 필요한 경우 환경 변수로 확장할 수 있다.

```java
@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;

    // Super Admin Email — 하드코딩
    private static final String SUPER_ADMIN_EMAIL = "aisamdasu1204@gmail.com";

    // 환경 변수로 추가 Admin 설정 가능
    @Value("${stylelens.admin.emails:}")
    private List<String> adminEmails;

    /**
     * Google OAuth 로그인 시 호출
     * 사용자가 없으면 생성, 있으면 정보 업데이트
     */
    public User findOrCreateUser(GoogleUserInfo userInfo) {
        User user = userRepository.findByEmail(userInfo.getEmail())
            .orElseGet(() -> createNewUser(userInfo));

        // 프로필 정보 업데이트
        user.setName(userInfo.getName());
        user.setPictureUrl(userInfo.getPicture());
        user.setLastLoginAt(Instant.now());

        // Role 결정 (매 로그인마다 재확인)
        user.setRole(determineRole(userInfo.getEmail()));

        return userRepository.save(user);
    }

    private UserRole determineRole(String email) {
        if (SUPER_ADMIN_EMAIL.equals(email)) {
            return UserRole.SUPER_ADMIN;
        }
        if (adminEmails.contains(email)) {
            return UserRole.ADMIN;
        }
        return UserRole.USER;
    }

    private User createNewUser(GoogleUserInfo userInfo) {
        User user = new User();
        user.setEmail(userInfo.getEmail());
        user.setGoogleId(userInfo.getGoogleId());
        user.setName(userInfo.getName());
        user.setPictureUrl(userInfo.getPicture());
        user.setCreatedAt(Instant.now());
        return user;
    }
}
```

### 6.3 User Entity

```java
@Entity
@Table(name = "users")
@Getter @Setter
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private String email;

    @Column(unique = true, nullable = false)
    private String googleId;

    @Column(nullable = false)
    private String name;

    private String pictureUrl;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private UserRole role = UserRole.USER;

    private Instant createdAt;
    private Instant lastLoginAt;

    // Storage 관련 (see File_Storage.md)
    private Long storageUsedBytes = 0L;

    @Column(nullable = false)
    private Long storageQuotaBytes = 5L * 1024 * 1024 * 1024;  // 5GB default
}

public enum UserRole {
    USER,
    ADMIN,
    SUPER_ADMIN;

    public String getSpringRole() {
        return "ROLE_" + this.name();
    }
}
```

### 6.4 UserPrincipal (Security Context)

```java
@Data
@Builder
public class UserPrincipal implements UserDetails {

    private Long id;
    private String email;
    private String name;
    private UserRole role;

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        List<SimpleGrantedAuthority> authorities = new ArrayList<>();

        // 상위 Role은 하위 Role 권한을 포함
        switch (role) {
            case SUPER_ADMIN:
                authorities.add(new SimpleGrantedAuthority("ROLE_SUPER_ADMIN"));
                // fall through
            case ADMIN:
                authorities.add(new SimpleGrantedAuthority("ROLE_ADMIN"));
                // fall through
            case USER:
                authorities.add(new SimpleGrantedAuthority("ROLE_USER"));
                break;
        }

        return authorities;
    }

    @Override public String getPassword() { return null; }
    @Override public String getUsername() { return email; }
    @Override public boolean isAccountNonExpired() { return true; }
    @Override public boolean isAccountNonLocked() { return true; }
    @Override public boolean isCredentialsNonExpired() { return true; }
    @Override public boolean isEnabled() { return true; }
}
```

---

## 7. Session Mapping (세션 매핑)

### 7.1 Spring Session → AI Orchestrator Session

사용자가 로그인하면 Spring Boot에서 관리하는 세션과
AI Orchestrator의 세션이 매핑된다.

```
User Login
    │
    ▼
┌─────────────────────────────────────────────┐
│           Spring Boot                        │
│                                              │
│  JWT Token ──────> UserPrincipal             │
│      │                  │                    │
│      │         userId: 12345                 │
│      │                  │                    │
│      ▼                  ▼                    │
│  SessionMapping                              │
│  ┌──────────────────────────────────┐        │
│  │ userId: 12345                    │        │
│  │ aiSessionId: "uuid-abc-123"     │        │
│  │ createdAt: 2026-02-11T10:00:00  │        │
│  │ lastAccessedAt: 2026-02-11T...  │        │
│  └──────────────────────────────────┘        │
│      │                                       │
│      │  ?session_id=uuid-abc-123             │
│      ▼                                       │
│  AI Orchestrator Request                     │
└─────────────────────────────────────────────┘
```

### 7.2 User-Based Session Service

JWT 기반 Stateless 아키텍처에서는 Spring HttpSession 대신
**userId 기반**으로 AI 세션을 관리한다.

```java
@Service
public class UserAiSessionService {

    private final ConcurrentHashMap<Long, AiSessionInfo> userSessions
        = new ConcurrentHashMap<>();

    @Data @AllArgsConstructor
    static class AiSessionInfo {
        private String aiSessionId;
        private Instant createdAt;
        private Instant lastAccessedAt;
    }

    /**
     * 사용자의 AI Session ID를 반환한다.
     * Avatar 생성 등으로 세션이 필요한 시점에 생성된다.
     */
    public String getOrCreateAiSessionId(Long userId) {
        AiSessionInfo session = userSessions.computeIfAbsent(
            userId,
            key -> new AiSessionInfo(
                UUID.randomUUID().toString(),
                Instant.now(),
                Instant.now()
            )
        );
        session.setLastAccessedAt(Instant.now());
        return session.getAiSessionId();
    }

    /**
     * 사용자의 AI 세션을 초기화한다.
     * (새 아바타 생성 시작 시)
     */
    public String resetAiSession(Long userId) {
        AiSessionInfo newSession = new AiSessionInfo(
            UUID.randomUUID().toString(),
            Instant.now(),
            Instant.now()
        );
        userSessions.put(userId, newSession);
        return newSession.getAiSessionId();
    }

    /**
     * 만료된 세션 정리 (2시간 미접근)
     */
    @Scheduled(fixedRate = 600_000)
    public void cleanupExpired() {
        Instant cutoff = Instant.now().minus(Duration.ofHours(2));
        userSessions.entrySet().removeIf(
            entry -> entry.getValue().getLastAccessedAt().isBefore(cutoff));
    }
}
```

### 7.3 Controller에서 사용

```java
@RestController
@RequestMapping("/api/v6/avatar")
@RequiredArgsConstructor
public class AvatarController {

    private final UserAiSessionService userAiSessionService;
    private final AiOrchestratorProxyService proxyService;

    @PostMapping("/generate")
    public Mono<ResponseEntity<String>> generate(
            @RequestParam("video") MultipartFile video,
            @AuthenticationPrincipal UserPrincipal user
    ) {
        // JWT에서 추출된 userId로 AI 세션 매핑
        String aiSessionId = userAiSessionService
            .getOrCreateAiSessionId(user.getId());

        // ... proxy call with aiSessionId
    }
}
```

---

## 8. Security Configuration Summary (보안 설정 요약)

### 8.1 application.yml

```yaml
stylelens:
  jwt:
    secret: ${JWT_SECRET}                            # 환경 변수 필수
    access-token-expiry: 3600000                     # 1시간 (ms)
    refresh-token-expiry: 2592000000                 # 30일 (ms)

  auth:
    super-admin-email: aisamdasu1204@gmail.com        # 하드코딩
    admin-emails: ${ADMIN_EMAILS:}                    # 콤마 구분

  frontend:
    url: ${FRONTEND_URL:http://localhost:3000}

spring:
  security:
    oauth2:
      client:
        registration:
          google:
            client-id: ${GOOGLE_CLIENT_ID}
            client-secret: ${GOOGLE_CLIENT_SECRET}
            scope: openid, profile, email
            redirect-uri: "{baseUrl}/api/v6/auth/google/callback"
```

### 8.2 Environment Variables Checklist

| Variable | Required | Description | Example |
|---|---|---|---|
| `JWT_SECRET` | **Yes** | JWT 서명 키 (256-bit 이상) | `openssl rand -hex 64` |
| `GOOGLE_CLIENT_ID` | **Yes** | Google OAuth Client ID | `xxx.apps.googleusercontent.com` |
| `GOOGLE_CLIENT_SECRET` | **Yes** | Google OAuth Client Secret | `GOCSPX-xxx` |
| `FRONTEND_URL` | No | Frontend URL (기본: localhost:3000) | `https://stylelens.app` |
| `ADMIN_EMAILS` | No | Admin 이메일 목록 (콤마 구분) | `admin1@g.com,admin2@g.com` |

### 8.3 보안 체크리스트

| 항목 | 상태 | 설명 |
|---|---|---|
| JWT Secret 환경변수 분리 | Required | 코드에 하드코딩 금지 |
| HTTPS 강제 (Production) | Required | Cookie secure=true |
| CORS Origin 제한 | Required | 프로덕션 도메인만 허용 |
| CSRF 비활성화 | Done | JWT Stateless이므로 불필요 |
| Rate Limiting | Required | See API_Contract.md |
| AI Orchestrator 네트워크 격리 | Required | 외부 접근 차단 |
| Refresh Token DB 저장 | Required | 무효화 지원 |
| Google OAuth state 검증 | Done | CSRF 방어 |

---

## 9. Frontend Integration Guide (프론트엔드 연동 가이드)

### 9.1 로그인 흐름

```typescript
// 1. 로그인 시작 (새 창 또는 redirect)
window.location.href = '/api/v6/auth/google/login';

// 2. Callback 페이지에서 Token 수신
// /auth/callback?token=xxx&refresh_token=yyy
const params = new URLSearchParams(window.location.search);
const accessToken = params.get('token');
const refreshToken = params.get('refresh_token');

// 3. Local Storage에 저장
localStorage.setItem('access_token', accessToken);
localStorage.setItem('refresh_token', refreshToken);
```

### 9.2 API 요청

```typescript
// 모든 API 요청에 Bearer Token 포함
const api = axios.create({
    baseURL: '/api/v6',
    headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
    }
});

// 401 응답 시 자동 Token Refresh
api.interceptors.response.use(
    response => response,
    async error => {
        if (error.response?.status === 401) {
            const refreshToken = localStorage.getItem('refresh_token');
            const { data } = await axios.post('/api/v6/auth/refresh', {
                refreshToken
            });
            localStorage.setItem('access_token', data.accessToken);

            // 원래 요청 재시도
            error.config.headers.Authorization = `Bearer ${data.accessToken}`;
            return axios(error.config);
        }
        return Promise.reject(error);
    }
);
```

### 9.3 로그아웃

```typescript
async function logout() {
    await api.post('/api/v6/auth/logout');
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    window.location.href = '/';
}
```
