# StyleLens V6 File Storage Specification

> S3/MinIO 기반 파일 저장소 설계 문서
> Version: 1.0 | Last Updated: 2026-02-11

---

## 1. Storage Architecture Overview (저장소 아키텍처 개요)

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Frontend   │     │  Spring Boot     │     │  S3 / MinIO      │
│   (React)    │────>│  (Tier 2)        │────>│  Object Storage  │
│              │<────│                  │<────│                  │
└──────────────┘     └──────────────────┘     └──────────────────┘
       │                     │                        │
       │  CDN URL            │  S3 SDK                │  Bucket
       │  (static assets)    │  (upload/download)     │  구조
       │                     │                        │
       ▼                     ▼                        ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  CloudFront  │     │  AI Orchestrator │     │  Lifecycle Rules │
│  CDN         │     │  (Tier 3)        │     │  (TTL Cleanup)   │
└──────────────┘     └──────────────────┘     └──────────────────┘
```

**핵심 원칙:**
- 모든 사용자 파일은 **S3(Production) / MinIO(Development)**에 저장한다
- Spring Boot가 파일 저장/조회의 유일한 Gateway이다
- AI Orchestrator의 결과물(GLB, 이미지)은 Spring Boot를 통해 S3에 영구 저장한다
- 임시 파일은 TTL 기반 자동 삭제한다
- CDN을 통해 정적 자산을 서비스한다

---

## 2. Bucket Structure (버킷 구조)

### 2.1 Bucket 설계

단일 Bucket에 Prefix(폴더)로 구분한다.
환경별 Bucket을 분리한다.

```
stylelens-assets-{env}/           # env: dev, staging, prod
│
├── avatars/                       # GLB 아바타 모델
│   └── {userId}/
│       ├── {avatarId}/
│       │   ├── avatar.glb         # 최종 GLB 파일
│       │   ├── thumbnail.webp     # 아바타 썸네일 (256x256)
│       │   └── metadata.json      # 생성 파라미터
│       └── latest -> {avatarId}/  # 최신 아바타 심볼릭 참조
│
├── photos/                        # 사용자 업로드 사진
│   └── {userId}/
│       ├── face/                   # 얼굴 사진 (V5 Face Identity)
│       │   └── {uuid}.webp
│       └── body/                   # 전신 사진/비디오 (Avatar 생성용)
│           ├── {uuid}.mp4
│           └── {uuid}.webp
│
├── wardrobe/                      # 의류 이미지 라이브러리
│   └── {userId}/
│       └── {clothingId}/
│           ├── original/           # 원본 이미지 (최대 10장)
│           │   ├── {uuid}_01.webp
│           │   ├── {uuid}_02.webp
│           │   └── ...
│           ├── thumbnail.webp      # 의류 썸네일 (256x256)
│           └── metadata.json       # 의류 메타데이터 (색상, 카테고리 등)
│
├── fitting/                       # Try-On 결과 이미지
│   └── {userId}/
│       └── {fittingId}/
│           ├── angle_0.webp        # 정면
│           ├── angle_45.webp       # 45도
│           ├── angle_90.webp       # 90도
│           ├── angle_135.webp      # 135도
│           ├── angle_180.webp      # 후면
│           ├── angle_225.webp      # 225도
│           ├── angle_270.webp      # 270도
│           ├── angle_315.webp      # 315도
│           └── metadata.json       # fitting 파라미터 + 의류 정보
│
├── viewer3d/                      # 3D 의류 모델 (Hunyuan3D)
│   └── {userId}/
│       └── {modelId}/
│           ├── model.glb           # 3D 의류 GLB
│           ├── thumbnail.webp      # 모델 썸네일
│           └── metadata.json
│
└── temp/                          # 임시 파일 (TTL: 24시간)
    └── {sessionId}/
        ├── upload/                 # 업로드 중간 파일
        ├── processing/             # AI 처리 중간 결과
        └── cache/                  # 캐시 파일
```

### 2.2 Bucket Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SpringBootFullAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::123456789012:role/stylelens-backend"
            },
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::stylelens-assets-prod",
                "arn:aws:s3:::stylelens-assets-prod/*"
            ]
        },
        {
            "Sid": "CDNReadOnly",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::cloudfront:user/CloudFront Origin Access Identity ..."
            },
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::stylelens-assets-prod/*"
        }
    ]
}
```

---

## 3. GLB Model Storage (GLB 모델 저장)

### 3.1 Avatar GLB 저장 흐름

```
AI Orchestrator                   Spring Boot                    S3
     │                                │                           │
     │  POST /avatar/generate         │                           │
     │  (video + images)              │                           │
     │ ────────────────────>          │                           │
     │                                │                           │
     │  { "status": "complete",       │                           │
     │    "glb_base64": "..." }       │                           │
     │ <────────────────────          │                           │
     │                                │                           │
     │                                │  1. base64 → byte[]       │
     │                                │  2. Generate avatarId     │
     │                                │  3. Upload GLB            │
     │                                │ ────────────────────────> │
     │                                │  avatars/{userId}/{id}/   │
     │                                │  avatar.glb               │
     │                                │                           │
     │                                │  4. Generate thumbnail    │
     │                                │ ────────────────────────> │
     │                                │  thumbnail.webp           │
     │                                │                           │
     │                                │  5. Save metadata         │
     │                                │ ────────────────────────> │
     │                                │  metadata.json            │
     │                                │                           │
     │                                │  6. DB에 레코드 저장       │
     │                                │                           │
     │                                │  7. Return to Frontend    │
     │                                │  { avatarId, thumbnailUrl }│
```

### 3.2 Storage Service

```java
@Service
@RequiredArgsConstructor
public class FileStorageService {

    private final S3Client s3Client;

    @Value("${stylelens.storage.s3.bucket}")
    private String bucket;

    /**
     * Avatar GLB 저장
     * @return S3 Key (avatars/{userId}/{avatarId}/avatar.glb)
     */
    public String storeAvatarGlb(Long userId, String avatarId, byte[] glbData) {
        String key = buildKey("avatars", userId, avatarId, "avatar.glb");

        PutObjectRequest request = PutObjectRequest.builder()
            .bucket(bucket)
            .key(key)
            .contentType("model/gltf-binary")
            .contentLength((long) glbData.length)
            .build();

        s3Client.putObject(request,
            RequestBody.fromBytes(glbData));

        log.info("Stored avatar GLB: {} ({} bytes)", key, glbData.length);
        return key;
    }

    /**
     * GLB 파일 조회 (다운로드)
     */
    public byte[] getAvatarGlb(Long userId, String avatarId) {
        String key = buildKey("avatars", userId, avatarId, "avatar.glb");

        GetObjectRequest request = GetObjectRequest.builder()
            .bucket(bucket)
            .key(key)
            .build();

        try (ResponseInputStream<GetObjectResponse> response =
                s3Client.getObject(request)) {
            return response.readAllBytes();
        } catch (NoSuchKeyException e) {
            throw new FileNotFoundException("Avatar GLB not found: " + avatarId);
        } catch (IOException e) {
            throw new StorageException("Failed to read GLB: " + key, e);
        }
    }

    /**
     * Presigned URL 생성 (CDN 우회 직접 다운로드)
     * GLB는 대용량이므로 Presigned URL 사용 권장
     */
    public String generatePresignedUrl(String key, Duration expiry) {
        GetObjectPresignRequest presignRequest = GetObjectPresignRequest.builder()
            .signatureDuration(expiry)
            .getObjectRequest(GetObjectRequest.builder()
                .bucket(bucket)
                .key(key)
                .build())
            .build();

        return s3Presigner.presignGetObject(presignRequest).url().toString();
    }

    /**
     * S3 Key 빌더
     */
    private String buildKey(String prefix, Long userId, String entityId,
                            String filename) {
        return String.format("%s/%d/%s/%s", prefix, userId, entityId, filename);
    }
}
```

---

## 4. User Photo Storage (사용자 사진 저장)

### 4.1 업로드 흐름

사용자가 업로드하는 사진은 즉시 S3에 저장하고, UUID 기반 경로를 사용한다.
원본 파일명은 메타데이터에만 보관하고 저장 경로에는 사용하지 않는다 (보안).

```java
@Service
@RequiredArgsConstructor
public class PhotoStorageService {

    private final FileStorageService storageService;
    private final ImageProcessingService imageService;

    /**
     * 사용자 얼굴 사진 저장
     * - WebP로 변환 (용량 절감)
     * - 768x768 리사이즈 (AI Orchestrator 입력 크기)
     */
    public PhotoUploadResult storeFacePhoto(Long userId, MultipartFile file) {
        String photoId = UUID.randomUUID().toString();

        // 1. 이미지 검증
        validateImage(file);

        // 2. WebP 변환 + 리사이즈
        byte[] processed = imageService.processAndConvertToWebP(
            file.getBytes(), 768, 768);

        // 3. S3 업로드
        String key = String.format("photos/%d/face/%s.webp", userId, photoId);
        storageService.upload(key, processed, "image/webp");

        // 4. 메타데이터 저장
        PhotoMetadata metadata = PhotoMetadata.builder()
            .photoId(photoId)
            .userId(userId)
            .originalFilename(file.getOriginalFilename())
            .s3Key(key)
            .sizeBytes(processed.length)
            .width(768)
            .height(768)
            .uploadedAt(Instant.now())
            .build();

        return new PhotoUploadResult(photoId, key, metadata);
    }

    /**
     * 사용자 비디오 저장 (Avatar 생성용)
     * - 원본 그대로 저장 (AI Orchestrator가 프레임 추출)
     */
    public VideoUploadResult storeBodyVideo(Long userId, MultipartFile file) {
        String videoId = UUID.randomUUID().toString();

        validateVideo(file);

        String key = String.format("photos/%d/body/%s.mp4", userId, videoId);
        storageService.upload(key, file.getBytes(), file.getContentType());

        return new VideoUploadResult(videoId, key, file.getSize());
    }
}
```

### 4.2 UUID 기반 경로 규칙

| 파일 종류 | S3 Key Pattern | Content-Type |
|---|---|---|
| 얼굴 사진 | `photos/{userId}/face/{uuid}.webp` | `image/webp` |
| 전신 사진 | `photos/{userId}/body/{uuid}.webp` | `image/webp` |
| 전신 비디오 | `photos/{userId}/body/{uuid}.mp4` | `video/mp4` |

**중요**: 원본 파일명은 절대 S3 Key에 포함하지 않는다.
- 보안: 파일명에 개인정보가 포함될 수 있음 (예: `김태규_증명사진.jpg`)
- 충돌 방지: UUID로 고유성 보장
- URL 안전: 한글/특수문자 이슈 방지

---

## 5. Try-On Result Image Storage (피팅 결과 이미지 저장)

### 5.1 Base64 → S3 변환

AI Orchestrator는 Try-On 결과를 Base64로 반환한다.
Spring Boot에서 디코딩 후 S3에 저장한다.

```java
@Service
@RequiredArgsConstructor
public class FittingResultStorageService {

    private final FileStorageService storageService;

    /**
     * AI Orchestrator의 Fitting 결과를 S3에 저장
     *
     * @param result AI 응답 (각 앵글별 base64 이미지)
     * @return 저장된 이미지 URL 목록
     */
    public FittingStorageResult storeFittingResult(
            Long userId,
            String fittingId,
            FittingAiResponse result
    ) {
        List<FittingImageInfo> images = new ArrayList<>();

        for (FittingAngleResult angle : result.getAngles()) {
            // 1. Base64 디코딩
            byte[] imageBytes = Base64.getDecoder()
                .decode(angle.getImageBase64());

            // 2. S3 Key 생성
            String angleName = String.format("angle_%d", angle.getDegree());
            String key = String.format("fitting/%d/%s/%s.webp",
                userId, fittingId, angleName);

            // 3. S3 업로드
            storageService.upload(key, imageBytes, "image/webp");

            // 4. CDN URL 생성
            String cdnUrl = buildCdnUrl(key);

            images.add(FittingImageInfo.builder()
                .angle(angle.getDegree())
                .s3Key(key)
                .cdnUrl(cdnUrl)
                .sizeBytes(imageBytes.length)
                .build());
        }

        // 5. 메타데이터 저장
        FittingMetadata metadata = FittingMetadata.builder()
            .fittingId(fittingId)
            .userId(userId)
            .clothingId(result.getClothingId())
            .qualityMode(result.getQualityMode())
            .generationMethod(result.getGenerationMethod())
            .totalDurationMs(result.getTotalDurationMs())
            .createdAt(Instant.now())
            .build();

        String metadataKey = String.format("fitting/%d/%s/metadata.json",
            userId, fittingId);
        storageService.upload(metadataKey,
            objectMapper.writeValueAsBytes(metadata),
            "application/json");

        return new FittingStorageResult(fittingId, images, metadata);
    }
}
```

### 5.2 결과 이미지 형식

| Angle | Filename | 설명 |
|---|---|---|
| 0 | `angle_0.webp` | 정면 |
| 45 | `angle_45.webp` | 우측 45도 |
| 90 | `angle_90.webp` | 우측 90도 (측면) |
| 135 | `angle_135.webp` | 우측 135도 |
| 180 | `angle_180.webp` | 후면 |
| 225 | `angle_225.webp` | 좌측 135도 |
| 270 | `angle_270.webp` | 좌측 90도 (측면) |
| 315 | `angle_315.webp` | 좌측 45도 |

---

## 6. Clothing Image Library (의류 이미지 라이브러리)

### 6.1 사용자별 의류 저장

```java
@Service
@RequiredArgsConstructor
public class WardrobeStorageService {

    private final FileStorageService storageService;
    private final ImageProcessingService imageService;

    /**
     * 의류 이미지 다건 등록 (최대 10장)
     */
    public ClothingStorageResult storeClothingImages(
            Long userId,
            String clothingId,
            List<MultipartFile> images
    ) {
        List<String> uploadedKeys = new ArrayList<>();

        for (int i = 0; i < images.size(); i++) {
            MultipartFile image = images.get(i);
            String uuid = UUID.randomUUID().toString().substring(0, 8);

            // WebP 변환 (원본 비율 유지, 최대 1024px)
            byte[] processed = imageService.processAndConvertToWebP(
                image.getBytes(), 1024, 1024);

            String key = String.format("wardrobe/%d/%s/original/%s_%02d.webp",
                userId, clothingId, uuid, i + 1);

            storageService.upload(key, processed, "image/webp");
            uploadedKeys.add(key);
        }

        // 썸네일 생성 (첫 번째 이미지 기반)
        byte[] thumbnail = imageService.processAndConvertToWebP(
            images.get(0).getBytes(), 256, 256);
        String thumbnailKey = String.format("wardrobe/%d/%s/thumbnail.webp",
            userId, clothingId);
        storageService.upload(thumbnailKey, thumbnail, "image/webp");

        return new ClothingStorageResult(clothingId, uploadedKeys, thumbnailKey);
    }

    /**
     * 사용자의 전체 Wardrobe 목록 조회
     */
    public List<ClothingSummary> listUserWardrobe(Long userId) {
        String prefix = String.format("wardrobe/%d/", userId);

        ListObjectsV2Request request = ListObjectsV2Request.builder()
            .bucket(bucket)
            .prefix(prefix)
            .delimiter("/")
            .build();

        ListObjectsV2Response response = s3Client.listObjectsV2(request);

        return response.commonPrefixes().stream()
            .map(p -> {
                String clothingId = extractClothingId(p.prefix());
                String thumbnailUrl = buildCdnUrl(
                    prefix + clothingId + "/thumbnail.webp");
                return new ClothingSummary(clothingId, thumbnailUrl);
            })
            .toList();
    }

    /**
     * 의류 삭제
     */
    public void deleteClothing(Long userId, String clothingId) {
        String prefix = String.format("wardrobe/%d/%s/", userId, clothingId);

        // 해당 prefix 하위 모든 파일 삭제
        ListObjectsV2Response objects = s3Client.listObjectsV2(
            ListObjectsV2Request.builder()
                .bucket(bucket)
                .prefix(prefix)
                .build());

        List<ObjectIdentifier> toDelete = objects.contents().stream()
            .map(obj -> ObjectIdentifier.builder().key(obj.key()).build())
            .toList();

        if (!toDelete.isEmpty()) {
            s3Client.deleteObjects(DeleteObjectsRequest.builder()
                .bucket(bucket)
                .delete(Delete.builder().objects(toDelete).build())
                .build());
        }

        log.info("Deleted clothing {}/{}: {} files",
            userId, clothingId, toDelete.size());
    }
}
```

### 6.2 의류 메타데이터

```json
{
    "clothingId": "clth_abc123",
    "userId": 12345,
    "category": "top",
    "subCategory": "t-shirt",
    "color": {
        "primary": "#2E4057",
        "secondary": "#FFFFFF",
        "hex": "#2E4057"
    },
    "brand": "Nike",
    "name": "Dri-FIT T-Shirt",
    "imageCount": 3,
    "viewClassification": {
        "front": "clth_abc123_01.webp",
        "back": "clth_abc123_02.webp",
        "detail": "clth_abc123_03.webp"
    },
    "extractedInfo": {
        "pattern": "solid",
        "material": "polyester",
        "fit": "regular",
        "features": ["round neck", "short sleeve", "logo print"]
    },
    "createdAt": "2026-02-11T10:30:00Z",
    "updatedAt": "2026-02-11T10:30:00Z"
}
```

---

## 7. Temporary File Cleanup (임시 파일 정리)

### 7.1 TTL 기반 정리 전략

| 파일 종류 | S3 Prefix | TTL | 정리 방법 |
|---|---|---|---|
| 업로드 임시 파일 | `temp/{sessionId}/upload/` | **2시간** | S3 Lifecycle Rule |
| AI 처리 중간 결과 | `temp/{sessionId}/processing/` | **24시간** | S3 Lifecycle Rule |
| 캐시 파일 | `temp/{sessionId}/cache/` | **24시간** | S3 Lifecycle Rule |
| 만료된 Fitting 결과 | `fitting/{userId}/{fittingId}/` | **90일** | Scheduled Job |
| 삭제된 사용자 데이터 | 전체 사용자 prefix | **30일 유예** | Manual / Scheduled |

### 7.2 S3 Lifecycle Rules

```xml
<!-- S3 Lifecycle Configuration -->
<LifecycleConfiguration>
    <!-- 임시 파일: 1일 후 자동 삭제 -->
    <Rule>
        <ID>cleanup-temp-files</ID>
        <Filter>
            <Prefix>temp/</Prefix>
        </Filter>
        <Status>Enabled</Status>
        <Expiration>
            <Days>1</Days>
        </Expiration>
    </Rule>

    <!-- 오래된 Fitting 결과: 90일 후 Glacier로 이동 -->
    <Rule>
        <ID>archive-old-fittings</ID>
        <Filter>
            <Prefix>fitting/</Prefix>
        </Filter>
        <Status>Enabled</Status>
        <Transition>
            <Days>90</Days>
            <StorageClass>GLACIER</StorageClass>
        </Transition>
        <Expiration>
            <Days>365</Days>
        </Expiration>
    </Rule>

    <!-- 멀티파트 업로드 미완료: 1일 후 정리 -->
    <Rule>
        <ID>cleanup-incomplete-uploads</ID>
        <Filter>
            <Prefix></Prefix>
        </Filter>
        <Status>Enabled</Status>
        <AbortIncompleteMultipartUpload>
            <DaysAfterInitiation>1</DaysAfterInitiation>
        </AbortIncompleteMultipartUpload>
    </Rule>
</LifecycleConfiguration>
```

### 7.3 Scheduled Cleanup Job

```java
@Component
@RequiredArgsConstructor
public class StorageCleanupJob {

    private final S3Client s3Client;
    private final UserRepository userRepository;
    private final FittingRepository fittingRepository;

    @Value("${stylelens.storage.s3.bucket}")
    private String bucket;

    /**
     * 매일 새벽 3시 실행: 만료된 데이터 정리
     */
    @Scheduled(cron = "0 0 3 * * *")
    public void cleanupExpiredData() {
        log.info("[CLEANUP] Starting daily storage cleanup");

        int tempCleaned = cleanupTempFiles();
        int orphanCleaned = cleanupOrphanFiles();

        log.info("[CLEANUP] Completed: temp={}, orphan={}",
            tempCleaned, orphanCleaned);
    }

    /**
     * 24시간 이상 된 임시 파일 삭제
     * (S3 Lifecycle과 중복 보호)
     */
    private int cleanupTempFiles() {
        Instant cutoff = Instant.now().minus(Duration.ofHours(24));
        // S3 Lifecycle이 주로 처리하지만, 누락 방지용
        return deleteObjectsOlderThan("temp/", cutoff);
    }

    /**
     * DB에 레코드가 없는 S3 파일 삭제 (고아 파일)
     * 업로드 후 DB 저장 실패 시 발생
     */
    private int cleanupOrphanFiles() {
        // 구현: S3 목록과 DB 레코드 비교
        // 주의: 대량 데이터 시 배치 처리 필요
        return 0;
    }

    private int deleteObjectsOlderThan(String prefix, Instant cutoff) {
        // ... S3 ListObjects + DeleteObjects
        return 0;
    }
}
```

---

## 8. CDN Integration (CDN 연동)

### 8.1 CloudFront 설정

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ Frontend │────>│ CloudFront   │────>│ S3 Bucket    │
│          │     │ CDN          │     │              │
│          │     │ d111.cf.net  │     │ stylelens-*  │
└──────────┘     └──────────────┘     └──────────────┘
                       │
                 Cache Policy:
                 - avatars/*:    max-age=86400 (1일)
                 - wardrobe/*:   max-age=86400 (1일)
                 - fitting/*:    max-age=604800 (7일)
                 - viewer3d/*:   max-age=604800 (7일)
                 - temp/*:       no-cache
```

### 8.2 CDN URL 빌더

```java
@Service
public class CdnUrlService {

    @Value("${stylelens.cdn.base-url:}")
    private String cdnBaseUrl;

    @Value("${stylelens.storage.s3.bucket}")
    private String bucket;

    @Value("${stylelens.cdn.enabled:false}")
    private boolean cdnEnabled;

    /**
     * CDN URL 생성
     * CDN이 비활성화된 경우 Spring Boot API URL 반환
     */
    public String buildUrl(String s3Key) {
        if (cdnEnabled && !cdnBaseUrl.isBlank()) {
            return cdnBaseUrl + "/" + s3Key;
        }
        // CDN 미사용 시 Spring Boot API 경유
        return "/api/v6/storage/files/" + s3Key;
    }

    /**
     * Signed CDN URL 생성 (비공개 콘텐츠)
     * 사용자별 Fitting 결과 등 접근 제한이 필요한 콘텐츠
     */
    public String buildSignedUrl(String s3Key, Duration expiry) {
        if (cdnEnabled) {
            return generateCloudFrontSignedUrl(s3Key, expiry);
        }
        return buildPresignedS3Url(s3Key, expiry);
    }

    private String generateCloudFrontSignedUrl(String s3Key, Duration expiry) {
        // CloudFront Signed URL 생성
        // AWS SDK CloudFront Signer 사용
        Date expirationDate = Date.from(Instant.now().plus(expiry));
        // ... CloudFront 서명 로직
        return cdnBaseUrl + "/" + s3Key + "?Expires=...&Signature=...&Key-Pair-Id=...";
    }

    private String buildPresignedS3Url(String s3Key, Duration expiry) {
        // S3 Presigned URL (CDN 미사용 시 대안)
        // ... S3 Presigner 로직
        return "";
    }
}
```

### 8.3 Cache 정책

| Content Type | Cache-Control | CDN TTL | 이유 |
|---|---|---|---|
| Avatar GLB | `max-age=86400, private` | 1일 | 재생성 가능, 큰 파일 |
| Avatar Thumbnail | `max-age=86400, public` | 1일 | 작은 이미지 |
| Wardrobe Images | `max-age=86400, private` | 1일 | 사용자 콘텐츠 |
| Fitting Results | `max-age=604800, private` | 7일 | 변경 없음 |
| 3D Models | `max-age=604800, private` | 7일 | 변경 없음 |
| Temp Files | `no-cache, no-store` | 0 | 임시 |

**주의**: 모든 사용자 콘텐츠는 `private`을 사용한다. `public` 캐시는 썸네일 등
공유 가능한 콘텐츠에만 적용한다.

---

## 9. File Naming Conventions (파일 명명 규칙)

### 9.1 ID 생성 규칙

| Entity | ID Format | Example | 생성 방식 |
|---|---|---|---|
| Avatar | `avt_{nanoid(12)}` | `avt_V1StGXR8_Z5j` | NanoID |
| Clothing | `clth_{nanoid(12)}` | `clth_2nYpBq0s_mKx` | NanoID |
| Fitting | `fit_{nanoid(12)}` | `fit_xH3k9mL_pQr` | NanoID |
| 3D Model | `mdl_{nanoid(12)}` | `mdl_7Yw2Fc_sT4n` | NanoID |
| Photo | `{uuid}` | `a1b2c3d4-e5f6-...` | UUID v4 |

### 9.2 파일명 규칙

```
Rule 1: 원본 파일명 사용 금지
  Bad:  wardrobe/123/김태규_나이키티셔츠.jpg
  Good: wardrobe/123/clth_abc/original/f8e2a1b3_01.webp

Rule 2: UUID 또는 시퀀스 번호 사용
  photos/{userId}/face/{uuid}.webp
  wardrobe/{userId}/{clothingId}/original/{uuid}_{seq}.webp

Rule 3: 확장자는 항상 실제 포맷과 일치
  WebP 파일은 .webp, GLB 파일은 .glb

Rule 4: 메타데이터는 항상 metadata.json
  각 엔티티 폴더에 metadata.json 포함

Rule 5: 소문자만 사용
  Bad:  Avatars/User123/
  Good: avatars/123/
```

### 9.3 이미지 포맷 통일

| 용도 | 저장 포맷 | 품질 | 최대 해상도 | 이유 |
|---|---|---|---|---|
| 얼굴 사진 | WebP | 90% | 768x768 | AI 입력 최적 크기 |
| 의류 원본 | WebP | 85% | 1024x1024 | 품질/용량 균형 |
| 썸네일 | WebP | 80% | 256x256 | 목록 표시용 |
| Fitting 결과 | WebP | 90% | 1024x1024 | 고품질 결과물 |
| 비디오 | MP4 (원본) | - | 원본 유지 | AI가 프레임 추출 |

---

## 10. Storage Quotas (사용자별 저장 용량)

### 10.1 Quota 정책

| Plan | Total Quota | Avatar 수 | Wardrobe 수 | Fitting 결과 | 3D 모델 |
|---|---|---|---|---|---|
| Free | **1 GB** | 1개 | 20벌 | 최근 10건 | 1개 |
| Standard | **5 GB** | 3개 | 100벌 | 최근 50건 | 5개 |
| Premium | **20 GB** | 10개 | 무제한 | 무제한 | 20개 |

### 10.2 Quota Service

```java
@Service
@RequiredArgsConstructor
public class StorageQuotaService {

    private final UserRepository userRepository;
    private final FileStorageService storageService;

    /**
     * 파일 저장 전 Quota 확인
     * @throws StorageQuotaExceededException Quota 초과 시
     */
    public void checkQuota(Long userId, long additionalBytes) {
        User user = userRepository.findById(userId).orElseThrow();

        long currentUsage = user.getStorageUsedBytes();
        long quota = user.getStorageQuotaBytes();

        if (currentUsage + additionalBytes > quota) {
            long remainingBytes = quota - currentUsage;

            throw new StorageQuotaExceededException(
                String.format(
                    "저장 공간이 부족합니다. 현재 사용: %s / %s, 필요: %s",
                    formatBytes(currentUsage),
                    formatBytes(quota),
                    formatBytes(additionalBytes)
                ),
                currentUsage,
                quota,
                remainingBytes
            );
        }
    }

    /**
     * 파일 저장 후 사용량 업데이트
     */
    @Transactional
    public void addUsage(Long userId, long bytes) {
        userRepository.incrementStorageUsed(userId, bytes);
    }

    /**
     * 파일 삭제 후 사용량 업데이트
     */
    @Transactional
    public void reduceUsage(Long userId, long bytes) {
        userRepository.decrementStorageUsed(userId, bytes);
    }

    /**
     * 사용자 저장소 사용 현황 조회
     */
    public StorageUsageInfo getUsageInfo(Long userId) {
        User user = userRepository.findById(userId).orElseThrow();

        // S3에서 실제 사용량 계산 (정확도 높음, 비용 높음)
        // 일반적으로는 DB의 storageUsedBytes 사용
        long dbUsage = user.getStorageUsedBytes();
        long quota = user.getStorageQuotaBytes();

        return StorageUsageInfo.builder()
            .userId(userId)
            .usedBytes(dbUsage)
            .quotaBytes(quota)
            .usedPercent((double) dbUsage / quota * 100)
            .remainingBytes(quota - dbUsage)
            .build();
    }

    /**
     * S3 실제 사용량과 DB 동기화 (매일 새벽 4시)
     */
    @Scheduled(cron = "0 0 4 * * *")
    public void syncStorageUsage() {
        List<User> allUsers = userRepository.findAll();

        for (User user : allUsers) {
            long actualUsage = storageService.calculateUserStorageSize(user.getId());
            if (Math.abs(actualUsage - user.getStorageUsedBytes()) > 1024 * 1024) {
                log.warn("[QUOTA-SYNC] User {} drift: DB={}, S3={}",
                    user.getId(),
                    formatBytes(user.getStorageUsedBytes()),
                    formatBytes(actualUsage));
                user.setStorageUsedBytes(actualUsage);
                userRepository.save(user);
            }
        }
    }

    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024) + " KB";
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)) + " MB";
        return String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024));
    }
}
```

### 10.3 Quota 초과 응답

```json
{
    "error": {
        "code": "STORAGE_QUOTA_EXCEEDED",
        "message": "저장 공간이 부족합니다.",
        "detail": {
            "currentUsage": "4.2 GB",
            "quota": "5.0 GB",
            "remaining": "0.8 GB",
            "required": "1.2 GB"
        },
        "suggestion": "불필요한 Fitting 결과나 의류를 삭제하여 공간을 확보해주세요."
    }
}
```

---

## 11. S3/MinIO Configuration (저장소 설정)

### 11.1 application.yml

```yaml
stylelens:
  storage:
    type: ${STORAGE_TYPE:local}       # local | s3 | minio

    # AWS S3 (Production)
    s3:
      bucket: ${S3_BUCKET:stylelens-assets-prod}
      region: ${AWS_REGION:ap-northeast-2}
      # AWS SDK가 자동으로 AWS credentials 탐색
      # (환경변수, ~/.aws/credentials, IAM Role 등)

    # MinIO (Development)
    minio:
      endpoint: ${MINIO_ENDPOINT:http://localhost:9000}
      access-key: ${MINIO_ACCESS_KEY:minioadmin}
      secret-key: ${MINIO_SECRET_KEY:minioadmin}
      bucket: ${MINIO_BUCKET:stylelens-assets-dev}

    # Local (테스트용)
    local:
      base-path: ${LOCAL_STORAGE_PATH:/data/stylelens}

  cdn:
    enabled: ${CDN_ENABLED:false}
    base-url: ${CDN_BASE_URL:}        # https://d111.cloudfront.net

  quota:
    default-bytes: 5368709120           # 5 GB (기본)
    free-bytes: 1073741824              # 1 GB (무료)
    premium-bytes: 21474836480          # 20 GB (프리미엄)
```

### 11.2 S3 Client Configuration

```java
@Configuration
public class StorageConfig {

    @Bean
    @ConditionalOnProperty(name = "stylelens.storage.type", havingValue = "s3")
    public S3Client s3Client(
            @Value("${stylelens.storage.s3.region}") String region
    ) {
        return S3Client.builder()
            .region(Region.of(region))
            .build();
    }

    @Bean
    @ConditionalOnProperty(name = "stylelens.storage.type", havingValue = "minio")
    public S3Client minioClient(
            @Value("${stylelens.storage.minio.endpoint}") String endpoint,
            @Value("${stylelens.storage.minio.access-key}") String accessKey,
            @Value("${stylelens.storage.minio.secret-key}") String secretKey
    ) {
        return S3Client.builder()
            .endpointOverride(URI.create(endpoint))
            .region(Region.AP_NORTHEAST_2)  // MinIO는 region 무관
            .credentialsProvider(StaticCredentialsProvider.create(
                AwsBasicCredentials.create(accessKey, secretKey)))
            .forcePathStyle(true)           // MinIO는 Path Style 필수
            .build();
    }

    @Bean
    @ConditionalOnProperty(name = "stylelens.storage.type", havingValue = "s3")
    public S3Presigner s3Presigner(
            @Value("${stylelens.storage.s3.region}") String region
    ) {
        return S3Presigner.builder()
            .region(Region.of(region))
            .build();
    }
}
```

### 11.3 Docker Compose (MinIO 개발환경)

```yaml
# docker-compose.dev.yml
services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"       # S3 API
      - "9001:9001"       # Console UI
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

  # MinIO 초기 버킷 생성
  minio-init:
    image: minio/mc:latest
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
        mc alias set local http://minio:9000 minioadmin minioadmin;
        mc mb local/stylelens-assets-dev --ignore-existing;
        mc anonymous set none local/stylelens-assets-dev;
        echo 'MinIO bucket created';
      "

volumes:
  minio_data:
```

---

## 12. Storage API Endpoints (저장소 API)

Spring Boot에서 제공하는 파일 관련 API 목록:

```java
@RestController
@RequestMapping("/api/v6/storage")
@RequiredArgsConstructor
public class StorageController {

    /**
     * 사용자 저장소 사용 현황
     */
    @GetMapping("/usage")
    public ResponseEntity<StorageUsageInfo> getUsage(
            @AuthenticationPrincipal UserPrincipal user
    ) { ... }

    /**
     * 파일 직접 다운로드 (CDN 미사용 시 대안)
     */
    @GetMapping("/files/{*key}")
    public ResponseEntity<byte[]> getFile(
            @PathVariable String key,
            @AuthenticationPrincipal UserPrincipal user
    ) {
        // 사용자 소유 파일만 접근 가능
        validateOwnership(user.getId(), key);
        // ...
    }

    /**
     * 파일 삭제
     */
    @DeleteMapping("/files/{*key}")
    public ResponseEntity<Void> deleteFile(
            @PathVariable String key,
            @AuthenticationPrincipal UserPrincipal user
    ) {
        validateOwnership(user.getId(), key);
        // ...
    }
}
```

---

## Appendix A: Storage Size Estimates (용량 추정)

### 사용자 1명 기준 예상 저장 용량

| 항목 | 파일 크기 | 수량 | 합계 |
|---|---|---|---|
| Avatar GLB | ~5 MB | 1 | 5 MB |
| Avatar Thumbnail | ~20 KB | 1 | 20 KB |
| Face Photo | ~200 KB | 1 | 200 KB |
| Body Video | ~30 MB | 1 | 30 MB |
| Wardrobe (per item) | ~500 KB x 3장 | 20벌 | 30 MB |
| Fitting Result (per set) | ~300 KB x 8장 | 10건 | 24 MB |
| 3D Model | ~10 MB | 1 | 10 MB |
| Metadata JSON | ~2 KB | 30+ | 60 KB |
| **합계** | | | **~100 MB** |

### 서비스 규모별 예상

| 규모 | 사용자 수 | 총 저장 용량 | 월 S3 비용 (ap-northeast-2) |
|---|---|---|---|
| 초기 | 100명 | ~10 GB | ~$0.25 |
| 성장기 | 1,000명 | ~100 GB | ~$2.50 |
| 성숙기 | 10,000명 | ~1 TB | ~$25.00 |
| 대규모 | 100,000명 | ~10 TB | ~$250.00 |

*S3 Standard: $0.025/GB/month (ap-northeast-2)*
*CDN 전송 비용 별도*
