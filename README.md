# StyleLens AI Server -- 360도 가상 피팅 + 3D 바디 복원 시스템

사용자 사진 한 장에서 3D 바디 메쉬를 복원하고, 8개 각도의 가상 피팅 이미지를 생성하는 AI 파이프라인이다.
4-Tier 분산 아키텍처로 설계되어 있으며, GPU 연산은 Modal 서버리스 인프라에 위임한다.


## 아키텍처

```
Frontend (React 19 + Three.js)
    |
Backend (Spring Boot, JWT/S3)
    |
Orchestrator (FastAPI, CPU)        <-- 이 저장소
    |
GPU Worker (Modal, H200 141GB)
```

Orchestrator는 전체 파이프라인을 조율하는 중앙 컨트롤러 역할을 한다.
체형 분석, 메쉬 렌더링 등 CPU 작업은 직접 처리하고, SDXL/VTON/Face Swap 등 GPU 작업은 Modal Worker에 위임한다.


## 파이프라인 흐름

```mermaid
graph LR
    A[사용자 사진] --> B[SAM 3D Body]
    B --> C[CPU 메쉬 렌더링]
    C --> D[SDXL + ControlNet Depth]
    D --> E[FLUX.2-klein 리파인]
    E --> F[FASHN VTON v1.5]
    F --> G[InsightFace Face Swap]
    G --> H[8각도 피팅 결과]
```

| Phase | 설명 | 실행 위치 |
|-------|------|----------|
| Phase 1 | SAM 3D Body로 사진에서 3D 메쉬 복원 | GPU Worker |
| Phase 1R | CPU 소프트웨어 렌더러로 8각도 depth map 생성 | Orchestrator |
| Phase 1.5A | SDXL + ControlNet Depth로 사실적 인체 이미지 생성 | GPU Worker |
| Phase 1.5B | FLUX.2-klein-4B img2img 텍스처 리파인 (선택) | GPU Worker |
| Phase 3 | FASHN VTON v1.5로 의류 가상 착용 | GPU Worker |
| Phase 4 | InsightFace antelopev2로 얼굴 일관성 유지 | GPU Worker |
| Phase 5 | P2P(Physics-to-Prompt) 핏 분석 | Orchestrator |


## 주요 기능

- **360도 가상 피팅**: 8개 각도(0~315도, 45도 간격)에서 의류 착용 이미지를 생성한다.
- **체형 반영 메쉬 변형**: 컵 사이즈별 가슴 볼륨 조절, 자세 보정, 팔 접기 등을 적용한다.
- **동적 ControlNet 스케일**: 체형에 따라 cn_scale을 자동 조정하여 VTON 단계에서 실루엣을 보존한다.
- **얼굴 일관성 유지**: InsightFace 기반 face swap으로 모든 각도에서 동일한 얼굴을 유지한다.
- **P2P 핏 분석**: 체형-의류 치수 차이를 물리 기반으로 분석하여 핏 프롬프트를 생성한다.
- **FLUX 리파인 선택적 적용**: config 플래그로 FLUX 단계를 on/off 전환하여 비용을 제어한다.


## 기술 스택

| 분류 | 기술 | 용도 |
|------|------|------|
| Orchestrator | FastAPI + Uvicorn | 비동기 API 서버 |
| GPU Worker | Modal Serverless (H200) | AI 모델 실행 |
| 3D 복원 | SAM 3D Body DINOv3 | 단일 사진 3D 메쉬 복원 |
| 이미지 생성 | SDXL + ControlNet Depth | depth 기반 사실적 인체 생성 |
| 텍스처 리파인 | FLUX.2-klein-4B | img2img 질감 개선 (선택) |
| 가상 착용 | FASHN VTON v1.5 | maskless 의류 합성 |
| 얼굴 보존 | InsightFace antelopev2 | 각도별 face swap |
| 메쉬 렌더링 | NumPy CPU Renderer | 소프트웨어 기반 depth map 생성 |
| 핏 분석 | P2P Engine | 물리 기반 핏 프롬프트 변환 |


## 실행 방법

```bash
# 1. 환경 설정
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 환경 변수
cp .env.example .env
# .env.example 참고하여 API 키 설정

# 3. Modal 인증 (GPU Worker 사용 시)
pip install modal
modal setup

# 4. Orchestrator 실행
python -m orchestrator.main
# http://localhost:8000/docs 에서 API 확인
```


## 디렉토리 구조

```
ai-service/
├── orchestrator/               # Tier 3: FastAPI Orchestrator
│   ├── main.py                 #   앱 진입점 + lifespan
│   ├── config.py               #   환경 기반 설정
│   ├── session.py              #   파이프라인 세션 관리
│   ├── worker_client.py        #   Modal GPU Worker RPC 클라이언트
│   ├── serialization.py        #   numpy/image base64 코덱
│   └── routes/
│       ├── avatar.py           #   Phase 1: 아바타 생성
│       ├── wardrobe.py         #   Phase 2: 의류 분석
│       ├── fitting.py          #   Phase 3: 가상 피팅 (5-phase)
│       ├── viewer3d.py         #   Phase 4: 3D GLB 생성
│       ├── visualization.py    #   렌더 결과 시각화
│       ├── face_bank.py        #   얼굴 레퍼런스 관리
│       ├── p2p.py              #   핏 분석 API
│       └── quality.py          #   품질 게이트
├── worker/                     # Tier 4: Modal GPU Worker
│   ├── modal_app.py            #   H200 서버리스 GPU 함수
│   └── _upload_*.py            #   모델 볼륨 업로드 유틸리티
├── core/                       # AI 핵심 로직
│   ├── config.py               #   모델 경로, 파이프라인 플래그
│   ├── body_analyzer.py        #   체형 분석 + 컵 사이즈 변환
│   ├── sw_renderer.py          #   CPU 소프트웨어 메쉬 렌더러
│   ├── fitting.py              #   로컬 피팅 파이프라인
│   ├── pipeline.py             #   아바타 생성 파이프라인
│   ├── face_bank.py            #   다중 참조 얼굴 관리
│   ├── face_identity.py        #   InsightFace 연동
│   ├── p2p_engine.py           #   Physics-to-Prompt 엔진
│   ├── p2p_ensemble.py         #   다중 전략 P2P 앙상블
│   └── ...
├── scripts/                    # 관리 스크립트
├── tests/                      # 테스트 코드
└── requirements.txt
```
