# StyleLens V6: Distributed 4-Tier AI Architecture for 360° High-Fidelity Virtual Try-On

**StyleLens V6** is a distributed AI pipeline that combines **7 SOTA deep learning models** with **physics-based fit analysis** to deliver photorealistic 360° virtual try-on with full 3D body reconstruction.

---

## Technical Complexity: 4-Tier Architecture

I have designed and implemented a **Distributed 4-Tier Architecture** to manage the complex computational demands of real-time 3D avatar generation.

```
┌─────────────────────────┬────────────────────────────────────────┐
│  Tier 1 (Frontend)      │  React 19 + Three.js                  │
│                         │  Interactive 3D GLB visualization      │
├─────────────────────────┼────────────────────────────────────────┤
│  Tier 2 (Backend)       │  Spring Boot                          │
│                         │  JWT auth + S3 data management         │
├─────────────────────────┼────────────────────────────────────────┤
│  Tier 3 (Orchestrator)  │  FastAPI AI Brain                     │
│                         │  P2P physics analysis + scheduling     │
├─────────────────────────┼────────────────────────────────────────┤
│  Tier 4 (Worker)        │  Modal GPU (NVIDIA H100 80GB)         │
│                         │  Serverless SOTA model pipeline        │
└─────────────────────────┴────────────────────────────────────────┘
```

- **Tier 1 (Frontend):** React 19 and Three.js for interactive 3D GLB visualization.
- **Tier 2 (Backend):** Spring Boot for secure user authentication and S3-based data management.
- **Tier 3 (Orchestrator):** A FastAPI-based AI brain that performs P2P (Physics-to-Prompt) physical analysis and handles worker scheduling.
- **Tier 4 (Worker):** Serverless GPU nodes on Modal (NVIDIA H100) executing a sequential pipeline of 7 SOTA models.

---

## Resource Requirements: 55GB VRAM

The StyleLens V6 pipeline is uniquely resource-intensive, requiring a cumulative **55GB of VRAM** for full-stack model residency.

| Stack | Models | VRAM |
|:------|:-------|-----:|
| **Vision Stack** | YOLO26-L (detection) + SAM 3 (segmentation) + SAM 3D Body (skeletal reconstruction) | ~12GB |
| **Generation Stack** | FLUX.1-dev GGUF Q8 (base diffusion) + CatVTON-FLUX (garment transfer) | ~21GB |
| **Identity & 3D Stack** | InsightFace (facial preservation) + Hunyuan3D 2.0 (PBR texture synthesis) | ~22GB |
| **Total** | **7 Models** | **~55GB** |

Due to this 55GB VRAM requirement, high-memory GPU nodes like the **NVIDIA H100 (80GB)** are essential to avoid memory fragmentation and latency during model swapping. The Tier 4 worker runs on **Modal serverless GPU infrastructure** with H100 nodes.

---

## Research Innovation: Multi-Reference Identity Bank

My research focuses on **"Zero-shot Identity Preservation"** across extreme 360-degree rotations. Unlike standard single-image methods, I am developing a **Multi-Reference Face ID Bank** that utilizes **11+ images** (10 historical + 1 current) to maintain 100% ID consistency in 3D reconstruction. This approach solves the **"identity drift"** problem prevalent in current 3D generative AI.

**Core implementation** (`core/face_bank.py`):
- Manages up to 11 reference images with InsightFace embedding extraction
- Classifies face angles and selects optimal references per target angle
- Cosine similarity-based identity verification across all 8 viewing angles (0° ~ 315°)

---

## 4-Phase Pipeline

### Phase 1 — Avatar Generation
> Person detection + 3D body reconstruction

- **YOLO26-L**: NMS-free single-shot person detection
- **SAM 3D Body DINOv3**: Single-image to 3D mesh (SMPL vertices, joints, betas)
- **Gemini Feedback Inspector**: Quality gate validation per stage

### Phase 2 — Wardrobe Analysis
> Clothing segmentation + physics extraction

- **SAM 3**: Concept-aware clothing segmentation
- **FASHN Parser**: 18-class fashion body parsing
- **Gemini Vision**: Size chart OCR + fabric physics inference (elasticity, thickness, texture)

### Phase 3 — Virtual Try-On
> Physics-aware 8-angle fitting

- **P2P Engine**: Converts physical deltas (garment size - body size) into visual cue keywords
- **P2P Ensemble**: Multi-strategy fusion with configurable timeout
- **CatVTON-FLUX**: FLUX-based 8-angle virtual try-on with physics prompt injection
- **Face Bank**: Multi-reference identity preservation across all angles

### Phase 4 — 3D Visualization
> Textured 3D model generation

- **Hunyuan3D 2.0**: Multi-view to textured 3D GLB with PBR materials
- **Software Renderer**: CPU/MPS fallback for mesh preview

---

## Tech Stack

| Category | Technology | Role |
|:---------|:-----------|:-----|
| **Orchestrator** | FastAPI 0.115+ / Uvicorn | Async API server (Tier 3) |
| **GPU Worker** | Modal Serverless (H100 80GB) | SOTA model execution (Tier 4) |
| **Detection** | YOLO26-L | NMS-free person detection |
| **Segmentation** | SAM 3 + FASHN Parser | Concept-aware seg + 18-class parsing |
| **3D Body** | SAM 3D Body DINOv3 | Single-image 3D mesh reconstruction |
| **Try-On** | CatVTON-FLUX + FLUX.1-dev GGUF | High-fidelity virtual garment transfer |
| **3D Generation** | Hunyuan3D 2.0 | PBR-textured 3D GLB export |
| **Face Identity** | InsightFace buffalo_l | Multi-reference face embedding |
| **AI Brain** | Google Gemini | Quality gates + clothing analysis + P2P |
| **Physics** | P2P Engine / Ensemble | Measurement-based fit prompt injection |

---

## Project Structure

```
ai-service/
├── orchestrator/                    # Tier 3: FastAPI AI Orchestrator
│   ├── main.py                      #   App entrypoint + lifespan
│   ├── config.py                    #   Environment-based configuration
│   ├── session.py                   #   Pipeline session management
│   ├── worker_client.py             #   Modal GPU worker RPC client
│   ├── serialization.py             #   numpy/image <-> base64 codec
│   └── routes/
│       ├── avatar.py                #   Phase 1: Avatar generation
│       ├── wardrobe.py              #   Phase 2: Wardrobe analysis
│       ├── fitting.py               #   Phase 3: Virtual try-on
│       ├── viewer3d.py              #   Phase 4: 3D visualization
│       ├── face_bank.py             #   Face identity management
│       ├── p2p.py                   #   Physics-to-Prompt analysis
│       └── quality.py               #   Gemini quality inspection
├── worker/                          # Tier 4: Modal GPU Worker
│   ├── modal_app.py                 #   H100 serverless GPU functions
│   └── serialization.py             #   Worker-side serialization
├── core/                            # Shared AI Logic
│   ├── config.py                    #   Model paths + device + constants
│   ├── loader.py                    #   Lazy model registry
│   ├── pipeline.py                  #   Phase 1: Avatar pipeline
│   ├── fitting.py                   #   Phase 3: Fitting pipeline
│   ├── catvton_pipeline.py          #   CatVTON-FLUX wrapper
│   ├── p2p_engine.py                #   Physics-to-Prompt engine
│   ├── p2p_ensemble.py              #   Multi-strategy P2P fusion
│   ├── face_bank.py                 #   Multi-reference face ID bank
│   ├── face_identity.py             #   InsightFace integration
│   ├── gemini_client.py             #   Gemini API client
│   ├── gemini_feedback.py           #   Quality gate inspector
│   ├── wardrobe.py                  #   Wardrobe analysis logic
│   ├── body_deformation.py          #   Mesh deformation
│   ├── clothing_merger.py           #   Garment mesh merging
│   ├── image_preprocess.py          #   Image preprocessing
│   ├── multiview.py                 #   Multi-view rendering
│   ├── sw_renderer.py               #   Software mesh renderer
│   └── viewer3d.py                  #   3D viewer pipeline
├── scripts/
│   └── upload_models_to_volume.py   # Modal Volume model upload
├── tests/                           # Unit & integration tests
├── setup_models.py                  # Interactive model download script
├── requirements.txt                 # Python dependencies
└── static/index.html                # Test console UI
```

---

## Quick Start

```bash
# 1. Setup
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Environment
cp .env.example .env
# Set GEMINI_API_KEY in .env

# 3. Download models (~25GB)
python setup_models.py

# 4. Run orchestrator (local mode)
python -m orchestrator.main

# 5. Test
open http://localhost:8000/docs   # Swagger UI
open http://localhost:8000/ui     # Test console
```

### Modal GPU Mode

```bash
# Install Modal SDK
pip install modal

# Authenticate
modal setup

# Upload model weights to Modal Volume (one-time)
python scripts/upload_models_to_volume.py

# Run — orchestrator auto-detects Modal and delegates GPU tasks to H100
python -m orchestrator.main
```

---

## API Endpoints

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/health` | System health + model status |
| `POST` | `/avatar/create` | Phase 1: Avatar generation |
| `POST` | `/wardrobe/analyze` | Phase 2: Clothing analysis |
| `POST` | `/fitting/run` | Phase 3: 8-angle virtual try-on |
| `POST` | `/viewer3d/generate` | Phase 4: 3D GLB generation |
| `POST` | `/face-bank/register` | Register face reference |
| `POST` | `/p2p/analyze` | Physics-to-Prompt analysis |
| `POST` | `/quality/inspect` | Gemini quality gate |
| `GET` | `/sessions` | Active session list |
