"""
StyleLens V6 — E2E Test with IMG_Data samples
Runs the full pipeline: Phase 1 (Avatar) → Phase 2 (Wardrobe) → Phase 3 (Fitting)
Uses actual sample images from IMG_Data/ directory.
Results saved to output/e2e_test/ for visual inspection.
"""

import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# ── Config ──────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
IMG_DATA = Path(__file__).resolve().parent.parent.parent / "IMG_Data"
OUTPUT = Path(__file__).resolve().parent.parent / "output" / "e2e_test"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Sample files
USER_IMAGES = sorted(
    [f for f in (IMG_DATA / "User_IMG").iterdir() if f.suffix.lower() in (".jpg", ".png")]
)
WEAR_IMAGES = sorted(
    [f for f in (IMG_DATA / "wear").iterdir() if f.suffix.lower() in (".jpg", ".png")]
)
WEAR_SIZE_IMAGES = sorted(
    [f for f in (IMG_DATA / "wearSize").iterdir() if f.suffix.lower() in (".jpg", ".png")]
)


def save_b64_image(b64_str: str, path: Path):
    """Decode base64 image and save to disk."""
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(str(path), img)
        print(f"  Saved: {path.name} ({img.shape})")
    else:
        print(f"  WARN: Could not decode image for {path.name}")


def check_server():
    """Check if the server is running."""
    try:
        r = requests.get(f"{API_BASE}/", timeout=5)
        data = r.json()
        print(f"Server: {data.get('service', 'unknown')} v{data.get('version', '?')}")
        print(f"Mode: {data.get('mode', '?')}")
        return True
    except Exception as e:
        print(f"Server not available: {e}")
        return False


def test_phase1_avatar(session_id: str) -> dict:
    """Phase 1: Avatar generation from user image."""
    print("\n" + "=" * 60)
    print("PHASE 1: Avatar Generation")
    print("=" * 60)

    # Use the first user image
    user_img = USER_IMAGES[0]
    print(f"Input: {user_img.name}")

    t0 = time.time()
    with open(user_img, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/avatar/generate",
            params={"session_id": session_id},
            files={"image": (user_img.name, f, "image/jpeg")},
            data={
                "gender": "male",
                "height_cm": "175",
                "weight_kg": "70",
                "body_type": "standard",
            },
            timeout=300,
        )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"FAILED: {resp.status_code} — {resp.text[:500]}")
        return {}

    data = resp.json()
    print(f"Time: {elapsed:.1f}s")
    print(f"Session: {data.get('session_id')}")
    print(f"Has mesh: {data.get('has_mesh')}")
    print(f"Vertex count: {data.get('vertex_count')}")
    print(f"GLB size: {data.get('glb_size_bytes')} bytes")
    print(f"Renders: {list(data.get('renders', {}).keys())}")

    # Save renders
    for angle, b64 in data.get("renders", {}).items():
        save_b64_image(b64, OUTPUT / f"phase1_render_{angle}.jpg")

    # Quality gates
    for g in data.get("quality_gates", []):
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  Gate [{g['stage']}]: {g['score']:.2f} {status}")

    return data


def test_phase2_wardrobe(session_id: str) -> dict:
    """Phase 2: Wardrobe analysis with multiple clothing images + size info."""
    print("\n" + "=" * 60)
    print("PHASE 2: Wardrobe Analysis")
    print("=" * 60)

    print(f"Clothing images: {len(WEAR_IMAGES)}")
    print(f"Size info images: {len(WEAR_SIZE_IMAGES)}")

    # Build multipart files
    files = []
    for wf in WEAR_IMAGES:
        files.append(("images", (wf.name, open(wf, "rb"), "image/png")))

    # Add size chart (first wearSize image)
    if WEAR_SIZE_IMAGES:
        files.append(
            ("size_chart", (WEAR_SIZE_IMAGES[0].name, open(WEAR_SIZE_IMAGES[0], "rb"), "image/png"))
        )

    # Add product info (remaining wearSize images)
    for i, ws in enumerate(WEAR_SIZE_IMAGES[1:3]):
        key = f"product_info_{i+1}"
        files.append(
            (key, (ws.name, open(ws, "rb"), "image/png"))
        )

    t0 = time.time()
    resp = requests.post(
        f"{API_BASE}/wardrobe/add-images",
        params={"session_id": session_id},
        files=files,
        timeout=600,
    )
    elapsed = time.time() - t0

    # Close file handles
    for _, (_, fh, _) in files:
        fh.close()

    if resp.status_code != 200:
        print(f"FAILED: {resp.status_code} — {resp.text[:500]}")
        return {}

    data = resp.json()
    print(f"Time: {elapsed:.1f}s")
    print(f"Session: {data.get('session_id')}")

    analysis = data.get("analysis", {})
    print(f"\nClothing Analysis:")
    print(f"  Name: {analysis.get('name', 'N/A')}")
    print(f"  Category: {analysis.get('category', 'N/A')}")
    print(f"  Color: {analysis.get('color', 'N/A')}")
    print(f"  Fabric: {analysis.get('fabric', 'N/A')}")
    print(f"  Fit: {analysis.get('fit_type', 'N/A')}")
    print(f"  Style: {analysis.get('style_keywords', 'N/A')}")

    size_chart = data.get("size_chart")
    if size_chart:
        print(f"\nSize Chart: {json.dumps(size_chart, ensure_ascii=False)[:200]}")

    product_info = data.get("product_info")
    if product_info:
        print(f"\nProduct Info: {json.dumps(product_info, ensure_ascii=False)[:200]}")

    fitting_model_info = data.get("fitting_model_info")
    if fitting_model_info:
        print(f"\nFitting Model: {json.dumps(fitting_model_info, ensure_ascii=False)[:200]}")

    # Quality gates
    for g in data.get("quality_gates", []):
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  Gate [{g['stage']}]: {g['score']:.2f} {status}")

    return data


def test_phase3_fitting(session_id: str) -> dict:
    """Phase 3: Virtual try-on (8-angle fitting)."""
    print("\n" + "=" * 60)
    print("PHASE 3: Virtual Try-On (Fitting)")
    print("=" * 60)

    # Use a clear face photo
    face_img = USER_IMAGES[0]
    print(f"Face photo: {face_img.name}")

    t0 = time.time()
    with open(face_img, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/fitting/try-on",
            params={"session_id": session_id},
            files={"face_photo": (face_img.name, f, "image/jpeg")},
            timeout=600,
        )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"FAILED: {resp.status_code} — {resp.text[:500]}")
        return {}

    data = resp.json()
    print(f"Time: {elapsed:.1f}s")
    print(f"Session: {data.get('session_id')}")
    print(f"Images generated: {list(data.get('images', {}).keys())}")
    print(f"Methods: {data.get('methods', {})}")

    # Save all fitting images
    for angle, b64 in data.get("images", {}).items():
        save_b64_image(b64, OUTPUT / f"phase3_fitting_{angle}.jpg")

    # P2P analysis
    p2p = data.get("p2p")
    if p2p:
        print(f"\nP2P Analysis:")
        print(f"  Physics prompt: {p2p.get('physics_prompt', 'N/A')[:100]}")
        print(f"  Overall tightness: {p2p.get('overall_tightness', 'N/A')}")
        print(f"  Mask expansion: {p2p.get('mask_expansion_factor', 'N/A')}")
        print(f"  Confidence: {p2p.get('confidence', 'N/A')}")
        for d in p2p.get("deltas", []):
            print(f"    {d['body_part']}: {d['delta_cm']:.1f}cm ({d['tightness']})")

    # Quality gates
    for g in data.get("quality_gates", []):
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  Gate [{g['stage']}]: {g['score']:.2f} {status}")

    # Face Bank
    fb = data.get("face_bank")
    if fb:
        print(f"\nFace Bank: {fb}")

    return data


def main():
    """Run full E2E pipeline test."""
    print("StyleLens V6 — E2E Test with IMG_Data")
    print(f"IMG_Data: {IMG_DATA}")
    print(f"Output: {OUTPUT}")
    print(f"User images: {len(USER_IMAGES)}")
    print(f"Wear images: {len(WEAR_IMAGES)}")
    print(f"WearSize images: {len(WEAR_SIZE_IMAGES)}")

    if not check_server():
        print("\nERROR: Server not running. Start with:")
        print("  cd ai-service && .venv/bin/python -m orchestrator.main")
        sys.exit(1)

    session_id = f"e2e-test-{int(time.time())}"

    # Phase 1
    p1 = test_phase1_avatar(session_id)
    if not p1:
        print("\nPhase 1 failed — cannot continue")
        sys.exit(1)
    session_id = p1.get("session_id", session_id)

    # Phase 2
    p2 = test_phase2_wardrobe(session_id)
    if not p2:
        print("\nPhase 2 failed — cannot continue")
        sys.exit(1)

    # Phase 3
    p3 = test_phase3_fitting(session_id)

    # Summary
    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)
    print(f"Phase 1 (Avatar):   {'OK' if p1 else 'FAIL'}")
    print(f"Phase 2 (Wardrobe): {'OK' if p2 else 'FAIL'}")
    print(f"Phase 3 (Fitting):  {'OK' if p3 else 'FAIL'}")
    print(f"\nResults saved to: {OUTPUT}")
    print("Open the output directory to visually inspect fitting results.")


if __name__ == "__main__":
    main()
