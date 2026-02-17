#!/usr/bin/env python3
"""Quick face swap debug test — reuses v33 VTON results."""

import base64
import io
import sys
from pathlib import Path

from PIL import Image, ExifTags

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent

from worker.modal_app import (
    app as modal_app,
    run_face_swap,
)

V33_DIR = ROOT / "tests" / "v33_face_consistency"
IMG_DATA = PROJECT_ROOT / "IMG_Data"
FACE_DIR = IMG_DATA / "User_New" / "User_Face"

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]


def fix_exif_rotation(img):
    try:
        exif = img._getexif()
        if exif is None:
            return img
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        else:
            return img
        if orientation_key not in exif:
            return img
        o = exif[orientation_key]
        if o == 3: return img.rotate(180, expand=True)
        elif o == 6: return img.rotate(270, expand=True)
        elif o == 8: return img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def load_b64(path):
    pil = Image.open(path)
    pil = fix_exif_rotation(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def angle_label(a):
    return str(int(a)) if a == int(a) else str(a)


def main():
    print("=== Face Swap Debug Test ===")

    # Load face reference
    face_imgs = sorted(list(FACE_DIR.glob("*.jpg")) + list(FACE_DIR.glob("*.png")))
    if not face_imgs:
        print("ERROR: No face reference images")
        return
    face_b64 = load_b64(face_imgs[0])
    print(f"Face ref: {face_imgs[0].name}")

    # Load VTON results (before face swap) — use just 3 hero angles for quick test
    test_angles = [0, 45, 270]  # front, 45-degree, side
    fitted_b64s = []
    for a in test_angles:
        p = V33_DIR / f"fitted_before_face_{angle_label(a)}.png"
        if not p.exists():
            print(f"  Missing: {p}")
            return
        fitted_b64s.append(load_b64(p))
        print(f"  Loaded fitted_before_face_{angle_label(a)}.png")

    print(f"\nRunning face swap on {len(test_angles)} angles...")

    with modal_app.run():
        result = run_face_swap.remote(
            images_b64=fitted_b64s,
            face_reference_b64=face_b64,
            angles=test_angles,
            blend_radius=25,
            face_scale=1.0,
        )

    print(f"\nResult keys: {list(result.keys())}")
    if "error" in result:
        print(f"ERROR: {result['error'][:1000]}")
    else:
        print(f"Success! {result['num_images']} images, {result['elapsed_sec']:.1f}s")
        print(f"Face detected: {result['face_detected']}")
        # Save results
        for a, b64 in zip(test_angles, result["swapped_b64"]):
            out_path = V33_DIR / f"debug_faceswap_{angle_label(a)}.png"
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"  Saved: {out_path.name}")


if __name__ == "__main__":
    main()
