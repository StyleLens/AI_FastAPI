#!/usr/bin/env python3
"""Modal Volume cleanup - runs remotely on Modal to list/delete files."""
import sys
import modal

app = modal.App("stylelens-volume-cleanup")
model_volume = modal.Volume.from_name("stylelens-models")

@app.function(
    volumes={"/models": model_volume},
    timeout=120,
)
def list_and_cleanup(do_cleanup: bool = False) -> str:
    import os
    import shutil

    lines = []
    lines.append("=== Modal Volume Contents ===\n")

    # Walk /models and calculate sizes
    dir_sizes = {}
    total_size = 0
    for root, dirs, files in os.walk("/models"):
        for f in files:
            fp = os.path.join(root, f)
            try:
                sz = os.path.getsize(fp)
                total_size += sz
                parts = fp.split("/")
                if len(parts) >= 3:
                    top = f"/models/{parts[2]}"
                    dir_sizes[top] = dir_sizes.get(top, 0) + sz
            except:
                pass

    for d, sz in sorted(dir_sizes.items()):
        gb = sz / (1024**3)
        lines.append(f"  {d:50s} {gb:8.2f} GB")

    lines.append(f"\nTotal: {total_size / (1024**3):.2f} GB")

    if not do_cleanup:
        lines.append("\nUse --cleanup to remove dead models")
        return "\n".join(lines)

    lines.append("\n=== Cleanup: Removing ALL non-commercial / legacy models ===")
    dead_paths = [
        "/models/flux_dev",              # 53.91GB — FLUX.1-dev 비상업
        "/models/flux_fill_dev",         # 54.07GB — FLUX.1-Fill-dev 비상업
        "/models/catvton_flux",          # 22.17GB — CC BY-NC-SA
        "/models/hunyuan3d",             # 17.57GB — 한국 제외
        "/models/flux_controlnet_depth", # 3.34GB  — FLUX 전용
        "/models/sam3",                  # 3.21GB  — SAM 2.1 (불필요, sam3d_body 사용)
        "/models/pulid_flux",            # 1.86GB  — InsightFace 의존
        "/models/catvton",               # 1.33GB  — 비상업
        "/models/insightface",           # 0.72GB  — 비상업
        "/models/fashn_parser",          # 0.20GB  — SegFormer 비상업
        "/models/Hunyuan3D-2",           # 0.15GB  — 한국 제외
        "/models/yolo26",                # 0.05GB  — AGPL
        "/models/gfpgan",                # if exists
    ]

    freed = 0
    for path in dead_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                dir_sz = sum(
                    os.path.getsize(os.path.join(r, f))
                    for r, _, files in os.walk(path)
                    for f in files
                )
                shutil.rmtree(path)
                freed += dir_sz
                lines.append(f"  DELETED dir:  {path} ({dir_sz / (1024**2):.1f} MB)")
            else:
                sz = os.path.getsize(path)
                os.remove(path)
                freed += sz
                lines.append(f"  DELETED file: {path} ({sz / (1024**2):.1f} MB)")
        else:
            lines.append(f"  NOT FOUND:    {path}")

    model_volume.commit()
    lines.append(f"\nFreed: {freed / (1024**2):.1f} MB")
    lines.append("Volume changes committed.")
    return "\n".join(lines)


if __name__ == "__main__":
    do_cleanup = "--cleanup" in sys.argv

    with app.run():
        result = list_and_cleanup.remote(do_cleanup=do_cleanup)
        print(result)
