"""Upload GFPGAN face restoration model weights to Modal Volume."""
import modal

app = modal.App("upload-gfpgan")
model_volume = modal.Volume.from_name("stylelens-models")

@app.function(
    image=modal.Image.debian_slim().pip_install("requests"),
    volumes={"/models": model_volume},
    timeout=300,
)
def download_gfpgan():
    import os
    import requests
    
    gfpgan_dir = "/models/gfpgan"
    os.makedirs(gfpgan_dir, exist_ok=True)
    
    # Also create facexlib weights directory
    facexlib_det_dir = "/models/gfpgan/weights"
    os.makedirs(facexlib_det_dir, exist_ok=True)
    
    files = {
        "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "weights/detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "weights/parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
    }
    
    results = {}
    for filename, url in files.items():
        filepath = os.path.join(gfpgan_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"Already exists: {filepath} ({size_mb:.1f}MB)")
            results[filename] = {"status": "exists", "size_mb": size_mb}
            continue
            
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Downloaded: {filepath} ({size_mb:.1f}MB)")
        results[filename] = {"status": "downloaded", "size_mb": size_mb}
    
    # Verify
    print("\nContents of /models/gfpgan:")
    for item in os.listdir(gfpgan_dir):
        full_path = os.path.join(gfpgan_dir, item)
        if os.path.isdir(full_path):
            print(f"  {item}/ (dir)")
            for sub in os.listdir(full_path):
                sub_path = os.path.join(full_path, sub)
                sub_size = os.path.getsize(sub_path) / (1024 * 1024)
                print(f"    {sub}: {sub_size:.1f}MB")
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  {item}: {size_mb:.1f}MB")
    
    model_volume.commit()
    print("\nVolume committed!")
    
    return results

@app.local_entrypoint()
def main():
    import json, time
    t0 = time.time()
    result = download_gfpgan.remote()
    elapsed = time.time() - t0
    print(f"\nResult: {json.dumps(result, indent=2)}")
    print(f"Elapsed: {elapsed:.1f}s")
