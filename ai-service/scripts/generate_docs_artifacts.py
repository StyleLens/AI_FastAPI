import base64
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import httpx
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVICE_DIR.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from orchestrator.main import app

PORT = 8089
BASE = f"http://127.0.0.1:{PORT}"
OUT_DIR = REPO_DIR / "docs" / "images"
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
PARTS = [
    ("shoulder", "어깨"),
    ("chest", "가슴둘레"),
    ("waist", "허리둘레"),
    ("hip", "엉덩이둘레"),
    ("sleeve", "소매길이"),
]
TIGHTNESS_COLOR = {
    "critical_tight": (231, 76, 60),
    "tight": (231, 76, 60),
    "optimal": (46, 204, 113),
    "loose": (241, 196, 15),
    "very_loose": (255, 140, 170),
}
MEASUREMENTS = {
    "gender": "female",
    "shoulder_width_cm": "38.5",
    "chest_cm": "88.0",
    "waist_cm": "68.0",
    "hip_cm": "94.0",
}


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        buf = ""
        for ch in paragraph:
            trial = buf + ch
            if buf and draw.textlength(trial, font=font) > width:
                lines.append(buf)
                buf = ch
            else:
                buf = trial
        lines.append(buf)
    return lines


def _jpeg(draw_fn) -> bytes:
    image = Image.new("RGB", (480, 960), (236, 236, 236))
    draw_fn(ImageDraw.Draw(image))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _sample_person() -> bytes:
    # 샘플 전신 이미지 생성.
    root = (SERVICE_DIR / os.getenv("IMG_DATA_DIR", "../IMG_Data") / "User_IMG").resolve()
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                return path.read_bytes()

    def draw(pen: ImageDraw.ImageDraw) -> None:
        pen.ellipse((190, 36, 290, 136), fill=(214, 184, 164))
        pen.polygon([(150, 170), (330, 170), (300, 470), (180, 470)], fill=(55, 78, 130))
        pen.rectangle((175, 470, 230, 900), fill=(42, 44, 58))
        pen.rectangle((250, 470, 305, 900), fill=(42, 44, 58))

    return _jpeg(draw)


def _wait_health(client: httpx.Client) -> None:
    last = ""
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            resp = client.get("/health")
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                return
            last = resp.text
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(0.4)
    raise RuntimeError(f"health 확인 실패: {last}")


def _post_avatar(client: httpx.Client) -> dict:
    resp = client.post(
        "/avatar/generate",
        data=MEASUREMENTS,
        files={"image": ("person.jpg", _sample_person(), "image/jpeg")},
    )
    resp.raise_for_status()
    body = resp.json()
    missing = [angle for angle in ANGLES if str(angle) not in body.get("renders", {})]
    if missing:
        raise RuntimeError(f"누락된 앵글: {missing}")
    return body


def _post_fit(client: httpx.Client, session_id: str) -> dict:
    # 세션에 의류 아이템 등록 후 P2P 분석 수행.
    from core.wardrobe import ClothingItem
    from core.gemini_client import ClothingAnalysis
    analysis = ClothingAnalysis(
        name="슬림핏 실크 블라우스",
        category="top",
        color="navy",
        fit_type="regular",
        elasticity="low",
    )
    size_chart = {
        "sizes": {
            "M": {
                "shoulder_cm": 39.0,
                "chest_cm": 90.0,
                "waist_cm": 72.0,
                "hip_cm": 95.0,
                "sleeve_cm": 58.0,
                "length_cm": 62.0,
            }
        }
    }
    app.state.session_manager.get(session_id).clothing_item = ClothingItem(
        analysis=analysis,
        size_chart=size_chart,
        product_info={"title": "Slim Fit Silk Blouse"},
        fitting_model_info={"model_height_cm": 172},
    )

    resp = client.post("/p2p/analyze", params={"session_id": session_id})
    resp.raise_for_status()
    return resp.json()


def _save_mesh_grid(renders: dict, path: Path) -> None:
    # 8각도 렌더 그리드 생성.
    cell_w, cell_h, header = 256, 346, 48
    canvas = Image.new("RGB", (cell_w * 4, header + cell_h * 2), (232, 232, 232))
    pen = ImageDraw.Draw(canvas)
    pen.rectangle((0, 0, canvas.width, header), fill=(17, 17, 23))
    title = _font(16)
    label = _font(14)
    pen.text((12, 14), "StyleLens AI 3D Body Mesh 8-Angle Depth Renders (0° ~ 315°)", fill=(230, 230, 230), font=title)
    for index, angle in enumerate(ANGLES):
        col, row = index % 4, index // 4
        x, y = col * cell_w, header + row * cell_h
        image = Image.open(BytesIO(base64.b64decode(renders[str(angle)]))).convert("RGB")
        image.thumbnail((cell_w - 16, cell_h - 40))
        canvas.paste(image, (x + (cell_w - image.width) // 2, y + 30 + (cell_h - 40 - image.height) // 2))
        pen.rectangle((x + 8, y + 8, x + 58, y + 26), fill=(0, 0, 0))
        pen.text((x + 14, y + 8), f"{angle}°", fill=(0, 255, 156), font=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_fit_card(payload: dict, path: Path) -> None:
    # 핏 분석 인포그래픽 생성.
    report = payload.get("fit_report") or {}
    shape = payload.get("body_shape_name_ko") or report.get("body_shape_name") or "표준 체형"
    guide_bits = [
        report.get("fit_grade") or "",
        report.get("fit_summary") or "",
        report.get("recommended_size_guidance") or "",
        *(report.get("styling_tips") or []),
    ]
    guide = "\n".join(bit for bit in guide_bits if bit) or "핏 가이드가 없습니다."
    by_part = {item.get("body_part"): item for item in payload.get("deltas") or []}

    width = 960
    title_font = _font(18)
    shape_font = _font(28)
    body_font = _font(16)
    small_font = _font(14)
    probe = ImageDraw.Draw(Image.new("RGB", (width, 8)))
    guide_lines = []
    for paragraph in guide.split("\n"):
        guide_lines.extend(_wrap(probe, paragraph, body_font, width - 80))
    row_h = 46
    header, shape_h, table_top = 52, 78, 36
    guide_top = 40
    height = header + 16 + shape_h + table_top + row_h * 5 + 16 + guide_top + 28 * max(len(guide_lines), 1) + 28

    image = Image.new("RGB", (width, height), (13, 17, 28))
    pen = ImageDraw.Draw(image)
    pen.rectangle((0, 0, width, header), fill=(17, 22, 36))
    pen.text((24, 16), "StyleLens P2P 체형 · 핏 분석", fill=(0, 212, 255), font=title_font)
    y = header + 18
    pen.text((24, y), shape, fill=(255, 255, 255), font=shape_font)
    y += shape_h
    pen.rectangle((24, y, width - 24, y + table_top + row_h * 5), fill=(24, 28, 42))
    columns = [(40, "부위"), (220, "델타"), (400, "조임")]
    for x, name in columns:
        pen.text((x, y + 8), name, fill=(0, 212, 255), font=small_font)
    for index, (key, name) in enumerate(PARTS):
        row_y = y + table_top + index * row_h
        item = by_part.get(key) or {}
        delta = item.get("delta_cm")
        tightness = item.get("tightness") or "-"
        delta_text = f"{delta:+.1f} cm" if isinstance(delta, (int, float)) else "-"
        pen.text((40, row_y + 12), name, fill=(230, 230, 236), font=body_font)
        pen.text((220, row_y + 12), delta_text, fill=(241, 196, 15), font=body_font)
        pen.text((400, row_y + 12), tightness, fill=TIGHTNESS_COLOR.get(tightness, (180, 180, 190)), font=body_font)
    y += table_top + row_h * 5 + 16
    pen.text((24, y), "핏 가이드", fill=(0, 212, 255), font=title_font)
    y += guide_top
    for line in guide_lines:
        pen.text((40, y), line, fill=(220, 224, 232), font=body_font)
        y += 28
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _capture_pages() -> None:
    # 문서 화면 캡처 생성.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        for url, name in ((f"{BASE}/docs", "api_swagger_docs.png"), (f"{BASE}/ui", "test_console_ui.png")):
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(800)
            page.screenshot(path=str(OUT_DIR / name))
        browser.close()


def main() -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=BASE, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            _wait_health(client)
            avatar = _post_avatar(client)
            fit = _post_fit(client, avatar["session_id"])
        _save_mesh_grid(avatar["renders"], OUT_DIR / "avatar_3d_mesh_render.png")
        _save_fit_card(fit, OUT_DIR / "p2p_fit_analysis.png")
        _capture_pages()
        print("모든 아티팩트 캡처 완료.")
    finally:
        server.should_exit = True
        thread.join(timeout=15)


if __name__ == "__main__":
    main()
