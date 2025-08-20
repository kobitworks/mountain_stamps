import sys, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
import cv2
from urllib.request import urlopen, Request
from urllib.parse import quote, urlparse, unquote
import json
import time
import random
import io
import zipfile

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "output_stamps"
COMPLETE_DIR = BASE_DIR / "complete_stamp"
DEFAULT_CSV = BASE_DIR / "dat/top100mountains_v4.csv"

USER_AGENT = "MountainStamps/1.0 (+https://github.com/kaizen-bot/mountain_stamps)"
SLEEP_SEC = 1.0


def open_url(url, timeout=10):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(req, timeout=timeout)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_mountain_mask(img_pil: Image.Image) -> Image.Image:
    """Canny + 形態学で最大コンポーネント抽出し、上側20%は空とみなして切り落とす。"""
    img = np.array(img_pil)
    h, w, _ = img.shape
    scale = 640 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 50, 120)

    k = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, k, 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        def score(cnt):
            area = cv2.contourArea(cnt)
            cy = cnt[:, :, 1].mean()
            return area * (1.0 + 0.5 * (cy / gray.shape[0]))  # 下側ほど加点
        best = max(contours, key=score)
        cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)

    cut = int(mask.shape[0] * 0.2)
    mask[:cut, :] = 0

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask)


def crop_to_mask(img_pil: Image.Image, mask_pil: Image.Image, pad_ratio: float = 0.02):
    """マスクの外接矩形で切り抜き、周囲に少し余白を持たせる。"""
    bbox = mask_pil.getbbox()
    if not bbox:
        return img_pil, mask_pil
    l, t, r, b = bbox
    w, h = r - l, b - t
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    l = max(l - pad_x, 0)
    t = max(t - pad_y, 0)
    r = min(r + pad_x, img_pil.width)
    b = min(b + pad_y, img_pil.height)
    return img_pil.crop((l, t, r, b)), mask_pil.crop((l, t, r, b))

FONT_PATH = Path(__file__).resolve().parent / "YujiBoku-Regular.ttf"
FONT_DOWNLOAD_URL = "https://fonts.google.com/download?family=Yuji%20Boku"
# 標準フォント（DejaVu Sans）を使用
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def get_font(font_size: int) -> ImageFont.FreeTypeFont:
    """指定サイズのフォントを取得する。未取得の場合はGoogle FontsからDLする。"""
    path = FONT_PATH
    if not path.exists():
        try:
            with open_url(FONT_DOWNLOAD_URL, timeout=20) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if name.endswith("YujiBoku-Regular.ttf"):
                        path.write_bytes(zf.read(name))
                        break
        except Exception:
            path = Path(DEFAULT_FONT_PATH)
    try:
        return ImageFont.truetype(str(path), font_size)
    except OSError:
        return ImageFont.load_default()


def make_stamp(img_pil: Image.Image, mask_pil: Image.Image, name: str = "") -> Image.Image:
    size = 768
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    margin = 20
    border = 24  # 円枠線を従来の2倍に太くする
    circle_bbox = [margin, margin, size - margin, size - margin]
    draw.ellipse(circle_bbox, fill=(255, 255, 255, 255))

    inner_size = size - 2 * (margin + border)
    scale = inner_size / max(img_pil.width, img_pil.height)
    new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
    img_sq = img_pil.resize(new_size, Image.LANCZOS)
    mask_sq = mask_pil.resize(new_size, Image.LANCZOS)
    m = mask_sq.convert("L").point(lambda v: 255 if v > 80 else 0)

    # --- 網掛けと輪郭強調によるシルエット生成 ---
    # グレースケール化してディザ処理を適用
    gray = img_sq.convert("L")
    dither = gray.convert("1")  # Floyd-Steinberg ディザ
    dither = dither.convert("L")
    halftone = ImageOps.colorize(dither, black="black", white="white").convert("RGBA")
    halftone.putalpha(m)

    # 輪郭線を抽出して強調
    mask_np = np.array(m)
    edges = cv2.Canny(mask_np, 80, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    edge_img = Image.fromarray(edges)
    edge_rgba = Image.new("RGBA", new_size, (0, 0, 0, 255))
    edge_rgba.putalpha(edge_img)

    # キャンバスへ合成
    offset = (margin + border + (inner_size - new_size[0]) // 2,
              margin + border + (inner_size - new_size[1]) // 2)
    tmp2 = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    tmp2.paste(halftone, offset, halftone.split()[3])
    tmp2.paste(edge_rgba, offset, edge_rgba.split()[3])

    # 山のシルエットが円の外にはみ出さないようにクリッピング
    circle_mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(circle_mask).ellipse(circle_bbox, fill=255)
    tmp2 = Image.composite(tmp2, Image.new("RGBA", canvas.size, (0, 0, 0, 0)), circle_mask)
    canvas.alpha_composite(tmp2)

    # --- かすれ・インクのにじみを追加 ---
    smudge = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(smudge)
    for _ in range(120):
        r = random.randint(1, 4)
        x = random.randint(circle_bbox[0], circle_bbox[2])
        y = random.randint(circle_bbox[1], circle_bbox[3])
        alpha = random.randint(20, 60)
        sdraw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, alpha))
    smudge = smudge.filter(ImageFilter.GaussianBlur(0.8))
    smudge = Image.composite(smudge, Image.new("RGBA", canvas.size, (0, 0, 0, 0)), circle_mask)
    canvas.alpha_composite(smudge)

    if name:
        font_size = 42
        font = get_font(font_size)
        bbox = draw.textbbox((0, 0), name, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) // 2
        y = size - margin - border - h - 10
        draw.text((x, y), name, font=font, fill=(0, 0, 0, 255))

    draw.ellipse(circle_bbox, outline=(0, 0, 0, 255), width=border)
    return canvas

def filename_to_name(path: str) -> str:
    stem = Path(path).stem.replace("_", " ").strip()
    return stem[:24] if len(stem) > 24 else stem


def download_mountain_photos(csv_path: Path, img_dir: Path = INPUT_DIR):
    """CSVの note_url からWikipediaの画像を取得し保存する。"""
    ensure_dir(img_dir)
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gis_id = row.get("gis_id") or row.get("id")
                name = row.get("a_name") or row.get("name")
                note_url = row.get("note_url") or row.get("wiki_url")
                if not (gis_id and name and note_url):
                    continue
                stem = f"{gis_id}_{name}"
                # 画像ファイルが存在しなければダウンロード
                if any((Path(img_dir) / f"{stem}{ext}").exists() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    continue
                title = unquote(note_url.rstrip("/").split("/")[-1])
                info = fetch_mountain_data(title)
                img_url = info.get("image_url")
                if not img_url:
                    print(f"WARN: 画像取得失敗 ({name})")
                    continue
                ext = Path(urlparse(img_url).path).suffix or ".jpg"
                fname = f"{stem}{ext}"
                save_path = Path(img_dir) / fname
                try:
                    with open_url(img_url, timeout=20) as r, open(save_path, "wb") as out:
                        out.write(r.read())
                    lower_path = save_path.with_suffix(save_path.suffix.lower())
                    if save_path != lower_path:
                        save_path.rename(lower_path)
                        save_path = lower_path
                    print(f"DOWNLOADED: {save_path}")
                    time.sleep(SLEEP_SEC)
                except Exception as e:
                    print(f"WARN: 画像保存失敗 ({name}) {e}")
                    time.sleep(SLEEP_SEC)
    except FileNotFoundError:
        print(f"WARN: CSVが見つかりません: {csv_path}")

def fetch_mountain_data(name: str) -> dict:
    """日本語Wikipediaから山の概要とサムネイル画像URLを取得する。"""
    url = f"https://ja.wikipedia.org/api/rest_v1/page/summary/{quote(name)}"
    try:
        with open_url(url, timeout=10) as r:
            data = json.load(r)
        time.sleep(SLEEP_SEC)
        thumb = data.get("thumbnail") or {}
        return {
            "title": data.get("title", name),
            "summary": data.get("extract", ""),
            "image_url": thumb.get("source", ""),
        }
    except Exception as e:
        print(f"WARN: Wikipedia取得失敗 ({name}) {e}")
        time.sleep(SLEEP_SEC)
        return {"title": name, "summary": "", "image_url": ""}

def generate_stamps(in_dir: str, out_dir: str):
    """入力画像からスタンプPNGを作成する。"""
    ensure_dir(out_dir)
    paths = [p for p in Path(in_dir).glob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    paths = sorted(map(str, paths))
    for p in paths:
        stem = Path(p).stem
        out_stamp = Path(out_dir) / f"{stem}.png"
        if out_stamp.exists():
            continue
        try:
            img = load_image(p)
            mask = extract_mountain_mask(img)
            img, mask = crop_to_mask(img, mask)
            stamp = make_stamp(img, mask)
            stamp.save(out_stamp)
            print(f"STAMP: {p} -> {out_stamp}")
        except Exception as e:
            print(f"FAIL: {p} ({e})")


def add_mountain_names(in_dir: str, out_dir: str, complete_dir: str):
    """スタンプに山名を追加した画像を生成する。"""
    ensure_dir(complete_dir)
    for stamp_path in sorted(Path(out_dir).glob("*.png")):
        stem = stamp_path.stem
        complete_path = Path(complete_dir) / f"{stem}.png"
        if complete_path.exists():
            continue
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = Path(in_dir) / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"WARN: 元画像なし ({stem})")
            continue
        try:
            img = load_image(str(img_path))
            mask = extract_mountain_mask(img)
            img, mask = crop_to_mask(img, mask)
            a_name = stem.split('_', 1)[1] if '_' in stem else stem
            info = fetch_mountain_data(a_name)
            named = make_stamp(img, mask, name=a_name)
            named.save(complete_path)
            if info["summary"]:
                out_txt = Path(complete_dir) / f"{stem}_wiki.txt"
                out_txt.write_text(info["summary"], encoding="utf-8")
            print(f"COMPLETE: {stamp_path} -> {complete_path}")
        except Exception as e:
            print(f"FAIL: {stamp_path} ({e})")

if __name__ == "__main__":
    # 引数は以下のいずれかの形式を受け付ける:
    #   python tools/batch_stamp.py                           (全てデフォルト)
    #   python tools/batch_stamp.py input_dir output_dir      (CSVはデフォルト)
    #   python tools/batch_stamp.py csv_path input_dir output_dir
    args = sys.argv[1:]

    csv_path = DEFAULT_CSV
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    complete_dir = COMPLETE_DIR

    if len(args) == 1:
        first = Path(args[0])
        if first.suffix.lower() == ".csv":
            csv_path = first
        else:
            in_dir = first
    elif len(args) == 2:
        first, second = map(Path, args)
        if first.suffix.lower() == ".csv":
            csv_path = first
            in_dir = second
        else:
            in_dir = first
            out_dir = second
    elif len(args) >= 3:
        csv_path, in_dir, out_dir = map(Path, args[:3])
        if len(args) >= 4:
            complete_dir = Path(args[3])

    download_mountain_photos(csv_path, in_dir)
    time.sleep(5)
    generate_stamps(in_dir, out_dir)
    add_mountain_names(in_dir, out_dir, complete_dir)
