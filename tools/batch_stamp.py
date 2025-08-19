import sys, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from urllib.request import urlopen
from urllib.parse import quote, urlparse, unquote
import json

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "output_stamps"
COMPLETE_DIR = BASE_DIR / "complete_stamp"
DEFAULT_CSV = BASE_DIR / "dat/top100mountains_v4.csv"

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

FONT_PATH = Path(__file__).resolve().parent / "YujiSyuku-Regular.ttf"


def make_stamp(img_pil: Image.Image, mask_pil: Image.Image, name: str = "") -> Image.Image:
    size = 768
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    margin = 20
    border = 12
    circle_bbox = [margin, margin, size - margin, size - margin]
    draw.ellipse(circle_bbox, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=border)

    inner_size = size - 2 * (margin + border)
    scale = inner_size / max(img_pil.width, img_pil.height)
    new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
    img_sq = img_pil.resize(new_size, Image.LANCZOS)
    mask_sq = mask_pil.resize(new_size, Image.LANCZOS)
    m = mask_sq.convert("L").point(lambda v: 255 if v > 80 else 0)

    sil_layer = Image.new("RGBA", new_size, (0, 0, 0, 255))
    silhouette = Image.composite(sil_layer, Image.new("RGBA", new_size, (0, 0, 0, 0)), m)

    offset = (margin + border + (inner_size - new_size[0]) // 2,
              margin + border + (inner_size - new_size[1]) // 2)
    tmp2 = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    tmp2.paste(silhouette, offset, silhouette.split()[3])
    canvas.alpha_composite(tmp2)

    if name:
        try:
            font_size = 80
            font = ImageFont.truetype(str(FONT_PATH), font_size)
        except OSError:
            font = ImageFont.load_default()
            font_size = 20
        max_width = size - 2 * (margin + border)
        stroke = 4
        bbox = draw.textbbox((0, 0), name, font=font, stroke_width=stroke)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        while w > max_width and font_size > 10:
            font_size -= 2
            try:
                font = ImageFont.truetype(str(FONT_PATH), font_size)
            except OSError:
                font = ImageFont.load_default()
                break
            bbox = draw.textbbox((0, 0), name, font=font, stroke_width=stroke)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) // 2
        y = int(size * 0.55)
        draw.text((x, y), name, font=font, fill=(0, 0, 0, 255),
                  stroke_width=stroke, stroke_fill=(255, 255, 255, 255))

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
                    with urlopen(img_url, timeout=20) as r, open(save_path, "wb") as out:
                        out.write(r.read())
                    print(f"DOWNLOADED: {save_path}")
                except Exception as e:
                    print(f"WARN: 画像保存失敗 ({name}) {e}")
    except FileNotFoundError:
        print(f"WARN: CSVが見つかりません: {csv_path}")

def fetch_mountain_data(name: str) -> dict:
    """日本語Wikipediaから山の概要とサムネイル画像URLを取得する。"""
    url = f"https://ja.wikipedia.org/api/rest_v1/page/summary/{quote(name)}"
    try:
        with urlopen(url, timeout=10) as r:
            data = json.load(r)
        thumb = data.get("thumbnail") or {}
        return {
            "title": data.get("title", name),
            "summary": data.get("extract", ""),
            "image_url": thumb.get("source", ""),
        }
    except Exception as e:
        print(f"WARN: Wikipedia取得失敗 ({name}) {e}")
        return {"title": name, "summary": "", "image_url": ""}

def generate_stamps(in_dir: str, out_dir: str):
    """入力画像からスタンプPNGを作成する。"""
    ensure_dir(out_dir)
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths += list(Path(in_dir).glob(ext))
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
            name = filename_to_name(str(img_path))
            info = fetch_mountain_data(name)
            named = make_stamp(img, mask, name=info["title"])
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

    download_mountain_photos(csv_path, INPUT_DIR)
    generate_stamps(in_dir, out_dir)
    add_mountain_names(in_dir, out_dir, complete_dir)
