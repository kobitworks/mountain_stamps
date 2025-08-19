import sys, glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def dominant_colors(img_pil: Image.Image, k: int = 3):
    small = img_pil.resize((128, 128))
    arr = np.array(small).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(arr, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    order = counts.argsort()[::-1]
    return [tuple(map(int, centers[i])) for i in order]  # [(r,g,b), ...]

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
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(mask)

def try_load_font(size: int = 44):
    # Ubuntu (GH Actions) に入る DejaVuSans の標準パスを試し、無ければデフォルト
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def make_stamp(img_pil: Image.Image, mask_pil: Image.Image, name: str = "") -> Image.Image:
    size = 768
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(base)

    margin = 28
    bbox = [margin, margin, size - margin, size - margin]
    draw.ellipse(bbox, fill=(255, 255, 255, 255))
    draw.ellipse([margin + 8, margin + 8, size - margin - 8, size - margin - 8], outline=(0, 0, 0, 40), width=2)
    canvas.alpha_composite(base)

    # 背景グラデ用の主要色
    palette = dominant_colors(img_pil, k=3)
    base_col = tuple(palette[0]) + (255,)
    dark_col = tuple(palette[1]) + (255,)

    # 丸の内側に入る正方サークル領域
    inner_w = inner_h = size - 2 * margin - 40
    center = ((size - inner_w) // 2, (size - inner_h) // 2)

    # グラデーション背景
    grad = Image.new("RGBA", (1, inner_h), (0, 0, 0, 0))
    for y in range(inner_h):
        t = y / (inner_h - 1)
        col = tuple(int(base_col[i] * (0.85 + 0.15 * t)) for i in range(3)) + (255,)
        grad.putpixel((0, y), col)
    grad = grad.resize((inner_w, inner_h))
    circle_mask = Image.new("L", (inner_w, inner_h), 0)
    ImageDraw.Draw(circle_mask).ellipse([0, 0, inner_w, inner_h], fill=255)
    tmp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    tmp.paste(grad, center, circle_mask)
    canvas.alpha_composite(tmp)

    # 画像→シルエット
    img_sq = img_pil.copy()
    img_sq.thumbnail((inner_w, inner_h))
    mask_sq = mask_pil.copy()
    mask_sq.thumbnail(img_sq.size)
    m = mask_sq.convert("L").point(lambda v: 255 if v > 100 else 0)
    sil_layer = Image.new("RGBA", img_sq.size, dark_col)
    silhouette = Image.composite(sil_layer, Image.new("RGBA", img_sq.size, (0, 0, 0, 0)), m)

    # ちょい下寄せ
    offset = (center[0] + (inner_w - silhouette.size[0]) // 2,
              center[1] + (inner_h - silhouette.size[1]) // 2 + 30)
    tmp2 = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    tmp2.paste(silhouette, offset, silhouette.split()[3])
    canvas.alpha_composite(tmp2)

    # 白い外枠
    outline = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(outline)
    od.ellipse([margin, margin, size - margin, size - margin], outline=(255, 255, 255, 255), width=14)
    canvas.alpha_composite(outline)

    # テキスト
    if name:
        font = try_load_font(44)
        tw, th = font.getsize(name)
        text_img = Image.new("RGBA", (tw + 24, th + 12), (0, 0, 0, 0))
        td = ImageDraw.Draw(text_img)
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            td.text((12 + dx, 6 + dy), name, font=font, fill=(0, 0, 0, 190))
        td.text((12, 6), name, font=font, fill=(255, 255, 255, 255))
        pos = ((size - text_img.size[0]) // 2, size - text_img.size[1] - 22)
        canvas.alpha_composite(text_img, pos)

    return canvas

def filename_to_name(path: str) -> str:
    stem = Path(path).stem.replace("_", " ").strip()
    return stem[:24] if len(stem) > 24 else stem

def process_folder(in_dir: str, out_dir: str):
    ensure_dir(out_dir)
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths += list(Path(in_dir).glob(ext))
    paths = sorted(map(str, paths))
    for p in paths:
        try:
            img = load_image(p)
            mask = extract_mountain_mask(img)
            name = filename_to_name(p)
            stamp = make_stamp(img, mask, name=name)
            out = Path(out_dir) / (Path(p).stem + "_stamp.png")
            stamp.save(out)
            print(f"OK: {p} -> {out}")
        except Exception as e:
            print(f"FAIL: {p} ({e})")

if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "input_images"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output_stamps"
    process_folder(in_dir, out_dir)
