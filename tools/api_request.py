import os
import uuid
import urllib.request

def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        raise RuntimeError("GITHUB_REPOSITORY 環境変数が未設定です")
    wiki_url = f"https://raw.githubusercontent.com/{repo}/wiki/sample.jpg"

    # Wikiからサンプル画像を取得
    with urllib.request.urlopen(wiki_url) as resp:
        img_data = resp.read()
    with open("sample.jpg", "wb") as f:
        f.write(img_data)

    # multipart/form-data のボディを構築
    boundary = uuid.uuid4().hex
    lines = []
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(b"Content-Disposition: form-data; name=\"file\"; filename=\"sample.jpg\"\r\n")
    lines.append(b"Content-Type: image/jpeg\r\n\r\n")
    lines.append(img_data)
    lines.append(b"\r\n")

    for name, value in {"threshold": "160", "ink_density": "0.85"}.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())

    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)

    req = urllib.request.Request(
        "https://kazooshino-mountainstamp.hf.space/process",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))},
    )
    with urllib.request.urlopen(req) as resp:
        result = resp.read().decode("utf-8")
    with open("response.json", "w", encoding="utf-8") as f:
        f.write(result)

if __name__ == "__main__":
    main()
