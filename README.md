# Mountain Stamps (free, CPU-only, batch)

山の写真からスタンプ風PNGを **無料/CPUのみ** でバッチ生成します。
- リスト: `dat/top100mountains_v4.csv`
- 入力: Wikipediaから自動取得し `input_images/` に保存
- 出力: `output_stamps/`（入力と同名のPNG、透過768px）
- 自動化: GitHub Actions（毎日UTC20:00/JST05:00 & 手動実行）

## ローカル実行
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python tools/batch_stamp.py dat/top100mountains_v4.csv input_images output_stamps
```

## GitHub Actions

* 公開リポ推奨（無料分を安心して使えるため）
* 生成後は自動コミットされます（差分が無い場合はスキップ）

## 仕組み（概要）

* OpenCV(CPU)の Canny + 形態学 + 最大コンポーネントで **山シルエット** 抽出
* OpenCVの k-means で **主要色抽出**
* Pillowで **丸背景/白フチ/テキスト** 合成、透過PNGで保存

## 注意

* 雲やコントラストによってはシルエット抽出が弱い場合があります（`tools/batch_stamp.py` の Canny 等を調整してください）
