#!/usr/bin/env python3
"""output_stamps に存在しないスタンプのみ生成する補助スクリプト."""
from pathlib import Path

from batch_stamp import INPUT_DIR, OUTPUT_DIR, generate_stamps


def main() -> None:
    before = set(Path(OUTPUT_DIR).glob("*.png"))
    generate_stamps(INPUT_DIR, OUTPUT_DIR)
    after = set(Path(OUTPUT_DIR).glob("*.png"))
    added = len(after) - len(before)
    print(f"{added} 件のスタンプを生成しました。")


if __name__ == "__main__":
    main()
