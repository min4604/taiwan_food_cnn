import argparse
import os
from pathlib import Path

def rename_images(folder, start_num=1, prefix="img"):
    folder = Path(folder)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    images = [p for p in folder.iterdir() if p.suffix.lower() in image_exts and p.is_file()]
    images.sort()  # 可依檔名排序

    num = start_num
    for img_path in images:
        new_name = f"{prefix}{num:03d}{img_path.suffix.lower()}"
        new_path = folder / new_name
        # 若新檔名已存在，跳過
        if new_path.exists():
            print(f"跳過已存在: {new_name}")
            num += 1
            continue
        img_path.rename(new_path)
        print(f"{img_path.name} -> {new_name}")
        num += 1

    print(f"✅ 完成，共處理 {num - start_num} 張圖片。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批次重新命名資料夾內所有圖片為連續編號")
    parser.add_argument("--folder", type=str, required=True, help="要處理的圖片資料夾路徑")
    parser.add_argument("--start", type=int, default=1, help="起始編號 (預設1)")
    parser.add_argument("--prefix", type=str, default="", help="檔名前綴 (預設img)")
    args = parser.parse_args()
    rename_images(args.folder, args.start, args.prefix)