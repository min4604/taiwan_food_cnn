#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bing 圖片爬蟲 (免費，不需 API Key)

特色：
- 完全免費，不需 API Key
- 依據 CSV 類別下載圖片
- 自動建立資料夾
- 可設定下載數量、延遲、ID 範圍

原理：
模擬瀏覽器向 Bing 圖片搜尋發送請求，解析其回傳的 HTML，
從中提取一個內嵌的 JSON 物件，該物件包含了搜尋結果的圖片資訊。
"""

import os
import csv
import time
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("需要 'beautifulsoup4' 套件，請執行: pip install beautifulsoup4 lxml")
    exit(1)

# 預設常數
DEFAULT_CLASSES_CSV = os.path.join("archive", "tw_food_101", "tw_food_101_classes.csv")
DEFAULT_OUTPUT_DIR = os.path.join("downloads", "bing_images")

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)

class BingImageCrawler:
    def __init__(
        self,
        classes_csv: str = DEFAULT_CLASSES_CSV,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        images_per_class: int = 20,
        request_delay: float = 0.5,
        query_prefix: str = "",
        query_suffix: str = "",
        adult_filter: str = "off", # off | moderate | strict
    ) -> None:
        self.classes_csv = classes_csv
        self.output_dir = output_dir
        self.images_per_class = int(images_per_class)
        self.request_delay = float(request_delay)
        self.query_prefix = query_prefix.strip()
        self.query_suffix = query_suffix.strip()
        self.adult_filter = adult_filter.capitalize() # API expects Off, Moderate, Strict

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
        })

        ensure_dir(self.output_dir)

    def load_food_classes(self) -> Dict[int, str]:
        classes: Dict[int, str] = {}
        if not os.path.exists(self.classes_csv):
            raise FileNotFoundError(f"找不到類別檔案: {self.classes_csv}")

        with open(self.classes_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                try:
                    classes[int(row[0])] = row[1].strip()
                except (ValueError, IndexError):
                    continue
        return classes

    def search_bing_images(self, query: str, max_images: int) -> List[str]:
        """從 Bing 搜尋並返回圖片 URL 列表，包含正確的分頁邏輯"""
        urls = set()
        search_url = "https://www.bing.com/images/search"
        page_size = 35  # Bing 的大約頁面大小
        current_offset = 0
        retries = 3

        print(f"  - 開始搜尋 '{query}'，目標 {max_images} 張圖片...")

        while len(urls) < max_images:
            params = {
                'q': query,
                'form': 'HDRSC3',
                'first': current_offset,
                'count': page_size,
                'mmasync': 1,
                'adlt': self.adult_filter,
            }

            try:
                res = self.session.get(search_url, params=params)
                res.raise_for_status()
                
                soup = BeautifulSoup(res.text, 'lxml')
                image_links = soup.find_all('a', class_='iusc')
                
                if not image_links:
                    print("  - ⚠ 找不到更多圖片結果，可能已達搜尋結尾。")
                    break

                new_images_found_this_page = 0
                for link in image_links:
                    if 'm' in link.attrs:
                        try:
                            m_data = json.loads(link['m'])
                            if image_url := m_data.get("murl"):
                                if image_url not in urls:
                                    urls.add(image_url)
                                    new_images_found_this_page += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                print(f"  - 第 {current_offset // page_size + 1} 頁找到 {new_images_found_this_page} 張新圖片 (目前總數: {len(urls)})")

                if new_images_found_this_page == 0 and current_offset > 0:
                    print("  - ⚠ 本頁未找到新圖片，可能已達搜尋結尾。")
                    break
                
                # 正確的分頁邏輯
                current_offset += page_size
                
                # 禮貌性延遲
                time.sleep(self.request_delay)

            except requests.RequestException as e:
                print(f"  - ✗ 網路請求失敗: {e}")
                retries -= 1
                if retries <= 0:
                    print("  - ✗ 連續請求失敗，跳過此查詢。")
                    break
                time.sleep(3) # 發生錯誤時等待更長時間
            except Exception as e:
                print(f"  - ✗ 處理搜尋結果時發生未知錯誤: {e}")
                break
        
        return list(urls)[:max_images]

    def download_image(self, url: str, dest_path: str, timeout: int = 15) -> Tuple[bool, str]:
        try:
            r = self.session.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            
            ensure_dir(os.path.dirname(dest_path))
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True, "ok"
        except requests.exceptions.RequestException as e:
            return False, f"request error: {e}"
        except Exception as e:
            return False, f"error: {e}"

    def build_query(self, cls_name: str) -> str:
        parts = [p for p in [self.query_prefix, cls_name.replace("_", " "), self.query_suffix] if p]
        return " ".join(parts)

    def run(self, start_id: Optional[int] = None, end_id: Optional[int] = None) -> None:
        classes = self.load_food_classes()
        ids = sorted([i for i in classes.keys() if (start_id is None or i >= start_id) and (end_id is None or i <= end_id)])

        print(f"找到 {len(classes)} 個類別，將處理 {len(ids)} 個。")
        print(f"目標每類 {self.images_per_class} 張，輸出到: {self.output_dir}")

        for i in ids:
            cls_name = classes[i]
            query = self.build_query(cls_name)
            target_dir = os.path.join(self.output_dir, safe_filename(cls_name))
            ensure_dir(target_dir)

            print(f"\n[#{i:03d}] {cls_name} -> '{query}'")

            try:
                image_urls = self.search_bing_images(query, self.images_per_class)
            except Exception as e:
                print(f"  ✗ 搜尋時發生嚴重錯誤: {e}")
                continue

            if not image_urls:
                print("  ⚠ 無搜尋結果")
                continue
            
            print(f"  → 找到 {len(image_urls)} 個候選圖片")

            downloaded_count = 0
            for idx, url in enumerate(image_urls):
                if downloaded_count >= self.images_per_class:
                    break

                ext = os.path.splitext(url.split("?")[0])[-1].lower()
                if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                    ext = ".jpg"
                
                filename = f"{safe_filename(cls_name)}_{downloaded_count + 1:03d}{ext}"
                dest_path = os.path.join(target_dir, filename)

                ok, msg = self.download_image(url, dest_path)
                if ok:
                    downloaded_count += 1
                    print(f"    ✓ ({downloaded_count}/{self.images_per_class}) 保存: {filename}")
                else:
                    print(f"    ✗ 失敗: {url[:50]}... ({msg})")

                time.sleep(self.request_delay)
            
            print(f"  → 完成 {downloaded_count}/{self.images_per_class} 張")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bing 圖片爬蟲 (免費，不需 API Key)")
    p.add_argument("--classes-csv", default=DEFAULT_CLASSES_CSV, help="類別 CSV 檔路徑")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="輸出資料夾")
    p.add_argument("--images-per-class", type=int, default=20, help="每個類別要下載的張數")
    p.add_argument("--request-delay", type=float, default=0.5, help="每次下載之間的秒數延遲")
    p.add_argument("--start-id", type=int, default=None, help="起始類別 ID（包含）")
    p.add_argument("--end-id", type=int, default=None, help="結束類別 ID（包含）")
    p.add_argument("--query-prefix", default="", help="查詢關鍵字前綴")
    p.add_argument("--query-suffix", default="麻花捲", help="查詢關鍵字後綴")
    p.add_argument("--adult-filter", default="off", choices=["off", "moderate", "strict"], help="成人內容過濾")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    crawler = BingImageCrawler(
        classes_csv=args.classes_csv,
        output_dir=args.output_dir,
        images_per_class=args.images_per_class,
        request_delay=args.request_delay,
        query_prefix=args.query_prefix,
        query_suffix=args.query_suffix,
        adult_filter=args.adult_filter,
    )
    try:
        crawler.run(start_id=args.start_id, end_id=args.end_id)
    except KeyboardInterrupt:
        print("\n使用者中斷。")
    except Exception as e:
        print(f"\n執行失敗：{e}")

if __name__ == "__main__":
    main()
