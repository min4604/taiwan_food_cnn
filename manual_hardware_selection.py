#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台灣美食 CNN 分類 - 手動硬體選擇推理工具
Taiwan Food CNN Classification - Manual Hardware Selection Inference Tool

支援手動選擇推理硬體，包括 AMD Ryzen AI NPU、GPU、CPU
"""

import os
import sys
import glob
from evaluate_test_set import detect_available_devices, choose_device, evaluate_with_amd_npu, evaluate_standard_mode

def manual_device_inference():
    """
    手動選擇硬體進行推理的主函數
    """
    print("🍜 台灣美食 CNN 分類 - 手動硬體選擇模式")
    print("Taiwan Food CNN Classification - Manual Hardware Selection")
    print("=" * 80)
    
    # 檢查模型檔案
    if not os.path.exists('models'):
        print("❌ 找不到 models 資料夾")
        print("請先執行 python train_pytorch.py 進行訓練")
        return
    
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
    if not model_files:
        print("❌ 找不到訓練好的模型檔案")
        print("請先執行 python train_pytorch.py 進行訓練")
        return
    
    # 顯示可用模型
    print("📂 可用的模型檔案:")
    for i, model_file in enumerate(model_files):
        model_path = os.path.join('models', model_file)
        model_size = os.path.getsize(model_path) / (1024*1024)  # MB
        print(f"  {i}. {model_file} ({model_size:.1f} MB)")
    
    # 選擇模型
    while True:
        try:
            if len(model_files) == 1:
                model_idx = 0
                print(f"\n🎯 自動選擇唯一模型: {model_files[0]}")
                break
            else:
                model_input = input(f"\n👉 請選擇模型 (0-{len(model_files)-1}) 或按 Enter 使用最新模型: ").strip()
                
                if model_input == "":
                    # 使用最新模型
                    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join('models', x)))
                    model_idx = model_files.index(latest_model)
                    print(f"🎯 使用最新模型: {latest_model}")
                    break
                else:
                    model_idx = int(model_input)
                    if 0 <= model_idx < len(model_files):
                        print(f"🎯 選擇模型: {model_files[model_idx]}")
                        break
                    else:
                        print(f"⚠️  請輸入 0-{len(model_files)-1} 之間的數字")
                        
        except ValueError:
            print("⚠️  請輸入有效的數字")
        except KeyboardInterrupt:
            print("\n\n👋 程式結束")
            return
    
    model_path = os.path.join('models', model_files[model_idx])
    
    # 檢測可用硬體
    available_devices, npu_available, gpu_available, amd_npu_available = detect_available_devices()
    
    # 提供 NPU 使用率選項
    if amd_npu_available:
        print(f"\n🚀 AMD NPU 推理模式選擇:")
        print("  1. 🔥 高效能模式 (最大化 NPU 使用率)")
        print("  2. 🔧 標準模式 (平衡效能)")
        print("  3. 💻 其他硬體 (GPU/CPU)")
        
        while True:
            try:
                npu_choice = input("\n👉 請選擇 NPU 模式 (1-3): ").strip()
                
                if npu_choice == "1":
                    print("🔥 使用高效能 AMD NPU 模式")
                    device_str = 'amd_npu_optimized'
                    break
                elif npu_choice == "2":
                    print("🔧 使用標準 AMD NPU 模式")
                    device_str = 'amd_npu'
                    break
                elif npu_choice == "3":
                    print("💻 選擇其他硬體...")
                    device_str = choose_device(available_devices, npu_available, gpu_available, amd_npu_available, manual_mode=True)
                    break
                else:
                    print("⚠️  請輸入 1、2 或 3")
                    
            except KeyboardInterrupt:
                print("\n\n👋 程式結束")
                return
    else:
        # 手動選擇硬體
        device_str = choose_device(available_devices, npu_available, gpu_available, amd_npu_available, manual_mode=True)
    
    if device_str is None:
        print("❌ 未選擇裝置，程式結束")
        return
    
    # 設定測試參數
    test_csv = 'archive/tw_food_101/tw_food_101_test_list.csv'
    test_img_dir = 'archive/tw_food_101/test'
    num_classes = 101
    batch_size = 32
    img_size = 224
    
    # NPU 最佳化參數調整
    if device_str == 'amd_npu_optimized':
        batch_size = 32  # 最佳化批次大小
        print(f"🔥 高效能模式：使用批次大小 {batch_size}")
    
    # 檢查測試資料
    if not os.path.exists(test_csv):
        print(f"❌ 找不到測試清單檔案: {test_csv}")
        return
    
    if not os.path.exists(test_img_dir):
        print(f"❌ 找不到測試圖片目錄: {test_img_dir}")
        return
    
    print(f"\n📋 推理設定:")
    print(f"   模型檔案: {model_files[model_idx]}")
    print(f"   推理硬體: {device_str}")
    print(f"   測試圖片: {test_img_dir}")
    print(f"   批次大小: {batch_size}")
    print(f"   圖片尺寸: {img_size}x{img_size}")
    print("=" * 60)
    
    # 確認開始推理
    confirm = input("🚀 是否開始推理？ (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ 使用者取消推理")
        return
    
    # 執行推理
    try:
        if device_str == 'amd_npu_optimized':
            print("\n� 使用高效能 AMD NPU 模式進行推理...")
            from optimized_amd_npu import OptimizedAMDNPUInference
            
            # 建立最佳化推理引擎
            npu_inference = OptimizedAMDNPUInference(
                model_path, 
                img_size=img_size,
                batch_size=batch_size,
                num_threads=6  # 高並行設定
            )
            
            # 準備測試圖片路徑
            import glob
            test_images = []
            for i in range(100):  # 測試前100張圖片
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_path = os.path.join(test_img_dir, f"{i}{ext}")
                    if os.path.exists(img_path):
                        test_images.append(img_path)
                        break
            
            if test_images:
                print(f"📸 找到 {len(test_images)} 張測試圖片")
                predictions = npu_inference.predict_image_batch(test_images)
                
                # 儲存結果
                results_file = "test_predictions_optimized_npu.csv"
                with open(results_file, 'w', encoding='utf-8') as f:
                    f.write("Id,Category,Path\n")
                    for pred in predictions:
                        f.write(f"{pred['id']},{pred['prediction']},{pred['path']}\n")
                
                print(f"✅ 結果已儲存至: {results_file}")
            
            # 清理
            npu_inference.shutdown()
            
        elif device_str == 'amd_npu':
            print("\n🔧 使用標準 AMD NPU 模式進行推理...")
            evaluate_with_amd_npu(model_path, test_csv, test_img_dir, num_classes, batch_size, img_size)
        else:
            print(f"\n🔧 使用 {device_str.upper()} 進行推理...")
            evaluate_standard_mode(model_path, test_csv, test_img_dir, num_classes, batch_size, img_size, device_str)
            
        print("\n🎉 推理完成！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷推理")
    except Exception as e:
        print(f"\n❌ 推理過程發生錯誤: {e}")
        print("💡 建議檢查硬體相容性和資料完整性")

def show_hardware_info():
    """
    顯示硬體資訊的輔助函數
    """
    print("🔍 硬體資訊檢測")
    print("=" * 40)
    
    available_devices, npu_available, gpu_available, amd_npu_available = detect_available_devices()
    
    print(f"\n📊 硬體支援總結:")
    print(f"   AMD NPU: {'✅ 可用' if amd_npu_available else '❌ 不可用'}")
    print(f"   傳統 NPU: {'✅ 可用' if npu_available else '❌ 不可用'}")
    print(f"   GPU (CUDA): {'✅ 可用' if gpu_available else '❌ 不可用'}")
    print(f"   CPU: ✅ 可用")
    
    return available_devices, npu_available, gpu_available, amd_npu_available

def main():
    """
    主選單
    """
    while True:
        print("\n🍜 台灣美食 CNN - 手動硬體選擇工具")
        print("=" * 50)
        print("1. 🚀 開始手動硬體推理")
        print("2. 🔍 檢視硬體資訊")
        print("3. ❌ 退出程式")
        
        try:
            choice = input("\n👉 請選擇功能 (1-3): ").strip()
            
            if choice == "1":
                manual_device_inference()
            elif choice == "2":
                show_hardware_info()
                input("\n按 Enter 鍵繼續...")
            elif choice == "3":
                print("\n👋 程式結束，感謝使用！")
                break
            else:
                print("⚠️  請輸入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n👋 程式結束，感謝使用！")
            break

if __name__ == '__main__':
    main()