#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD NPU 效能最佳化工具
NPU Performance Optimization Tool

專門用於測試和最佳化 AMD Ryzen AI 9HX NPU 的使用率和效能
"""

import os
import time
import glob
import numpy as np
from optimized_amd_npu import OptimizedAMDNPUInference, benchmark_npu_utilization

def find_model_files():
    """尋找可用的模型檔案"""
    if not os.path.exists('models'):
        print("❌ 找不到 models 資料夾")
        return []
    
    model_files = glob.glob('models/*.pth')
    return model_files

def find_test_images(max_images=100):
    """尋找測試圖片"""
    test_dir = 'archive/tw_food_101/test'
    
    if not os.path.exists(test_dir):
        print(f"❌ 找不到測試圖片目錄: {test_dir}")
        return []
    
    # 尋找各種格式的圖片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    
    for ext in extensions:
        pattern = os.path.join(test_dir, ext)
        images = glob.glob(pattern)
        all_images.extend(images)
    
    # 限制數量並排序
    all_images = sorted(all_images)[:max_images]
    print(f"📸 找到 {len(all_images)} 張測試圖片")
    
    return all_images

def test_npu_utilization():
    """測試 NPU 使用率"""
    print("🚀 AMD NPU 使用率測試工具")
    print("=" * 60)
    
    # 檢查模型
    model_files = find_model_files()
    if not model_files:
        print("❌ 找不到訓練好的模型檔案")
        print("請先執行 python train_pytorch.py 進行訓練")
        return
    
    # 選擇最新的模型
    latest_model = max(model_files, key=os.path.getctime)
    print(f"🎯 使用模型: {os.path.basename(latest_model)}")
    
    # 尋找測試圖片
    test_images = find_test_images(200)  # 使用更多圖片測試
    if not test_images:
        return
    
    # 執行基準測試
    print(f"\n🧪 開始 NPU 效能基準測試...")
    results = benchmark_npu_utilization(
        latest_model, 
        test_images, 
        batch_sizes=[1, 2, 4, 8, 16, 24, 32, 48, 64]
    )
    
    return results

def run_optimized_inference():
    """執行最佳化推理"""
    print("⚡ AMD NPU 最佳化推理測試")
    print("=" * 50)
    
    # 檢查模型
    model_files = find_model_files()
    if not model_files:
        print("❌ 找不到訓練好的模型檔案")
        return
    
    latest_model = max(model_files, key=os.path.getctime)
    print(f"🎯 使用模型: {os.path.basename(latest_model)}")
    
    # 尋找測試圖片
    test_images = find_test_images(50)
    if not test_images:
        return
    
    # 測試不同的最佳化設定
    configs = [
        {'batch_size': 16, 'num_threads': 4, 'name': '標準設定'},
        {'batch_size': 32, 'num_threads': 6, 'name': '高並行設定'},
        {'batch_size': 24, 'num_threads': 8, 'name': '最大並行設定'},
        {'batch_size': 8, 'num_threads': 2, 'name': '保守設定'},
    ]
    
    print(f"\n📊 測試不同最佳化設定...")
    
    results = []
    
    for config in configs:
        print(f"\n🔧 測試: {config['name']}")
        print(f"   批次大小: {config['batch_size']}")
        print(f"   執行緒數: {config['num_threads']}")
        print("-" * 40)
        
        try:
            # 建立最佳化推理引擎
            npu_inference = OptimizedAMDNPUInference(
                latest_model,
                batch_size=config['batch_size'],
                num_threads=config['num_threads']
            )
            
            # 執行推理測試
            start_time = time.time()
            predictions = npu_inference.predict_image_batch(test_images)
            total_time = time.time() - start_time
            
            if predictions:
                throughput = len(test_images) / total_time
                avg_time = total_time / len(test_images)
                
                results.append({
                    'name': config['name'],
                    'batch_size': config['batch_size'],
                    'num_threads': config['num_threads'],
                    'throughput': throughput,
                    'total_time': total_time,
                    'avg_time': avg_time
                })
                
                print(f"✅ 吞吐量: {throughput:.1f} 圖片/秒")
                print(f"📊 總時間: {total_time:.3f}s")
                print(f"⏱️  平均時間: {avg_time:.3f}s/圖片")
            
            # 清理
            npu_inference.shutdown()
            del npu_inference
            
        except Exception as e:
            print(f"❌ 設定測試失敗: {e}")
    
    # 顯示結果比較
    if results:
        print(f"\n📈 最佳化設定比較結果")
        print("=" * 80)
        print(f"{'設定名稱':<12} {'批次':<6} {'執行緒':<8} {'吞吐量':<12} {'總時間':<10} {'平均時間':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['name']:<12} {result['batch_size']:<6} "
                  f"{result['num_threads']:<8} {result['throughput']:<12.1f} "
                  f"{result['total_time']:<10.3f} {result['avg_time']:<10.3f}")
        
        best_result = max(results, key=lambda x: x['throughput'])
        print(f"\n🏆 最佳設定: {best_result['name']}")
        print(f"   最高吞吐量: {best_result['throughput']:.1f} 圖片/秒")
        print(f"   建議批次大小: {best_result['batch_size']}")
        print(f"   建議執行緒數: {best_result['num_threads']}")

def monitor_npu_real_time():
    """即時監控 NPU 使用率"""
    print("📊 NPU 即時效能監控")
    print("=" * 50)
    
    model_files = find_model_files()
    if not model_files:
        print("❌ 找不到模型檔案")
        return
    
    latest_model = max(model_files, key=os.path.getctime)
    test_images = find_test_images(20)
    
    if not test_images:
        return
    
    print(f"🎯 使用模型: {os.path.basename(latest_model)}")
    print(f"📸 測試圖片: {len(test_images)} 張")
    print("\n🔄 開始即時監控 (Ctrl+C 停止)...")
    
    try:
        # 建立高效能設定
        npu_inference = OptimizedAMDNPUInference(
            latest_model,
            batch_size=32,
            num_threads=6
        )
        
        iteration = 1
        total_images = 0
        total_time = 0
        
        while True:
            print(f"\n📊 第 {iteration} 次推理:")
            
            start_time = time.time()
            predictions = npu_inference.predict_image_batch(test_images)
            iteration_time = time.time() - start_time
            
            if predictions:
                throughput = len(test_images) / iteration_time
                total_images += len(test_images)
                total_time += iteration_time
                avg_throughput = total_images / total_time
                
                print(f"   本次吞吐量: {throughput:.1f} 圖片/秒")
                print(f"   累計平均: {avg_throughput:.1f} 圖片/秒")
                print(f"   NPU 狀態: {'🟢 高使用率' if throughput > 20 else '🟡 中等使用率' if throughput > 10 else '🔴 低使用率'}")
            
            iteration += 1
            time.sleep(2)  # 暫停 2 秒
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  監控已停止")
        print(f"📊 總計處理: {total_images} 張圖片")
        print(f"⏱️  總計時間: {total_time:.3f}s")
        if total_time > 0:
            print(f"🚀 平均效能: {total_images/total_time:.1f} 圖片/秒")
        
        # 清理
        npu_inference.shutdown()
    
    except Exception as e:
        print(f"❌ 監控失敗: {e}")

def optimize_npu_settings():
    """自動最佳化 NPU 設定"""
    print("🔧 NPU 自動最佳化設定")
    print("=" * 50)
    
    model_files = find_model_files()
    if not model_files:
        print("❌ 找不到模型檔案")
        return None
    
    latest_model = max(model_files, key=os.path.getctime)
    test_images = find_test_images(30)
    
    if not test_images:
        return None
    
    print(f"🎯 使用模型: {os.path.basename(latest_model)}")
    print(f"📸 測試圖片: {len(test_images)} 張")
    print("\n🔍 自動尋找最佳設定...")
    
    # 測試參數組合
    batch_sizes = [8, 16, 24, 32, 48]
    thread_counts = [2, 4, 6, 8]
    
    best_config = None
    best_throughput = 0
    
    total_tests = len(batch_sizes) * len(thread_counts)
    current_test = 0
    
    for batch_size in batch_sizes:
        for num_threads in thread_counts:
            current_test += 1
            print(f"\n🧪 測試 {current_test}/{total_tests}: 批次={batch_size}, 執行緒={num_threads}")
            
            try:
                npu_inference = OptimizedAMDNPUInference(
                    latest_model,
                    batch_size=batch_size,
                    num_threads=num_threads
                )
                
                # 執行多次測試取平均
                throughputs = []
                for _ in range(3):
                    start_time = time.time()
                    predictions = npu_inference.predict_image_batch(test_images)
                    elapsed = time.time() - start_time
                    
                    if predictions:
                        throughput = len(test_images) / elapsed
                        throughputs.append(throughput)
                
                if throughputs:
                    avg_throughput = np.mean(throughputs)
                    print(f"   平均吞吐量: {avg_throughput:.1f} 圖片/秒")
                    
                    if avg_throughput > best_throughput:
                        best_throughput = avg_throughput
                        best_config = {
                            'batch_size': batch_size,
                            'num_threads': num_threads,
                            'throughput': avg_throughput
                        }
                        print(f"   🏆 新的最佳設定！")
                
                npu_inference.shutdown()
                del npu_inference
                
            except Exception as e:
                print(f"   ❌ 測試失敗: {e}")
    
    if best_config:
        print(f"\n🎉 自動最佳化完成！")
        print(f"🏆 最佳設定:")
        print(f"   批次大小: {best_config['batch_size']}")
        print(f"   執行緒數: {best_config['num_threads']}")
        print(f"   最高吞吐量: {best_config['throughput']:.1f} 圖片/秒")
        
        # 儲存最佳設定
        config_file = "optimal_npu_config.txt"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(f"AMD NPU 最佳設定\n")
            f.write(f"批次大小: {best_config['batch_size']}\n")
            f.write(f"執行緒數: {best_config['num_threads']}\n")
            f.write(f"最高吞吐量: {best_config['throughput']:.1f} 圖片/秒\n")
            f.write(f"測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"💾 最佳設定已儲存至: {config_file}")
        
        return best_config
    else:
        print(f"❌ 未找到有效的最佳設定")
        return None

def main():
    """主選單"""
    while True:
        print("\n🚀 AMD NPU 效能最佳化工具")
        print("=" * 50)
        print("1. 🧪 NPU 使用率基準測試")
        print("2. ⚡ 最佳化推理測試")
        print("3. 📊 即時效能監控")
        print("4. 🔧 自動最佳化設定")
        print("5. 📋 檢視硬體資訊")
        print("6. ❌ 退出")
        
        try:
            choice = input("\n👉 請選擇功能 (1-6): ").strip()
            
            if choice == "1":
                test_npu_utilization()
                input("\n按 Enter 鍵繼續...")
                
            elif choice == "2":
                run_optimized_inference()
                input("\n按 Enter 鍵繼續...")
                
            elif choice == "3":
                monitor_npu_real_time()
                input("\n按 Enter 鍵繼續...")
                
            elif choice == "4":
                optimize_npu_settings()
                input("\n按 Enter 鍵繼續...")
                
            elif choice == "5":
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    print(f"\n📋 ONNX Runtime 提供者: {providers}")
                    
                    if 'DmlExecutionProvider' in providers:
                        print("✅ DirectML 可用 - AMD NPU 支援正常")
                    else:
                        print("❌ DirectML 不可用")
                        
                except ImportError:
                    print("❌ ONNX Runtime 未安裝")
                    
                input("\n按 Enter 鍵繼續...")
                
            elif choice == "6":
                print("\n👋 退出程式，感謝使用！")
                break
                
            else:
                print("⚠️  請輸入 1-6")
                
        except KeyboardInterrupt:
            print("\n\n👋 程式結束，感謝使用！")
            break

if __name__ == '__main__':
    main()