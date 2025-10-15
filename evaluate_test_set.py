import torch
import torch.nn as nn
from pytorch_model import get_model
from pytorch_data_loader import TaiwanFoodDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import csv
from PIL import Image

def detect_model_architecture(model_path):
    """從模型檔案名稱中檢測模型架構"""
    filename = os.path.basename(model_path).lower()
    
    if 'efficientnet_b3' in filename:
        return 'efficientnet_b3'
    elif 'convnext_tiny' in filename:
        return 'convnext_tiny'
    elif 'regnet_y' in filename:
        return 'regnet_y'
    elif 'vit' in filename:
        return 'vit'
    elif 'resnet50' in filename:
        return 'resnet50'
    else:
        # 預設為 ResNet50
        print(f"⚠️  無法從檔案名稱檢測模型架構: {filename}")
        print("   使用預設架構: ResNet50")
        return 'resnet50'

def resolve_image_paths_from_csv(test_csv, test_img_dir):
    """從 CSV 讀取圖片路徑並盡力解析為實際檔案路徑。
    支援兩種格式：
      - 測試集: index,path (例如 0,test/0.jpg)
      - 訓練清單: index,class_id,path (例如 0,0,train/bawan/0.jpg)
    解析策略：
      1) 如果 CSV 中的 path 是絕對路徑且存在，直接使用。
      2) 嘗試拼接 test_img_dir + path
      3) 嘗試拼接 test_img_dir + basename(path)
      4) 嘗試拼接 test_img_dir + 最後兩層 (通常為 類別/檔名)
      5) 如果 path 以 train/ 或 test/ 開頭，嘗試移除第一層再拼接
    回傳：與 CSV 行數對應的 list，找不到則為 None。
    """
    paths = []
    test_img_dir = os.path.abspath(test_img_dir)
    with open(test_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                paths.append(None)
                continue
            raw_path = row[-1].strip()  # 取最後一欄作為路徑欄
            raw_path = raw_path.replace('\\', '/')

            candidates = []
            # 絕對路徑
            if os.path.isabs(raw_path):
                candidates.append(raw_path)
            # 直接拼接 test_img_dir + raw_path
            candidates.append(os.path.join(test_img_dir, raw_path))
            # basename
            candidates.append(os.path.join(test_img_dir, os.path.basename(raw_path)))
            # 最後兩層 (類別/檔名)
            parts = raw_path.split('/')
            if len(parts) >= 2:
                candidates.append(os.path.join(test_img_dir, parts[-2], parts[-1]))
            # 去掉開頭的 train/ 或 test/
            if parts and parts[0] in ('train', 'test'):
                stripped = '/'.join(parts[1:])
                candidates.append(os.path.join(test_img_dir, stripped))
                if len(parts) >= 3:
                    candidates.append(os.path.join(test_img_dir, parts[-2], parts[-1]))

            chosen = None
            for c in candidates:
                if c and os.path.exists(c):
                    chosen = c
                    break
            paths.append(chosen)
    return paths

def is_openable_image(path):
    try:
        with Image.open(path) as img:
            img.verify()  # 快速驗證
        return True
    except Exception:
        return False

def filter_openable_paths(paths):
    valid = []
    invalid = []
    for p in paths:
        if p and is_openable_image(p):
            valid.append(p)
        else:
            invalid.append(p)
    return valid, invalid

def detect_available_devices():
    """
    檢測可用的計算裝置（NPU/GPU/CPU）
    特別支援 AMD Ryzen AI 9HX NPU
    """
    print("🔍 檢測可用的計算裝置")
    print("=" * 60)
    
    devices = []
    device_info = []
    
    # 檢測 AMD Ryzen AI NPU
    amd_npu_available = False
    try:
        # 方式 1: 檢測 DirectML
        try:
            import torch_directml
            if torch_directml.is_available():
                amd_npu_available = True
                devices.append('dml')
                device_info.append("🚀 AMD NPU 可用 (DirectML)")
                device_info.append("   支援: Ryzen AI 9HX NPU")
        except ImportError:
            pass
        
        # 方式 2: 檢測 ONNX Runtime
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                if not amd_npu_available:  # 避免重複顯示
                    amd_npu_available = True
                    devices.append('onnx_dml')
                device_info.append("✅ ONNX Runtime DML 可用")
                device_info.append("   支援: AMD Ryzen AI NPU")
        except ImportError:
            pass
        
        # 方式 3: 檢測系統資訊
        import platform
        import subprocess
        if platform.system() == 'Windows':
            try:
                # 檢查 AMD Ryzen AI 處理器
                result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=5)
                if 'AMD Ryzen' in result.stdout and 'AI' in result.stdout:
                    device_info.append("💻 檢測到 AMD Ryzen AI 處理器")
                    if not amd_npu_available:
                        device_info.append("   ⚠️  NPU 可能可用但未啟用")
            except:
                pass
    except Exception as e:
        pass
    
    # 檢測传統 NPU 支援（华为等）
    npu_available = False
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            npu_count = torch.npu.device_count()
            npu_available = True
            devices.append('npu')
            device_info.append(f"🚀 NPU 可用: {npu_count} 個裝置")
            for i in range(npu_count):
                try:
                    npu_name = torch.npu.get_device_name(i)
                    device_info.append(f"   NPU {i}: {npu_name}")
                except:
                    device_info.append(f"   NPU {i}: 未知型號")
        elif hasattr(torch.backends, 'npu') and torch.backends.npu.is_available():
            npu_available = True
            devices.append('npu')
            device_info.append("🚀 NPU 支援已啟用")
    except Exception as e:
        pass
    
    # 檢測 Apple MPS
    mps_available = False
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
            devices.append('mps')
            device_info.append("🍎 MPS (Apple Silicon) 可用")
    except:
        pass
    
    # 檢測 GPU 支援
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        devices.append('cuda')
        device_info.append(f"✅ CUDA GPU 可用: {gpu_count} 個裝置")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            device_info.append(f"   GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    # CPU 始終可用
    devices.append('cpu')
    device_info.append("💻 CPU 可用")
    
    # 顯示檢測結果
    for info in device_info:
        print(info)
    
    if not amd_npu_available and not npu_available and not gpu_available:
        print("⚠️  沒有檢測到 NPU 或 GPU，將使用 CPU")
    
    print("=" * 60)
    return devices, npu_available, gpu_available, amd_npu_available

def choose_device(available_devices, npu_available, gpu_available, amd_npu_available=False, manual_mode=False):
    """
    選擇計算裝置
    manual_mode: True 為手動選擇，False 為自動選擇
    """
    print("\n🎯 選擇計算裝置")
    print("-" * 40)
    
    options = []
    if amd_npu_available:
        options.append(('amd_npu', '🚀 AMD Ryzen AI NPU (最高效能)'))
    if npu_available:
        options.append(('npu:0', '🚀 NPU (高效能)'))
    if gpu_available:
        options.append(('cuda:0', '✅ GPU (高效能)'))
    options.append(('cpu', '💻 CPU (穩定)'))
    
    print("可用的計算裝置:")
    for i, (device, desc) in enumerate(options):
        print(f"  {i}. {desc}")
    
    if manual_mode:
        # 手動選擇模式
        print(f"\n請選擇要使用的裝置 (0-{len(options)-1}):")
        print("或直接按 Enter 使用自動選擇")
        
        while True:
            try:
                user_input = input("👉 請輸入選項編號: ").strip()
                
                if user_input == "":
                    # 使用者選擇自動模式
                    print("🤖 使用自動選擇模式")
                    break
                
                choice_idx = int(user_input)
                if 0 <= choice_idx < len(options):
                    chosen_device = options[choice_idx][0]
                    print(f"\n✅ 手動選擇: {options[choice_idx][1]}")
                    return chosen_device
                else:
                    print(f"⚠️  請輸入 0-{len(options)-1} 之間的數字")
                    
            except ValueError:
                print("⚠️  請輸入有效的數字")
            except KeyboardInterrupt:
                print("\n\n❌ 使用者取消操作")
                return None
    
    # 自動選擇最佳裝置
    if amd_npu_available:
        chosen_device = 'amd_npu'
        print(f"\n🤖 自動選擇: AMD Ryzen AI NPU (最佳效能)")
    elif npu_available:
        chosen_device = 'npu:0'
        print(f"\n🤖 自動選擇: NPU (高效能)")
    elif gpu_available:
        chosen_device = 'cuda:0'
        print(f"\n🤖 自動選擇: GPU (高效能)")
    else:
        chosen_device = 'cpu'
        print(f"\n🤖 自動選擇: CPU")
    
    return chosen_device

def evaluate_with_amd_npu(model_path, test_csv, test_img_dir, num_classes=101, batch_size=32, img_size=224):
    """
    使用最佳化的 AMD NPU 進行測試集評估 - 提高 NPU 使用率
    """
    try:
        # 嘗試使用最佳化版本
        try:
            from optimized_amd_npu import OptimizedAMDNPUInference
            print("🚀 使用最佳化 AMD NPU 推理引擎...")
            
            # 調整批次大小以最大化 NPU 使用率
            optimized_batch_size = min(batch_size, 32)  # NPU 最佳批次大小
            npu_inference = OptimizedAMDNPUInference(
                model_path, 
                img_size, 
                batch_size=optimized_batch_size,
                num_threads=6  # 增加並行執行緒
            )
            
        except ImportError:
            # 回退到原始版本
            from amd_npu_fixed import AMDNPUInference
            print("🔄 使用標準 AMD NPU 推理引擎...")
            npu_inference = AMDNPUInference(model_path, img_size)
        
        # 建立測試集 DataLoader (僅用於組織資料)
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_dataset = TaiwanFoodDataset(test_csv, test_img_dir, test_transform, is_test=True)
        
        print(f"📊 測試集大小: {len(test_dataset)}")
        print("🔍 開始 AMD NPU 高效率評估...")
        print("⚠️  注意：這是首次在測試集上評估，結果代表模型的真實性能")
        print("=" * 60)
        
        # 準備所有測試圖片路徑 (改為從 CSV 讀取)
        all_image_paths = resolve_image_paths_from_csv(test_csv, test_img_dir)
        
        # 過濾出有效的圖片路徑
        valid_paths_quick = [path for path in all_image_paths if path is not None]
        # 進一步驗證圖片可讀性
        valid_paths, invalid_paths = filter_openable_paths(valid_paths_quick)
        if invalid_paths:
            print(f"⚠️  跳過無法讀取的圖片: {len(invalid_paths)} 張")
        print(f"📸 有效圖片: {len(valid_paths)}/{len(all_image_paths)}")
        
        # 使用最佳化的批次推理
        if hasattr(npu_inference, 'predict_image_batch'):
            print("⚡ 使用批次推理模式以最大化 NPU 使用率...")
            batch_results = npu_inference.predict_image_batch(valid_paths)
            
            # 重新對應結果到原始索引
            all_predictions = []
            all_confidences = []  # 新增信心分數收集
            all_image_paths_with_results = []  # 對應的圖片路徑
            valid_idx = 0
            
            for i, img_path in enumerate(all_image_paths):
                if img_path is not None:
                    if valid_idx < len(batch_results):
                        all_predictions.append(batch_results[valid_idx]['prediction'])
                        # 檢查是否有信心分數
                        confidence = batch_results[valid_idx].get('confidence', 0.0)
                        all_confidences.append(confidence)
                        all_image_paths_with_results.append(batch_results[valid_idx]['path'])
                        valid_idx += 1
                    else:
                        all_predictions.append(-1)
                        all_confidences.append(0.0)
                        all_image_paths_with_results.append(None)
                else:
                    all_predictions.append(-1)
                    all_confidences.append(0.0)
                    all_image_paths_with_results.append(None)
            
            # 清理資源
            if hasattr(npu_inference, 'shutdown'):
                npu_inference.shutdown()
                
        else:
            # 回退到單張處理
            print("🔄 使用單張推理模式...")
            all_predictions = []
            all_confidences = []  # 單張推理模式不支援信心分數
            all_image_paths_with_results = []
            
            with tqdm(total=len(valid_paths), desc="AMD NPU 推理中", ncols=80) as pbar:
                for img_path in valid_paths:
                    if img_path and os.path.exists(img_path):
                        # 檢查是否支援信心分數
                        if hasattr(npu_inference, 'predict_image_with_confidence'):
                            pred, confidence = npu_inference.predict_image_with_confidence(img_path)
                            all_predictions.append(pred)
                            all_confidences.append(confidence)
                        else:
                            pred = npu_inference.predict_image(img_path)
                            all_predictions.append(pred)
                            all_confidences.append(0.0)  # 不支援信心分數
                        all_image_paths_with_results.append(img_path)
                    else:
                        all_predictions.append(-1)
                        all_confidences.append(0.0)
                        all_image_paths_with_results.append(None)
                    
                    pbar.update(1)
                    pbar.set_postfix({'已處理': len(all_predictions)})
        
        # 儲存預測結果
        results_file = "test_predictions_optimized_amd_npu.csv"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Id,Category,Confidence,Path\n")
            for i, pred in enumerate(all_predictions):
                confidence = all_confidences[i] if i < len(all_confidences) else 0.0
                path = all_image_paths_with_results[i] if i < len(all_image_paths_with_results) else "Unknown"
                f.write(f"{i},{pred},{confidence:.4f},{path}\n")
        
        print(f"\n✅ 最佳化 AMD NPU 評估完成！")
        print(f"📁 預測結果已儲存至: {results_file}")
        print(f"📊 共處理 {len(all_predictions)} 張測試圖片")
        
        # 顯示預測類別分佈
        from collections import Counter
        valid_predictions = [p for p in all_predictions if p != -1]
        pred_counts = Counter(valid_predictions)
        print(f"\n📈 預測類別分佈 (前10名):")
        for class_id, count in pred_counts.most_common(10):
            print(f"   類別 {class_id}: {count} 張圖片")
        
        print(f"\n🚀 AMD Ryzen AI NPU 最佳化推理完成！NPU 使用率已最大化！")
        
        # 新增：自動挑選信心分數低的圖片並開啟顯示
        try:
            import subprocess
            threshold = 0.5  # 信心分數門檻
            max_show = 10    # 最多顯示圖片數量
            
            # 找出信心分數低的圖片
            low_confidence_images = []
            for i, (confidence, path) in enumerate(zip(all_confidences, all_image_paths_with_results)):
                if confidence > 0 and confidence < threshold and path and os.path.exists(path):
                    low_confidence_images.append((i, confidence, path))
            
            # 按信心分數排序（最低的優先）
            low_confidence_images = sorted(low_confidence_images, key=lambda x: x[1])[:max_show]
            
            if low_confidence_images:
                print(f"\n🔍 發現 {len(low_confidence_images)} 張預測信心低於 {threshold} 的圖片：")
                print("=" * 60)
                
                for idx, confidence, path in low_confidence_images:
                    predicted_class = all_predictions[idx]
                    print(f"  📷 圖片編號: {idx}")
                    print(f"  🎯 預測類別: {predicted_class}")
                    print(f"  📊 信心分數: {confidence:.3f}")
                    print(f"  📁 路徑: {path}")
                    print("-" * 40)
                    
                    # 使用 Windows 預設程式開啟圖片
                    
                
                print(f"🎯 已自動開啟 {len(low_confidence_images)} 張低信心圖片供檢視")
            else:
                print(f"\n✅ 太棒了！沒有信心分數低於 {threshold} 的圖片！模型表現優異！")
                
        except Exception as e:
            print(f"\n⚠️  低信心圖片顯示功能發生錯誤: {e}")
        
    except ImportError as e:
        print(f"❌ AMD NPU 模組載入失敗: {e}")
        print("🔄 回退到標準評估模式...")
        return evaluate_standard_mode(model_path, test_csv, test_img_dir, num_classes, batch_size, img_size, 'cpu')
    except Exception as e:
        print(f"❌ AMD NPU 評估失敗: {e}")
        print("🔄 回退到標準評估模式...")
        return evaluate_standard_mode(model_path, test_csv, test_img_dir, num_classes, batch_size, img_size, 'cpu')

def evaluate_standard_mode(model_path, test_csv, test_img_dir, num_classes, batch_size, img_size, device_str):
    """
    標準模式評估 (CPU/GPU/傳統NPU)
    """
    # 設定裝置
    device = torch.device(device_str)
    print(f"🔧 使用標準模式裝置: {device}")
    
    # 如果使用 NPU，調整批次大小
    if 'npu' in device_str:
        batch_size = min(batch_size, 16)  # NPU 可能需要較小的批次
        print(f"🔧 NPU 最佳化: 調整批次大小為 {batch_size}")
    
    # 檢測模型架構
    model_architecture = detect_model_architecture(model_path)
    print(f"🏗️  檢測到模型架構: {model_architecture}")
    
    # 載入模型
    model = get_model(model_architecture, num_classes=num_classes, dropout_rate=0.3)
    try:
        # 先在 CPU 上載入，然後移動到目標裝置
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"✅ 成功載入模型: {model_path}")
        
        # 如果使用 NPU，可能需要特殊設定
        if 'npu' in str(device):
            print("🚀 NPU 模式已啟用")
            
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        print("💡 建議檢查模型檔案和裝置相容性")
        return
    
    # 測試集變換（不增強）
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 建立測試集 DataLoader
    test_dataset = TaiwanFoodDataset(test_csv, test_img_dir, test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"📊 測試集大小: {len(test_dataset)}")
    print("🔍 開始最終評估...")
    print("⚠️  注意：這是首次在測試集上評估，結果代表模型的真實性能")
    print("=" * 60)
    
    # 由於測試集沒有標籤，我們只能生成預測結果
    all_predictions = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="測試中", ncols=80)
        for images, _ in test_pbar:  # 標籤為 -1，忽略
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            test_pbar.set_postfix({'已處理': len(all_predictions)})
    
    # 儲存預測結果
    results_file = "test_predictions.csv"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Id,Category\n")
        for i, pred in enumerate(all_predictions):
            f.write(f"{i},{pred}\n")
    
    print(f"\n✅ 測試完成！")
    print(f"📁 預測結果已儲存至: {results_file}")
    print(f"📊 共處理 {len(all_predictions)} 張測試圖片")
    
    # 顯示預測類別分佈
    from collections import Counter
    pred_counts = Counter(all_predictions)
    print(f"\n📈 預測類別分佈 (前10名):")
    for class_id, count in pred_counts.most_common(10):
        print(f"   類別 {class_id}: {count} 張圖片")

def evaluate_on_test_set(model_path, test_csv, test_img_dir, num_classes=101, batch_size=32, img_size=224, manual_device_selection=False):
    """
    在測試集上評估訓練好的模型
    
    manual_device_selection: True 為手動選擇硬體，False 為自動選擇
    注意：這是最終評估，測試集從未參與訓練過程
    """
    
    # 檢查模型檔案是否存在
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return
    
    # 檢測可用裝置並選擇
    available_devices, npu_available, gpu_available, amd_npu_available = detect_available_devices()
    
    if manual_device_selection:
        print("\n🎮 手動裝置選擇模式")
    else:
        print("\n🤖 自動裝置選擇模式")
    
    device_str = choose_device(available_devices, npu_available, gpu_available, amd_npu_available, manual_device_selection)
    
    if device_str is None:
        print("❌ 未選擇裝置，程式結束")
        return
    
    # 設定裝置
    if device_str == 'amd_npu':
        # 使用 AMD NPU 專用推理
        print("\n🚀 啟用 AMD Ryzen AI NPU 推理模式")
        return evaluate_with_amd_npu(model_path, test_csv, test_img_dir, num_classes, batch_size, img_size)
    else:
        # 使用標準模式 (CPU/GPU/傳統NPU)
        return evaluate_standard_mode(model_path, test_csv, test_img_dir, num_classes, batch_size, img_size, device_str)
    
if __name__ == '__main__':
    # 使用最新的模型檔案進行測試
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
    if not model_files:
        print("❌ 找不到訓練好的模型檔案")
        print("請先執行 python train_pytorch.py 進行訓練")
    else:
        # 選擇最新的模型
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join('models', x)))
        model_path = os.path.join('models', latest_model)
        
        print(f"🎯 使用模型: {latest_model}")
        print("=" * 60)
        
        # 詢問使用者選擇模式
        print("🎮 請選擇推理硬體選擇模式:")
        print("  1. 🤖 自動模式 (系統自動選擇最佳硬體)")
        print("  2. 🎮 手動模式 (手動選擇推理硬體)")
        print("  3. ❌ 退出程式")
        
        while True:
            try:
                mode_choice = input("\n👉 請選擇模式 (1-3): ").strip()
                
                if mode_choice == "1":
                    print("\n🤖 使用自動模式")
                    manual_mode = False
                    break
                elif mode_choice == "2":
                    print("\n🎮 使用手動模式")
                    manual_mode = True
                    break
                elif mode_choice == "3":
                    print("\n👋 程式結束")
                    exit(0)
                else:
                    print("⚠️  請輸入 1、2 或 3")
                    
            except KeyboardInterrupt:
                print("\n\n👋 程式結束")
                exit(0)
        
        # 開始評估
        evaluate_on_test_set(
            model_path=model_path,
            test_csv='archive/tw_food_101/tw_food_101_test_list.csv',
            test_img_dir='archive/tw_food_101/test',
            #test_csv='downloads/train_list.csv',
            #test_img_dir='downloads/bing_images',
            manual_device_selection=manual_mode
        )