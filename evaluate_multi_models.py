#!/usr/bin/env python3
"""
多模型集成評估腳本
自動從 models 目錄抓取多個模型，並使用集成學習進行預測
"""

import torch
import torch.nn as nn
from pytorch_model import get_model
from pytorch_data_loader import TaiwanFoodDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import csv
import time
from collections import Counter
import platform
import subprocess
import numpy as np

def detect_available_devices():
    """檢測可用的計算裝置（NPU/GPU/CPU）"""
    print("\n🔍 檢測可用的計算裝置")
    print("=" * 60)
    
    devices = []
    device_info = []
    
    # 檢測 ONNX Runtime DirectML - NPU 加速推薦方式
    onnx_dml_available = False
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            onnx_dml_available = True
            devices.append(('onnx_dml', 'AMD Ryzen AI NPU (ONNX Runtime DirectML)'))
            device_info.append("✅ ONNX Runtime DirectML 可用")
            device_info.append("   支援: AMD Ryzen AI NPU 硬體加速")
            device_info.append("   系列: Ryzen AI 7040/8040/9HX")
    except ImportError:
        device_info.append("⚠️  ONNX Runtime 未安裝")
        device_info.append("   建議安裝: pip install onnxruntime-directml")
    
    # 檢測 AMD Ryzen AI 處理器
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                  capture_output=True, text=True, timeout=5)
            if 'AMD Ryzen' in result.stdout and 'AI' in result.stdout:
                device_info.append("💻 檢測到 AMD Ryzen AI 處理器")
                if not onnx_dml_available:
                    device_info.append("   ⚠️  NPU 可能可用但未啟用")
                    device_info.append("   請安裝: pip install onnxruntime-directml")
        except:
            pass
    
    # 檢測 Apple MPS (Neural Engine)
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append(('mps', 'Apple Neural Engine (MPS)'))
            device_info.append("🍎 Apple Neural Engine (MPS) 可用")
    except:
        pass
    
    # 檢測 CUDA GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            devices.append((f'cuda:{i}', f'GPU {i}: {gpu_name} ({memory:.1f}GB)'))
            device_info.append(f"✅ GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    # CPU 始終可用
    devices.append(('cpu', 'CPU'))
    device_info.append("💻 CPU 可用")
    
    # 顯示檢測結果
    for info in device_info:
        print(info)
    
    if not devices[:-1]:  # 除了CPU以外沒有其他設備
        print("\n⚠️  沒有檢測到 NPU 或 GPU，將使用 CPU")
        print("💡 提示: 可以安裝 NPU 支援以加速推理")
    
    print("=" * 60)
    return devices

def get_all_models_from_directory(models_dir='models', max_models=None, min_models=2):
    """從目錄中獲取所有模型檔案
    
    Args:
        models_dir: 模型目錄路徑
        max_models: 最多使用的模型數量，None表示使用所有模型
        min_models: 最少需要的模型數量
    
    Returns:
        list: 模型檔案路徑列表
    """
    if not os.path.exists(models_dir):
        print(f"❌ 模型目錄不存在: {models_dir}")
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print(f"❌ 在 {models_dir} 中沒有找到任何 .pth 模型檔案")
        return []
    
    if len(model_files) < min_models:
        print(f"⚠️  只找到 {len(model_files)} 個模型，少於最少需求 {min_models} 個")
        print(f"   建議至少訓練 {min_models} 個不同的模型以獲得更好的集成效果")
    
    # 按修改時間排序（最新的優先）
    model_files = sorted(model_files, 
                        key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), 
                        reverse=True)
    
    if max_models:
        model_files = model_files[:max_models]
    
    model_paths = [os.path.join(models_dir, f) for f in model_files]
    
    print(f"\n📁 在 {models_dir} 中找到 {len(model_paths)} 個模型:")
    print("=" * 60)
    for i, (path, filename) in enumerate(zip(model_paths, model_files), 1):
        file_time = os.path.getmtime(path)
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time))
        file_size = os.path.getsize(path) / 1024 / 1024  # MB
        print(f"   {i}. {filename}")
        print(f"      修改時間: {time_str}, 大小: {file_size:.1f} MB")
    print("=" * 60)
    
    return model_paths

def convert_model_to_onnx(model, model_name, input_shape=(1, 3, 224, 224), output_path=None):
    """將 PyTorch 模型轉換為 ONNX 格式
    
    Args:
        model: PyTorch 模型
        model_name: 模型名稱
        input_shape: 輸入張量形狀
        output_path: ONNX 檔案輸出路徑
    
    Returns:
        onnx_path: ONNX 檔案路徑
    """
    if output_path is None:
        output_path = f"temp_onnx_{model_name}.onnx"
    
    try:
        # 確保模型在 CPU 上
        model = model.cpu()
        model.eval()
        
        # 創建虛擬輸入
        dummy_input = torch.randn(*input_shape)
        
        # 導出為 ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True
        )
        
        return output_path
    
    except Exception as e:
        print(f"   ⚠️  ONNX 轉換失敗: {e}")
        return None

def warmup_onnx_session(session, input_shape):
    """預熱 ONNX 會話以優化首次推理性能
    
    Args:
        session: ONNX Runtime 會話
        input_shape: 輸入形狀 (batch_size, channels, height, width)
    """
    try:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        # 執行幾次預熱推理
        for _ in range(3):
            _ = session.run(None, {input_name: dummy_input})
    except:
        pass  # 預熱失敗不影響正常使用

def create_onnx_session(onnx_path, use_dml=True, enable_profiling=False):
    """創建優化的 ONNX Runtime 推理會話（NPU 加速優化）
    
    Args:
        onnx_path: ONNX 模型路徑
        use_dml: 是否使用 DirectML 執行提供者
    
    Returns:
        session: ONNX Runtime 推理會話
    """
    try:
        import onnxruntime as ort
        
        # 創建會話選項 - 優化配置
        sess_options = ort.SessionOptions()
        
        # 圖優化等級 - 使用最高優化（擴展優化）
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        
        # 啟用記憶體優化模式
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        sess_options.enable_cpu_mem_arena = True
        
        # 並行執行優化
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.inter_op_num_threads = 2  # 操作間並行
        sess_options.intra_op_num_threads = 4  # 操作內並行
        
        # 性能分析（可選）
        if enable_profiling:
            sess_options.enable_profiling = True
        
        # 設置執行提供者 - DirectML 優化配置
        providers = []
        provider_options = []
        
        if use_dml:
            # DirectML 提供者配置（針對 AMD Ryzen AI NPU 優化）
            dml_options = {
                'device_id': 0,  # 使用第一個 NPU 設備
                'disable_metacommands': False,  # 啟用 metacommands 加速
                'enable_dynamic_graph_fusion': True,  # 啟用動態圖融合
            }
            providers.append('DmlExecutionProvider')
            provider_options.append(dml_options)
        
        # CPU 作為回退
        providers.append('CPUExecutionProvider')
        provider_options.append({})
        
        # 創建推理會話
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options
        )
        
        return session
    
    except Exception as e:
        print(f"   ⚠️  創建 ONNX 會話失敗: {e}")
        return None

def ensemble_predict_onnx(onnx_sessions, images, strategy='weighted_average'):
    """使用 ONNX Runtime 進行多模型集成預測（NPU 加速）
    
    Args:
        onnx_sessions: ONNX 推理會話列表 [(session, weight, name), ...]
        images: 輸入圖片張量 (PyTorch)
        strategy: 集成策略
    
    Returns:
        predictions: 預測類別
        confidences: 預測信心度
        details: 詳細資訊
    """
    all_outputs = []
    all_predictions = []
    all_confidences = []
    successful_models = []
    
    # 轉換 PyTorch 張量為 NumPy（連續記憶體布局）
    if images.is_cuda:
        images_np = images.cpu().numpy()
    else:
        images_np = images.numpy()
    
    # 確保連續記憶體布局（優化性能）
    if not images_np.flags['C_CONTIGUOUS']:
        images_np = np.ascontiguousarray(images_np)
    
    # 確保正確的數據類型
    images_np = images_np.astype(np.float32)
    
    # 收集所有模型的預測（並行推理優化）
    for session, weight, name in onnx_sessions:
        try:
            # ONNX Runtime 推理（使用 NPU 加速）
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: images_np})[0]
            
            # 轉換回 PyTorch 張量進行後處理
            outputs_torch = torch.from_numpy(outputs).float()
            probs = torch.softmax(outputs_torch, dim=1)
            max_probs, predicted = probs.max(1)
            
            all_outputs.append(outputs_torch * weight)
            all_predictions.append(predicted)
            all_confidences.append(max_probs)
            successful_models.append(name)
            
        except Exception as e:
            # 靜默跳過失敗的模型
            continue
    
    # 確保至少有一個模型成功
    if not all_outputs:
        raise RuntimeError("所有模型都無法進行預測")
    
    # 單模型情況
    if len(all_outputs) == 1:
        probs = torch.softmax(all_outputs[0], dim=1)
        max_probs, predictions = probs.max(1)
        details = {
            'individual_predictions': [all_predictions[0].numpy()],
            'individual_confidences': [all_confidences[0].numpy()],
            'model_names': successful_models
        }
        return predictions, max_probs, details
    
    # 集成策略
    if strategy == 'weighted_average':
        ensemble_outputs = all_outputs[0]
        for output in all_outputs[1:]:
            ensemble_outputs = ensemble_outputs + output
        probs = torch.softmax(ensemble_outputs, dim=1)
        max_probs, predictions = probs.max(1)
        
    elif strategy == 'voting':
        predictions_stack = torch.stack(all_predictions)
        predictions = torch.mode(predictions_stack, dim=0)[0]
        max_probs = torch.stack(all_confidences).mean(0)
        
    elif strategy == 'max_confidence':
        all_confidences_stack = torch.stack(all_confidences)
        max_conf_indices = all_confidences_stack.argmax(0)
        
        predictions = torch.zeros_like(all_predictions[0])
        max_probs = torch.zeros_like(all_confidences[0])
        
        for i in range(len(predictions)):
            best_model_idx = max_conf_indices[i]
            predictions[i] = all_predictions[best_model_idx][i]
            max_probs[i] = all_confidences[best_model_idx][i]
    
    # 返回詳細資訊
    details = {
        'individual_predictions': [p.numpy() for p in all_predictions],
        'individual_confidences': [c.numpy() for c in all_confidences],
        'model_names': successful_models
    }
    
    return predictions, max_probs, details

def detect_model_architecture(model_path):
    """從模型檔案名稱中檢測模型架構"""
    filename = os.path.basename(model_path).lower()
    
    if 'efficientnet_b3' in filename or 'efficientnet' in filename:
        return 'efficientnet_b3'
    elif 'convnext_tiny' in filename or 'convnext' in filename:
        return 'convnext_tiny'
    elif 'regnet_y' in filename or 'regnet' in filename:
        return 'regnet_y'
    elif 'vit' in filename or 'vision_transformer' in filename:
        return 'vit'
    elif 'resnet50' in filename or 'resnet' in filename:
        return 'resnet50'
    else:
        # 嘗試從模型內容檢測
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            keys = list(state_dict.keys())
            
            if any('features' in key for key in keys):
                if any('block' in key for key in keys):
                    return 'efficientnet_b3'
                elif any('stages' in key for key in keys):
                    return 'convnext_tiny'
            elif any('layer' in key for key in keys):
                return 'resnet50'
            elif any('blocks' in key for key in keys):
                return 'vit'
        except:
            pass
        
        print(f"⚠️  無法從檔案名稱檢測模型架構: {filename}")
        print("   使用預設架構: ResNet50")
        return 'resnet50'

def ensemble_predict(models_info, images, device, strategy='weighted_average'):
    """多模型集成預測
    
    Args:
        models_info: 模型資訊列表 [(model, weight, name), ...]
        images: 輸入圖片張量
        device: 計算設備
        strategy: 集成策略 ('weighted_average', 'voting', 'max_confidence')
    
    Returns:
        predictions: 預測類別
        confidences: 預測信心度
        details: 每個模型的預測詳情
    """
    all_outputs = []
    all_predictions = []
    all_confidences = []
    all_probs = []
    successful_models = []  # 記錄成功的模型
    
    # 收集所有模型的預測
    for model, weight, name in models_info:
        with torch.no_grad():
            try:
                outputs = model(images.to(device))
                probs = torch.softmax(outputs, dim=1)
                max_probs, predicted = probs.max(1)
                
                all_outputs.append(outputs * weight)
                all_predictions.append(predicted)
                all_confidences.append(max_probs)
                all_probs.append(probs)
                successful_models.append(name)
                
            except Exception as e:
                # 靜默跳過失敗的模型
                continue
    
    # 確保至少有一個模型成功預測
    if not all_outputs:
        raise RuntimeError("所有模型都無法進行預測")
    
    # 如果只有一個模型成功，直接使用它的結果
    if len(all_outputs) == 1:
        probs = torch.softmax(all_outputs[0], dim=1)
        max_probs, predictions = probs.max(1)
        details = {
            'individual_predictions': [all_predictions[0].cpu().numpy() if all_predictions[0].is_cuda else all_predictions[0].numpy()],
            'individual_confidences': [all_confidences[0].cpu().numpy() if all_confidences[0].is_cuda else all_confidences[0].numpy()],
            'model_names': successful_models
        }
        return predictions, max_probs, details
    
    # 所有集成計算在CPU上進行（已經是CPU張量）
    try:
        if strategy == 'weighted_average':
            # 策略1: 加權平均所有模型的輸出
            ensemble_outputs = all_outputs[0]
            for output in all_outputs[1:]:
                ensemble_outputs = ensemble_outputs + output
            probs = torch.softmax(ensemble_outputs, dim=1)
            max_probs, predictions = probs.max(1)
            
        elif strategy == 'voting':
            # 策略2: 投票法（每個模型一票）
            predictions_stack = torch.stack(all_predictions)
            predictions = torch.mode(predictions_stack, dim=0)[0]
            max_probs = torch.stack(all_confidences).mean(0)
            
        elif strategy == 'max_confidence':
            # 策略3: 選擇最高信心度的預測
            all_confidences_stack = torch.stack(all_confidences)
            max_conf_indices = all_confidences_stack.argmax(0)
            
            predictions = torch.zeros_like(all_predictions[0])
            max_probs = torch.zeros_like(all_confidences[0])
            
            for i in range(len(predictions)):
                best_model_idx = max_conf_indices[i]
                predictions[i] = all_predictions[best_model_idx][i]
                max_probs[i] = all_confidences[best_model_idx][i]
    
    except Exception as e:
        print(f"\n❌ 集成策略 '{strategy}' 執行失敗: {e}")
        print(f"   成功的模型數: {len(successful_models)}")
        print(f"   輸出張量數: {len(all_outputs)}")
        if all_outputs:
            print(f"   輸出張量形狀: {all_outputs[0].shape}")
        raise
    
    # 返回詳細資訊
    details = {
        'individual_predictions': [p.cpu().numpy() if p.is_cuda else p.numpy() for p in all_predictions],
        'individual_confidences': [c.cpu().numpy() if c.is_cuda else c.numpy() for c in all_confidences],
        'model_names': successful_models
    }
    
    return predictions, max_probs, details

def evaluate_multi_models(model_paths, test_csv, test_img_dir, num_classes=101, 
                          batch_size=32, img_size=224, device_str='cpu', 
                          strategy='weighted_average', use_onnx_npu=False):
    """使用多個模型進行集成評估
    
    Args:
        model_paths: 模型檔案路徑列表
        test_csv: 測試集CSV檔案
        test_img_dir: 測試集圖片目錄
        num_classes: 類別數量
        batch_size: 批次大小
        img_size: 圖片大小
        device_str: 計算設備
        strategy: 集成策略
        use_onnx_npu: 是否使用 ONNX Runtime DirectML 進行 NPU 加速
    """
    print(f"\n🎯 多模型集成評估模式")
    print(f"📊 使用 {len(model_paths)} 個模型進行集成預測")
    print(f"🎲 集成策略: {strategy}")
    if use_onnx_npu:
        print(f"🚀 NPU 加速: ONNX Runtime DirectML")
    print("=" * 60)
    
    # 設定裝置
    # 注意：當 use_onnx_npu=True 時，PyTorch 模型使用 CPU，推理由 ONNX Runtime 在 NPU 上執行
    if use_onnx_npu:
        device = torch.device('cpu')
        print(f"💻 PyTorch 使用裝置: CPU (模型載入用)")
        print(f"🚀 ONNX Runtime 將使用: NPU (DirectML 推理)")
    else:
        # PyTorch 模式
        try:
            if isinstance(device_str, str):
                if device_str.startswith('cuda'):
                    device = torch.device(device_str)
                elif device_str == 'mps':
                    device = torch.device('mps')
                elif device_str.startswith('npu'):
                    device = torch.device(device_str)
                else:
                    device = torch.device('cpu')
            else:
                device = device_str
        except Exception as e:
            print(f"⚠️  設備初始化失敗: {e}，使用 CPU")
            device = torch.device('cpu')
        
        print(f"💻 使用裝置: {device}")
    
    # 顯示設備詳細資訊
    if not use_onnx_npu:
        if str(device).startswith('cuda'):
            gpu_id = int(str(device).split(':')[1]) if ':' in str(device) else 0
            print(f"🚀 GPU: {torch.cuda.get_device_name(gpu_id)}")
            print(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
        elif str(device) == 'mps':
            print(f"🍎 使用 Apple Neural Engine (MPS)")
    print("=" * 60)
    
    # 載入所有模型
    models_info = []
    onnx_sessions = []  # 用於ONNX Runtime模式
    
    print(f"\n📦 開始載入 {len(model_paths)} 個模型...")
    if use_onnx_npu:
        print("🚀 模式: ONNX Runtime DirectML (NPU加速)")
    print("=" * 60)
    
    for i, model_path in enumerate(model_paths, 1):
        model_name = os.path.basename(model_path)
        print(f"\n📦 [{i}/{len(model_paths)}] 載入: {model_name}")
        
        # 檢測模型架構
        architecture = detect_model_architecture(model_path)
        print(f"   🏗️  架構: {architecture}")
        
        # 建立模型
        try:
            model = get_model(architecture, num_classes=num_classes, dropout_rate=0.3)
            
            # 載入權重到CPU
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            model = model.cpu()
            model.eval()
            
            # 計算模型權重
            weight = 1.0 / len(model_paths)
            
            # 如果使用ONNX Runtime NPU加速
            if use_onnx_npu:
                try:
                    # 轉換為ONNX
                    print(f"   🔄 轉換為 ONNX 格式...")
                    onnx_path = convert_model_to_onnx(
                        model, 
                        model_name.replace('.pth', ''),
                        input_shape=(batch_size, 3, img_size, img_size)
                    )
                    
                    if onnx_path:
                        # 創建 ONNX Runtime 會話
                        session = create_onnx_session(onnx_path, use_dml=True)
                        if session:
                            # 檢查執行提供者
                            providers = session.get_providers()
                            if 'DmlExecutionProvider' in providers:
                                print(f"   � ONNX Runtime 已啟用 DirectML (NPU加速)")
                            else:
                                print(f"   💻 ONNX Runtime 使用 CPU")
                            
                            onnx_sessions.append((session, weight, model_name))
                            print(f"   ✅ ONNX 轉換成功 (權重: {weight:.4f})")
                        else:
                            print(f"   ⚠️  ONNX 會話創建失敗，跳過此模型")
                    else:
                        print(f"   ⚠️  ONNX 轉換失敗，跳過此模型")
                        
                except Exception as e:
                    print(f"   ❌ ONNX 處理失敗: {e}")
                    print(f"   ⚠️  跳過此模型")
                    continue
            else:
                # PyTorch 模式（非 NPU）
                model = model.to(device)
                models_info.append((model, weight, model_name))
                print(f"   ✅ 載入成功 (權重: {weight:.4f})")
            
        except Exception as e:
            print(f"   ❌ 載入失敗: {e}")
            print(f"   ⚠️  跳過此模型")
            continue
    
    # 檢查是否有成功載入的模型
    if use_onnx_npu:
        if not onnx_sessions:
            print("\n❌ 沒有成功載入任何 ONNX 模型")
            return
        if len(onnx_sessions) < 2:
            print(f"\n⚠️  只成功載入 {len(onnx_sessions)} 個模型")
            print("   集成效果可能有限，建議使用至少2個模型")
        print(f"\n✅ 成功載入 {len(onnx_sessions)} 個 ONNX 模型（NPU加速）")
    else:
        if not models_info:
            print("\n❌ 沒有成功載入任何模型")
            return
        if len(models_info) < 2:
            print(f"\n⚠️  只成功載入 {len(models_info)} 個模型")
            print("   集成效果可能有限，建議使用至少2個模型")
        print(f"\n✅ 成功載入 {len(models_info)} 個模型")
    print("=" * 60)
    
    # 資料轉換
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # NPU 批次大小優化建議
    if use_onnx_npu:
        # NPU 通常在較大批次下性能更好
        original_batch_size = batch_size
        if batch_size < 32:
            batch_size = 32
            print(f"\n💡 NPU 優化: 批次大小從 {original_batch_size} 調整為 {batch_size}")
            print(f"   較大批次能更好利用 NPU 並行計算能力")
    
    # 建立測試集 DataLoader
    print("\n📊 載入測試集資料...")
    test_dataset = TaiwanFoodDataset(test_csv, test_img_dir, test_transform, is_test=True)
    
    # NPU 優化：使用 pin_memory 加速數據傳輸
    pin_memory = use_onnx_npu
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=pin_memory
    )
    
    print(f"   測試集大小: {len(test_dataset)} 張圖片")
    print(f"   批次大小: {batch_size}")
    if use_onnx_npu:
        print(f"   NPU 優化: 已啟用記憶體固定 (pin_memory)")
    print("=" * 60)
    
    # 執行集成預測
    print(f"\n🔍 開始多模型集成評估...")
    print(f"   策略: {strategy}")
    
    all_predictions = []
    all_confidences = []
    all_details = []
    
    start_time = time.time()
    
    # 追蹤失敗的模型（只在第一批顯示）
    first_batch_failed_models = set()
    first_batch_processed = False
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="集成預測中", ncols=100)
        for batch_idx, (images, _) in enumerate(test_pbar):
            
            # 根據模式選擇預測方法
            try:
                if use_onnx_npu:
                    # ONNX Runtime DirectML 模式（NPU加速）
                    predictions, confidences, details = ensemble_predict_onnx(
                        onnx_sessions, images, strategy
                    )
                    
                    # 第一批時顯示信息
                    if not first_batch_processed:
                        print(f"\n✅ 使用 {len(details['model_names'])} 個模型進行 NPU 加速推理\n")
                        first_batch_processed = True
                        
                else:
                    # PyTorch 模式
                    is_directml = 'privateuseone' in str(device).lower() or 'dml' in str(device).lower()
                    
                    if not is_directml:
                        images = images.to(device)
                    
                    predictions, confidences, details = ensemble_predict(
                        models_info, images, device, strategy
                    )
                    
                    # 第一批時檢查哪些模型失敗了
                    if not first_batch_processed:
                        all_model_names = [name for _, _, name in models_info]
                        successful_models = details['model_names']
                        failed_models = set(all_model_names) - set(successful_models)
                        
                        if failed_models:
                            print(f"\n⚠️  以下模型無法在DirectML上運行，已跳過:")
                            for model_name in failed_models:
                                print(f"   - {model_name}")
                            print(f"\n✅ 使用 {len(successful_models)} 個模型繼續集成預測\n")
                        
                        first_batch_processed = True
                
            except Exception as e:
                print(f"\n❌ 集成預測失敗: {type(e).__name__}")
                print(f"   錯誤信息: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_details.append(details)
            
            # 更新進度條
            avg_conf = sum(all_confidences) / len(all_confidences)
            test_pbar.set_postfix({
                '已處理': len(all_predictions),
                '平均信心': f'{avg_conf:.3f}'
            })
    
    elapsed_time = time.time() - start_time
    
    # 儲存預測結果
    results_file = f"test_predictions_ensemble_{strategy}.csv"
    print(f"\n💾 儲存預測結果...")
    
    with open(results_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category', 'Confidence'])
        for i, (pred, conf) in enumerate(zip(all_predictions, all_confidences)):
            writer.writerow([i, int(pred), f'{conf:.6f}'])
    
    # 統計結果
    print("\n" + "=" * 60)
    print("📊 集成評估結果統計")
    print("=" * 60)
    print(f"✅ 評估完成！")
    print(f"📁 結果檔案: {results_file}")
    print(f"📊 處理圖片: {len(all_predictions)} 張")
    print(f"⏱️  總耗時: {elapsed_time:.2f} 秒")
    print(f"⚡ 速度: {len(all_predictions)/elapsed_time:.2f} 張/秒")
    
    avg_confidence = sum(all_confidences) / len(all_confidences)
    print(f"\n📈 平均信心度: {avg_confidence:.4f}")
    print(f"📈 最高信心度: {max(all_confidences):.4f}")
    print(f"📉 最低信心度: {min(all_confidences):.4f}")
    
    # 顯示預測類別分佈
    pred_counts = Counter(all_predictions)
    print(f"\n📊 預測類別分佈 (前10名):")
    for class_id, count in pred_counts.most_common(10):
        percentage = count / len(all_predictions) * 100
        print(f"   類別 {class_id:3d}: {count:4d} 張 ({percentage:5.2f}%)")
    
    # 顯示信心度分佈
    conf_ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0)]
    print(f"\n📊 信心度分佈:")
    for low, high in conf_ranges:
        count = sum(1 for c in all_confidences if low <= c < high)
        percentage = count / len(all_confidences) * 100
        print(f"   {low:.2f} ~ {high:.2f}: {count:4d} 張 ({percentage:5.2f}%)")
    
    # 標記低信心度預測
    low_conf_threshold = 0.5
    low_conf_predictions = [(i, pred, conf) for i, (pred, conf) in enumerate(zip(all_predictions, all_confidences)) 
                            if conf < low_conf_threshold]
    
    if low_conf_predictions:
        print(f"\n⚠️  信心度低於 {low_conf_threshold} 的預測: {len(low_conf_predictions)} 張 ({len(low_conf_predictions)/len(all_predictions)*100:.1f}%)")
        print(f"   建議人工檢查這些預測")
        
        # 儲存低信心度預測清單
        low_conf_file = f"low_confidence_predictions_{strategy}.csv"
        with open(low_conf_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ImageId', 'PredictedClass', 'Confidence'])
            for img_id, pred, conf in low_conf_predictions[:20]:  # 只顯示前20個
                writer.writerow([img_id, int(pred), f'{conf:.6f}'])
        print(f"   詳細清單已儲存至: {low_conf_file}")
    else:
        print(f"\n✅ 太棒了！所有預測的信心度都高於 {low_conf_threshold}")
    
    print("=" * 60)
    
    return all_predictions, all_confidences

def main():
    """主程式"""
    print("=" * 60)
    print("🎯 多模型集成評估工具")
    print("=" * 60)
    
    # 獲取所有模型
    model_paths = get_all_models_from_directory(models_dir='models', max_models=None, min_models=2)
    
    if len(model_paths) < 2:
        print("\n❌ 需要至少2個模型才能進行集成評估")
        print("💡 請先訓練多個不同的模型")
        return
    
    # 選擇集成策略
    print("\n🎲 請選擇集成策略:")
    print("=" * 60)
    print("1. 加權平均 (Weighted Average) - 推薦")
    print("   所有模型的輸出加權平均，平衡各模型意見")
    print("\n2. 投票法 (Voting)")
    print("   每個模型投一票，多數決定最終預測")
    print("\n3. 最高信心度 (Max Confidence)")
    print("   選擇信心度最高的模型的預測結果")
    print("=" * 60)
    
    strategy_map = {
        '1': 'weighted_average',
        '2': 'voting',
        '3': 'max_confidence'
    }
    
    while True:
        try:
            choice = input("\n請選擇策略 (1-3) [預設=1]: ").strip()
            if choice == '':
                choice = '1'
            
            if choice in strategy_map:
                strategy = strategy_map[choice]
                print(f"✅ 選擇策略: {strategy}")
                break
            else:
                print("⚠️  請輸入 1、2 或 3")
        except KeyboardInterrupt:
            print("\n\n👋 程式結束")
            return
    
    # 檢測並選擇計算設備
    available_devices = detect_available_devices()
    
    print("\n💻 請選擇計算設備:")
    print("=" * 60)
    
    for i, (device_id, device_name) in enumerate(available_devices, 1):
        prefix = "🚀" if 'npu' in device_id.lower() or 'dml' in device_id.lower() or 'mps' in device_id.lower() else \
                 "⚡" if 'cuda' in device_id else "💻"
        suffix = " - 推薦" if i == 1 and device_id != 'cpu' else ""
        print(f"{i}. {prefix} {device_name}{suffix}")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"\n請選擇設備 (1-{len(available_devices)}) [預設=1]: ").strip()
            if choice == '':
                choice = '1'
            
            idx = int(choice) - 1
            if 0 <= idx < len(available_devices):
                device_str, device_name = available_devices[idx]
                print(f"✅ 選擇設備: {device_name}")
                break
            else:
                print(f"⚠️  請輸入 1-{len(available_devices)}")
        except ValueError:
            print("⚠️  請輸入有效數字")
        except KeyboardInterrupt:
            print("\n\n👋 程式結束")
            return
    
    # 如果選擇了 ONNX DML 設備，詢問是否使用 NPU 加速
    use_onnx_npu = False
    if device_str == 'onnx_dml':
        print("\n🚀 AMD Ryzen AI NPU 加速")
        print("=" * 60)
        print("是否啟用 ONNX Runtime DirectML NPU 硬體加速？")
        print("1. ✅ 是 - 使用 ONNX Runtime DirectML (推薦)")
        print("2. ❌ 否 - 使用 PyTorch CPU 模式")
        print("=" * 60)
        
        while True:
            try:
                npu_choice = input("\n請選擇 (1-2) [預設=1]: ").strip()
                if npu_choice == '' or npu_choice == '1':
                    use_onnx_npu = True
                    print("✅ 已啟用 ONNX Runtime NPU 加速")
                    print("💡 模型將轉換為 ONNX 格式並在 NPU 上執行")
                    device_str = 'cpu'  # PyTorch 端使用 CPU 載入模型
                    break
                elif npu_choice == '2':
                    use_onnx_npu = False
                    print("✅ 使用 PyTorch CPU 模式")
                    device_str = 'cpu'
                    break
                else:
                    print("⚠️  請輸入 1 或 2")
            except KeyboardInterrupt:
                print("\n\n👋 程式結束")
                return
    
    # 開始評估
    print("\n" + "=" * 60)
    print("🚀 開始多模型集成評估")
    print("=" * 60)
    
    evaluate_multi_models(
        model_paths=model_paths,
        test_csv='archive/tw_food_101/tw_food_101_test_list.csv',
        test_img_dir='archive/tw_food_101/test',
        num_classes=101,
        batch_size=32,
        img_size=224,
        device_str=device_str,
        strategy=strategy,
        use_onnx_npu=use_onnx_npu
    )
    
    print("\n" + "=" * 60)
    print("🎉 評估完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()