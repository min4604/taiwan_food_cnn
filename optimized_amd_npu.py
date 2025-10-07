import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import concurrent.futures
import threading
from queue import Queue
import time

class OptimizedAMDNPUInference:
    """
    最佳化的 AMD Ryzen AI 9HX NPU 推理類
    專注於提高 NPU 使用率和並行處理
    """
    
    def __init__(self, model_path, img_size=224, batch_size=16, num_threads=4):
        self.img_size = img_size
        self.batch_size = batch_size  # NPU 批次處理大小
        self.num_threads = num_threads  # 並行處理執行緒數
        self.onnx_model_path = None
        self.ort_session = None
        self.pytorch_model = None
        
        # 設定轉換
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 執行緒安全的佇列系統
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.processing_thread = None
        self.shutdown_flag = threading.Event()
        
        # 初始化模型
        self._setup_model(model_path)
        
        # 啟動後台處理執行緒
        self._start_background_processing()
    
    def _detect_model_architecture(self, model_path):
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
            # 嘗試從模型內容檢測
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                keys = list(state_dict.keys())
                
                # 根據 state_dict 的鍵值檢測架構
                if any('features' in key for key in keys):
                    if any('block' in key for key in keys):
                        return 'efficientnet_b3'  # EfficientNet 特徵
                    elif any('stages' in key for key in keys):
                        return 'convnext_tiny'    # ConvNeXt 特徵
                    else:
                        return 'efficientnet_b3'  # 預設為 EfficientNet
                elif any('layer' in key for key in keys):
                    return 'resnet50'             # ResNet 特徵
                elif any('blocks' in key for key in keys):
                    return 'vit'                  # ViT 特徵
                else:
                    return 'resnet50'             # 預設為 ResNet50
            except:
                return 'resnet50'                 # 預設為 ResNet50
    
    def _setup_model(self, pytorch_model_path):
        """設定 AMD NPU 推理模型，針對最大使用率最佳化"""
        try:
            # 檢查是否支援 ONNX Runtime DirectML
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'DmlExecutionProvider' not in providers:
                raise RuntimeError("DirectML 不可用，無法使用 AMD NPU")
            
            print("✅ 檢測到 DirectML 支援")
            print("🚀 為 AMD Ryzen AI 9 HX 370 NPU 最大使用率最佳化...")
            
            # 轉換 PyTorch 模型為 ONNX (如果需要)
            onnx_path = self._convert_to_onnx_optimized(pytorch_model_path)
            
            # 建立 ONNX Runtime 會話，專為最大 NPU 使用率最佳化
            providers = [
                ('DmlExecutionProvider', {
                    'device_id': 0,
                    'enable_dynamic_shapes': True,
                    'disable_memory_arena': False,  # 啟用記憶體池
                    'enable_graph_capture': True,  # 啟用圖形捕獲
                    'enable_dynamic_shapes': True,
                    'force_sequential_execution': False,  # 允許並行執行
                }),
                'CPUExecutionProvider'   # CPU 作為備援
            ]
            
            # 建立會話選項以最大化效能
            session_options = ort.SessionOptions()
            
            # 效能最佳化設定
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_reuse = True
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 並行設定 - 最大化 NPU 使用率
            session_options.intra_op_num_threads = self.num_threads * 2  # 增加內部並行
            session_options.inter_op_num_threads = self.num_threads      # 增加操作間並行
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 並行執行模式
            
            # 記憶體最佳化
            session_options.enable_profiling = False  # 關閉分析以提高效能
            session_options.log_severity_level = 3   # 減少日誌輸出
            
            self.ort_session = ort.InferenceSession(
                onnx_path, 
                providers=providers,
                sess_options=session_options
            )
            
            print(f"🚀 NPU 最佳化推理會話已建立")
            print(f"📊 使用提供者: {self.ort_session.get_providers()}")
            print(f"⚡ 批次大小: {self.batch_size}")
            print(f"🔄 並行執行緒: {self.num_threads}")
            
            # 預熱 NPU
            self._warmup_npu()
            
        except ImportError:
            print("❌ ONNX Runtime 未安裝，請執行 install_amd_npu.bat")
            self._setup_cpu_fallback(pytorch_model_path)
        except Exception as e:
            print(f"❌ AMD NPU 設定失敗: {e}")
            print("💡 將回退到 CPU 推理")
            self._setup_cpu_fallback(pytorch_model_path)
    
    def _convert_to_onnx_optimized(self, pytorch_model_path):
        """將 PyTorch 模型轉換為最佳化的 ONNX 格式"""
        onnx_path = pytorch_model_path.replace('.pth', '_optimized_npu.onnx')
        
        if os.path.exists(onnx_path):
            print(f"📁 使用現有的最佳化 ONNX 模型: {onnx_path}")
            return onnx_path
        
        print("🔄 轉換 PyTorch 模型為最佳化 ONNX...")
        
        try:
            # 檢測模型架構
            model_architecture = self._detect_model_architecture(pytorch_model_path)
            print(f"🏗️  檢測到模型架構: {model_architecture}")
            
            # 載入對應的模型類型
            from pytorch_model import get_model
            model = get_model(model_architecture, num_classes=101, dropout_rate=0.3)
            model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
            model.eval()
            
            # 建立支援批次處理的範例輸入
            dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
            
            # 匯出為最佳化的 ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,  # 使用較新的 opset 版本
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                # 最佳化設定
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False
            )
            
            print(f"✅ 最佳化 ONNX 模型已儲存: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX 轉換失敗: {e}")
            raise
    
    def _warmup_npu(self):
        """預熱 NPU 以確保最佳效能"""
        if self.ort_session:
            print("🔥 預熱 AMD NPU...")
            try:
                # 建立虛擬輸入進行預熱
                dummy_input = np.random.randn(self.batch_size, 3, self.img_size, self.img_size).astype(np.float32)
                input_name = self.ort_session.get_inputs()[0].name
                
                # 執行幾次預熱推理
                for i in range(3):
                    start_time = time.time()
                    _ = self.ort_session.run(None, {input_name: dummy_input})
                    elapsed = time.time() - start_time
                    print(f"   預熱 {i+1}/3: {elapsed:.3f}s")
                
                print("✅ NPU 預熱完成，效能最佳化")
                
            except Exception as e:
                print(f"⚠️  NPU 預熱失敗: {e}")
    
    def _start_background_processing(self):
        """啟動後台批次處理執行緒"""
        if self.ort_session:
            self.processing_thread = threading.Thread(target=self._batch_processing_worker, daemon=True)
            self.processing_thread.start()
            print("🔄 後台批次處理執行緒已啟動")
    
    def _batch_processing_worker(self):
        """後台批次處理工作執行緒"""
        while not self.shutdown_flag.is_set():
            batch_items = []
            
            # 收集批次資料
            for _ in range(self.batch_size):
                try:
                    if self.shutdown_flag.is_set():
                        break
                    item = self.input_queue.get(timeout=0.1)
                    batch_items.append(item)
                except:
                    break
            
            if batch_items:
                self._process_batch(batch_items)
    
    def _process_batch(self, batch_items):
        """處理一個批次的資料"""
        try:
            # 準備批次輸入
            batch_input = np.stack([item['input'] for item in batch_items])
            
            # NPU 批次推理
            input_name = self.ort_session.get_inputs()[0].name
            start_time = time.time()
            result = self.ort_session.run(None, {input_name: batch_input})
            inference_time = time.time() - start_time
            
            output = result[0]
            predictions = np.argmax(output, axis=1)
            
            # 將結果放回輸出佇列
            for i, item in enumerate(batch_items):
                self.output_queue.put({
                    'id': item['id'],
                    'prediction': predictions[i],
                    'inference_time': inference_time / len(batch_items)
                })
                
            print(f"⚡ 批次推理完成: {len(batch_items)} 張圖片, {inference_time:.3f}s")
            
        except Exception as e:
            print(f"❌ 批次處理失敗: {e}")
            # 回退到單張處理
            for item in batch_items:
                self.output_queue.put({
                    'id': item['id'],
                    'prediction': -1,
                    'inference_time': 0
                })
    
    def predict_image_batch(self, image_paths):
        """批次處理多張圖片以最大化 NPU 使用率"""
        if not self.ort_session:
            return self._predict_cpu_fallback(image_paths)
        
        print(f"🚀 NPU 批次推理: {len(image_paths)} 張圖片")
        start_time = time.time()
        
        # 並行載入和預處理圖片
        def preprocess_image(img_path_id):
            img_path, img_id = img_path_id
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0)
                return {
                    'id': img_id,
                    'input': input_tensor.numpy()[0],
                    'path': img_path
                }
            except Exception as e:
                print(f"⚠️  圖片預處理失敗 {img_path}: {e}")
                return None
        
        # 使用多執行緒預處理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            image_path_ids = [(path, i) for i, path in enumerate(image_paths)]
            preprocessed = list(executor.map(preprocess_image, image_path_ids))
        
        # 過濾失敗的項目
        valid_items = [item for item in preprocessed if item is not None]
        
        if not valid_items:
            print("❌ 沒有有效的圖片可處理")
            return []
        
        # 批次推理
        predictions = []
        
        for i in range(0, len(valid_items), self.batch_size):
            batch = valid_items[i:i + self.batch_size]
            batch_input = np.stack([item['input'] for item in batch])
            
            try:
                input_name = self.ort_session.get_inputs()[0].name
                batch_start = time.time()
                result = self.ort_session.run(None, {input_name: batch_input})
                batch_time = time.time() - batch_start
                
                output = result[0]
                # 計算 softmax 以獲得信心分數
                batch_softmax = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                batch_predictions = np.argmax(output, axis=1)
                batch_confidences = np.max(batch_softmax, axis=1)
                
                for j, item in enumerate(batch):
                    predictions.append({
                        'id': item['id'],
                        'path': item['path'],
                        'prediction': int(batch_predictions[j]),
                        'confidence': float(batch_confidences[j]),
                        'inference_time': batch_time / len(batch)
                    })
                
                print(f"⚡ 批次 {i//self.batch_size + 1}: {len(batch)} 張圖片, {batch_time:.3f}s")
                
            except Exception as e:
                print(f"❌ 批次推理失敗: {e}")
                for item in batch:
                    predictions.append({
                        'id': item['id'],
                        'path': item['path'],
                        'prediction': -1,
                        'confidence': 0.0,
                        'inference_time': 0
                    })
        
        total_time = time.time() - start_time
        throughput = len(valid_items) / total_time
        
        print(f"🎉 NPU 批次推理完成!")
        print(f"📊 總時間: {total_time:.3f}s")
        print(f"⚡ 吞吐量: {throughput:.1f} 圖片/秒")
        print(f"🚀 NPU 使用率: 最佳化批次處理")
        
        return predictions
    
    def predict_image(self, image_path):
        """單張圖片推理 (相容性方法)"""
        results = self.predict_image_batch([image_path])
        return results[0]['prediction'] if results else -1
    
    def _predict_cpu_fallback(self, image_paths):
        """CPU 備援推理"""
        if not self.pytorch_model:
            return []
        
        print("🔄 使用 CPU 備援推理...")
        predictions = []
        
        for i, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.pytorch_model(input_tensor)
                    # 計算 softmax 以獲得信心分數
                    softmax_output = torch.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(softmax_output, dim=1)
                
                predictions.append({
                    'id': i,
                    'path': img_path,
                    'prediction': predicted_class.item(),
                    'confidence': confidence.item(),
                    'inference_time': 0
                })
                
            except Exception as e:
                print(f"❌ CPU 推理失敗 {img_path}: {e}")
                predictions.append({
                    'id': i,
                    'path': img_path,
                    'prediction': -1,
                    'confidence': 0.0,
                    'inference_time': 0
                })
        
        return predictions
    
    def _setup_cpu_fallback(self, pytorch_model_path):
        """設定 CPU 備援推理"""
        try:
            # 檢測模型架構
            model_architecture = self._detect_model_architecture(pytorch_model_path)
            print(f"🏗️  CPU 備援模式檢測到架構: {model_architecture}")
            
            # 載入對應的模型類型
            from pytorch_model import get_model
            self.pytorch_model = get_model(model_architecture, num_classes=101, dropout_rate=0.3)
            self.pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
            self.pytorch_model.eval()
            print("🔄 已設定 CPU 備援推理")
        except Exception as e:
            print(f"❌ CPU 備援設定失敗: {e}")
    
    def shutdown(self):
        """關閉處理執行緒"""
        if self.processing_thread:
            self.shutdown_flag.set()
            self.processing_thread.join(timeout=1)
            print("🔄 後台處理執行緒已關閉")
    
    def __del__(self):
        """解構函數"""
        self.shutdown()

def benchmark_npu_utilization(model_path, test_images, batch_sizes=[1, 4, 8, 16, 32]):
    """NPU 使用率基準測試"""
    print("🧪 NPU 使用率基準測試")
    print("=" * 60)
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n📊 測試批次大小: {batch_size}")
        print("-" * 40)
        
        try:
            # 建立最佳化推理引擎
            npu_inference = OptimizedAMDNPUInference(
                model_path, 
                batch_size=batch_size,
                num_threads=4
            )
            
            # 選擇測試圖片
            test_batch = test_images[:min(len(test_images), batch_size * 3)]
            
            # 執行基準測試
            start_time = time.time()
            predictions = npu_inference.predict_image_batch(test_batch)
            total_time = time.time() - start_time
            
            if predictions:
                throughput = len(test_batch) / total_time
                avg_time = total_time / len(test_batch)
                
                results.append({
                    'batch_size': batch_size,
                    'images': len(test_batch),
                    'total_time': total_time,
                    'throughput': throughput,
                    'avg_time': avg_time
                })
                
                print(f"✅ 完成: {throughput:.1f} 圖片/秒")
            else:
                print("❌ 測試失敗")
            
            # 清理
            npu_inference.shutdown()
            del npu_inference
            
        except Exception as e:
            print(f"❌ 批次大小 {batch_size} 測試失敗: {e}")
    
    # 顯示結果摘要
    print(f"\n📈 NPU 效能基準測試結果")
    print("=" * 60)
    print(f"{'批次大小':<8} {'圖片數':<8} {'總時間':<10} {'吞吐量':<12} {'平均時間':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['batch_size']:<8} {result['images']:<8} "
              f"{result['total_time']:<10.3f} {result['throughput']:<12.1f} "
              f"{result['avg_time']:<10.3f}")
    
    if results:
        best_result = max(results, key=lambda x: x['throughput'])
        print(f"\n🏆 最佳效能: 批次大小 {best_result['batch_size']}, "
              f"吞吐量 {best_result['throughput']:.1f} 圖片/秒")
    
    return results

if __name__ == '__main__':
    # 測試最佳化的 NPU 推理
    print("🚀 測試最佳化 AMD NPU 推理")
    
    # 檢查基本支援
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"📋 可用提供者: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            print("✅ DirectML 可用 - 準備最佳化測試")
        else:
            print("❌ DirectML 不可用")
            
    except ImportError:
        print("❌ ONNX Runtime 未安裝")