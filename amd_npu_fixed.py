import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

class AMDNPUInference:
    """
    AMD Ryzen AI 9HX NPU 推理類
    使用 ONNX Runtime DirectML 進行 NPU 推理
    """
    
    def __init__(self, model_path, img_size=224):
        self.img_size = img_size
        self.onnx_model_path = None
        self.ort_session = None
        self.pytorch_model = None
        
        # 設定轉換
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 初始化模型
        self._setup_model(model_path)
    
    def _setup_model(self, pytorch_model_path):
        """設定 AMD NPU 推理模型"""
        try:
            # 檢查是否支援 ONNX Runtime DirectML
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'DmlExecutionProvider' not in providers:
                raise RuntimeError("DirectML 不可用，無法使用 AMD NPU")
            
            print("✅ 檢測到 DirectML 支援")
            print("🚀 為 AMD Ryzen AI 9 HX 370 NPU 最佳化...")
            
            # 轉換 PyTorch 模型為 ONNX (如果需要)
            onnx_path = self._convert_to_onnx(pytorch_model_path)
            
            # 建立 ONNX Runtime 會話，專為 AMD NPU 最佳化
            providers = [
                ('DmlExecutionProvider', {
                    'device_id': 0,
                    'enable_dynamic_shapes': True
                }),
                'CPUExecutionProvider'   # CPU 作為備援
            ]
            
            # 建立會話選項以最佳化效能
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.ort_session = ort.InferenceSession(
                onnx_path, 
                providers=providers,
                sess_options=session_options
            )
            print(f"🚀 AMD NPU 推理會話已建立")
            print(f"📊 使用提供者: {self.ort_session.get_providers()}")
            
        except ImportError:
            print("❌ ONNX Runtime 未安裝，請執行 install_amd_npu.bat")
            self._setup_cpu_fallback(pytorch_model_path)
        except Exception as e:
            print(f"❌ AMD NPU 設定失敗: {e}")
            print("💡 將回退到 CPU 推理")
            self._setup_cpu_fallback(pytorch_model_path)
    
    def _convert_to_onnx(self, pytorch_model_path):
        """將 PyTorch 模型轉換為 ONNX 格式"""
        onnx_path = pytorch_model_path.replace('.pth', '_amd_npu.onnx')
        
        if os.path.exists(onnx_path):
            print(f"📁 使用現有的 ONNX 模型: {onnx_path}")
            return onnx_path
        
        print("🔄 轉換 PyTorch 模型為 ONNX...")
        
        try:
            from pytorch_model import TaiwanFoodResNet50
            
            # 載入 PyTorch 模型
            model = TaiwanFoodResNet50(num_classes=101)
            model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
            model.eval()
            
            # 建立範例輸入
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
            
            # 匯出為 ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"✅ ONNX 模型已儲存: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX 轉換失敗: {e}")
            raise
    
    def _setup_cpu_fallback(self, pytorch_model_path):
        """設定 CPU 備援推理"""
        try:
            from pytorch_model import TaiwanFoodResNet50
            
            self.pytorch_model = TaiwanFoodResNet50(num_classes=101)
            self.pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
            self.pytorch_model.eval()
            print("🔄 已設定 CPU 備援推理")
        except Exception as e:
            print(f"❌ CPU 備援設定失敗: {e}")
    
    def predict_image(self, image_path):
        """使用 AMD NPU 進行圖片推理"""
        try:
            # 載入和預處理圖片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            input_array = input_tensor.numpy()
            
            if self.ort_session:
                # 使用 ONNX Runtime (AMD NPU)
                input_name = self.ort_session.get_inputs()[0].name
                result = self.ort_session.run(None, {input_name: input_array})
                output = result[0]
                predicted_class = np.argmax(output, axis=1)[0]
            else:
                # 使用 PyTorch CPU 備援
                with torch.no_grad():
                    output = self.pytorch_model(input_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
            
            return predicted_class
            
        except Exception as e:
            print(f"❌ 推理失敗: {e}")
            return -1

def test_amd_npu():
    """測試 AMD NPU 功能"""
    print("🧪 測試 AMD Ryzen AI 9HX NPU 功能")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"📋 可用提供者: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            print("✅ DirectML 可用 - AMD NPU 支援正常")
        else:
            print("❌ DirectML 不可用")
            print("💡 請執行 install_amd_npu.bat 安裝支援套件")
            
    except ImportError:
        print("❌ ONNX Runtime 未安裝")
        print("💡 請執行 install_amd_npu.bat 安裝")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

if __name__ == '__main__':
    test_amd_npu()