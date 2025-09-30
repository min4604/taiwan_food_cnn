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
            
            # 轉換 PyTorch 模型為 ONNX (如果需要)
            onnx_path = self._convert_to_onnx(pytorch_model_path)
            
            # 建立 ONNX Runtime 會話
            providers = [
                'DmlExecutionProvider',  # DirectML (AMD NPU)
                'CPUExecutionProvider'   # CPU 作為備援
            ]
            
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
            print(f"🚀 AMD NPU 推理會話已建立")
            print(f"📊 使用提供者: {self.ort_session.get_providers()}")
            
        except ImportError:
            print("❌ ONNX Runtime 未安裝，請執行 install_amd_npu.bat")
            raise
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
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size)\n            \n            # 匯出為 ONNX\n            torch.onnx.export(\n                model,\n                dummy_input,\n                onnx_path,\n                export_params=True,\n                opset_version=11,\n                do_constant_folding=True,\n                input_names=['input'],\n                output_names=['output'],\n                dynamic_axes={\n                    'input': {0: 'batch_size'},\n                    'output': {0: 'batch_size'}\n                }\n            )\n            \n            print(f\"✅ ONNX 模型已儲存: {onnx_path}\")\n            return onnx_path\n            \n        except Exception as e:\n            print(f\"❌ ONNX 轉換失敗: {e}\")\n            raise\n    \n    def _setup_cpu_fallback(self, pytorch_model_path):\n        \"\"\"設定 CPU 備援推理\"\"\"\n        from pytorch_model import TaiwanFoodResNet50\n        \n        self.pytorch_model = TaiwanFoodResNet50(num_classes=101)\n        self.pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))\n        self.pytorch_model.eval()\n        print(\"🔄 已設定 CPU 備援推理\")\n    \n    def predict_image(self, image_path):\n        \"\"\"使用 AMD NPU 進行圖片推理\"\"\"\n        try:\n            # 載入和預處理圖片\n            image = Image.open(image_path).convert('RGB')\n            input_tensor = self.transform(image).unsqueeze(0)\n            input_array = input_tensor.numpy()\n            \n            if self.ort_session:\n                # 使用 ONNX Runtime (AMD NPU)\n                input_name = self.ort_session.get_inputs()[0].name\n                result = self.ort_session.run(None, {input_name: input_array})\n                output = result[0]\n                predicted_class = np.argmax(output, axis=1)[0]\n            else:\n                # 使用 PyTorch CPU 備援\n                with torch.no_grad():\n                    output = self.pytorch_model(input_tensor)\n                    predicted_class = torch.argmax(output, dim=1).item()\n            \n            return predicted_class\n            \n        except Exception as e:\n            print(f\"❌ 推理失敗: {e}\")\n            return -1\n    \n    def predict_batch(self, image_paths):\n        \"\"\"批次推理\"\"\"\n        predictions = []\n        \n        for img_path in image_paths:\n            pred = self.predict_image(img_path)\n            predictions.append(pred)\n        \n        return predictions\n\ndef test_amd_npu():\n    \"\"\"測試 AMD NPU 功能\"\"\"\n    print(\"🧪 測試 AMD Ryzen AI 9HX NPU 功能\")\n    print(\"=\" * 50)\n    \n    try:\n        import onnxruntime as ort\n        providers = ort.get_available_providers()\n        print(f\"📋 可用提供者: {providers}\")\n        \n        if 'DmlExecutionProvider' in providers:\n            print(\"✅ DirectML 可用 - AMD NPU 支援正常\")\n            \n            # 測試建立會話\n            dummy_model = \"test_model.onnx\"\n            # 這裡可以添加實際的模型測試\n            \n        else:\n            print(\"❌ DirectML 不可用\")\n            print(\"💡 請執行 install_amd_npu.bat 安裝支援套件\")\n            \n    except ImportError:\n        print(\"❌ ONNX Runtime 未安裝\")\n        print(\"💡 請執行 install_amd_npu.bat 安裝\")\n    except Exception as e:\n        print(f\"❌ 測試失敗: {e}\")\n\nif __name__ == '__main__':\n    test_amd_npu()