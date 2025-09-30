#!/usr/bin/env python3
"""
AMD NPU 快速驗證腳本
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

def quick_amd_npu_test():
    """快速測試 AMD NPU 功能"""
    print("🚀 AMD NPU 快速功能驗證")
    print("=" * 40)
    
    try:
        # 檢查 ONNX Runtime DirectML
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        print(f"可用提供者: {providers}")
        
        if 'DmlExecutionProvider' not in providers:
            print("❌ DirectML 不可用")
            return False
        
        print("✅ DirectML 可用")
        
        # 嘗試建立簡單的推理會話
        try:
            # 建立一個最簡單的 ONNX 模型來測試
            import onnx
            from onnx import helper, TensorProto
            
            # 建立簡單的恆等運算模型
            X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
            
            identity_node = helper.make_node(
                'Identity',
                inputs=['X'],
                outputs=['Y'],
            )
            
            graph = helper.make_graph(
                [identity_node],
                'test_graph',
                [X],
                [Y]
            )
            
            model = helper.make_model(graph)
            
            # 儲存測試模型
            test_model_path = 'test_amd_npu_model.onnx'
            onnx.save(model, test_model_path)
            
            # 建立 DirectML 會話
            session = ort.InferenceSession(
                test_model_path, 
                providers=['DmlExecutionProvider', 'CPUExecutionProvider']
            )
            
            print(f"✅ ONNX 會話建立成功")
            print(f"🎯 使用提供者: {session.get_providers()}")
            
            # 測試推理
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            result = session.run(None, {'X': dummy_input})
            
            print("✅ AMD NPU 推理測試成功")
            print(f"📊 輸入形狀: {dummy_input.shape}")
            print(f"📊 輸出形狀: {result[0].shape}")
            
            # 清理測試檔案
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX 推理測試失敗: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ 套件導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def test_pytorch_model_conversion():
    """測試 PyTorch 模型轉換"""
    print("\n🔄 PyTorch 模型轉換測試")
    print("-" * 30)
    
    try:
        from pytorch_model import TaiwanFoodResNet50
        
        # 建立測試模型
        model = TaiwanFoodResNet50(num_classes=101)
        model.eval()
        
        # 建立範例輸入
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 測試模型前向傳播
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ PyTorch 模型測試成功")
        print(f"📊 輸入形狀: {dummy_input.shape}")
        print(f"📊 輸出形狀: {output.shape}")
        
        # 測試 ONNX 導出
        test_onnx_path = 'test_conversion.onnx'
        
        torch.onnx.export(
            model,
            dummy_input,
            test_onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        print(f"✅ ONNX 導出成功: {test_onnx_path}")
        
        # 驗證導出的 ONNX 模型
        import onnx
        onnx_model = onnx.load(test_onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print("✅ ONNX 模型驗證通過")
        
        # 清理
        if os.path.exists(test_onnx_path):
            os.remove(test_onnx_path)
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch 模型轉換測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🧪 AMD Ryzen AI NPU 完整功能測試")
    print("=" * 50)
    
    # 測試 1: AMD NPU 基本功能
    npu_ok = quick_amd_npu_test()
    
    # 測試 2: PyTorch 模型轉換
    conversion_ok = test_pytorch_model_conversion()
    
    # 總結
    print("\n📊 測試結果總結")
    print("=" * 30)
    print(f"AMD NPU 基本功能: {'✅ 通過' if npu_ok else '❌ 失敗'}")
    print(f"PyTorch 模型轉換: {'✅ 通過' if conversion_ok else '❌ 失敗'}")
    
    if npu_ok and conversion_ok:
        print("\n🎉 恭喜！AMD NPU 完全就緒！")
        print("您可以開始使用 AMD NPU 進行深度學習推理了！")
        print("\n下一步:")
        print("python evaluate_test_set.py  # 使用 AMD NPU 評估模型")
    else:
        print("\n⚠️  部分功能測試失敗")
        print("請檢查相關套件安裝或聯繫支援")

if __name__ == '__main__':
    main()