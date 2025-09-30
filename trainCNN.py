#!/usr/bin/env python3
"""
台灣美食 CNN 訓練主程式
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import TaiwanFoodDataLoader
from cnn_model import TaiwanFoodCNN

def check_gpu():
    """檢查 GPU 可用性並設定"""
    print("\n🔍 檢查計算環境...")
    
    # 檢查 TensorFlow 版本
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 檢查 GPU 可用性
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✅ 發現 {len(gpus)} 個 GPU 裝置:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            
        try:
            # 啟用記憶體增長（避免一次性佔用所有 GPU 記憶體）
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 設定預設 GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            print("✅ GPU 設定完成，將使用 GPU 進行訓練")
            
            # 顯示 GPU 記憶體資訊
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if 'device_name' in gpu_details:
                print(f"   GPU 型號: {gpu_details['device_name']}")
                
            return True
            
        except RuntimeError as e:
            print(f"⚠️  GPU 設定失敗: {e}")
            print("⚠️  將使用 CPU 進行訓練")
            return False
    else:
        print("ℹ️  未發現 GPU 裝置，將使用 CPU 進行訓練")
        print("💡 如需 GPU 加速，請安裝 CUDA 和 cuDNN")
        return False

def print_device_info():
    """顯示目前使用的計算裝置"""
    print("\n📊 計算裝置資訊:")
    
    # 檢查可用的裝置
    devices = tf.config.list_logical_devices()
    for device in devices:
        print(f"   {device.device_type}: {device.name}")
    
    # 檢查預設裝置
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # 建立一個簡單的張量來測試裝置
        test_tensor = tf.constant([1.0, 2.0, 3.0])
        print(f"✅ 預設計算裝置: {test_tensor.device}")

def main():
    """主要訓練流程"""
    
    print("=" * 60)
    print("台灣美食 CNN 訓練程式")
    print("=" * 60)
    
    # === 檢查 GPU 環境 ===
    gpu_available = check_gpu()
    print_device_info()
    
    # === 設定參數 ===
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32 if gpu_available else 16  # GPU 可用時使用較大批次
    EPOCHS = 50
    MODEL_TYPE = 'resnet50'  # 'custom', 'resnet50', 'efficientnet', 'mobilenet'
    LEARNING_RATE = 0.001
    
    print(f"圖片大小: {IMG_SIZE}")
    print(f"批次大小: {BATCH_SIZE} {'(GPU 最佳化)' if gpu_available else '(CPU 最佳化)'}")
    print(f"訓練輪數: {EPOCHS}")
    print(f"模型類型: {MODEL_TYPE}")
    print(f"學習率: {LEARNING_RATE}")
    print(f"計算裝置: {'GPU 🚀' if gpu_available else 'CPU 💻'}")
    
    # === 建立資料載入器 ===
    print("\n1. 建立資料載入器...")
    loader = TaiwanFoodDataLoader(
        data_dir="archive/tw_food_101",
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # === 載入資料 ===
    print("\n2. 載入訓練資料...")
    try:
        (X_train, y_train), (X_val, y_val) = loader.load_dataset_from_csv('train')
        
        print(f"訓練集: {X_train.shape}")
        print(f"驗證集: {X_val.shape}")
        print(f"類別數: {loader.num_classes}")
        
        # 顯示類別分布
        train_dist = loader.get_class_distribution(y_train)
        print(f"訓練集樣本數: {len(X_train)}")
        print(f"驗證集樣本數: {len(X_val)}")
        
    except Exception as e:
        print(f"載入訓練資料失敗: {e}")
        return
    
    # === 載入測試資料 ===
    print("\n3. 載入測試資料...")
    try:
        X_test, y_test = loader.load_dataset_from_csv('test')
        print(f"測試集: {X_test.shape}")
        print("注意：測試集沒有真實標籤，僅用於預測")
    except Exception as e:
        print(f"載入測試資料失敗: {e}")
        X_test, y_test = None, None
    
    # === 建立資料生成器 ===
    print("\n4. 建立資料生成器...")
    train_generator, val_generator = loader.create_data_generators(
        (X_train, y_train), (X_val, y_val), augment=True
    )
    
    # === 建立 CNN 模型 ===
    print(f"\n5. 建立 {MODEL_TYPE.upper()} 模型...")
    cnn = TaiwanFoodCNN(
        num_classes=loader.num_classes,
        img_size=IMG_SIZE,
        model_type=MODEL_TYPE
    )
    
    model = cnn.build_model()
    cnn.compile_model(learning_rate=LEARNING_RATE)
    
    print(f"模型參數數量: {model.count_params():,}")
    
    # 顯示模型摘要
    print("\n模型架構:")
    cnn.get_model_summary()
    
    # === 開始訓練 ===
    print(f"\n6. 開始訓練 ({EPOCHS} 輪)...")
    
    # 建立模型儲存目錄
    os.makedirs('models', exist_ok=True)
    
    # 獲取回調函數
    callbacks = cnn.get_callbacks(f'taiwan_food_{MODEL_TYPE}')
    
    # 訓練模型
    history = cnn.train(
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # === 繪製訓練歷程 ===
    print("\n7. 繪製訓練歷程...")
    cnn.plot_training_history(history)
    
    # === 評估模型 ===
    print("\n8. 評估模型...")
    
    # 在驗證集上評估
    val_loss, val_acc, val_top5_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"驗證集準確率: {val_acc:.4f}")
    print(f"驗證集 Top-5 準確率: {val_top5_acc:.4f}")
    
    # 在測試集上評估（如果有的話）
    if X_test is not None:
        # 注意：測試集沒有真實標籤，這裡僅作為預測範例
        print("對測試集進行預測...")
        test_predictions = model.predict(X_test[:10])  # 預測前10張
        
        print("前10張測試圖片的預測結果:")
        for i in range(min(10, len(test_predictions))):
            pred_class = np.argmax(test_predictions[i])
            confidence = test_predictions[i][pred_class]
            class_name = loader.id_to_name[pred_class]
            print(f"  圖片 {i+1}: {pred_class} ({class_name}) - 信心度: {confidence:.4f}")
    else:
        print("沒有測試集資料可供預測")
    
    # === 微調（僅適用於遷移學習模型） ===
    if MODEL_TYPE != 'custom':
        print(f"\n9. 微調 {MODEL_TYPE.upper()} 模型...")
        fine_tune_history = cnn.fine_tune(
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=10,
            learning_rate=1e-5
        )
        
        if fine_tune_history:
            # 重新評估
            val_loss, val_acc, val_top5_acc = model.evaluate(X_val, y_val, verbose=0)
            print(f"微調後驗證集準確率: {val_acc:.4f}")
            print(f"微調後驗證集 Top-5 準確率: {val_top5_acc:.4f}")
    
    # === 儲存最終模型 ===
    print("\n10. 儲存模型...")
    model_path = f'models/taiwan_food_{MODEL_TYPE}_final.h5'
    cnn.save_model(model_path)
    
    print("\n=" * 60)
    print("訓練完成！")
    print("=" * 60)
    print(f"最佳模型已儲存至: models/taiwan_food_{MODEL_TYPE}_best.h5")
    print(f"最終模型已儲存至: {model_path}")

def predict_sample():
    """預測範例圖片"""
    print("\n=== 預測範例 ===")
    
    # 載入訓練好的模型
    model_path = 'models/taiwan_food_resnet50_best.h5'
    if not os.path.exists(model_path):
        print(f"找不到模型檔案: {model_path}")
        return
    
    # 載入模型
    cnn = TaiwanFoodCNN()
    cnn.load_model(model_path)
    
    # 載入資料載入器（用於類別映射）
    loader = TaiwanFoodDataLoader()
    
    # 載入測試資料
    try:
        X_test, _ = loader.load_dataset_from_csv('test')
        
        # 預測前幾張圖片
        num_samples = 5
        predictions = cnn.model.predict(X_test[:num_samples])
        
        print("預測結果:")
        for i in range(num_samples):
            pred_label = np.argmax(predictions[i])
            confidence = predictions[i][pred_label]
            pred_name = loader.id_to_name[pred_label]
            
            print(f"測試圖片 {i+1}:")
            print(f"  預測: {pred_label} ({pred_name})")
            print(f"  信心度: {confidence:.4f}")
            print()
            
    except Exception as e:
        print(f"預測時發生錯誤: {e}")

if __name__ == "__main__":
    # 訓練模型
    main()
    
    # 預測範例
    # predict_sample()