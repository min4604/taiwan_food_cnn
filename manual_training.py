#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台灣美食 CNN 分類 - 手動硬體選擇訓練工具
Taiwan Food CNN Classification - Manual Hardware Selection Training Tool

支援手動選擇訓練硬體，包括 GPU、CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_model import TaiwanFoodResNet50
from pytorch_data_loader import get_dataloaders
from tqdm import tqdm
import os

def detect_training_devices():
    """檢測可用的訓練裝置"""
    print("🔍 檢測可用的訓練裝置")
    print("=" * 50)
    
    devices = []
    device_info = []
    
    # 檢測 GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            device_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            devices.append(f'cuda:{i}')
            device_info.append(f"✅ GPU {i}: {device_name} ({memory:.1f} GB)")
    else:
        device_info.append("❌ 未檢測到 CUDA GPU")
    
    # 檢測 MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
        device_info.append("🍎 MPS (Apple Silicon) 可用")
    
    # CPU 始終可用
    devices.append('cpu')
    device_info.append("💻 CPU 可用")
    
    # 顯示檢測結果
    for info in device_info:
        print(info)
    
    print("=" * 50)
    return devices

def choose_training_device(available_devices, manual_mode=False):
    """選擇訓練裝置"""
    print("\n🎯 選擇訓練裝置")
    print("-" * 40)
    
    # 準備選項
    options = []
    for device in available_devices:
        if device.startswith('cuda:'):
            gpu_idx = device.split(':')[1]
            gpu_name = torch.cuda.get_device_name(int(gpu_idx))
            memory = torch.cuda.get_device_properties(int(gpu_idx)).total_memory / 1024**3
            options.append((device, f"🚀 {gpu_name} ({memory:.1f} GB)"))
        elif device == 'mps':
            options.append((device, "🍎 Apple Silicon MPS"))
        elif device == 'cpu':
            options.append((device, "💻 CPU (較慢但穩定)"))
    
    print("可用的訓練裝置:")
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
    for device, desc in options:
        if device.startswith('cuda:'):
            print(f"\n🤖 自動選擇: {desc}")
            return device
    
    if 'mps' in available_devices:
        print(f"\n🤖 自動選擇: Apple Silicon MPS")
        return 'mps'
    
    print(f"\n🤖 自動選擇: CPU")
    return 'cpu'

def choose_model_to_continue():
    """選擇要繼續訓練的模型"""
    print("\n🎯 選擇訓練模式")
    print("-" * 40)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 已建立 models 資料夾")
        return None, None
    
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
    
    if not model_files:
        print("📝 沒有找到已訓練的模型，將開始新的訓練")
        return None, None
    
    print("可用的模型檔案:")
    print("  0. 🆕 開始新的訓練")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. 📁 {model_file}")
    
    while True:
        try:
            choice = input(f"\n👉 請選擇 (0-{len(model_files)}): ").strip()
            choice_idx = int(choice)
            
            if choice_idx == 0:
                return None, None
            elif 1 <= choice_idx <= len(model_files):
                selected_model = model_files[choice_idx - 1]
                model_path = os.path.join('models', selected_model)
                
                # 嘗試從檔名解析 epoch
                try:
                    if 'epoch' in selected_model.lower():
                        epoch_str = selected_model.lower().split('epoch')[1].split('_')[0].split('.')[0]
                        start_epoch = int(epoch_str) + 1
                    else:
                        start_epoch = 1
                except:
                    start_epoch = 1
                
                print(f"✅ 將從 {selected_model} 繼續訓練 (從第 {start_epoch} epoch 開始)")
                return model_path, start_epoch
            else:
                print(f"⚠️  請輸入 0-{len(model_files)} 之間的數字")
                
        except ValueError:
            print("⚠️  請輸入有效的數字")
        except KeyboardInterrupt:
            print("\n\n❌ 使用者取消操作")
            return None, None

def train_model_with_device(device, model_path=None, start_epoch=1, num_epochs=50, batch_size=32):
    """使用指定裝置訓練模型"""
    
    print(f"\n🚀 開始訓練設定")
    print(f"   訓練裝置: {device}")
    print(f"   批次大小: {batch_size}")
    print(f"   總 Epochs: {num_epochs}")
    print(f"   起始 Epoch: {start_epoch}")
    
    # 如果是 CPU，調整批次大小
    if device == 'cpu':
        batch_size = min(batch_size, 16)
        print(f"   CPU 最佳化: 調整批次大小為 {batch_size}")
    
    # 設定裝置
    device_obj = torch.device(device)
    
    # 建立資料載入器
    print("\n📂 載入資料集...")
    train_loader, val_loader, num_classes = get_dataloaders(
        train_csv='archive/tw_food_101/tw_food_101_train.csv',
        train_img_dir='archive/tw_food_101/train',
        batch_size=batch_size,
        val_split=0.2
    )
    
    print(f"✅ 訓練集: {len(train_loader.dataset)} 張圖片")
    print(f"✅ 驗證集: {len(val_loader.dataset)} 張圖片")
    print(f"✅ 類別數: {num_classes}")
    
    # 建立模型
    model = TaiwanFoodResNet50(num_classes=num_classes)
    
    # 載入預訓練模型（如果有）
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"✅ 成功載入預訓練模型: {model_path}")
        except Exception as e:
            print(f"⚠️  載入預訓練模型失敗: {e}")
            print("🔄 將開始新的訓練")
            start_epoch = 1
    
    model = model.to(device_obj)
    
    # 設定損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 訓練迴圈
    best_val_acc = 0.0
    
    print(f"\n🎯 開始訓練 (裝置: {device})")
    print("=" * 60)
    
    for epoch in range(start_epoch, num_epochs + 1):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - 訓練", ncols=100)
        
        for images, labels in train_pbar:
            images, labels = images.to(device_obj), labels.to(device_obj)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / ((train_pbar.n + 1) * batch_size):.2f}%'
            })
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - 驗證", ncols=100)
            
            for images, labels in val_pbar:
                images, labels = images.to(device_obj), labels.to(device_obj)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / ((val_pbar.n + 1) * batch_size):.2f}%'
                })
        
        # 計算平均指標
        train_acc = 100. * train_correct / len(train_loader.dataset)
        val_acc = 100. * val_correct / len(val_loader.dataset)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch:3d} | 訓練 Loss: {avg_train_loss:.4f} | 訓練 Acc: {train_acc:.2f}% | 驗證 Loss: {avg_val_loss:.4f} | 驗證 Acc: {val_acc:.2f}%")
        
        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f'models/best_model_epoch{epoch}_acc{val_acc:.2f}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 新的最佳模型！驗證準確率: {val_acc:.2f}%")
        
        # 每10個epoch儲存一次
        if epoch % 10 == 0:
            checkpoint_path = f'models/checkpoint_epoch{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 檢查點已儲存: {checkpoint_path}")
        
        scheduler.step()
        print("-" * 60)
    
    print(f"\n🎉 訓練完成！最佳驗證準確率: {best_val_acc:.2f}%")

def main():
    """主程式"""
    print("🍜 台灣美食 CNN 分類 - 手動硬體選擇訓練工具")
    print("Taiwan Food CNN Classification - Manual Hardware Selection Training")
    print("=" * 80)
    
    # 檢測可用裝置
    available_devices = detect_training_devices()
    
    # 選擇模式
    print("\n🎮 請選擇訓練硬體選擇模式:")
    print("  1. 🤖 自動模式 (系統自動選擇最佳硬體)")
    print("  2. 🎮 手動模式 (手動選擇訓練硬體)")
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
                return
            else:
                print("⚠️  請輸入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n👋 程式結束")
            return
    
    # 選擇訓練裝置
    device = choose_training_device(available_devices, manual_mode)
    if device is None:
        print("❌ 未選擇裝置，程式結束")
        return
    
    # 選擇模型
    model_path, start_epoch = choose_model_to_continue()
    
    # 設定訓練參數
    print(f"\n⚙️  訓練參數設定:")
    
    try:
        epochs = input("📅 訓練 Epochs (預設 50): ").strip()
        num_epochs = int(epochs) if epochs else 50
        
        batch = input("📦 批次大小 (預設 32): ").strip()
        batch_size = int(batch) if batch else 32
        
    except ValueError:
        print("⚠️  使用預設參數")
        num_epochs = 50
        batch_size = 32
    
    print(f"\n📋 最終設定:")
    print(f"   訓練裝置: {device}")
    print(f"   訓練 Epochs: {num_epochs}")
    print(f"   批次大小: {batch_size}")
    if model_path:
        print(f"   繼續訓練: {model_path}")
    else:
        print(f"   訓練模式: 全新訓練")
    
    # 確認開始訓練
    confirm = input("\n🚀 是否開始訓練？ (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ 使用者取消訓練")
        return
    
    # 開始訓練
    try:
        train_model_with_device(device, model_path, start_epoch, num_epochs, batch_size)
    except KeyboardInterrupt:
        print("\n\n⚠️  訓練被使用者中斷")
    except Exception as e:
        print(f"\n❌ 訓練過程發生錯誤: {e}")

if __name__ == '__main__':
    main()