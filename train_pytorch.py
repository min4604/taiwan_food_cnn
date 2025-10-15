import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from pytorch_model import get_model, freeze_backbone
from pytorch_data_loader import get_dataloaders
from tqdm import tqdm
import os

def check_gpu():
    """詳細檢查 GPU 可用性"""
    print("=" * 60)
    print("🔍 GPU 環境檢測")
    print("=" * 60)
    
    # 嘗試直接測試 CUDA 是否可用
    try:
        gpu_available = torch.cuda.is_available()
        # 進一步驗證，嘗試進行一次 CUDA 操作
        if gpu_available:
            print("📊 測試 CUDA 功能...", end="")
            # 使用 CUDA 事件測試 CUDA 實際可用性
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # 執行矩陣乘法測試
            start_event.record()
            test_tensor1 = torch.rand(1000, 1000, device='cuda')
            test_tensor2 = torch.rand(1000, 1000, device='cuda')
            result = torch.mm(test_tensor1, test_tensor2)
            torch.cuda.synchronize()  # 確保計算完成
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            
            print(f" ✅ 成功! 耗時: {elapsed_time:.2f} ms")
            
            # 強制啟用 cuDNN
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            
            torch.cuda.empty_cache()  # 清理
            
    except Exception as e:
        print(f"\n❌ 嘗試 CUDA 操作時發生錯誤: {e}")
        gpu_available = False
    
    if gpu_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"✅ CUDA 可用: {torch.version.cuda}")
        print(f"🎯 GPU 數量: {device_count}")
        print(f"🚀 當前 GPU: {device_name}")
        print(f"✅ cuDNN 啟用: {torch.backends.cudnn.enabled}")
        print(f"✅ cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        
        # 顯示 GPU 記憶體資訊
        if device_count > 0:
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory = props.total_memory / 1024**3  # 轉換為 GB
                print(f"   GPU {i}: {props.name} ({memory:.1f} GB)")
                
        print(f"🔧 PyTorch 版本: {torch.__version__}")
    else:
        print("❌ CUDA 不可用")
        print("💡 建議:")
        print("   1. 確認您的電腦有 NVIDIA GPU")
        print("   2. 安裝 NVIDIA 驅動程式")
        print("   3. 執行 install_pytorch_gpu.bat 安裝 CUDA 版本的 PyTorch")
        print(f"🔧 當前 PyTorch 版本: {torch.__version__}")
    
    print("=" * 60)
    return gpu_available

def detect_model_architecture_from_file(model_path):
    """從模型檔案檢測架構類型"""
    filename = os.path.basename(model_path).lower()
    
    if 'efficientnet_b7' in filename:
        return 'efficientnet_b7'
    elif 'efficientnet_b3' in filename:
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

def choose_model_architecture():
    """選擇要使用的模型架構"""
    print("\n" + "=" * 60)
    print("🏗️  選擇模型架構")
    print("=" * 60)
    
    models = {
        '1': ('resnet50', 'ResNet50 (基礎模型，速度快)', '~23M', '~2GB', '快'),
        '2': ('efficientnet_b3', 'EfficientNet-B3 (推薦，效能佳)', '~12M', '~3GB', '中等'),
        '3': ('efficientnet_b7', 'EfficientNet-B7 (最強性能，需大顯存)', '~66M', '~8GB+', '慢'),
        '4': ('convnext_tiny', 'ConvNeXt-Tiny (現代架構，精度高)', '~28M', '~3GB', '中等'),
        '5': ('regnet_y', 'RegNet-Y (高效網路，平衡性好)', '~4M', '~2GB', '快'),
        '6': ('vit', 'Vision Transformer (注意力機制，需較大資料集)', '~86M', '~4GB', '慢')
    }
    
    for key, (model_name, description, params, memory, speed) in models.items():
        print(f"{key}. {description}")
        print(f"   參數量: {params} | 顯存需求: {memory} | 速度: {speed}")
        if key == '3':  # EfficientNet-B7 警告
            print("   ⚠️  建議: RTX 4060 8GB 可能需要減小批次大小")
        elif key == '6':  # ViT 警告
            print("   💡 適合: 大型資料集，需要長時間訓練")
    
    print("\n💡 提示: 所有模型都使用 ImageNet 預訓練權重")
    
    while True:
        try:
            choice = input("\n請選擇模型架構 (1-6) [預設=2]: ").strip()
            if choice == '':
                choice = '2'
            if choice in models:
                selected_model, description, params, memory, speed = models[choice]
                print(f"✅ 選擇了: {description}")
                
                # B7 特殊提醒
                if choice == '3':
                    print("\n🔥 EfficientNet-B7 注意事項:")
                    print("   • 需要 8GB+ 顯存")
                    print("   • 建議批次大小: 8-16")
                    print("   • 訓練時間較長，但精度更高")
                    confirm = input("   確定使用 B7 模型? (y/n) [y]: ").lower()
                    if confirm == 'n':
                        continue
                
                return selected_model
            else:
                print("❌ 無效選擇，請輸入 1-6")
        except KeyboardInterrupt:
            print("\n🚫 取消選擇，退出程式")
            exit(0)

def choose_training_strategy():
    """選擇訓練策略"""
    print("\n" + "=" * 60)
    print("🎯 選擇訓練策略")
    print("=" * 60)
    
    strategies = {
        '1': ('fine_tune', '微調訓練 (推薦)', '先凍結預訓練層訓練分類器，後解凍全模型微調'),
        '2': ('full_train', '全模型訓練', '從頭訓練所有層（需要更多時間和數據）'),
        '3': ('freeze_train', '凍結骨幹訓練', '只訓練分類器，預訓練層保持不變（適合小數據集）')
    }
    
    for key, (strategy, name, description) in strategies.items():
        print(f"{key}. {name}")
        print(f"   {description}")
    
    print("\n💡 推薦:")
    print("   - 資料集較小 (<5000張) → 選擇 3 (凍結骨幹)")
    print("   - 資料集中等 (5000-20000張) → 選擇 1 (微調訓練)")
    print("   - 資料集較大 (>20000張) → 選擇 2 (全模型訓練)")
    
    while True:
        try:
            choice = input("\n請選擇訓練策略 (1-3) [預設=1]: ").strip()
            if choice == '':
                choice = '1'
            
            if choice in strategies:
                selected_strategy, name, _ = strategies[choice]
                print(f"✅ 選擇了: {name}")
                return selected_strategy
            else:
                print("❌ 無效選擇，請輸入 1-3")
        except KeyboardInterrupt:
            print("\n🚫 取消選擇，退出程式")
            exit(0)

def choose_model_to_continue():
    """選擇要繼續訓練的模型，並自動檢測架構"""
    print("\n" + "=" * 60)
    print("🎯 選擇訓練模式")
    print("=" * 60)
    
    # 檢查是否有已保存的模型
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("📁 建立 models 目錄")
        model_architecture = choose_model_architecture()
        return None, 0, model_architecture
    
    # 獲取所有 .pth 模型檔案
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("📋 沒有找到已保存的模型，將從頭開始訓練")
        model_architecture = choose_model_architecture()
        return None, 0, model_architecture
    
    # 顯示選項
    print("請選擇訓練模式:")
    print("0. 🆕 從頭開始訓練 (新模型)")
    
    # 顯示可用的模型檔案
    for i, model_file in enumerate(model_files, 1):
        # 嘗試從檔名中提取 epoch 資訊
        if 'epoch' in model_file:
            print(f"{i}. 🔄 繼續訓練: {model_file}")
        else:
            print(f"{i}. 🔄 繼續訓練: {model_file}")
    
    print("-" * 60)
    
    # 獲取用戶選擇
    while True:
        try:
            choice = input(f"請輸入選擇 (0-{len(model_files)}): ").strip()
            choice = int(choice)
            
            if choice == 0:
                print("✅ 選擇從頭開始訓練")
                # 讓用戶選擇架構
                model_architecture = choose_model_architecture()
                return None, 0, model_architecture
            elif 1 <= choice <= len(model_files):
                selected_model = model_files[choice - 1]
                print(f"✅ 選擇繼續訓練: {selected_model}")
                
                # 檢測模型架構
                model_path = os.path.join(models_dir, selected_model)
                model_architecture = detect_model_architecture_from_file(model_path)
                print(f"🏗️  檢測到模型架構: {model_architecture}")
                
                # 嘗試從檔名中提取 epoch 數
                start_epoch = 0
                try:
                    if 'epoch' in selected_model:
                        # 假設檔名格式為 taiwan_food_resnet50_epoch10.pth
                        epoch_part = selected_model.split('epoch')[1].split('.')[0]
                        start_epoch = int(epoch_part)
                        print(f"📊 檢測到從第 {start_epoch} 個 epoch 繼續訓練")
                    else:
                        print("⚠️  無法從檔名檢測 epoch，將從 epoch 0 開始計數")
                except:
                    print("⚠️  無法解析 epoch 資訊，將從 epoch 0 開始計數")
                
                return os.path.join(models_dir, selected_model), start_epoch, model_architecture
            else:
                print(f"❌ 無效選擇，請輸入 0-{len(model_files)} 之間的數字")
        except ValueError:
            print("❌ 請輸入有效的數字")
        except KeyboardInterrupt:
            print("\n🚫 取消選擇，退出程式")
            exit(0)

def main():
    # 資料路徑
    train_csv = 'archive/tw_food_101/tw_food_101_train.csv'
    # 注意：測試集不參與訓練，僅用於最終評估
    test_csv = 'archive/tw_food_101/tw_food_101_test_list.csv'  
    train_img_dir = 'archive/tw_food_101/train'
    test_img_dir = 'archive/tw_food_101/test'
    num_classes = 101
    total_epochs = 50
    lr = 1e-3
    img_size = 224

    # 確保 CUDA 可用性
    print("\n" + "=" * 60)
    print("🔧 設備配置驗證")
    print("=" * 60)
    
    # 強制重新檢測 CUDA 並直接指定設備
    if torch.cuda.is_available():
        # 直接強制使用 CUDA，不經過 check_gpu()
        device = torch.device('cuda:0')
        print(f"✅ 強制設置設備為 CUDA: {device}")
        
        # 顯示 GPU 信息
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"� GPU 數量: {gpu_count}")
        print(f"📍 當前 GPU: {current_gpu} - {gpu_name}")
        print(f"📍 CUDA 版本: {torch.version.cuda}")
        
        # 強制執行一次 CUDA 操作
        try:
            print("⏳ 執行 CUDA 測試...")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            result = z.sum().item()
            print(f"✅ CUDA 測試成功! 結果: {result:.4f}")
            print(f"✅ 張量設備: {z.device}")
            
            # 清理記憶體
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU 測試失敗: {str(e)}")
            print("⚠️  無法使用 GPU，降級使用 CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("⚠️  無法使用 GPU，設備設置為 CPU")
    
    print("\n📍 最終設備設置: " + str(device) + f" ({device.type.upper()})")
    
    # 設置 CUDA 性能優化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"✅ cuDNN benchmark 模式已啟用")
    
    print("=" * 60 + "\n")

    # 選擇要繼續訓練的模型（這會自動檢測架構）
    model_path, start_epoch, model_architecture = choose_model_to_continue()
    remaining_epochs = total_epochs - start_epoch
    
    # 根據模型架構調整批次大小
    if device.type == 'cuda':
        if model_architecture == 'efficientnet_b7':
            batch_size = 8   # B7 需要更小批次
            print(f"🔥 EfficientNet-B7 檢測到，調整批次大小為 {batch_size}")
        elif model_architecture == 'vit':
            batch_size = 16  # ViT 也需要較小批次
            print(f"🤖 ViT 檢測到，調整批次大小為 {batch_size}")
        else:
            batch_size = 32  # 其他模型使用標準批次
            print(f"🚀 使用 GPU 訓練，批次大小: {batch_size}")
    else:
        batch_size = 8   # CPU 使用小批次
        print(f"💻 使用 CPU 訓練，批次大小: {batch_size}")
    
    # 選擇訓練策略（只在新訓練時詢問）
    if model_path is None:
        training_strategy = choose_training_strategy()
    else:
        # 繼續訓練時，預設使用微調策略
        training_strategy = 'fine_tune'
        print(f"🔄 繼續訓練模式，使用微調策略")

    # 調整 DataLoader 參數，確保數據加載高效
    num_workers = 0
    pin_memory = False
    
    if device.type == 'cuda':
        num_workers = 4  # 多線程加載數據
        pin_memory = True  # 數據直接加載到固定記憶體，加速 GPU 傳輸
        print(f"✅ 數據加載優化已啟用: {num_workers} 工作線程, pin_memory={pin_memory}")
    
    # DataLoader - 只使用訓練集進行訓練和驗證
    # 測試集不參與任何訓練過程
    train_loader, val_loader, _ = get_dataloaders(
        train_csv, test_csv, train_img_dir, test_img_dir, 
        batch_size=batch_size, 
        img_size=img_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print("📊 資料集資訊:")
    print(f"   訓練集大小: {len(train_loader.dataset)} (用於訓練)")
    print(f"   驗證集大小: {len(val_loader.dataset)} (從訓練集分割，用於驗證)")
    print(f"   批次大小: {batch_size}")
    print(f"   ⚠️  測試集: 不參與訓練過程，僅供最終評估使用")
    print()

    # 根據選擇創建模型（使用預訓練權重）
    print(f"🏗️  創建模型: {model_architecture}")
    print(f"📦 載入 ImageNet 預訓練權重...")
    
    # 強制指定 CPU 先創建模型，然後再移到 GPU
    model = get_model(model_architecture, num_classes=num_classes, dropout_rate=0.3, pretrained=True)
    print(f"📌 模型初始化在: {next(model.parameters()).device}")
    
    # 明確移至目標設備
    model = model.to(device)
    print(f"📌 模型已移至: {next(model.parameters()).device}")
    
    # 確保損失函數也在相同設備上
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 載入已存在的模型（如果選擇了）
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ 成功載入模型: {model_path}")
            print(f"🔄 從 epoch {start_epoch + 1} 開始繼續訓練")
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            print("🆕 將從頭開始訓練")
            start_epoch = 0
            remaining_epochs = total_epochs
    else:
        print("🆕 從頭開始訓練新模型")
    
    # 根據訓練策略設定模型
    print("\n" + "=" * 60)
    print(f"🎯 訓練策略: {training_strategy}")
    print("=" * 60)
    
    if training_strategy == 'freeze_train':
        # 凍結骨幹，只訓練分類器
        model = freeze_backbone(model, freeze=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        print(f"📚 訓練階段: 只訓練分類器層")
        
    elif training_strategy == 'fine_tune':
        # 微調訓練：分兩階段
        # 階段1: 凍結骨幹訓練分類器 (前 20% epochs)
        # 階段2: 解凍全模型微調 (後 80% epochs)
        freeze_epochs = max(5, total_epochs // 5)  # 至少5個epoch用於預訓練分類器
        
        if start_epoch < freeze_epochs:
            # 階段1: 凍結骨幹
            model = freeze_backbone(model, freeze=True)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            print(f"📚 階段1: 凍結骨幹訓練分類器 (epoch 1-{freeze_epochs})")
            print(f"📚 階段2: 將在 epoch {freeze_epochs + 1} 後解凍全模型微調")
        else:
            # 已經過了凍結階段，直接微調
            model = freeze_backbone(model, freeze=False)
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)  # 使用較小學習率
            print(f"📚 階段2: 全模型微調（使用較小學習率）")
        
        fine_tune_strategy = True
        fine_tune_epoch_threshold = freeze_epochs
        
    else:  # full_train
        # 全模型訓練
        model = freeze_backbone(model, freeze=False)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f"📚 訓練階段: 全模型訓練")
        fine_tune_strategy = False
        fine_tune_epoch_threshold = 0
        
    print(f"📈 訓練計畫: 從 epoch {start_epoch + 1} 到 epoch {total_epochs} (共 {remaining_epochs} 個 epochs)")
    print("=" * 60)
    
    # 驗證模型和數據在正確的設備上
    print("\n🔍 最終設備檢查:")
    print(f"   模型設備: {next(model.parameters()).device}")
    print(f"   目標設備: {device}")
    if next(model.parameters()).device.type != device.type:
        print(f"   ⚠️  警告: 模型設備不匹配！重新移動模型到 {device}")
        model = model.to(device)
        print(f"   ✅ 模型已移至: {next(model.parameters()).device}")
    else:
        print(f"   ✅ 模型設備配置正確")
    print()

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        # 微調策略：在指定 epoch 後解凍並調整學習率
        if training_strategy == 'fine_tune' and epoch == fine_tune_epoch_threshold:
            print("\n" + "=" * 60)
            print(f"🔓 解凍模型，開始全模型微調 (epoch {epoch + 1})")
            print("=" * 60)
            model = freeze_backbone(model, freeze=False)
            # 重新創建優化器，使用較小學習率
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
            print(f"📉 調整學習率: {lr} → {lr * 0.1}")
            print()
        
        # 在每個 epoch 開始時驗證設備（第一個 epoch）
        if epoch == start_epoch:
            print(f"🎯 Epoch {epoch+1} 設備驗證:")
            print(f"   模型在: {next(model.parameters()).device}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 第一個 batch 的詳細檢查
        first_batch = True
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", ncols=100)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # 第一個 batch 時檢查數據設備
            if first_batch and epoch == start_epoch:
                print(f"   數據批次在: {images.device}")
                print(f"   標籤在: {labels.device}")
                if device.type == 'cuda':
                    print(f"   GPU 記憶體已使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                print()
                first_batch = False
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{correct/total:.4f}'
            })
        print(f"訓練 Loss: {running_loss/total:.4f} | 訓練 Acc: {correct/total:.4f}")

        # 驗證
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="驗證中", ncols=80, leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                val_pbar.set_postfix({
                    'val_loss': f'{val_loss/val_total:.4f}',
                    'val_acc': f'{val_correct/val_total:.4f}'
                })
        
        print(f"驗證 Loss: {val_loss/val_total:.4f} | 驗證 Acc: {val_correct/val_total:.4f}")
        
        # 顯示 GPU 記憶體使用情況（如果使用 GPU）
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"🎮 GPU 記憶體: 已分配 {memory_allocated:.2f} MB | 已保留 {memory_reserved:.2f} MB")
        
        print("-" * 60)

        # 保存模型
        os.makedirs('models', exist_ok=True)
        model_filename = f'taiwan_food_{model_architecture}_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), f'models/{model_filename}')
        print(f"💾 模型已保存: models/{model_filename}")
        print()

if __name__ == '__main__':
    main()
