import torch
import torch.nn as nn
import torchvision.models as models

class TaiwanFoodResNet50(nn.Module):
    def __init__(self, num_classes=101, dropout_rate=0.3, pretrained=True):
        super().__init__()
        # 使用預訓練權重
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.base = models.resnet50(weights=weights)
        in_features = self.base.fc.in_features
        
        # 增強的分類器設計
        self.base.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class TaiwanFoodEfficientNetB3(nn.Module):
    """EfficientNet-B3: 更強的特徵提取能力，適合中等規模資料集"""
    def __init__(self, num_classes=101, dropout_rate=0.3, pretrained=True):
        super().__init__()
        # 使用預訓練權重
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.base = models.efficientnet_b3(weights=weights)
        in_features = self.base.classifier[1].in_features
        
        # 增強的分類器設計
        self.base.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class TaiwanFoodEfficientNetB7(nn.Module):
    """EfficientNet-B7: 最強的 EfficientNet 模型，適合大規模資料集和高精度需求"""
    def __init__(self, num_classes=101, dropout_rate=0.5, pretrained=True):
        super().__init__()
        # 使用預訓練權重
        weights = models.EfficientNet_B7_Weights.DEFAULT if pretrained else None
        self.base = models.efficientnet_b7(weights=weights)
        in_features = self.base.classifier[1].in_features  # 2560
        
        # 增強的分類器設計（B7 需要更複雜的分類器）
        self.base.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),  # 更大的隱藏層
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # 較低的 dropout
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class TaiwanFoodConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny: 現代化CNN架構，性能優秀"""
    def __init__(self, num_classes=101, dropout_rate=0.3, pretrained=True):
        super().__init__()
        # 使用預訓練權重
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.base = models.convnext_tiny(weights=weights)
        in_features = self.base.classifier[2].in_features
        
        # 增強的分類器設計
        self.base.classifier = nn.Sequential(
            self.base.classifier[0],  # LayerNorm
            self.base.classifier[1],  # Flatten
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class TaiwanFoodRegNetY(nn.Module):
    """RegNet-Y: Facebook開發的高效網路"""
    def __init__(self, num_classes=101, dropout_rate=0.3, pretrained=True):
        super().__init__()
        # 使用預訓練權重
        weights = models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
        self.base = models.regnet_y_400mf(weights=weights)
        in_features = self.base.fc.in_features
        
        # 增強的分類器設計
        self.base.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class TaiwanFoodViT(nn.Module):
    """Vision Transformer: 基於注意力機制的現代架構"""
    def __init__(self, num_classes=101, dropout_rate=0.2, pretrained=True):
        super().__init__()
        # 使用預訓練權重
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.base = models.vit_b_16(weights=weights)
        in_features = self.base.heads.head.in_features
        
        # 增強的分類器設計
        self.base.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

# 模型選擇工廠函數
def get_model(model_name='efficientnet_b3', num_classes=101, dropout_rate=0.3, pretrained=True):
    """
    取得指定的模型
    
    Args:
        model_name: 模型名稱 ('resnet50', 'efficientnet_b3', 'efficientnet_b7', 'convnext_tiny', 'regnet_y', 'vit')
        num_classes: 分類數量
        dropout_rate: Dropout比率
        pretrained: 是否使用預訓練權重
    
    Returns:
        PyTorch模型
    """
    models_dict = {
        'resnet50': TaiwanFoodResNet50,
        'efficientnet_b3': TaiwanFoodEfficientNetB3,
        'efficientnet_b7': TaiwanFoodEfficientNetB7,  # 新增 B7
        'convnext_tiny': TaiwanFoodConvNeXtTiny,
        'regnet_y': TaiwanFoodRegNetY,
        'vit': TaiwanFoodViT
    }
    
    if model_name not in models_dict:
        raise ValueError(f"不支援的模型: {model_name}. 支援的模型: {list(models_dict.keys())}")
    
    return models_dict[model_name](num_classes, dropout_rate, pretrained)

def freeze_backbone(model, freeze=True):
    """
    凍結或解凍模型的骨幹網路（預訓練部分）
    
    Args:
        model: PyTorch 模型
        freeze: True=凍結參數（不訓練），False=解凍（訓練）
    
    Returns:
        model: 修改後的模型
    """
    # 找出分類器層的名稱
    classifier_names = ['fc', 'classifier', 'head', 'heads']
    
    for name, param in model.named_parameters():
        # 如果不是分類器層，則根據 freeze 參數設定
        if not any(cls_name in name for cls_name in classifier_names):
            param.requires_grad = not freeze
    
    if freeze:
        print("🔒 已凍結預訓練層（只訓練分類器）")
    else:
        print("🔓 已解凍所有層（全模型訓練）")
    
    # 顯示可訓練參數統計
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 參數統計:")
    print(f"   總參數: {total_params:,}")
    print(f"   可訓練參數: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model
