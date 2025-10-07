import torch
import torch.nn as nn
import torchvision.models as models

class TaiwanFoodResNet50(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.base(x)

class TaiwanFoodEfficientNetB3(nn.Module):
    """EfficientNet-B3: 更強的特徵提取能力，適合中等規模資料集"""
    def __init__(self, num_classes=101, dropout_rate=0.3):
        super().__init__()
        self.base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
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

class TaiwanFoodConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny: 現代化CNN架構，性能優秀"""
    def __init__(self, num_classes=101, dropout_rate=0.3):
        super().__init__()
        self.base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
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
    def __init__(self, num_classes=101, dropout_rate=0.3):
        super().__init__()
        self.base = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
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
    def __init__(self, num_classes=101, dropout_rate=0.2):
        super().__init__()
        self.base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
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
def get_model(model_name='efficientnet_b3', num_classes=101, dropout_rate=0.3):
    """
    取得指定的模型
    
    Args:
        model_name: 模型名稱 ('resnet50', 'efficientnet_b3', 'convnext_tiny', 'regnet_y', 'vit')
        num_classes: 分類數量
        dropout_rate: Dropout比率
    
    Returns:
        PyTorch模型
    """
    models_dict = {
        'resnet50': TaiwanFoodResNet50,
        'efficientnet_b3': TaiwanFoodEfficientNetB3,
        'convnext_tiny': TaiwanFoodConvNeXtTiny,
        'regnet_y': TaiwanFoodRegNetY,
        'vit': TaiwanFoodViT
    }
    
    if model_name not in models_dict:
        raise ValueError(f"不支援的模型: {model_name}. 支援的模型: {list(models_dict.keys())}")
    
    if model_name == 'resnet50':
        return models_dict[model_name](num_classes)
    else:
        return models_dict[model_name](num_classes, dropout_rate)
