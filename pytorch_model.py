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

# 你可以根據需求擴充 EfficientNet/MobileNet 或自訂 CNN
