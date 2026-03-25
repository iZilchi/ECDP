import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18ForSkin(nn.Module):
    """ResNet‑18 adapted for skin cancer classification (7 classes)."""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = resnet18(weights=weights)
        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)