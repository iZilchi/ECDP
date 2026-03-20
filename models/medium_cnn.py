import torch.nn as nn

class MediumCNN(nn.Module):
    """
    A medium-sized CNN for HAM10000 with more capacity than TinyCNN.
    Designed to improve SNR while keeping training feasible.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 16 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 16 x 14 x 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 32 x 7 x 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 7 x 7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))                    # 64 x 4 x 4
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x