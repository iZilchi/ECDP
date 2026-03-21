import torch.nn as nn

class TinyCNN(nn.Module):
    """
    Extremely small CNN for HAM10000 to make DP feasible.
    Total parameters: ~ 3,200 (instead of >400k).
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),   # 8 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 8 x 14 x 14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16 x 14 x 14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7))                   # 16 x 7 x 7
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x