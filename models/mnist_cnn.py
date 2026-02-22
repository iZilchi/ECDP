# models/mnist_cnn.py - UPDATED FOR SKIN CANCER
import torch
import torch.nn as nn

class MNISTCNN(nn.Module):
    """
    Updated CNN model for Skin Cancer HAM10000
    - 3 input channels (RGB) instead of 1
    - 7 output classes (skin cancer types) instead of 10
    """
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # CHANGED: 3 input channels for RGB
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128),  # 28x28 -> 7x7 after 2 max pools
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)  # CHANGED: 7 classes for skin cancer
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def get_gradients(self):
        """For future use with gradient-based methods"""
        gradients = []
        for param in self.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
        return gradients
    
    def set_gradients(self, gradients):
        """For future use with gradient-based methods"""
        for param, grad in zip(self.parameters(), gradients):
            param.grad = grad.clone()