import torch
import torch.nn as nn
from torchvision import models

class CandlestickCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CandlestickCNN, self).__init__()
        
        self.resnet = models.resnet18(pretrained=False) # Local backend doesn't need to download weights from internet during init since we load .pth
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
