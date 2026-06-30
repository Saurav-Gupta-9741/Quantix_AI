import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import cv2

class CandlestickCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CandlestickCNN, self).__init__()
        
        self.resnet = models.resnet18(pretrained=False) # Local backend doesn't need to download weights from internet during init since we load .pth
        
        # Hooks for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Register hooks on the last convolutional layer (layer4 in ResNet18)
        target_layer = self.resnet.layer4[-1].conv2
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.resnet(x)
        
    def get_gradcam_heatmap(self, class_idx=None):
        """Generates Grad-CAM heatmap for the target class."""
        if self.gradients is None or self.activations is None:
            return None
            
        # Get the gradients and activations for the single image in the batch
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global average pooling on the gradients to get the weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU on the CAM (we only care about features that have a positive influence)
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        # Resize to 224x224 (original image size)
        cam = cv2.resize(cam, (224, 224))
        
        return cam
