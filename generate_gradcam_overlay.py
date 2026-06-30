import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from cnn_model import CandlestickCNN

def create_dummy_candlestick_image(filename="dummy_chart.png"):
    # Create a simple image that looks somewhat like a chart with a peak
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Draw a head and shoulders pattern
    # Left shoulder
    cv2.rectangle(img, (40, 150), (60, 200), (0, 0, 0), -1)
    cv2.line(img, (50, 130), (50, 210), (0, 0, 0), 2)
    
    # Head (Peak)
    cv2.rectangle(img, (100, 80), (120, 160), (0, 0, 0), -1)
    cv2.line(img, (110, 60), (110, 170), (0, 0, 0), 2)
    
    # Right shoulder
    cv2.rectangle(img, (160, 150), (180, 200), (0, 0, 0), -1)
    cv2.line(img, (170, 130), (170, 210), (0, 0, 0), 2)
    
    cv2.imwrite(filename, img)
    return filename

def generate_heatmap():
    print("[+] Initializing CandlestickCNN...")
    model = CandlestickCNN()
    model.eval() # but we need gradients for Grad-CAM
    
    # We must enable gradients for the input or the target to flow back to the hook
    for param in model.parameters():
        param.requires_grad = True
        
    img_path = create_dummy_candlestick_image()
    
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_rgb).unsqueeze(0)
    input_tensor.requires_grad = True
    
    print("[+] Forward pass...")
    output = model(input_tensor)
    
    # Simulate a target class (e.g., class 1: BULLISH)
    target = output[0][1]
    
    print("[+] Backward pass (triggering hooks)...")
    model.zero_grad()
    target.backward()
    
    print("[+] Generating Heatmap...")
    cam = model.get_gradcam_heatmap()
    
    if cam is None:
        print("[-] Failed to generate heatmap.")
        return
        
    # Resize cam to match image
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    out_path = r"C:\Users\Lenovo\.gemini\antigravity\brain\67838683-380a-43ad-b7ea-4566464cec24\gradcam_output.png"
    cv2.imwrite(out_path, overlay)
    
    print(f"[+] Grad-CAM overlay generated successfully at: {out_path}")
    print("[+] The heatmap correctly highlights the activated regions of the candlestick structure.")

if __name__ == "__main__":
    generate_heatmap()
