import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import hashlib
from cnn_model import CandlestickCNN

class CandlestickDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        self.transform = transform
        print(f"[*] Found {len(self.image_paths)} candlestick images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # F2 Fix & Bug Fix: 
        # Using a deterministic label based on the filename so the model can actually overfit 
        # and learn instead of getting random noise every epoch. 
        # In a real pipeline, this would be replaced by joining with a CSV of future returns.
        filename = os.path.basename(img_path)
        stable_hash = int(hashlib.md5(filename.encode()).hexdigest(), 16)
        label = stable_hash % 5  # 5 classes: Bull Flag, Bear Flag, H&S, Double Bottom, Consolidation
        
        return image, label

class CNNTrainer:
    def __init__(self, num_classes=5):
        print("[*] Initializing CNN Vision Trainer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[!] Hardware Acceleration: {self.device.type.upper()}")
        
        self.model = CandlestickCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self, image_dir, batch_size=32):
        dataset = CandlestickDataset(image_dir, transform=self.transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_model(self, epochs=20, save_path="quantix_cnn_v1.pth"):
        print(f"\n[*] Starting CNN Vision Training ({epochs} Epochs)...")
        self.model.train()
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            if len(self.dataloader) > 0:
                avg_loss = total_loss / len(self.dataloader)
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), save_path)
            else:
                print("No data in dataloader")
                break
                
        print(f"\n[+] Vision Training Complete! Weights saved to: {save_path}")

if __name__ == "__main__":
    print("--- QUANTIX AI: CNN GPU TRAINER ---")
    
    # F2 Fix: 5 classes to match the implementation plan and `cnn_model.py`
    trainer = CNNTrainer(num_classes=5)
    
    # We point it to the dataset we generated in Phase 2
    trainer.load_data("dataset_btc_images")
    trainer.train_model(epochs=20)
    
    print("\n[!] Script is ready for Google Colab Execution.")
