import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from lstm_model import TradingLSTM

class LSTMTrainer:
    def __init__(self, csv_file, sequence_length=30):
        print(f"[*] Initializing Highly Optimized GPU Trainer for {csv_file}")
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[!] Hardware Acceleration: {self.device.type.upper()}")
        
        # Load data
        self.df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        
        # [CRITICAL UPDATE]: We MUST scale the data between 0 and 1 for the LSTM to achieve optimal accuracy.
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Define the LSTM architecture
        self.model = TradingLSTM(input_size=5).to(self.device)
        self.criterion = nn.MSELoss()
        
        # Using AdamW optimizer with a lower learning rate for stable, optimal convergence
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        # Learning Rate Scheduler to drop the LR as it gets closer to the perfect answer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20, factor=0.5)

    def prepare_dataloaders(self, batch_size=64):
        print("[*] Scaling Data & Building PyTorch Tensors...")
        
        # Scale the 5 features
        raw_features = self.df[['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']].values
        scaled_features = self.feature_scaler.fit_transform(raw_features)
        
        # Scale the target (Close Price)
        raw_targets = self.df[['Close']].values
        scaled_targets = self.target_scaler.fit_transform(raw_targets)
        
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length - 1):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(scaled_targets[i + self.sequence_length])
            
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        
        # Split into 85% Train, 15% Test
        split = int(0.85 * len(X))
        
        train_dataset = TensorDataset(X[:split], y[:split])
        test_dataset = TensorDataset(X[split:], y[split:])
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"[+] Loaded {len(train_dataset)} training sequences and {len(test_dataset)} testing sequences.")

    def train_model(self, epochs=500, save_path="quantix_lstm_v1.pth"):
        print(f"\n[*] Starting Deep Learning Backpropagation ({epochs} Epochs)...")
        self.model.train()
        
        best_loss = float('inf')
        early_stopping_counter = 0
        patience = 50 # Stop if no improvement for 50 epochs
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            self.scheduler.step(avg_loss)
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {avg_loss:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
            # Save the best weights
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}. Maximum optimality reached.")
                break
                
        print(f"\n[+] Training Complete! Optimal model weights saved to: {save_path}")
        
        # [CRITICAL FIX]: Save the fitted scalers so main_api.py can use exact mathematical bounds
        joblib.dump(self.feature_scaler, 'feature_scaler.pkl')
        joblib.dump(self.target_scaler, 'target_scaler.pkl')
        print("[+] Stateful Scalers saved as 'feature_scaler.pkl' and 'target_scaler.pkl'")

if __name__ == "__main__":
    print("--- QUANTIX AI: ADVANCED GPU TRAINING ENGINE (NIFTY50) ---")
    
    trainer = LSTMTrainer("nifty50_processed_data.csv")
    trainer.prepare_dataloaders()
    
    # Train the Nifty50 Stock LSTM
    trainer.train_model(epochs=100, save_path="quantix_nifty_lstm.pth")
    
    print("\n[!] Script execution finished successfully.")
