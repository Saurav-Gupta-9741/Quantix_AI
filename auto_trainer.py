"""
QUANTIX AUTO-TRAINER
====================
Automated daily model training pipeline.
Downloads fresh market data from Yahoo Finance, trains LSTM + CNN models,
and saves updated .pth weights directly to the project folder.

Run manually:   python auto_trainer.py
Run scheduled:  Windows Task Scheduler runs this daily at 6:00 AM
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import mplfinance as mpf
from PIL import Image
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms, models

# Import our model architectures (ensures weights ALWAYS match)
sys.path.insert(0, os.path.dirname(__file__))
from lstm_model import TradingLSTM
from cnn_model import CandlestickCNN

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJECT_DIR, "training_log.txt")

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# =============================================================
# PHASE 1: LSTM TIME-SERIES MODEL TRAINING
# =============================================================
def train_lstm_model(ticker, output_name, epochs=50, seq_length=30):
    """
    Downloads real market data and trains the BiLSTM+Attention model.
    """
    log(f"--- LSTM Training: {ticker} ---")
    
    # Download 2 years of real daily data
    log(f"Downloading 2 years of {ticker} data from Yahoo Finance...")
    df = yf.Ticker(ticker).history(period="2y", interval="1d")
    
    if df.empty or len(df) < 100:
        log(f"ERROR: Not enough data for {ticker}. Skipping.")
        return False
    
    # Calculate technical indicators
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    
    log(f"Data loaded: {len(df)} trading days")
    
    # Prepare features and targets
    features = df[['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']].values
    targets = df[['Close']].values
    
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_targets = target_scaler.fit_transform(targets)
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_features) - 1):
        X.append(scaled_features[i - seq_length:i])
        y.append(scaled_targets[i + 1])  # Predict next day's close
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    log(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize model (same architecture as the server uses)
    device = torch.device("cpu")
    model = TradingLSTM(input_size=5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test_t)
            val_loss = criterion(val_predictions, y_test_t)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            model_path = os.path.join(PROJECT_DIR, f"{output_name}.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            log(f"  Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")
        
        # Early stopping
        if patience_counter >= 10:
            log(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Save scalers
    joblib.dump(feature_scaler, os.path.join(PROJECT_DIR, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(PROJECT_DIR, "target_scaler.pkl"))
    
    # Calculate accuracy metrics
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()
        test_actual = y_test_t.numpy()
    
    # Direction accuracy (did we predict up/down correctly?)
    pred_direction = np.diff(test_preds.flatten()) > 0
    actual_direction = np.diff(test_actual.flatten()) > 0
    direction_accuracy = np.mean(pred_direction == actual_direction) * 100
    
    log(f"  LSTM {ticker} Training Complete!")
    log(f"  Best Val Loss: {best_loss:.6f}")
    log(f"  Direction Accuracy: {direction_accuracy:.1f}%")
    log(f"  Saved to: {output_name}.pth + scalers")
    
    return True

# =============================================================
# PHASE 2: CNN CANDLESTICK PATTERN MODEL TRAINING
# =============================================================
def train_cnn_model(tickers_list, epochs=30):
    """
    Generates real candlestick chart images from market data and trains
    the ResNet18 CNN to classify chart patterns.
    """
    log("--- CNN Pattern Recognition Training ---")
    
    charts_dir = os.path.join(PROJECT_DIR, "training_charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_images = []
    all_labels = []
    
    for ticker in tickers_list:
        log(f"  Generating charts for {ticker}...")
        try:
            df = yf.Ticker(ticker).history(period="2y", interval="1d")
            if df.empty or len(df) < 60:
                continue
            
            # Calculate indicators for labeling
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss_col = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss_col
            df['RSI'] = 100 - (100 / (1 + rs))
            df.bfill(inplace=True)
            
            mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='in')
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle='', figcolor='black', facecolor='black')
            
            # Slide a 30-day window across the data to generate training samples
            for i in range(30, len(df) - 5, 5):
                window = df.iloc[i-30:i]
                future_return = (df['Close'].iloc[min(i+5, len(df)-1)] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                
                # Advanced 5-Class Labeling Logic
                # 0: Bull Flag, 1: Bear Flag, 2: Consolidation
                # 3: Head & Shoulders (Bearish), 4: Double Bottom (Bullish)
                
                highs = window['High'].values
                lows = window['Low'].values
                higher_highs = sum(1 for j in range(1, len(highs)) if highs[j] > highs[j-1])
                lower_lows = sum(1 for j in range(1, len(lows)) if lows[j] < lows[j-1])
                
                # Check for Head & Shoulders (3 peaks, middle is highest)
                is_hs = False
                if len(highs) >= 15:
                    p1, p2, p3 = max(highs[:5]), max(highs[5:10]), max(highs[10:15])
                    if p2 > p1 and p2 > p3 and future_return < -0.02:
                        is_hs = True
                
                # Check for Double Bottom (2 distinct lows roughly equal)
                is_db = False
                if len(lows) >= 15:
                    l1, l2 = min(lows[:7]), min(lows[7:15])
                    if abs(l1 - l2) / l1 < 0.02 and future_return > 0.02:
                        is_db = True

                if is_hs:
                    label = 3
                elif is_db:
                    label = 4
                elif future_return > 0.02 and higher_highs > lower_lows:
                    label = 0  # Bull Flag
                elif future_return < -0.02 and lower_lows > higher_highs:
                    label = 1  # Bear Flag
                else:
                    label = 2  # Consolidation
                
                # Generate chart image
                img_path = os.path.join(charts_dir, f"{ticker}_{i}.png")
                try:
                    mpf.plot(window, type='candle', style=style, axisoff=True,
                             savefig=dict(fname=img_path, dpi=50, bbox_inches='tight', pad_inches=0))
                    
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)
                    all_images.append(img_tensor)
                    all_labels.append(label)
                    os.remove(img_path)
                except:
                    continue
                    
        except Exception as e:
            log(f"  Error processing {ticker}: {e}")
            continue
    
    if len(all_images) < 50:
        log("ERROR: Not enough training images generated. Skipping CNN training.")
        return False
    
    log(f"  Generated {len(all_images)} training chart images")
    
    # Create tensors
    X = torch.stack(all_images)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Initialize CNN (Updated to 5 classes)
    device = torch.device("cpu")
    model = CandlestickCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    batch_size = 16
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            batch_idx = indices[start:end]
            
            batch_X = X_train[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test.to(device)).float().mean().item() * 100
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(PROJECT_DIR, "quantix_cnn_v1.pth"))
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(X_train) / batch_size)
            log(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.1f}%")
    
    log(f"  CNN Training Complete! Best Accuracy: {best_accuracy:.1f}%")
    
    try:
        os.rmdir(charts_dir)
    except:
        pass
    
    return True

# =============================================================
# MAIN: DAILY TRAINING PIPELINE
# =============================================================
def run_full_training():
    log("=" * 60)
    log("QUANTIX AUTO-TRAINER: Starting Daily Training Pipeline")
    log("=" * 60)
    
    start_time = datetime.now()
    
    # Train LSTM for BTC (global crypto model)
    train_lstm_model("BTC-USD", "quantix_btc_lstm", epochs=50)
    
    # Train LSTM for Nifty50 (Indian market model)
    train_lstm_model("^NSEI", "quantix_nifty_lstm", epochs=50)
    
    # Train CNN on diverse tickers for pattern recognition
    cnn_tickers = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA", "NVDA", 
                   "RELIANCE.NS", "TCS.NS", "AMZN", "GOOGL"]
    train_cnn_model(cnn_tickers, epochs=30)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    log(f"\nFull training pipeline completed in {elapsed:.1f} minutes")
    log(f"All .pth weights saved to: {PROJECT_DIR}")
    log("Restart the server to load the new weights.\n")

if __name__ == "__main__":
    run_full_training()
