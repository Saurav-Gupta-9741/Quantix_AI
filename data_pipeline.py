import yfinance as yf
import pandas as pd
import numpy as np

# Note: We will use the 'ta' library later for advanced technical indicators,
# but for now we manually calculate the core math to ensure deep understanding.

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = ema_fast - ema_slow
    data['MACD_Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    return data

def fetch_historical_data(ticker, period="10y", interval="1d"):
    """
    Downloads historical market data and engineers technical features
    required by the CNN and LSTM models.
    """
    print(f"[*] Downloading {period} of data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        print(f"[!] Warning: No data found for {ticker}")
        return None

    # Clean data
    df = df.dropna()
    
    # Feature Engineering for the LSTM
    print(f"[*] Engineering Mathematical Features for {ticker}...")
    df = calculate_rsi(df)
    df = calculate_macd(df)
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Drop rows with NaN values created by rolling windows
    df = df.dropna()
    
    print(f"[+] Successfully processed {len(df)} days of data for {ticker}.\n")
    return df

if __name__ == "__main__":
    print("--- OMNI-ASSET DATA PIPELINE INITIALIZED ---\n")
    
    # 1. Download NIFTY 50 Index (Indian Market)
    # Yahoo Finance ticker for Nifty 50 is ^NSEI
    nifty_data = fetch_historical_data(ticker="^NSEI", period="10y", interval="1d")
    if nifty_data is not None:
        nifty_data.to_csv("nifty50_processed_data.csv")
        print("Saved: nifty50_processed_data.csv")

    # 2. Download Bitcoin (Crypto Market)
    btc_data = fetch_historical_data(ticker="BTC-USD", period="10y", interval="1d")
    if btc_data is not None:
        btc_data.to_csv("bitcoin_processed_data.csv")
        print("Saved: bitcoin_processed_data.csv")
        
    print("\n--- PHASE 1: DATA PIPELINE COMPLETE ---")
