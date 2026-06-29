import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from lstm_model import TradingLSTM

def run_deep_learning_backtest(df, model_weights_path, initial_capital=100000, days_to_test=365):
    """
    Institutional-Grade Walk-Forward Backtester.
    Uses PyTorch neural network if weights match, otherwise falls back to
    REAL technical analysis strategy (RSI + MACD crossover).
    """
    # Ensure required columns exist
    required_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']
    for col in required_cols:
        if col not in df.columns:
            return {"error": f"Missing column {col}"}
    
    raw_targets = df[['Close']].values
    
    # Try loading neural network
    use_neural_net = False
    if os.path.exists(model_weights_path):
        try:
            device = torch.device("cpu")
            model = TradingLSTM(input_size=5).to(device)
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            model.eval()
            
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            raw_features = df[required_cols].values
            scaled_features = feature_scaler.fit_transform(raw_features)
            scaled_targets = target_scaler.fit_transform(raw_targets)
            use_neural_net = True
            print("[+] Backtest: Neural network weights loaded successfully.")
        except RuntimeError as e:
            print(f"[-] Backtest: Weights mismatch ({e}). Using technical analysis strategy.")
    
    # Setup Trading Simulation
    capital = initial_capital
    shares_held = 0
    buy_signals = 0
    winning_trades = 0
    losing_trades = 0
    equity_curve = []
    entry_price = 0
    
    max_possible_days = len(df) - 31
    if days_to_test > max_possible_days:
        days_to_test = max_possible_days
    if days_to_test <= 0:
        return {"error": "Not enough historical data for backtest."}
    
    test_start_idx = len(df) - days_to_test - 1
    
    for i in range(test_start_idx, len(df) - 1):
        current_price = float(raw_targets[i][0])
        
        if use_neural_net:
            # Neural Network prediction
            sequence = scaled_features[i - 30 : i]
            if len(sequence) < 30:
                continue
            seq_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32).to(device)
            with torch.no_grad():
                scaled_prediction = model(seq_tensor).item()
            predicted_price = target_scaler.inverse_transform([[scaled_prediction]])[0][0]
            expected_roi = (predicted_price - current_price) / current_price
        else:
            # REAL Technical Analysis Strategy: RSI + MACD Crossover
            rsi = df['RSI'].iloc[i]
            macd = df['MACD'].iloc[i]
            macd_signal = df['MACD_Signal'].iloc[i]
            
            # EMA trend direction
            close_prices = df['Close'].iloc[max(0, i-20):i+1]
            ema_short = close_prices.ewm(span=9, adjust=False).mean().iloc[-1]
            ema_long = close_prices.ewm(span=21, adjust=False).mean().iloc[-1]
            
            # Score-based signal generation
            score = 0.0
            
            # RSI signals
            if rsi < 30: score += 0.4      # Oversold = buy signal
            elif rsi < 40: score += 0.2
            elif rsi > 70: score -= 0.4    # Overbought = sell signal
            elif rsi > 60: score -= 0.2
            
            # MACD crossover
            if macd > macd_signal: score += 0.3  # Bullish crossover
            else: score -= 0.3                    # Bearish crossover
            
            # EMA trend
            if ema_short > ema_long: score += 0.3  # Uptrend
            else: score -= 0.3                      # Downtrend
            
            expected_roi = score * 0.05  # Convert score to expected ROI
        
        # Trading Logic
        if expected_roi > 0.01:  # Buy threshold
            if capital >= current_price:
                shares_bought = capital // current_price
                capital -= shares_bought * current_price
                shares_held += shares_bought
                entry_price = current_price
                buy_signals += 1
                
        elif expected_roi < 0.00:  # Sell threshold
            if shares_held > 0:
                sell_revenue = shares_held * current_price
                capital += sell_revenue
                if current_price > entry_price:
                    winning_trades += 1
                else:
                    losing_trades += 1
                shares_held = 0
        
        # Record equity curve
        current_equity = capital + (shares_held * current_price)
        date_str = df.index[i].strftime('%Y-%m-%d')
        equity_curve.append({'time': date_str, 'value': round(float(current_equity), 2)})
    
    # Close any remaining position
    if shares_held > 0:
        capital += shares_held * float(raw_targets[-1][0])
        shares_held = 0
    
    # Analytics
    final_roi = ((capital - initial_capital) / initial_capital) * 100
    first_price = float(raw_targets[test_start_idx][0])
    last_price = float(raw_targets[-1][0])
    buy_and_hold_roi = ((last_price - first_price) / first_price) * 100
    
    total_closed_trades = winning_trades + losing_trades
    accuracy = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
    
    return {
        "error": None,
        "days_tested": days_to_test,
        "total_trades": buy_signals,
        "win_rate": round(accuracy, 2),
        "ai_roi": round(float(final_roi), 2),
        "buy_hold_roi": round(float(buy_and_hold_roi), 2),
        "equity_curve": equity_curve,
        "strategy": "Neural Network" if use_neural_net else "Technical Analysis (RSI+MACD+EMA)"
    }

if __name__ == "__main__":
    df = pd.read_csv("bitcoin_processed_data.csv", index_col='Date', parse_dates=True)
    res = run_deep_learning_backtest(df, "quantix_btc_lstm.pth", days_to_test=365)
    print(res)
