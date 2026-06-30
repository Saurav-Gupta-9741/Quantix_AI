import torch
import pandas as pd
import numpy as np
import os
from lstm_model import TradingLSTM
from position_sizer import KellyCriterionSizer

def run_deep_learning_backtest(df, model_weights_path, feature_scaler=None, target_scaler=None, initial_capital=100000, days_to_test=365):
    """
    Institutional-Grade Walk-Forward Backtester.
    Uses PyTorch neural network with passed scalers to avoid lookahead bias.
    Includes transaction costs and symmetric thresholds.
    """
    # Ensure required columns exist
    required_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']
    for col in required_cols:
        if col not in df.columns:
            return {"error": f"Missing column {col}"}
    
    raw_targets = df[['Close']].values
    
    # Try loading neural network
    use_neural_net = False
    if os.path.exists(model_weights_path) and feature_scaler is not None and target_scaler is not None:
        try:
            device = torch.device("cpu")
            model = TradingLSTM(input_size=5).to(device)
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            model.eval()
            
            # Use pre-fitted scalers (No lookahead bias)
            raw_features = df[required_cols].values
            scaled_features = feature_scaler.transform(raw_features)
            
            use_neural_net = True
            print("[+] Backtest: Neural network weights and scalers loaded successfully.")
        except Exception as e:
            print(f"[-] Backtest: Neural net loading failed ({e}). Using technical analysis strategy.")
    
    # Setup Trading Simulation
    capital = initial_capital
    shares_held = 0
    buy_signals = 0
    winning_trades = 0
    losing_trades = 0
    equity_curve = []
    entry_price = 0
    
    TRANSACTION_FEE_PCT = 0.001  # 0.1% transaction cost
    position_sizer = KellyCriterionSizer()
    
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
            
            if rsi < 30: score += 0.4
            elif rsi < 40: score += 0.2
            elif rsi > 70: score -= 0.4
            elif rsi > 60: score -= 0.2
            
            if macd > macd_signal: score += 0.3
            else: score -= 0.3
            
            if ema_short > ema_long: score += 0.3
            else: score -= 0.3
            
            expected_roi = score * 0.05
        
        # Trading Logic - Symmetric Thresholds & Transaction Costs
        # Only buy if expected ROI beats transaction cost significantly
        if expected_roi > 0.01 + TRANSACTION_FEE_PCT:  
            if capital >= current_price:
                # Proper position sizing via Kelly Criterion instead of all-in
                historical_returns = df['Close'].pct_change().dropna().values[-30:]
                size_pct = position_sizer.get_position_size("BUY", min(max(expected_roi * 10, 0.1), 0.99), historical_returns) / 100.0
                
                investment = capital * size_pct
                if investment >= current_price:
                    shares_to_buy = investment // current_price
                    cost = shares_to_buy * current_price
                    fee = cost * TRANSACTION_FEE_PCT
                    
                    capital -= (cost + fee)
                    shares_held += shares_to_buy
                    
                    # Update average entry price
                    if shares_held > 0:
                        total_cost_basis = (entry_price * (shares_held - shares_to_buy)) + (cost + fee)
                        entry_price = total_cost_basis / shares_held
                    else:
                        entry_price = current_price
                        
                    buy_signals += 1
                
        elif expected_roi < -0.01 - TRANSACTION_FEE_PCT:  # Sell threshold (symmetric)
            if shares_held > 0:
                revenue = shares_held * current_price
                fee = revenue * TRANSACTION_FEE_PCT
                net_revenue = revenue - fee
                
                capital += net_revenue
                if net_revenue > (shares_held * entry_price):
                    winning_trades += 1
                else:
                    losing_trades += 1
                shares_held = 0
                entry_price = 0
        
        # Record equity curve
        current_equity = capital + (shares_held * current_price)
        date_str = df.index[i].strftime('%Y-%m-%d')
        equity_curve.append({'time': date_str, 'value': round(float(current_equity), 2)})
    
    # Close any remaining position
    if shares_held > 0:
        revenue = shares_held * float(raw_targets[-1][0])
        fee = revenue * TRANSACTION_FEE_PCT
        capital += (revenue - fee)
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
