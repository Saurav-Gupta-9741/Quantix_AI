from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import asyncio
import os
import io
import torch
import numpy as np
import yfinance as yf
import pandas as pd
import requests
import joblib
import mplfinance as mpf
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')
import threading
from transformers import pipeline
import torch
import torchvision.models as models
from torchvision.transforms import functional as TF
from io import BytesIO
from contextlib import contextmanager
from PIL import Image
from torchvision import transforms
from ticker_normalizer import TickerNormalizer
from lstm_model import TradingLSTM
from cnn_model import CandlestickCNN
from news_sentiment_agent import NewsSentimentAgent
from backtest import run_deep_learning_backtest
from rl_agent import RLPortfolioManager
from shared_state import init_db, get_conn, _db_lock, get_balance, update_balance
from circuit_breaker import fetch_history, fetch_last_price
import json
from datetime import datetime

app = FastAPI(title="Quantix AI Terminal")

# Setup SSE Event Queue
signal_queue = asyncio.Queue()

def push_event(data_dict):
    """Pushes event to the SSE stream."""
    try:
        asyncio.get_event_loop().call_soon_threadsafe(signal_queue.put_nowait, data_dict)
    except:
        pass

@app.on_event("startup")
def startup_event():
    init_db()
    import threading
    def run_risk():
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "risk_manager.py")
        subprocess.run(["python", script_path])
    
    t = threading.Thread(target=run_risk, daemon=True)
    t.start()
    print("[+] QUANTIX: Auto-Take-Profit Risk Manager started in background.")

normalizer = TickerNormalizer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
news_agent = NewsSentimentAgent(finbert_analyzer=finbert)
rl_manager = RLPortfolioManager()

# Global Model Registry (F2 Fix)
class ModelRegistry:
    def __init__(self, finbert):
        self._models = {}
        self.finbert = finbert
        
    def register(self, name, model, weights_path, device):
        if os.path.exists(weights_path):
            try:
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.eval()
                for param in model.parameters():
                    param.requires_grad_(False)
                self._models[name] = model
                print(f"[+] QUANTIX: {name} Loaded & Frozen.")
            except Exception as e:
                print(f"[-] QUANTIX: {name} Weights mismatch.")
        else:
            print(f"[-] QUANTIX: {name} weights not found.")

    def has(self, name):
        return name in self._models

    @contextmanager
    def infer(self, name):
        with torch.inference_mode():
            yield self._models[name]

registry = ModelRegistry(finbert)
device = torch.device("cpu")

registry.register("btc_lstm", TradingLSTM(input_size=5).to(device), os.path.join(os.path.dirname(__file__), "quantix_btc_lstm.pth"), device)
registry.register("nifty_lstm", TradingLSTM(input_size=5).to(device), os.path.join(os.path.dirname(__file__), "quantix_nifty_lstm.pth"), device)
registry.register("cnn", CandlestickCNN(num_classes=3).to(device), os.path.join(os.path.dirname(__file__), "quantix_cnn_v1.pth"), device)

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_technical_indicators(df):
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
    return df

# F3 Fix: Model Drift / Staleness Detection
def check_model_drift(ticker, model, live_data, device):
    """Computes RMSE of predictions over the last 15 days vs actuals."""
    try:
        from sklearn.preprocessing import MinMaxScaler
        closes = live_data['Close'].values
        if len(closes) < 45: return False
        
        test_actuals = closes[-15:]
        preds = []
        for i in range(15):
            idx = -15 + i
            window_df = live_data.iloc[idx-30:idx]
            if len(window_df) < 30: continue
            
            recent_30 = window_df[['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']].values
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            recent_30_scaled = feature_scaler.fit_transform(recent_30)
            seq_tensor = torch.tensor(np.array([recent_30_scaled]), dtype=torch.float32).to(device)
            
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaler.fit(window_df[['Close']].values)
            
            with torch.inference_mode():
                raw_pred = model(seq_tensor).numpy()
            unscaled_pred = target_scaler.inverse_transform(raw_pred)[0][0]
            preds.append(unscaled_pred)
            
        if len(preds) < 15: return False
        
        rmse = np.sqrt(np.mean((np.array(preds) - test_actuals)**2)) / np.mean(test_actuals)
        if rmse > 0.05: 
            print(f"[!] MODEL DRIFT DETECTED on {ticker}: RMSE {rmse:.4f}")
            return True
        return False
    except Exception as e:
        return False

@app.get("/", response_class=HTMLResponse)
def read_root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/search")
def search_ticker(q: str):
    if not q: return []
    try:
        url = f"https://symbol-search.tradingview.com/symbol_search/?text={q}&hl=1&exchange=&lang=en&type=&domain=production"
        headers = {'Origin': 'https://www.tradingview.com', 'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        results = []
        for item in data[:8]:
            symbol = item.get("symbol", "")
            exchange = item.get("exchange", "")
            asset_type = item.get("type", "")
            yahoo_ticker = symbol
            if exchange == "NSE": yahoo_ticker = f"{symbol}.NS"
            elif exchange == "BSE": yahoo_ticker = f"{symbol}.BO"
            elif asset_type == "crypto": yahoo_ticker = f"{symbol}-USD"
            elif exchange == "LSE": yahoo_ticker = f"{symbol}.L"
            elif exchange == "TSX": yahoo_ticker = f"{symbol}.TO"
            results.append({
                "ticker": yahoo_ticker,
                "display_symbol": symbol,
                "name": item.get("description", "Unknown"),
                "exchange": exchange,
                "type": asset_type
            })
        return results
    except Exception as e:
        return []

@app.get("/api/analyze")
def analyze(ticker: str, asset_class: str, beta: str, timeframe: str, risk: str):
    print(f"\n[>>>] INCOMING QUERY: {ticker}")
    clean_ticker = normalizer.normalize(ticker)
    
    try:
        live_data = fetch_history(yf.Ticker(clean_ticker), period="80d", interval="1d")
    except Exception as e:
        return {"error": str(e)}

    if live_data.empty:
        return {"error": f"No market data found for {clean_ticker}."}

    # F1 Fix: Data Recency Check
    last_date = live_data.index[-1].tz_localize(None)
    if (datetime.now() - last_date).days > 5:
        return {"error": f"Stale data for {clean_ticker}. Ticker may be halted or delisted."}
    
    live_data = calculate_technical_indicators(live_data)
    current_price = live_data['Close'].iloc[-1].item()
    
    recent_30 = live_data[['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']].tail(30).values
    from sklearn.preprocessing import MinMaxScaler
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    recent_30_scaled = feature_scaler.fit_transform(recent_30)
    seq_tensor = torch.tensor(np.array([recent_30_scaled]), dtype=torch.float32).to(device)
    
    exchange_rate = 83.5
    is_inr_native = "NS" in clean_ticker.upper() or "BO" in clean_ticker.upper()
    price_inr = float(current_price) if is_inr_native else float(current_price * exchange_rate)
    price_usd = float(current_price / exchange_rate) if is_inr_native else float(current_price)
    
    lstm_name = "nifty_lstm" if is_inr_native else "btc_lstm"
    
    if registry.has(lstm_name):
        is_drifting = False
        with registry.infer(lstm_name) as model:
            is_drifting = check_model_drift(clean_ticker, model, live_data, device)
            
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaler.fit(live_data[['Close']].tail(30).values)
            raw_prediction = model(seq_tensor).numpy()
            unscaled_pred = target_scaler.inverse_transform(raw_prediction)
            
        if is_inr_native:
            predicted_price_inr = float(unscaled_pred[0][0])
            predicted_price_usd = predicted_price_inr / exchange_rate
        else:
            predicted_price_usd = float(unscaled_pred[0][0])
            predicted_price_inr = predicted_price_usd * exchange_rate
            
        prediction_method = "Neural Network (BiLSTM+Attention)"
        
        # Override if drifting
        if is_drifting:
            predicted_price_usd = price_usd
            predicted_price_inr = price_inr
            prediction_method += " [SUPPRESSED DUE TO DRIFT]"
    else:
        rsi = float(live_data['RSI'].iloc[-1])
        macd = float(live_data['MACD'].iloc[-1])
        macd_signal = float(live_data['MACD_Signal'].iloc[-1])
        close_series = live_data['Close'].tail(30)
        ema_9 = float(close_series.ewm(span=9, adjust=False).mean().iloc[-1])
        ema_21 = float(close_series.ewm(span=21, adjust=False).mean().iloc[-1])
        
        ta_score = 0.0
        if rsi < 30: ta_score += 0.4
        elif rsi < 45: ta_score += 0.15
        elif rsi > 70: ta_score -= 0.4
        elif rsi > 55: ta_score -= 0.15
        
        if macd > macd_signal: ta_score += 0.3
        else: ta_score -= 0.3
        
        if ema_9 > ema_21: ta_score += 0.3
        else: ta_score -= 0.3
        
        predicted_price_usd = price_usd * (1 + ta_score * 0.05)
        prediction_method = f"Technical Analysis (RSI={rsi:.1f}, MACD={'Bullish' if macd > macd_signal else 'Bearish'}, EMA={'Up' if ema_9 > ema_21 else 'Down'})"
    
    lstm_signal = "BULLISH" if predicted_price_usd > price_usd else ("BEARISH" if predicted_price_usd < price_usd else "HOLD")
    predicted_price_inr = float(predicted_price_usd * exchange_rate)
    
    pattern_map = {0: "Bull Flag", 1: "Bear Flag", 2: "Consolidation", 3: "Head & Shoulders", 4: "Double Bottom"}
    
    if registry.has("cnn"):
        # F1 Fix: Thread-safe Matplotlib via Agg Buffer
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)
        fig, axes = mpf.plot(live_data.tail(30), type='candle', style=s, volume=False, returnfig=True)
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        import matplotlib.pyplot as plt
        plt.close(fig) # Prevent memory leaks
        
        img = Image.open(buf).convert("RGB")
        img_tensor = cnn_transform(img).unsqueeze(0).to(device)
        
        with registry.infer("cnn") as model:
            cnn_outputs = model(img_tensor)
            _, predicted_class = torch.max(cnn_outputs, 1)
        
        detected_pattern = pattern_map[predicted_class.item()]
    else:
        closes = live_data['Close'].tail(15).values
        highs = [float(x) for x in live_data['High'].tail(15).values]
        lows = [float(x) for x in live_data['Low'].tail(15).values]
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        price_range = max(highs) - min(lows)
        recent_range = max(highs[-5:]) - min(lows[-5:])
        
        if higher_highs >= 9 and recent_range < price_range * 0.4: detected_pattern = "Bull Flag"
        elif lower_lows >= 9 and recent_range < price_range * 0.4: detected_pattern = "Bear Flag"
        elif recent_range < price_range * 0.3: detected_pattern = "Consolidation"
        elif higher_highs > lower_lows: detected_pattern = "Bull Flag"
        elif lower_lows > higher_highs: detected_pattern = "Bear Flag"
        else: detected_pattern = "Consolidation"
    
    cnn_signal = "BULLISH" if ("Bull" in detected_pattern or "Bottom" in detected_pattern) else ("BEARISH" if ("Bear" in detected_pattern or "Head" in detected_pattern) else "NEUTRAL")
    
    start_day = 15; end_day = 20 
    
    # 3. News Sentiment NLP
    try:
        finbert_signal, finbert_summary = news_agent.analyze_fundamentals(clean_ticker)
    except Exception as e:
        finbert_signal, finbert_summary = "NEUTRAL", f"NLP Offline: {e}"
    
    recent_vol = float(live_data['Volume'].tail(3).mean())
    avg_vol = float(live_data['Volume'].mean())
    whale_detected = bool(recent_vol > (avg_vol * 1.5))
    
    score_breakdown = {
        "LSTM (40%)": 0.4 if lstm_signal == "BULLISH" else (-0.4 if lstm_signal == "BEARISH" else 0.0),
        "CNN (30%)": 0.3 if cnn_signal == "BULLISH" else (-0.3 if cnn_signal == "BEARISH" else 0.0),
        "FinBERT (30%)": 0.3 if finbert_signal == "BULLISH" else (-0.3 if finbert_signal == "BEARISH" else 0.0),
        "Whale Penalty": -0.1 if whale_detected else 0.0
    }
    
    score = sum(score_breakdown.values())
    
    if score >= 0.3: action = "BUY"
    elif score <= -0.3: action = "SELL"
    else: action = "HOLD"
    
    confidence = abs(score)
    roi_percent = float(((predicted_price_usd - price_usd) / price_usd) * 100)
    
    daily_returns = live_data['Close'].pct_change().dropna().tail(30).values
    rl_size = rl_manager.get_position_size(action, confidence, daily_returns)
    
    real_beta = "Dynamic"
    try:
        spy_data = fetch_history(yf.Ticker("SPY"), period="60d", interval="1d")
        if not spy_data.empty and len(spy_data) > 10:
            asset_returns = live_data['Close'].pct_change().dropna().tail(30)
            market_returns = spy_data['Close'].pct_change().dropna().tail(30)
            min_len = min(len(asset_returns), len(market_returns))
            if min_len > 5:
                cov = np.cov(asset_returns.values[-min_len:], market_returns.values[-min_len:])
                real_beta = str(round(float(cov[0][1] / cov[1][1]), 2))
    except:
        pass
    
    return {
        "asset": clean_ticker,
        "current_price": {"usd": round(price_usd, 2), "inr": round(price_inr, 2)},
        "filtered_data": {"asset_class": asset_class.replace('_', ' '), "beta_simulated": real_beta},
        "ai_analysis": {
            "lstm_predicted_price": {"usd": round(predicted_price_usd, 2), "inr": round(predicted_price_inr, 2)},
            "cnn_pattern": detected_pattern,
            "llm_advisor_summary": finbert_summary,
            "whale_manipulation_risk": "HIGH" if whale_detected else "LOW",
            "score_breakdown": score_breakdown
        },
        "strategy_summary": (
            f"Based on a consensus score of {score:.2f}, the system recommends a {action}. "
            f"Expected ROI: {roi_percent:.2f}%."
        ),
        "execution_suggestion": {
            "signal": action,
            "expected_roi": f"{roi_percent:.2f}%",
            "rl_position_size": f"{rl_size}%",
            "stop_loss": {
                "usd": round(price_usd * (0.97 if action == 'BUY' else 1.03), 2),
                "inr": round(price_inr * (0.97 if action == 'BUY' else 1.03), 2)
            }
        },
        "prediction_method": prediction_method
    }

@app.get("/api/backtest")
def backtest_ticker(ticker: str):
    clean_ticker = normalizer.normalize(ticker)
    try:
        df = fetch_history(yf.Ticker(clean_ticker), period="1y", interval="1d")
        if df.empty: return {"error": "No historical data found."}
        df = calculate_technical_indicators(df)
        is_inr_native = "NS" in clean_ticker.upper() or "BO" in clean_ticker.upper()
        model_path = os.path.join(os.path.dirname(__file__), "quantix_nifty_lstm.pth") if is_inr_native else os.path.join(os.path.dirname(__file__), "quantix_btc_lstm.pth")
        res = run_deep_learning_backtest(df, model_weights_path=model_path, days_to_test=365)
        return res
    except Exception as e:
        return {"error": str(e)}

# F4 Fix: SSE Stream
@app.get("/api/stream")
async def event_stream():
    async def generate():
        while True:
            event = await signal_queue.get()
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/trigger_refresh")
def trigger_refresh():
    push_event({"type": "refresh"})
    return {"status": "success"}

@app.get("/api/portfolio")
def get_portfolio():
    with _db_lock:
        conn = get_conn()
        balance = get_balance()
        holdings = {}
        cursor = conn.execute("SELECT * FROM holdings")
        for row in cursor.fetchall():
            ticker, qty, entry_price, total_cost, date = row
            try:
                live_price = fetch_last_price(yf.Ticker(ticker))
                if "NS" in ticker.upper() or "BO" in ticker.upper():
                    live_price = live_price / 83.5
                unrealized_pnl = (live_price - entry_price) * qty
                pnl_pct = ((live_price - entry_price) / entry_price) * 100
                holdings[ticker] = {
                    "qty": qty, "entry_price": entry_price, "total_cost": total_cost, "date": date,
                    "live_price": live_price, "unrealized_pnl": unrealized_pnl, "pnl_pct": pnl_pct
                }
            except:
                holdings[ticker] = {
                    "qty": qty, "entry_price": entry_price, "total_cost": total_cost, "date": date,
                    "live_price": entry_price, "unrealized_pnl": 0.0, "pnl_pct": 0.0
                }
                
        trade_history = []
        cursor = conn.execute("SELECT log_msg FROM trade_history ORDER BY id DESC LIMIT 50")
        for row in cursor.fetchall():
            trade_history.append(row[0])
            
        conn.close()
    return {"balance_usd": balance, "holdings": holdings, "trade_history": trade_history}

@app.post("/api/portfolio/run")
async def run_portfolio_tick():
    import asyncio
    script_path = os.path.join(os.path.dirname(__file__), "paper_trader.py")
    process = await asyncio.create_subprocess_exec("python", script_path)
    await process.communicate()
    push_event({"type": "refresh"})
    return {"status": "Paper trading sweep completed. Signals generated."}

@app.get("/api/signals")
def get_signals():
    with _db_lock:
        conn = get_conn()
        signals = []
        cursor = conn.execute("SELECT * FROM signals")
        for row in cursor.fetchall():
            signals.append({
                "ticker": row[0], "signal": row[1], "confidence": row[2],
                "predicted_roi": row[3], "current_price": row[4], "recommended_qty": row[5],
                "cost": row[6], "date": row[7]
            })
        conn.close()
    return signals

@app.post("/api/execute_trade")
async def execute_trade(request: Request):
    data = await request.json()
    ticker = data.get("ticker")
    
    with _db_lock:
        conn = get_conn()
        cursor = conn.execute("SELECT * FROM signals WHERE ticker=?", (ticker,))
        sig = cursor.fetchone()
        if not sig:
            conn.close()
            return {"error": "Signal not found."}
            
        ticker, signal, conf, roi, current_price, qty, cost, date = sig
        balance = get_balance()
        
        if balance >= cost:
            update_balance(balance - cost)
            conn.execute("INSERT OR REPLACE INTO holdings (ticker, qty, entry_price, total_cost, date) VALUES (?, ?, ?, ?, ?)",
                         (ticker, qty, current_price, cost, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.execute("DELETE FROM signals WHERE ticker=?", (ticker,))
            conn.execute("INSERT INTO trade_history (log_msg, date) VALUES (?, ?)", 
                         (f"BUY {ticker}: {qty:.4f} shares @ ${current_price:.2f}", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()
            push_event({"type": "refresh"})
            return {"status": "success"}
        else:
            conn.close()
            return {"error": "Insufficient funds to execute trade."}

@app.websocket('/ws/live_prices/{ticker}')
async def websocket_endpoint(websocket: WebSocket, ticker: str):
    await websocket.accept()
    clean_ticker = normalizer.normalize(ticker)
    try:
        ticker_obj = yf.Ticker(clean_ticker)
        base_price = fetch_last_price(ticker_obj)
    except:
        base_price = 1000.0

    try:
        while True:
            try:
                live_price = fetch_last_price(yf.Ticker(clean_ticker))
            except:
                live_price = base_price
            await websocket.send_json({'ticker': ticker, 'live_price': round(live_price, 2)})
            await asyncio.sleep(5.0)
    except WebSocketDisconnect:
        pass
