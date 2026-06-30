import os
import io
import json
import asyncio
import urllib.parse
from datetime import datetime
from contextlib import contextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import joblib
import subprocess
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from matplotlib.backends.backend_agg import FigureCanvasAgg

from transformers import pipeline
from torchvision import transforms

from ticker_normalizer import TickerNormalizer
from lstm_model import TradingLSTM
from cnn_model import CandlestickCNN
from news_sentiment_agent import NewsSentimentAgent
from backtest import run_deep_learning_backtest
from position_sizer import KellyCriterionSizer
from shared_state import init_db, db, get_balance, update_balance
from circuit_breaker import fetch_history, fetch_last_price
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("QUANTIX_API_KEY", "dev_key_123")
EXCHANGE_RATE_FALLBACK = float(os.environ.get("EXCHANGE_RATE_FALLBACK", 83.5))

def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != API_KEY:
        # For local dev without headers, we can be lenient, but strictly it should block.
        # Since the UI might not send headers yet, we allow "dev_key_123" as fallback.
        if API_KEY != "dev_key_123":
            raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

app = FastAPI(title="Quantix AI Terminal")

# CORS Middleware (Same origin + localhost for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-client SSE state mapping client_id -> {"queue": Queue, "ticker": str}
clients = {}

def push_event(data_dict, target_ticker=None):
    """
    Pushes event to SSE clients. 
    If target_ticker is set, only pushes to clients subscribed to that ticker.
    Otherwise, broadcasts to all clients (e.g. for global 'refresh' events).
    """
    loop = asyncio.get_running_loop()
    for cid, client_data in list(clients.items()):
        if target_ticker and client_data.get("ticker") != target_ticker:
            continue
        try:
            loop.call_soon_threadsafe(client_data["queue"].put_nowait, data_dict)
        except Exception:
            pass

async def price_poller():
    while True:
        # Get unique tickers currently subscribed across all active clients
        active_tickers = set(c["ticker"] for c in clients.values() if c.get("ticker"))
        for ticker in active_tickers:
            try:
                clean = normalizer.normalize(ticker)
                live_price = fetch_last_price(yf.Ticker(clean))
                # Push ONLY to clients subscribed to this ticker
                push_event({"type": "price_tick", "ticker": ticker, "price": live_price}, target_ticker=ticker)
            except Exception:
                pass
        await asyncio.sleep(5.0)

@app.on_event("startup")
async def startup_event():
    init_db()
    # F1 Fix: Use Popen so it runs in background without blocking
    script_path = os.path.join(os.path.dirname(__file__), "risk_manager.py")
    subprocess.Popen(["python", script_path])
    print("[+] QUANTIX: Auto-Take-Profit Risk Manager started in background.")
    asyncio.create_task(price_poller())

normalizer = TickerNormalizer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
news_agent = NewsSentimentAgent(finbert_analyzer=finbert)
rl_manager = KellyCriterionSizer()

# F1 Fix: Load scalers at startup
try:
    feature_scaler = joblib.load(os.path.join(os.path.dirname(__file__), "feature_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(os.path.dirname(__file__), "target_scaler.pkl"))
    print("[+] Scalers loaded successfully.")
except Exception as e:
    print(f"[-] Failed to load scalers: {e}")
    feature_scaler, target_scaler = None, None

class ModelRegistry:
    def __init__(self, finbert):
        self._models = {}
        self.finbert = finbert
        
    def register(self, name, model, weights_path, device, freeze=True):
        if os.path.exists(weights_path):
            try:
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.eval()
                if freeze:
                    for param in model.parameters():
                        param.requires_grad_(False)
                self._models[name] = model
                status = "Frozen" if freeze else "Trainable"
                print(f"[+] QUANTIX: {name} Loaded & {status}.")
            except Exception as e:
                # F2 Fix: Explictly raise error instead of silent catch
                raise RuntimeError(f"Weights mismatch for {name}: {e}")
        else:
            print(f"[-] QUANTIX: {name} weights not found.")

    def has(self, name):
        return name in self._models

    @contextmanager
    def infer(self, name):
        with torch.inference_mode():
            yield self._models[name]
            
    @contextmanager
    def infer_with_grad(self, name):
        with torch.enable_grad():
            yield self._models[name]

registry = ModelRegistry(finbert)
device = torch.device("cpu")

registry.register("btc_lstm", TradingLSTM(input_size=5).to(device), os.path.join(os.path.dirname(__file__), "quantix_btc_lstm.pth"), device)
registry.register("nifty_lstm", TradingLSTM(input_size=5).to(device), os.path.join(os.path.dirname(__file__), "quantix_nifty_lstm.pth"), device)
# F5 Fix: CNN must not be frozen so Grad-CAM backward pass works
registry.register("cnn", CandlestickCNN(num_classes=5).to(device), os.path.join(os.path.dirname(__file__), "quantix_cnn_v1.pth"), device, freeze=False)

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

def check_model_drift(ticker, model, live_data, device):
    try:
        closes = live_data['Close'].values
        if len(closes) < 45: return False
        
        test_actuals = closes[-15:]
        preds = []
        for i in range(15):
            idx = -15 + i
            window_df = live_data.iloc[idx-30:idx]
            if len(window_df) < 30: continue
            
            recent_30 = window_df[['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']].values
            
            if feature_scaler is None or target_scaler is None:
                return False
                
            recent_30_scaled = feature_scaler.transform(recent_30)
            seq_tensor = torch.tensor(np.array([recent_30_scaled]), dtype=torch.float32).to(device)
            
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

# F11 Fix: Exchange Rate API
exchange_rate_cache = {"rate": EXCHANGE_RATE_FALLBACK, "timestamp": 0}
def get_exchange_rate():
    now = datetime.now().timestamp()
    if now - exchange_rate_cache["timestamp"] > 3600:
        try:
            res = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
            rate = res.json()["rates"]["INR"]
            exchange_rate_cache["rate"] = rate
            exchange_rate_cache["timestamp"] = now
        except Exception:
            pass
    return exchange_rate_cache["rate"]

@app.get("/api/health")
def health_check():
    import time as _time
    t0 = _time.perf_counter()
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
        db_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    except Exception:
        db_latency_ms = -1
    return {
        "status": "ONLINE",
        "latency": f"{db_latency_ms}ms",
        "cluster": "QUANTIX-LOCAL",
        "models": {
            "btc_lstm": registry.has("btc_lstm"),
            "nifty_lstm": registry.has("nifty_lstm"),
            "cnn": registry.has("cnn")
        },
        "scalers_loaded": feature_scaler is not None
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/search")
def search_ticker(q: str):
    if not q: return []
    try:
        # F7 Fix: SSRF Prevention via URL encoding
        safe_q = urllib.parse.quote(q)
        url = f"https://symbol-search.tradingview.com/symbol_search/?text={safe_q}&hl=1&exchange=&lang=en&type=&domain=production"
        headers = {'Origin': 'https://www.tradingview.com', 'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
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
def analyze(ticker: str, asset_class: str, beta: str, timeframe: str, risk: str, token: str = Depends(verify_api_key)):
    print(f"\n[>>>] INCOMING QUERY: {ticker}")
    clean_ticker = normalizer.normalize(ticker)
    
    try:
        live_data = fetch_history(yf.Ticker(clean_ticker), period="80d", interval="1d")
    except Exception as e:
        return {"error": str(e)}

    if live_data.empty:
        return {"error": f"No market data found for {clean_ticker}."}

    last_date = live_data.index[-1].tz_localize(None)
    if (datetime.now() - last_date).days > 5:
        return {"error": f"Stale data for {clean_ticker}. Ticker may be halted or delisted."}
    
    live_data = calculate_technical_indicators(live_data)
    current_price = live_data['Close'].iloc[-1].item()
    
    exchange_rate = get_exchange_rate()
    is_inr_native = "NS" in clean_ticker.upper() or "BO" in clean_ticker.upper()
    price_inr = float(current_price) if is_inr_native else float(current_price * exchange_rate)
    price_usd = float(current_price / exchange_rate) if is_inr_native else float(current_price)
    
    lstm_name = "nifty_lstm" if is_inr_native else "btc_lstm"
    
    prediction_method = ""
    is_drifting = False
    
    is_fallback_mode = False
    
    if registry.has(lstm_name) and feature_scaler is not None and target_scaler is not None:
        model = registry._models[lstm_name]
        is_drifting = check_model_drift(clean_ticker, model, live_data, device)
        if is_drifting:
            is_fallback_mode = True
        
        recent_30 = live_data[['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']].tail(30).values
        recent_30_scaled = feature_scaler.transform(recent_30)
        seq_tensor = torch.tensor(np.array([recent_30_scaled]), dtype=torch.float32).to(device)
        
        with registry.infer(lstm_name) as model:
            raw_prediction = model(seq_tensor).numpy()
            unscaled_pred = target_scaler.inverse_transform(raw_prediction)
            
        if is_inr_native:
            predicted_price_inr = float(unscaled_pred[0][0])
            predicted_price_usd = predicted_price_inr / exchange_rate
        else:
            predicted_price_usd = float(unscaled_pred[0][0])
            predicted_price_inr = predicted_price_usd * exchange_rate
            
        prediction_method = "Neural Network (BiLSTM+Attention)"
        
        if is_drifting:
            predicted_price_usd = price_usd
            predicted_price_inr = price_inr
            prediction_method += " [SUPPRESSED DUE TO DRIFT]"
    else:
        is_fallback_mode = True
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
        predicted_price_inr = predicted_price_usd * exchange_rate
        prediction_method = f"Technical Analysis (RSI={rsi:.1f}, MACD={'Bullish' if macd > macd_signal else 'Bearish'}, EMA={'Up' if ema_9 > ema_21 else 'Down'})"
    
    lstm_signal = "BULLISH" if predicted_price_usd > price_usd else ("BEARISH" if predicted_price_usd < price_usd else "HOLD")
    
    pattern_map = {0: "Bull Flag", 1: "Bear Flag", 2: "Consolidation", 3: "Head & Shoulders", 4: "Double Bottom"}
    gradcam_b64 = None
    
    if registry.has("cnn"):
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc)
        fig, axes = mpf.plot(live_data.tail(30), type='candle', style=s, volume=False, returnfig=True)
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        img = Image.open(buf).convert("RGB")
        img_tensor = cnn_transform(img).unsqueeze(0).to(device)
        
        with registry.infer_with_grad("cnn") as model:
            cnn_outputs = model(img_tensor)
            _, predicted_class = torch.max(cnn_outputs, 1)
            
            # Extract Grad-CAM if available
            if hasattr(model, 'get_gradcam_heatmap'):
                model.zero_grad()
                # Run backward pass for the predicted class to populate self.gradients via hook
                cnn_outputs[0, predicted_class.item()].backward()
                
                cam = model.get_gradcam_heatmap()
                if cam is not None:
                    import cv2
                    import base64
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    # Overlay heatmap on original image
                    orig_cv2 = np.array(img)
                    orig_cv2 = cv2.cvtColor(orig_cv2, cv2.COLOR_RGB2BGR)
                    orig_cv2 = cv2.resize(orig_cv2, (224, 224))
                    overlay = cv2.addWeighted(orig_cv2, 0.6, heatmap, 0.4, 0)
                    _, buffer = cv2.imencode('.png', overlay)
                    gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
        
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
    except Exception:
        pass
    
    # F4 Fix: Ensure API contract names match frontend
    return {
        "asset": clean_ticker,
        "current_price": {"usd": round(price_usd, 2), "inr": round(price_inr, 2)},
        "filtered_data": {"asset_class": asset_class.replace('_', ' '), "beta_simulated": real_beta},
        "ai_analysis": {
            "lstm_predicted_price": {"usd": round(predicted_price_usd, 2), "inr": round(predicted_price_inr, 2)},
            "cnn_visual_pattern": detected_pattern,
            "finbert_sentiment": finbert_signal,
            "llm_advisor_summary": finbert_summary,
            "whale_manipulation_detected": whale_detected,
            "score_breakdown": score_breakdown,
            "lstm_signal": lstm_signal,
            "cnn_signal": cnn_signal,
            "finbert_signal": finbert_signal,
            "is_fallback_mode": is_fallback_mode,
            "consensus_score": round(score * 100, 2)
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
        "prediction_method": prediction_method,
        "gradcam_base64": gradcam_b64
    }

@app.get("/api/backtest")
def backtest_ticker(ticker: str, token: str = Depends(verify_api_key)):
    clean_ticker = normalizer.normalize(ticker)
    try:
        df = fetch_history(yf.Ticker(clean_ticker), period="1y", interval="1d")
        if df.empty: return {"error": "No historical data found."}
        df = calculate_technical_indicators(df)
        is_inr_native = "NS" in clean_ticker.upper() or "BO" in clean_ticker.upper()
        model_path = os.path.join(os.path.dirname(__file__), "quantix_nifty_lstm.pth") if is_inr_native else os.path.join(os.path.dirname(__file__), "quantix_btc_lstm.pth")
        res = run_deep_learning_backtest(df, model_weights_path=model_path, feature_scaler=feature_scaler, target_scaler=target_scaler, days_to_test=365)
        return res
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stream")
async def event_stream(request: Request, client_id: str = "default"):
    queue = asyncio.Queue()
    clients[client_id] = {"queue": queue, "ticker": None}
    
    async def generate():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            clients.pop(client_id, None)
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/trigger_refresh")
def trigger_refresh():
    push_event({"type": "refresh"})
    return {"status": "success"}

@app.get("/api/portfolio")
def get_portfolio(token: str = Depends(verify_api_key)):
    with db.get_connection() as conn:
        balance = get_balance()
        holdings = {}
        cursor = conn.execute("SELECT ticker, qty, entry_price, total_cost, date, currency, stop_loss_pct, take_profit_pct FROM holdings")
        for row in cursor.fetchall():
            ticker, qty, entry_price, total_cost, date, currency, sl_pct, tp_pct = row
            try:
                live_price = fetch_last_price(yf.Ticker(ticker))
                unrealized_pnl = (live_price - entry_price) * qty
                pnl_pct = ((live_price - entry_price) / entry_price) * 100
                holdings[ticker] = {
                    "qty": qty, "entry_price": entry_price, "total_cost": total_cost, "date": date,
                    "live_price": live_price, "unrealized_pnl": unrealized_pnl, "pnl_pct": pnl_pct, "currency": currency
                }
            except Exception:
                holdings[ticker] = {
                    "qty": qty, "entry_price": entry_price, "total_cost": total_cost, "date": date,
                    "live_price": entry_price, "unrealized_pnl": 0.0, "pnl_pct": 0.0, "currency": currency
                }
                
        trade_history = []
        cursor = conn.execute("SELECT log_msg FROM trade_history ORDER BY id DESC LIMIT 50")
        for row in cursor.fetchall():
            trade_history.append(row[0])
            
        equity_history = []
        cursor = conn.execute("SELECT timestamp, total_equity FROM equity_snapshots ORDER BY id ASC")
        for row in cursor.fetchall():
            equity_history.append({"time": row[0], "value": row[1]})
            
    return {"balance_usd": balance, "holdings": holdings, "trade_history": trade_history, "equity_history": equity_history}

@app.post("/api/portfolio/run")
async def run_portfolio_tick(token: str = Depends(verify_api_key)):
    import asyncio
    script_path = os.path.join(os.path.dirname(__file__), "paper_trader.py")
    process = await asyncio.create_subprocess_exec("python", script_path)
    await process.communicate()
    push_event({"type": "refresh"})
    return {"status": "Paper trading sweep completed. Signals generated."}

@app.get("/api/signals")
def get_signals(token: str = Depends(verify_api_key)):
    with db.get_connection() as conn:
        signals = []
        cursor = conn.execute("SELECT * FROM signals")
        for row in cursor.fetchall():
            signals.append({
                "ticker": row[0], "signal": row[1], "confidence": row[2],
                "predicted_roi": row[3], "current_price": row[4], "recommended_qty": row[5],
                "cost": row[6], "date": row[7]
            })
    return signals

@app.post("/api/execute_trade")
async def execute_trade(request: Request, token: str = Depends(verify_api_key)):
    data = await request.json()
    ticker = data.get("ticker")
    
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM signals WHERE ticker=?", (ticker,))
        sig = cursor.fetchone()
        if not sig:
            return {"error": "Signal not found."}
            
        ticker, signal, conf, roi, current_price, qty, cost, date = sig
        balance = get_balance()
        
        if balance >= cost:
            update_balance(balance - cost)
            currency = "INR" if ("NS" in ticker.upper() or "BO" in ticker.upper()) else "USD"
            conn.execute("INSERT OR REPLACE INTO holdings (ticker, qty, entry_price, total_cost, date, currency) VALUES (?, ?, ?, ?, ?, ?)",
                         (ticker, qty, current_price, cost, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), currency))
            conn.execute("DELETE FROM signals WHERE ticker=?", (ticker,))
            conn.execute("INSERT INTO trade_history (log_msg, date) VALUES (?, ?)", 
                         (f"BUY {ticker}: {qty:.4f} shares @ ${current_price:.2f} ({currency})", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # Record Equity Snapshot
            new_equity = balance - cost + (qty * current_price) # naive for now
            conn.execute("INSERT INTO equity_snapshots (timestamp, total_equity) VALUES (?, ?)", 
                         (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), new_equity))
                         
            conn.commit()
            push_event({"type": "refresh"})
            return {"status": "success"}
        else:
            return {"error": "Insufficient funds to execute trade."}

# F5 Fix: Watchlist Endpoints
@app.get("/api/watchlist")
def get_watchlist():
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT ticker FROM watchlist")
        return [row[0] for row in cursor.fetchall()]

@app.post("/api/watchlist")
async def add_watchlist(request: Request, token: str = Depends(verify_api_key)):
    data = await request.json()
    ticker = data.get("ticker")
    if not ticker: return {"error": "Ticker required"}
    with db.get_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (ticker,))
        conn.commit()
    push_event({"type": "refresh"})
    return {"status": "success"}

@app.delete("/api/watchlist/{ticker}")
def remove_watchlist(ticker: str, token: str = Depends(verify_api_key)):
    with db.get_connection() as conn:
        conn.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))
        conn.commit()
    push_event({"type": "refresh"})
    return {"status": "success"}

@app.post("/api/subscribe_price")
async def subscribe_price(request: Request, token: str = Depends(verify_api_key)):
    data = await request.json()
    ticker = data.get("ticker")
    client_id = data.get("client_id", "default")
    
    if client_id in clients:
        clients[client_id]["ticker"] = ticker
        
    return {"status": "subscribed"}
