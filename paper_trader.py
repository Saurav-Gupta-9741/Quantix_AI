import os
import json
import time
import requests
import yfinance as yf
from datetime import datetime

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "signals.json")
API_URL = "http://127.0.0.1:8000/api/analyze"

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA",
    "BTC-USD", "ETH-USD",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    "XOM", "CVX"
]

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {"balance_usd": 100000.0, "holdings": {}, "trade_history": []}
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_signals():
    if not os.path.exists(SIGNALS_FILE):
        return []
    with open(SIGNALS_FILE, "r") as f:
        return json.load(f)

def save_signals(data):
    with open(SIGNALS_FILE, "w") as f:
        json.dump(data, f, indent=4)

def run_paper_trading_tick():
    import concurrent.futures
    print(f"\n[======== PAPER TRADING TICK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ========]")
    portfolio = load_portfolio()
    current_signals = load_signals()
    new_signals = []
    
    for ticker in WATCHLIST:
        print(f"[*] Analyzing {ticker}...")
        try:
            params = {
                "ticker": ticker,
                "asset_class": "Global_Cluster",
                "sector": "all",
                "market_cap": "10",
                "pe_ratio": "50",
                "volume": "1000000",
                "beta": "any",
                "timeframe": "swing",
                "risk": "dynamic"
            }
            res = requests.get(API_URL, params=params, timeout=120)
            if res.status_code != 200:
                print(f"[-] API Error for {ticker}: {res.status_code}")
                continue
                
            data = res.json()
            signal = data['execution_suggestion']['signal']
            current_price = data['current_price']['usd']
            rl_size_str = data['execution_suggestion']['rl_position_size'].replace('%', '')
            rl_size = float(rl_size_str) / 100.0
            
            if signal == "BUY" and ticker not in portfolio['holdings']:
                allocate_amount = portfolio['balance_usd'] * rl_size
                if allocate_amount > 100:
                    qty = allocate_amount / current_price
                    sig_obj = {
                        "ticker": ticker,
                        "signal": "BUY",
                        "confidence": data.get("strategy_summary", "").split("Confidence:")[0][-5:] if "Confidence" in data.get("strategy_summary", "") else "High",
                        "predicted_roi": data.get("deep_learning_analysis", {}).get("lstm_expected_roi", "0.0%"),
                        "current_price": current_price,
                        "recommended_qty": qty,
                        "cost": allocate_amount,
                        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    if not any(s['ticker'] == ticker for s in current_signals) and not any(s['ticker'] == ticker for s in new_signals):
                        new_signals.append(sig_obj)
                        print(f"[+] SIGNAL GENERATED: {ticker}")
        except Exception as e:
            print(f"[-] Error processing {ticker}: {e}")
            
    # Calculate Total Equity
    total_equity = portfolio['balance_usd']
    for tkr, data in portfolio['holdings'].items():
        try:
            live_price = float(yf.Ticker(tkr).fast_info.last_price)
            if "NS" in tkr.upper() or "BO" in tkr.upper():
                live_price = live_price / 83.5
            total_equity += data['qty'] * live_price
        except:
            total_equity += data['total_cost']  # fallback to cost basis
            
    # Record equity history
    if 'equity_history' not in portfolio:
        portfolio['equity_history'] = []
    
    portfolio['equity_history'].append({
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "equity": total_equity
    })
    
    # Keep history clean
    portfolio['trade_history'] = portfolio['trade_history'][:50]
    
    save_portfolio(portfolio)
    
    # Save active pending signals
    all_signals = current_signals + new_signals
    save_signals(all_signals)
    
    print(f"[!] Tick Complete. New Signals Generated: {len(new_signals)}")

if __name__ == "__main__":
    run_paper_trading_tick()
