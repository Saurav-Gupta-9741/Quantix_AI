import os
import time
import requests
import yfinance as yf
from datetime import datetime
from shared_state import get_conn, _db_lock, get_balance

API_URL = "http://127.0.0.1:8000/api/analyze"

def get_watchlist():
    watchlist = []
    with _db_lock:
        conn = get_conn()
        cursor = conn.execute("SELECT ticker FROM watchlist")
        for row in cursor.fetchall():
            watchlist.append(row[0])
        conn.close()
    return watchlist

def get_holdings():
    holdings = set()
    with _db_lock:
        conn = get_conn()
        cursor = conn.execute("SELECT ticker FROM holdings")
        for row in cursor.fetchall():
            holdings.add(row[0])
        conn.close()
    return holdings

def get_signals():
    signals = set()
    with _db_lock:
        conn = get_conn()
        cursor = conn.execute("SELECT ticker FROM signals")
        for row in cursor.fetchall():
            signals.add(row[0])
        conn.close()
    return signals

def insert_signal(sig_obj):
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO signals 
            (ticker, signal, confidence, predicted_roi, current_price, recommended_qty, cost, date) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sig_obj['ticker'], sig_obj['signal'], sig_obj['confidence'], 
            sig_obj['predicted_roi'], sig_obj['current_price'], 
            sig_obj['recommended_qty'], sig_obj['cost'], sig_obj['date']
        ))
        conn.commit()
        conn.close()

def run_paper_trading_tick():
    print(f"\n[======== PAPER TRADING TICK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ========]")
    
    current_holdings = get_holdings()
    current_signals = get_signals()
    balance = get_balance()
    new_signals_count = 0
    
    watchlist = get_watchlist()
    
    for ticker in watchlist:
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
            
            if signal == "BUY" and ticker not in current_holdings:
                allocate_amount = balance * rl_size
                if allocate_amount > 100:
                    qty = allocate_amount / current_price
                    sig_obj = {
                        "ticker": ticker,
                        "signal": "BUY",
                        "confidence": data.get("strategy_summary", "").split("Confidence:")[0][-5:] if "Confidence" in data.get("strategy_summary", "") else "High",
                        "predicted_roi": data.get("ai_analysis", {}).get("lstm_predicted_price", {}).get("usd", "0.0"),
                        "current_price": current_price,
                        "recommended_qty": qty,
                        "cost": allocate_amount,
                        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    if ticker not in current_signals:
                        insert_signal(sig_obj)
                        new_signals_count += 1
                        print(f"[+] SIGNAL GENERATED: {ticker}")
        except Exception as e:
            print(f"[-] Error processing {ticker}: {e}")
            
    print(f"[!] Tick Complete. New Signals Generated: {new_signals_count}")

if __name__ == "__main__":
    run_paper_trading_tick()
