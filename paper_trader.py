import os
import time
from datetime import datetime
from shared_state import db, _db_lock, get_balance
# Import the analyze function directly to avoid self-referential HTTP calls
from main_api import analyze

def get_watchlist():
    watchlist = []
    with db.get_write_connection() as conn:
        cursor = conn.execute("SELECT ticker FROM watchlist")
        for row in cursor.fetchall():
            watchlist.append(row[0])
    return watchlist

def get_holdings():
    holdings = set()
    with db.get_write_connection() as conn:
        cursor = conn.execute("SELECT ticker FROM holdings")
        for row in cursor.fetchall():
            holdings.add(row[0])
    return holdings

def get_signals():
    signals = set()
    with db.get_write_connection() as conn:
        cursor = conn.execute("SELECT ticker FROM signals")
        for row in cursor.fetchall():
            signals.add(row[0])
    return signals

def insert_signal(sig_obj):
    with db.get_write_connection() as conn:
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
            # Flaw 32 Fix: Await the async analyze function
            import asyncio
            data = asyncio.run(analyze(
                ticker=ticker,
                asset_class="Global_Cluster",
                beta="any",
                timeframe="swing",
                risk="dynamic"
            ))
            
            if "error" in data:
                print(f"[-] Analysis Error for {ticker}: {data['error']}")
                continue
                
            signal = data['execution_suggestion']['signal']
            current_price = data['current_price']['usd']
            rl_size_str = data['execution_suggestion']['rl_position_size'].replace('%', '')
            rl_size = float(rl_size_str) / 100.0
            
            if signal == "BUY" and ticker not in current_holdings:
                allocate_amount = balance * rl_size
                if allocate_amount > 100:
                    qty = allocate_amount / current_price
                    
                    # Fix confidence parsing to use consensus score
                    confidence_val = data.get("ai_analysis", {}).get("consensus_score", "75")
                    
                    # Fix ROI parsing to be a percentage, not absolute price
                    try:
                        predicted_price = float(data.get("ai_analysis", {}).get("lstm_predicted_price", {}).get("usd", current_price))
                        roi_pct = ((predicted_price - current_price) / current_price) * 100
                        roi_str = f"{roi_pct:.2f}%"
                    except Exception:
                        roi_str = "0.0%"
                        
                    sig_obj = {
                        "ticker": ticker,
                        "signal": "BUY",
                        "confidence": f"{confidence_val}/100",
                        "predicted_roi": roi_str,
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
