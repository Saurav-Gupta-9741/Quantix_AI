import os
import time
import requests
import yfinance as yf
from datetime import datetime
from shared_state import get_conn, _db_lock, get_balance, update_balance

TAKE_PROFIT_PCT = 5.0  # Sell if +5% profit

def run_risk_manager():
    print(f"\n[======== AUTO-TAKE-PROFIT MONITOR STARTED ========]")
    print(f"Monitoring active holdings every 5 minutes. Take-Profit target: +{TAKE_PROFIT_PCT}%")
    
    while True:
        try:
            sold_something = False
            
            with _db_lock:
                conn = get_conn()
                cursor = conn.execute("SELECT * FROM holdings")
                holdings = cursor.fetchall()
                conn.close()
                
            if not holdings:
                time.sleep(300)
                continue
                
            for row in holdings:
                ticker, qty, entry_price, total_cost, date = row
                try:
                    # Fetch live price
                    live_price = float(yf.Ticker(ticker).fast_info.last_price)
                    if "NS" in ticker.upper() or "BO" in ticker.upper():
                        live_price = live_price / 83.5
                        
                    pnl_pct = ((live_price - entry_price) / entry_price) * 100
                    
                    if pnl_pct >= TAKE_PROFIT_PCT:
                        # EXECUTE AUTO-SELL
                        revenue = qty * live_price
                        profit = revenue - total_cost
                        
                        with _db_lock:
                            conn = get_conn()
                            current_balance = get_balance()
                            update_balance(current_balance + revenue)
                            conn.execute("DELETE FROM holdings WHERE ticker=?", (ticker,))
                            
                            log_msg = f"AUTO-SELL {ticker} (Take Profit triggered at +{pnl_pct:.2f}%): Profit ${profit:.2f}"
                            conn.execute("INSERT INTO trade_history (log_msg, date) VALUES (?, ?)", 
                                         (log_msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                            conn.commit()
                            conn.close()
                        
                        print(f"[!] {datetime.now().strftime('%H:%M:%S')} | {log_msg}")
                        sold_something = True
                        
                except Exception as e:
                    print(f"[-] Error tracking {ticker}: {e}")
                    
            if sold_something:
                try:
                    requests.post("http://127.0.0.1:8000/api/trigger_refresh")
                except:
                    pass
                
        except Exception as e:
            print(f"[-] Risk Manager Error: {e}")
            
        # Sleep for 5 minutes
        time.sleep(300)

if __name__ == "__main__":
    run_risk_manager()
