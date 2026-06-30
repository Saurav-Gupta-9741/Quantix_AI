import os
import time
import requests
import yfinance as yf
from datetime import datetime
from shared_state import db, _db_lock, get_balance, update_balance
from circuit_breaker import fetch_last_price

# Exchange rate cache (mirrors main_api.py logic)
_exchange_rate_cache = {"rate": 83.5, "timestamp": 0}
def get_exchange_rate():
    now = datetime.now().timestamp()
    if now - _exchange_rate_cache["timestamp"] > 3600:
        try:
            res = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
            rate = res.json()["rates"]["INR"]
            _exchange_rate_cache["rate"] = rate
            _exchange_rate_cache["timestamp"] = now
        except Exception:
            pass
    return _exchange_rate_cache["rate"]

def run_risk_manager():
    print(f"\n[======== AUTO-RISK-MANAGER STARTED ========]")
    print(f"Monitoring active holdings every 5 minutes for Take-Profit and Stop-Loss triggers.")
    
    while True:
        try:
            sold_something = False
            
            with db.get_write_connection() as conn:
                cursor = conn.execute("SELECT ticker, qty, entry_price, total_cost, date, currency, stop_loss_pct, take_profit_pct FROM holdings")
                holdings = cursor.fetchall()
                
            if not holdings:
                time.sleep(300)
                continue
                
            for row in holdings:
                ticker, qty, entry_price, total_cost, date, currency, sl_pct, tp_pct = row
                try:
                    # Fetch live price
                    live_price = fetch_last_price(yf.Ticker(ticker))
                    
                    # No hardcoded INR conversion. The database stores the entry_price in the native currency.
                    # e.g., if currency == 'INR', entry_price is already in INR.
                    
                    pnl_pct = ((live_price - entry_price) / entry_price) * 100
                    
                    is_stop_loss = pnl_pct <= sl_pct
                    is_take_profit = pnl_pct >= tp_pct
                    
                    if is_stop_loss or is_take_profit:
                        # EXECUTE AUTO-SELL
                        revenue = qty * live_price
                        profit = revenue - total_cost
                        action_type = "Stop Loss" if is_stop_loss else "Take Profit"
                        
                        # Convert revenue back to USD for the portfolio balance
                        usd_revenue = revenue
                        if currency == 'INR':
                            usd_revenue = revenue / get_exchange_rate()
                        
                        with db.get_write_connection() as conn:
                            # PROOF: Explicit transaction and rowcount check to prevent silent no-ops against stale data
                            conn.execute("BEGIN IMMEDIATE")
                            cursor = conn.execute("DELETE FROM holdings WHERE ticker=?", (ticker,))
                            if cursor.rowcount > 0:
                                current_balance = get_balance()
                                update_balance(current_balance + usd_revenue)
                                
                                log_msg = f"AUTO-SELL {ticker} ({action_type} triggered at {pnl_pct:.2f}%): Profit/Loss ${profit:.2f} (in native cur: {currency})"
                                conn.execute("INSERT INTO trade_history (log_msg, date) VALUES (?, ?)", 
                                             (log_msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                                conn.commit()
                                print(f"[!] {datetime.now().strftime('%H:%M:%S')} | {log_msg}")
                                sold_something = True
                            else:
                                conn.rollback()
                        
                except Exception as e:
                    print(f"[-] Error tracking {ticker}: {e}")
                    
            if sold_something:
                try:
                    requests.post("http://127.0.0.1:8000/api/trigger_refresh")
                except Exception:
                    pass
                
        except Exception as e:
            print(f"[-] Risk Manager Error: {e}")
            
        # Sleep for 5 minutes
        time.sleep(300)

if __name__ == "__main__":
    run_risk_manager()
