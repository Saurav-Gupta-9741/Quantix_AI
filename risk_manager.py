import os
import json
import time
import yfinance as yf
from datetime import datetime

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
TAKE_PROFIT_PCT = 5.0  # Sell if +5% profit

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return None
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

def run_risk_manager():
    print(f"\n[======== AUTO-TAKE-PROFIT MONITOR STARTED ========]")
    print(f"Monitoring active holdings every 5 minutes. Take-Profit target: +{TAKE_PROFIT_PCT}%")
    
    while True:
        try:
            portfolio = load_portfolio()
            if not portfolio or not portfolio.get("holdings"):
                time.sleep(300)
                continue
                
            holdings = portfolio["holdings"]
            sold_something = False
            
            for ticker, data in list(holdings.items()):
                try:
                    # Fetch live price
                    live_price = float(yf.Ticker(ticker).fast_info.last_price)
                    if "NS" in ticker.upper() or "BO" in ticker.upper():
                        live_price = live_price / 83.5
                        
                    entry_price = data['entry_price']
                    qty = data['qty']
                    
                    pnl_pct = ((live_price - entry_price) / entry_price) * 100
                    
                    if pnl_pct >= TAKE_PROFIT_PCT:
                        # EXECUTE AUTO-SELL
                        revenue = qty * live_price
                        profit = revenue - data['total_cost']
                        
                        portfolio['balance_usd'] += revenue
                        del portfolio['holdings'][ticker]
                        
                        log_msg = f"AUTO-SELL {ticker} (Take Profit triggered at +{pnl_pct:.2f}%): Profit ${profit:.2f}"
                        portfolio['trade_history'].insert(0, log_msg)
                        
                        print(f"[!] {datetime.now().strftime('%H:%M:%S')} | {log_msg}")
                        sold_something = True
                        
                except Exception as e:
                    print(f"[-] Error tracking {ticker}: {e}")
                    
            if sold_something:
                portfolio['trade_history'] = portfolio['trade_history'][:50]
                save_portfolio(portfolio)
                
        except Exception as e:
            print(f"[-] Risk Manager Error: {e}")
            
        # Sleep for 5 minutes
        time.sleep(300)

if __name__ == "__main__":
    run_risk_manager()
