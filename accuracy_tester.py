import os
import yfinance as yf
from main_api import calculate_technical_indicators, run_deep_learning_backtest
from main_api import nifty_model_path, btc_model_path, normalizer

TEST_UNIVERSE = [
    # US Tech
    "AAPL", "MSFT", "NVDA", "TSLA",
    # Crypto
    "BTC-USD", "ETH-USD",
    # Indian Market
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    # Commodities/Energy
    "XOM", "CVX", "GC=F"
]

def run_bulk_accuracy_test():
    print(f"\n[======== OMNI-ASSET BULK ACCURACY TEST ========]")
    print(f"Testing Model across {len(TEST_UNIVERSE)} diverse assets over 365 days...\n")
    
    total_win_rate = 0
    total_roi = 0
    valid_assets = 0
    total_trades_taken = 0
    
    for ticker in TEST_UNIVERSE:
        try:
            clean_ticker = normalizer.normalize(ticker)
            df = yf.Ticker(clean_ticker).history(period="1y", interval="1d")
            
            if df.empty or len(df) < 50:
                print(f"[-] {ticker:<12} | Not enough data. Skipping.")
                continue
                
            df = calculate_technical_indicators(df)
            
            is_inr = "NS" in clean_ticker.upper() or "BO" in clean_ticker.upper()
            model_path = nifty_model_path if is_inr and os.path.exists(nifty_model_path) else btc_model_path
            
            res = run_deep_learning_backtest(df, model_weights_path=model_path, days_to_test=365)
            
            wr = res['win_rate']
            roi = res['ai_roi']
            trades = res['total_trades']
            
            color = "\033[92m" if roi > 0 else "\033[91m"
            print(f"[*] {ticker:<12} | Trades: {trades:<3} | Win Rate: {wr:>5.1f}% | ROI: {color}{roi:>6.2f}%\033[0m")
            
            total_win_rate += wr
            total_roi += roi
            total_trades_taken += trades
            valid_assets += 1
            
        except Exception as e:
            print(f"[-] {ticker:<12} | Error: {e}")
            
    if valid_assets > 0:
        avg_wr = total_win_rate / valid_assets
        avg_roi = total_roi / valid_assets
        print(f"\n[======== FINAL SYSTEM ACCURACY ========]")
        print(f"Total Assets Tested : {valid_assets}")
        print(f"Total Trades Taken  : {total_trades_taken}")
        print(f"Average Win Rate    : {avg_wr:.2f}%")
        print(f"Average System ROI  : {avg_roi:.2f}%")
        print(f"========================================\n")
        
if __name__ == "__main__":
    run_bulk_accuracy_test()
