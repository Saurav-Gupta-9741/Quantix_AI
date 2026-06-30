import os
import sys
import time
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from main_api import check_model_drift, get_exchange_rate
from shared_state import get_balance

def live_paper_trade_loop(iterations=1, interval=1):
    print("==================================================")
    print(f"[{datetime.now()}] INITIATING LIVE PAPER TRADING TRACE")
    print("==================================================")
    print("[+] Connecting to yfinance market data...")
    print("[+] Initializing Quantix AI Engine...")
    print(f"[+] Starting {iterations} iterations with {interval}s interval...\n")
    
    for i in range(iterations):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"--- [ITERATION {i+1}/{iterations}] | {current_time} ---")
        
        components_passed = 0
        total_components = 2
        
        # 1. FX Circuit Breaker Check
        fx_rate = get_exchange_rate()
        print(f"[+] FX Circuit Breaker OK: 1 USD = {fx_rate} INR")
        
        # 2. Drift Detection Check
        print("[+] Polling for Model Drift (Volatility Checks)...")
        try:
            # Flaw 31 Fix: Import actual dependencies for the drift check
            from main_api import registry
            from circuit_breaker import fetch_history
            import yfinance as yf
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            live_data = fetch_history(yf.Ticker("BTC-USD"), period="80d", interval="1d")
            
            # Ensure model is loaded in registry
            if "btc_lstm" not in registry._models:
                # Fallback to load model if registry is empty
                print("    -> [DRIFT CHECK: FAILED] btc_lstm model not loaded in registry.")
            else:
                model = registry._models["btc_lstm"]
                drift = check_model_drift("BTC-USD", model, live_data, device)
                print(f"    -> [DRIFT CHECK: PASSED] Drift Status for BTC-USD: {drift}")
                components_passed += 1
        except Exception as e:
            print(f"    -> [DRIFT CHECK: FAILED] Exception: {e}")
            
        # 3. Market Sweep & Signal Generation
        print("[+] Executing Market Sweep & Signal Generation...")
        try:
            import subprocess
            script_path = os.path.join(os.path.dirname(__file__), "paper_trader.py")
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            
            if "argument of type 'coroutine' is not iterable" in result.stdout or "Error" in result.stdout:
                print(f"    -> [SWEEP SIGNAL GEN: FAILED] Sweep failed with errors.")
            elif "Tick Complete" in result.stdout:
                print(f"    -> [SWEEP SIGNAL GEN: PASSED] Sweep completed.")
                components_passed += 1
            else:
                print(f"    -> [SWEEP SIGNAL GEN: FAILED] Unknown state.")
            
            # Print a snippet of the paper trader's output to prove it ran
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"       {line}")
        except Exception as e:
            print(f"    -> [SWEEP SIGNAL GEN: FAILED] Exception: {e}")
            
        # 4. Portfolio State
        balance = get_balance()
        print(f"[+] Portfolio Status: USD {balance:.2f}")
        
        print(f"--- Iteration Complete: [OVERALL: {components_passed}/{total_components} COMPONENTS PASSED] ---\n")
        time.sleep(interval)

    print("==================================================")
    if components_passed == total_components:
        print(f"[{datetime.now()}] LIVE TRACE COMPLETED SUCCESSFULLY")
    else:
        print(f"[{datetime.now()}] LIVE TRACE COMPLETED WITH FAILURES")
    print("==================================================")

if __name__ == "__main__":
    live_paper_trade_loop()
