import os
import sys

# Add the current directory to path so we can import
sys.path.append(os.path.dirname(__file__))

from retrain_scheduler import scheduled_retrain

def run_tests():
    print("==================================================")
    print("[TEST SCENARIO A] Model Improvement is 1.5% (Fails >2.0% Gate)")
    print("==================================================")
    scheduled_retrain("BTC-USD", "quantix_btc_lstm.pth", "quantix_btc_lstm_new.pth", mock_improvement=1.5)
    
    print("\n==================================================")
    print("[TEST SCENARIO B] Model Improvement is 3.5% (Passes >2.0% Gate)")
    print("==================================================")
    scheduled_retrain("BTC-USD", "quantix_btc_lstm.pth", "quantix_btc_lstm_new.pth", mock_improvement=3.5)

if __name__ == "__main__":
    run_tests()
