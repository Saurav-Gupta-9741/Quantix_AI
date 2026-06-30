import os
import sys
import unittest
from unittest.mock import patch, ANY

# Add the current directory to path so we can import from main_api
sys.path.append(os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main_api import app, EMPIRICAL_CACHE

client = TestClient(app)

class TestBacktestEndpoint(unittest.TestCase):
    
    @patch('backtest.KellyCriterionSizer.get_position_size')
    def test_backtest_validates_kelly_and_cache(self, mock_get_position_size):
        # Mock the position sizer to track what arguments it receives
        mock_get_position_size.return_value = 15.0 # mock 15% allocation
        
        print("[+] Starting backtest trace via API endpoint for BTC-USD...")
        # Clear cache first to ensure it populates
        if "BTC-USD" in EMPIRICAL_CACHE:
            del EMPIRICAL_CACHE["BTC-USD"]
            
        response = client.get("/api/backtest?ticker=BTC-USD", headers={"x-api-key": "dev_key_123"})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsNone(data.get("error"))
        self.assertIn("win_rate", data)
        
        win_rate = data["win_rate"]
        print(f"[+] Backtest completed successfully. Extracted win_rate: {win_rate}%")
        
        # 1. Assert KellySizer is called with empirical win rate, TP, and SL
        # If there were trades, it should have been called. If 0 trades, we skip this assertion.
        if mock_get_position_size.call_count > 0:
            args, kwargs = mock_get_position_size.call_args
            # The kwargs we passed in backtest.py are:
            # signal, historical_win_rate, take_profit_pct, stop_loss_pct, historical_returns, target_volatility
            # Or kwargs directly. We can assert the kwargs have expected names
            self.assertIn('historical_win_rate', kwargs)
            self.assertIn('take_profit_pct', kwargs)
            self.assertIn('stop_loss_pct', kwargs)
            self.assertIsInstance(kwargs['historical_win_rate'], float)
            print(f"[+] KellySizer was passed historical_win_rate: {kwargs['historical_win_rate']}")
        else:
            print("[!] No trades generated in this window to trigger position sizer, but endpoint parsed correctly.")

        # 2. Assert EMPIRICAL_CACHE is populated correctly
        self.assertIn("BTC-USD", EMPIRICAL_CACHE)
        cached_val = EMPIRICAL_CACHE["BTC-USD"]["win_rate"]
        # Endpoint sets win_rate / 100.0
        self.assertAlmostEqual(cached_val, win_rate / 100.0, places=4)
        print(f"[+] EMPIRICAL_CACHE verified. Endpoint JSON win_rate {win_rate}% matches cached {cached_val}")

    @patch('main_api.run_deep_learning_backtest')
    def malformed_response_handling(self, mock_run):
        # 3. Simulate malformed response to test 23a Cache gating logic
        # We simulate a backtest that returns a truthy dict with a None error but missing 'win_rate'
        mock_run.return_value = {"error": None, "total_trades": 0} 
        
        print("[+] Testing malformed response caching logic (Flaw 23a Fix)...")
        # Ensure cache is clear
        if "ETH-USD" in EMPIRICAL_CACHE:
            del EMPIRICAL_CACHE["ETH-USD"]
            
        response = client.get("/api/backtest?ticker=ETH-USD", headers={"x-api-key": "dev_key_123"})
        self.assertEqual(response.status_code, 200)
        
        # We assert the endpoint didn't crash, but ETH-USD was NOT added to the cache because 'win_rate' was missing
        self.assertNotIn("ETH-USD", EMPIRICAL_CACHE)
        print("[+] Endpoint successfully short-circuited cache on malformed data without crashing.")

    def test_malformed_response(self):
        self.malformed_response_handling()

if __name__ == "__main__":
    unittest.main()
