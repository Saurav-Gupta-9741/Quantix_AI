import re

class TickerNormalizer:
    def __init__(self):
        # A dictionary mapping common user errors to the official Yahoo Finance Ticker
        self.crypto_map = {
            "BTCUSD": "BTC-USD",
            "BTC": "BTC-USD",
            "BITCOIN": "BTC-USD",
            "ETHUSD": "ETH-USD",
            "ETH": "ETH-USD",
            "ETHEREUM": "ETH-USD",
            "BTCINR": "BTC-INR"
        }
        
        self.indian_market_map = {
            "NIFTY": "^NSEI",
            "NIFTY50": "^NSEI",
            "SENSEX": "^BSESN",
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "HDFC": "HDFCBANK.NS"
        }
        
        self.us_market_map = {
            "APPLE": "AAPL",
            "TESLA": "TSLA",
            "AMAZON": "AMZN",
            "MICROSOFT": "MSFT",
            "GOOGLE": "GOOGL"
        }

    def normalize(self, raw_input):
        """
        Takes raw, messy user input, cleans it, and returns a safe, API-ready ticker.
        """
        # 1. Strip whitespace and convert to uppercase
        clean_input = str(raw_input).strip().upper()
        
        # 1.5 Strip any HTML tags (e.g., from browser dictionary extensions)
        clean_input = re.sub(r'<[^>]+>', '', clean_input)
        
        # 2. Remove special characters (except dashes/dots which are sometimes needed)
        clean_input = re.sub(r'[^A-Z0-9\-.]', '', clean_input)
        
        # 3. Check Crypto Maps
        if clean_input in self.crypto_map:
            return self.crypto_map[clean_input]
            
        # 4. Check Indian Market Maps
        if clean_input in self.indian_market_map:
            return self.indian_market_map[clean_input]
            
        # 5. Check US Market Maps
        if clean_input in self.us_market_map:
            return self.us_market_map[clean_input]
            
        # 6. Fallback: If it's a raw stock ticker (like AAPL) that isn't mapped, 
        # assume they typed it correctly and return it.
        return clean_input

if __name__ == "__main__":
    normalizer = TickerNormalizer()
    
    test_cases = ["   btc usd  ", "Nifty50", "Apple", "RELIANCE", "TSLA"]
    print("--- TICKER NORMALIZATION ENGINE TEST ---")
    for raw in test_cases:
        safe = normalizer.normalize(raw)
        print(f"User Input: '{raw}'  ->  Safe API Ticker: '{safe}'")
