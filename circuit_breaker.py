import time
from functools import wraps

class CircuitBreaker:
    def __init__(self, threshold=5, cooldown=300):
        self.failures = 0
        self.threshold = threshold
        self.open_until = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if time.time() < self.open_until:
                raise RuntimeError("yfinance circuit open - data unavailable")
            try:
                result = func(*args, **kwargs)
                self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                if self.failures >= self.threshold:
                    self.open_until = time.time() + self.cooldown
                raise
        return wrapper

yf_breaker = CircuitBreaker(threshold=5, cooldown=300)

@yf_breaker
def fetch_history(ticker_obj, period="80d", interval="1d"):
    return ticker_obj.history(period=period, interval=interval)

@yf_breaker
def fetch_last_price(ticker_obj):
    return float(ticker_obj.fast_info.last_price)
