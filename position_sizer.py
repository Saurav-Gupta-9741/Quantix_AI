import numpy as np

class KellyCriterionSizer:
    def __init__(self):
        self.agent_name = "Quantix Kelly Allocator"
        self.beta_lookback_window = 60  # Fixed 60-day rolling lookback window for beta covariance
        
    def get_position_size(self, signal, historical_win_rate, take_profit_pct, stop_loss_pct, historical_returns=None, target_volatility=0.20):
        """
        Calculates optimal position size using empirically derived volatility-adjusted Kelly Criterion.
        
        Args:
            signal: 'BUY', 'SELL', or 'HOLD'
            historical_win_rate: float (e.g. 0.65 for 65% win rate) derived from walk-forward backtest
            take_profit_pct: actual target take profit (e.g. 0.05 for 5%)
            stop_loss_pct: actual target stop loss (e.g. -0.02 for -2%)
            historical_returns: numpy array of recent daily returns for volatility calculation (requires minimum 30 days)
            target_volatility: portfolio-level target volatility band (e.g. 0.15 for conservative, 0.30 for aggressive)
        Returns:
            Position size as a percentage (e.g. 15.0 means 15% of portfolio)
        """
        # 1. Require empirical data, reject fabricated constants
        if historical_win_rate is None or historical_returns is None or len(historical_returns) < 30:
            print("[!] Kelly Sizer: Low confidence / insufficient sample. Returning 0.0% allocation.")
            return 0.0
            
        if signal == 'HOLD' or historical_win_rate <= 0.0:
            return 0.0
            
        if stop_loss_pct == 0.0:
            return 0.0 # Prevent division by zero
            
        # 2. Dynamic Reward/Risk Ratio from actual SL/TP configuration
        reward_risk_ratio = abs(take_profit_pct / stop_loss_pct)
        
        # 3. Kelly Fraction: f* = (p * b - q) / b
        win_prob = historical_win_rate
        kelly_fraction = (win_prob * reward_risk_ratio - (1 - win_prob)) / reward_risk_ratio
        kelly_fraction = max(0.0, kelly_fraction)
        
        # Apply half-Kelly for safety (standard institutional practice)
        half_kelly = kelly_fraction / 2.0
        
        # 4. Continuous Volatility Scaling (Inverse Volatility Weighting)
        daily_vol = np.std(historical_returns)
        annualized_vol = daily_vol * np.sqrt(252)
        if annualized_vol == 0: annualized_vol = 0.01 # prevent div by zero
        
        # Continuous multiplier driven by user portfolio risk tolerance
        vol_adjustment = min(1.5, target_volatility / annualized_vol)
        
        final_allocation = half_kelly * vol_adjustment
        
        # Clip between 3% min and 25% max (only if > 0)
        if final_allocation > 0:
            final_allocation = max(0.03, min(0.25, final_allocation))
            
        return round(final_allocation * 100, 2)

    def calculate_empirical_beta(self, asset_returns, market_returns):
        """
        Calculates Beta using explicit 60-day rolling lookback window.
        """
        min_len = min(len(asset_returns), len(market_returns), self.beta_lookback_window)
        if min_len < 10:
            return "Insufficient Data"
        cov = np.cov(asset_returns[-min_len:], market_returns[-min_len:])
        beta = cov[0][1] / cov[1][1]
        return round(float(beta), 2)

if __name__ == "__main__":
    rl = KellyCriterionSizer()
    returns = np.random.normal(0.001, 0.02, 30)  # Simulated 30 days of returns
    # win_rate=0.55, TP=6%, SL=-2%, risk="conservative" (target_vol=0.15)
    size = rl.get_position_size("BUY", 0.55, 0.06, -0.02, returns, 0.15)
    print(f"Optimal Empirical Kelly Position Size: {size}%")
