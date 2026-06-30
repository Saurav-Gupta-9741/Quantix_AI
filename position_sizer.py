import numpy as np

class KellyCriterionSizer:
    def __init__(self):
        self.agent_name = "Quantix Kelly Allocator"
        
    def get_position_size(self, signal, confidence_score, historical_returns=None):
        """
        Calculates optimal position size using real volatility-adjusted Kelly Criterion.
        
        Args:
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence_score: float between 0.0 and 1.0 representing ensemble confidence
            historical_returns: numpy array of recent daily returns for volatility calculation
        Returns:
            Position size as a percentage (e.g. 15.0 means 15% of portfolio)
        """
        if confidence_score <= 0.0:
            return 0.0
        
        # Calculate real historical volatility if returns data is provided
        if historical_returns is not None and len(historical_returns) > 5:
            daily_vol = np.std(historical_returns)
            annualized_vol = daily_vol * np.sqrt(252)
        else:
            annualized_vol = 0.30  # Default 30% annualized vol assumption
        
        # Kelly Criterion: f* = (p * b - q) / b
        # where p = win probability (mapped from confidence), b = reward/risk ratio, q = 1-p
        win_prob = 0.5 + (confidence_score * 0.3)  # Maps 0-1 confidence to 0.5-0.8 win prob
        reward_risk_ratio = 2.0  # Target 2:1 reward/risk
        
        kelly_fraction = (win_prob * reward_risk_ratio - (1 - win_prob)) / reward_risk_ratio
        kelly_fraction = max(0.0, kelly_fraction)
        
        # Apply half-Kelly for safety (standard institutional practice)
        half_kelly = kelly_fraction / 2.0
        
        # Volatility penalty: reduce position in high-vol environments
        vol_adjustment = 1.0
        if annualized_vol > 0.50:  # >50% annualized vol (crypto-level)
            vol_adjustment = 0.5
        elif annualized_vol > 0.30:  # >30% annualized vol
            vol_adjustment = 0.7
        
        final_allocation = half_kelly * vol_adjustment
        
        # Clip between 3% min and 25% max
        final_allocation = max(0.03, min(0.25, final_allocation))
        
        return round(final_allocation * 100, 2)

if __name__ == "__main__":
    rl = KellyCriterionSizer()
    returns = np.random.normal(0.001, 0.02, 30)  # Simulated 30 days of returns
    size = rl.get_position_size("BUY", 0.75, returns)
    print(f"Optimal Kelly Position Size: {size}%")
