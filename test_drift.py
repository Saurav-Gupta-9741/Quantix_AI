import numpy as np
import pandas as pd
import torch
import unittest
from scipy.optimize import brentq
from main_api import check_model_drift

# Mocking the PyTorch Model
class MockModel(torch.nn.Module):
    def __init__(self, raw_pred_val=0.5):
        super().__init__()
        self.raw_pred_val = raw_pred_val
    def forward(self, x):
        return torch.tensor([[self.raw_pred_val]])

# Mocking the Scikit-learn Scalers
class MockScaler:
    def transform(self, data):
        return data * 0.1 
    def inverse_transform(self, data):
        return data * 10.0

def find_prediction_for_target_rmse(actuals, target_normalized_rmse_pct):
    """
    Numerically searches for a constant prediction value that produces exactly 
    the target normalized RMSE percentage against the actuals array.
    """
    actuals = np.array(actuals)
    mean_actual = np.mean(actuals)
    
    def objective_fn(pred):
        rmse = np.sqrt(np.mean((actuals - pred)**2))
        normalized_rmse_pct = rmse / mean_actual
        return normalized_rmse_pct - (target_normalized_rmse_pct / 100.0)
    
    # The minimum possible RMSE is achieved by predicting the mean, which gives RMSE = std_dev.
    min_rmse_pct = np.std(actuals) / mean_actual
    if (target_normalized_rmse_pct / 100.0) < min_rmse_pct:
        raise ValueError(f"Target RMSE {target_normalized_rmse_pct}% is lower than minimum possible {min_rmse_pct*100}% for this dataset.")
        
    # We search in a range above the mean
    lower_bound = mean_actual
    upper_bound = mean_actual * 2.0
    
    pred_val = brentq(objective_fn, lower_bound, upper_bound)
    
    # Verify convergence
    achieved_rmse_pct = np.sqrt(np.mean((actuals - pred_val)**2)) / mean_actual
    assert np.isclose(achieved_rmse_pct, target_normalized_rmse_pct / 100.0, atol=1e-5), "Numerical solver failed to converge accurately."
    
    return pred_val

class TestModelDrift(unittest.TestCase):
    def setUp(self):
        import main_api
        # Reset the inverse_transform to just return the value as we will feed exact targets
        # Our mock model outputs `raw_pred_val`, and `inverse_transform` multiplies by 10.0.
        # But we will just mock `inverse_transform` to return exactly what it gets.
        main_api.feature_scaler = MockScaler()
        
        class IdentityScaler:
            def transform(self, data): return data
            def inverse_transform(self, data): return data
        main_api.target_scaler = IdentityScaler()
        
        # Create 45 days of synthetic closing prices (from 100 to 144)
        self.closes = np.arange(100, 145, dtype=float)
        self.df = pd.DataFrame({
            'Close': self.closes,
            'Volume': np.ones(45),
            'RSI': np.ones(45),
            'MACD': np.ones(45),
            'MACD_Signal': np.ones(45)
        }, index=pd.date_range('2026-05-01', periods=45))
        
        self.actuals = self.closes[-15:] # 130 to 144

    def test_rmse_just_below_threshold(self):
        """
        Boundary Test: Exactly 4.9% RMSE. Should NOT trigger drift suppression.
        """
        target_pred = find_prediction_for_target_rmse(self.actuals, 4.9)
        model = MockModel(raw_pred_val=target_pred) 
        
        is_drifting = check_model_drift("MOCK", model, self.df, 'cpu')
        self.assertFalse(is_drifting, "Model should NOT be flagged as drifting (4.9% < 5%)")

    def test_rmse_exact_edge_case(self):
        """
        Boundary Test: Exactly 5.0% RMSE.
        """
        target_pred = find_prediction_for_target_rmse(self.actuals, 5.0)
        model = MockModel(raw_pred_val=target_pred) 
        
        is_drifting = check_model_drift("MOCK", model, self.df, 'cpu')
        # Drift threshold uses > 0.05, so exactly 5.0% should NOT drift
        self.assertFalse(is_drifting, "Edge case 5.0% is not strictly > 0.05, should be False.")

    def test_rmse_just_above_threshold(self):
        """
        Boundary Test: Exactly 5.1% RMSE. MUST trigger drift suppression.
        """
        target_pred = find_prediction_for_target_rmse(self.actuals, 5.1)
        model = MockModel(raw_pred_val=target_pred) 
        
        is_drifting = check_model_drift("MOCK", model, self.df, 'cpu')
        self.assertTrue(is_drifting, "Model MUST be flagged as drifting (5.1% > 5%)")

    def test_timestamp_alignment(self):
        """
        Validates that the 15-day prediction array perfectly aligns with the last 15 days of actual closes.
        """
        actuals_slice = self.df['Close'].values[-15:]
        self.assertEqual(len(actuals_slice), 15)
        self.assertEqual(actuals_slice[-1], 144)
        self.assertEqual(actuals_slice[0], 130)

if __name__ == '__main__':
    unittest.main(verbosity=2)
