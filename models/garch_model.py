"""
GARCH Volatility Model
Fits a GARCH(1,1) model to forecast volatility and votes based on volatility trend.
"""

import pandas as pd
import numpy as np
import warnings


def get_garch_vote(data):
    """
    Fit GARCH(1,1) model and forecast T+1 volatility.
    
    Vote Logic:
    - +3 if forecast volatility is decreasing (Bullish)
    - -3 if forecast volatility is increasing (Bearish)
    - 0 if model fails to converge or insufficient data
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'current_vol': float,
            'forecast_vol': float,
            'explanation': str
        }
    """
    # Need at least 100 observations for reliable GARCH estimation
    if len(data) < 100:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'current_vol': None,
            'forecast_vol': None,
            'explanation': 'Need at least 100 days of data for GARCH model'
        }
    
    try:
        from arch import arch_model
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna() * 100  # Convert to percentage
        
        if len(returns) < 100:
            return {
                'vote': 0,
                'signal': 'Insufficient Data',
                'current_vol': None,
                'forecast_vol': None,
                'explanation': 'Insufficient return data for GARCH'
            }
        
        # Fit GARCH(1,1) model
        # Suppress convergence warnings during fitting
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            model_fit = model.fit(disp='off', show_warning=False)
        
        # Forecast volatility for T+1
        forecast = model_fit.forecast(horizon=1)
        forecast_variance = forecast.variance.values[-1, 0]
        forecast_vol = np.sqrt(forecast_variance)
        
        # Get current volatility (conditional volatility from the model)
        current_vol = model_fit.conditional_volatility.iloc[-1]
        
        # Determine vote based on volatility trend
        if forecast_vol < current_vol:
            vote = 3
            signal = 'Bullish (Vol Decreasing)'
        else:
            vote = -3
            signal = 'Bearish (Vol Increasing)'
        
        vol_change_pct = ((forecast_vol - current_vol) / current_vol) * 100
        explanation = f"Current Vol: {current_vol:.2f}%, Forecast: {forecast_vol:.2f}% ({vol_change_pct:+.1f}%)"
        
        return {
            'vote': vote,
            'signal': signal,
            'current_vol': round(current_vol, 2),
            'forecast_vol': round(forecast_vol, 2),
            'explanation': explanation
        }
        
    except Exception as e:
        # Handle any convergence or calculation errors gracefully
        return {
            'vote': 0,
            'signal': 'Model Failed',
            'current_vol': None,
            'forecast_vol': None,
            'explanation': f'GARCH model error: {str(e)[:50]}'
        }
