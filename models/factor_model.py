"""
Factor Model
Combines momentum and volatility factors for voting.
"""

import pandas as pd
import numpy as np


def get_factor_vote(data):
    """
    Calculate Factor Model vote based on momentum and volatility.
    
    Vote Logic:
    - +1 if Momentum is positive (Price > Price 20 days ago) AND
         Historical Volatility (20d) is in the lower 50th percentile
    - 0 otherwise
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'momentum_positive': bool,
            'vol_percentile': float,
            'explanation': str
        }
    """
    if len(data) < 21:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'momentum_positive': None,
            'vol_percentile': None,
            'explanation': 'Need at least 21 days of data'
        }
    
    # Calculate momentum (current price vs 20 days ago)
    current_price = data['Close'].iloc[-1]
    price_20d_ago = data['Close'].iloc[-21]
    momentum_positive = current_price > price_20d_ago
    
    # Calculate 20-day historical volatility
    returns = data['Close'].pct_change()
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized %
    
    current_vol = rolling_vol.iloc[-1]
    
    if pd.isna(current_vol):
        return {
            'vote': 0,
            'signal': 'Neutral',
            'momentum_positive': momentum_positive,
            'vol_percentile': None,
            'explanation': 'Volatility calculation returned NaN'
        }
    
    # Calculate percentile of current volatility across the entire dataset
    vol_percentile = (rolling_vol.dropna() < current_vol).sum() / len(rolling_vol.dropna()) * 100
    
    # Determine vote
    if momentum_positive and vol_percentile <= 50:
        vote = 1
        signal = 'Bullish'
        explanation = f"Positive Momentum + Low Vol (Percentile: {vol_percentile:.1f}%)"
    else:
        vote = 0
        signal = 'Neutral'
        if not momentum_positive:
            explanation = f"Negative Momentum (Vol Percentile: {vol_percentile:.1f}%)"
        else:
            explanation = f"High Volatility (Percentile: {vol_percentile:.1f}%)"
    
    return {
        'vote': vote,
        'signal': signal,
        'momentum_positive': momentum_positive,
        'vol_percentile': round(vol_percentile, 1),
        'explanation': explanation
    }
