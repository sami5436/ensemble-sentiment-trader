"""
Technical Support Model
Identifies support and resistance levels based on 50-day highs and lows.
"""

import pandas as pd
import numpy as np


def get_technical_support_vote(data):
    """
    Calculate Technical Support vote based on proximity to 50-day high/low.
    
    Vote Logic:
    - +3 if Close is within 1% of 50-day Low (Support level - Bullish)
    - -3 if Close is within 1% of 50-day High (Resistance level - Bearish)
    - 0 otherwise (Neutral)
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'high_50d': float,
            'low_50d': float,
            'current_price': float,
            'explanation': str
        }
    """
    if len(data) < 50:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'high_50d': None,
            'low_50d': None,
            'current_price': None,
            'explanation': 'Need at least 50 days of data'
        }
    
    # Calculate 50-day high and low
    high_50d = data['High'].iloc[-50:].max()
    low_50d = data['Low'].iloc[-50:].min()
    current_price = data['Close'].iloc[-1]
    
    # Calculate distance to high and low as percentage
    dist_to_high_pct = abs((current_price - high_50d) / high_50d) * 100
    dist_to_low_pct = abs((current_price - low_50d) / low_50d) * 100
    
    # Determine vote
    if dist_to_low_pct <= 1.0:
        vote = 3
        signal = 'Strong Bullish (Near Support)'
        explanation = f"Price ${current_price:.2f} within 1% of 50d Low ${low_50d:.2f}"
    elif dist_to_high_pct <= 1.0:
        vote = -3
        signal = 'Strong Bearish (Near Resistance)'
        explanation = f"Price ${current_price:.2f} within 1% of 50d High ${high_50d:.2f}"
    else:
        vote = 0
        signal = 'Neutral'
        explanation = f"Price ${current_price:.2f} (50d: ${low_50d:.2f}-${high_50d:.2f})"
    
    return {
        'vote': vote,
        'signal': signal,
        'high_50d': round(high_50d, 2),
        'low_50d': round(low_50d, 2),
        'current_price': round(current_price, 2),
        'explanation': explanation
    }
