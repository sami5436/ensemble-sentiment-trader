"""
Mean Reversion Model
Calculates 20-day SMA and votes based on price deviation from the mean.
"""

import pandas as pd
import numpy as np


def calculate_sma(data, period=20):
    """
    Calculate Simple Moving Average.
    
    Args:
        data (pd.DataFrame): Historical price data with 'Close' column
        period (int): SMA period (default 20)
        
    Returns:
        pd.Series: SMA values
    """
    return data['Close'].rolling(window=period).mean()


def get_mean_reversion_vote(data):
    """
    Calculate Mean Reversion vote based on deviation from 20-day SMA.
    
    Vote Logic:
    - +1 if Price is > 2% below SMA (Undervalued - Bullish)
    - -1 if Price is > 2% above SMA (Overvalued - Bearish)
    - 0 otherwise (Neutral)
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'sma_value': float,
            'price': float,
            'deviation_pct': float,
            'explanation': str
        }
    """
    if len(data) < 20:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'sma_value': None,
            'price': None,
            'deviation_pct': None,
            'explanation': 'Need at least 20 days of data for SMA calculation'
        }
    
    sma = calculate_sma(data)
    current_price = data['Close'].iloc[-1]
    current_sma = sma.iloc[-1]
    
    if pd.isna(current_sma):
        return {
            'vote': 0,
            'signal': 'Neutral',
            'sma_value': None,
            'price': current_price,
            'deviation_pct': None,
            'explanation': 'SMA calculation returned NaN'
        }
    
    # Calculate deviation percentage
    deviation_pct = ((current_price - current_sma) / current_sma) * 100
    
    # Determine vote
    if deviation_pct < -2.0:
        vote = 1
        signal = 'Bullish (Below SMA)'
    elif deviation_pct > 2.0:
        vote = -1
        signal = 'Bearish (Above SMA)'
    else:
        vote = 0
        signal = 'Neutral'
    
    explanation = f"Price: ${current_price:.2f}, SMA: ${current_sma:.2f}, Dev: {deviation_pct:.2f}%"
    
    return {
        'vote': vote,
        'signal': signal,
        'sma_value': round(current_sma, 2),
        'price': round(current_price, 2),
        'deviation_pct': round(deviation_pct, 2),
        'explanation': explanation
    }
