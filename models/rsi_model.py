"""
RSI Momentum Model
Calculates 14-day RSI and votes based on overbought/oversold conditions.
"""

import pandas as pd
import numpy as np


def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        data (pd.DataFrame): Historical price data with 'Close' column
        period (int): RSI period (default 14)
        
    Returns:
        pd.Series: RSI values
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def get_rsi_vote(data):
    """
    Calculate RSI vote based on the most recent data point.
    
    Vote Logic:
    - +1 if RSI < 30 (Oversold - Bullish)
    - -1 if RSI > 70 (Overbought - Bearish)
    - 0 otherwise (Neutral)
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'rsi_value': float,
            'explanation': str
        }
    """
    if len(data) < 15:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'rsi_value': None,
            'explanation': 'Need at least 15 days of data for RSI calculation'
        }
    
    rsi = calculate_rsi(data)
    current_rsi = rsi.iloc[-1]
    
    if pd.isna(current_rsi):
        return {
            'vote': 0,
            'signal': 'Neutral',
            'rsi_value': None,
            'explanation': 'RSI calculation returned NaN'
        }
    
    # Determine vote
    if current_rsi < 30:
        vote = 1
        signal = 'Bullish (Oversold)'
    elif current_rsi > 70:
        vote = -1
        signal = 'Bearish (Overbought)'
    else:
        vote = 0
        signal = 'Neutral'
    
    explanation = f"RSI: {current_rsi:.2f}"
    
    return {
        'vote': vote,
        'signal': signal,
        'rsi_value': round(current_rsi, 2),
        'explanation': explanation
    }
