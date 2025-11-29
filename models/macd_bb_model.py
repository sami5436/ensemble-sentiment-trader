"""
MACD + Bollinger Bands Combination Model
Combines MACD for trend identification with Bollinger Bands for volatility.
"""

import pandas as pd
import numpy as np


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data (pd.DataFrame): Historical price data with 'Close' column
        fast (int): Fast EMA period (default 12)
        slow (int): Slow EMA period (default 26)
        signal (int): Signal line period (default 9)
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.DataFrame): Historical price data with 'Close' column
        period (int): Moving average period (default 20)
        std_dev (float): Number of standard deviations (default 2)
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    middle_band = data['Close'].rolling(window=period).mean()
    rolling_std = data['Close'].rolling(window=period).std()
    
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band


def get_macd_bb_vote(data):
    """
    Calculate MACD + Bollinger Bands vote.
    
    Vote Logic:
    - +1 if MACD bullish cross AND price near lower BB (buy the dip in uptrend)
    - -1 if MACD bearish cross AND price near upper BB (sell the rip in downtrend)
    - 0 otherwise
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'macd_signal': str,
            'bb_signal': str,
            'explanation': str
        }
    """
    if len(data) < 50:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'macd_signal': None,
            'bb_signal': None,
            'explanation': 'Need at least 50 days of data for MACD+BB'
        }
    
    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(data)
    
    # Calculate Bollinger Bands
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data)
    
    # Get current values
    current_price = data['Close'].iloc[-1]
    current_macd = macd_line.iloc[-1]
    prev_macd = macd_line.iloc[-2]
    current_signal = signal_line.iloc[-1]
    prev_signal = signal_line.iloc[-2]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    
    # Check for NaN
    if pd.isna(current_macd) or pd.isna(current_upper):
        return {
            'vote': 0,
            'signal': 'Neutral',
            'macd_signal': None,
            'bb_signal': None,
            'explanation': 'Calculation returned NaN'
        }
    
    # MACD signal
    macd_bullish_cross = (prev_macd <= prev_signal) and (current_macd > current_signal)
    macd_bearish_cross = (prev_macd >= prev_signal) and (current_macd < current_signal)
    macd_bullish = current_macd > current_signal
    macd_bearish = current_macd < current_signal
    
    if macd_bullish_cross:
        macd_signal = 'Bullish Cross'
    elif macd_bearish_cross:
        macd_signal = 'Bearish Cross'
    elif macd_bullish:
        macd_signal = 'Bullish'
    else:
        macd_signal = 'Bearish'
    
    # Bollinger Band signal
    bb_width = current_upper - current_lower
    distance_to_lower = current_price - current_lower
    distance_to_upper = current_upper - current_price
    
    near_lower_bb = (distance_to_lower / bb_width) < 0.2  # Within 20% of lower band
    near_upper_bb = (distance_to_upper / bb_width) < 0.2  # Within 20% of upper band
    
    if near_lower_bb:
        bb_signal = 'Near Lower BB'
    elif near_upper_bb:
        bb_signal = 'Near Upper BB'
    else:
        bb_signal = 'Middle Range'
    
    # Determine vote
    vote = 0
    signal = 'Neutral'
    
    if (macd_bullish or macd_bullish_cross) and near_lower_bb:
        vote = 1
        signal = 'Bullish (Buy Dip)'
        explanation = f'MACD {macd_signal}, Price near lower BB'
    elif (macd_bearish or macd_bearish_cross) and near_upper_bb:
        vote = -1
        signal = 'Bearish (Sell Rip)'
        explanation = f'MACD {macd_signal}, Price near upper BB'
    else:
        explanation = f'MACD {macd_signal}, BB {bb_signal}'
    
    return {
        'vote': vote,
        'signal': signal,
        'macd_signal': macd_signal,
        'bb_signal': bb_signal,
        'explanation': explanation
    }
