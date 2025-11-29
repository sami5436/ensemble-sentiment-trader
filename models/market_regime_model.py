"""
Market Regime Model
Detects market regime (Bull/Bear/Sideways) and adapts strategy accordingly.
"""

import pandas as pd
import numpy as np


def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX) to measure trend strength.
    
    Args:
        data (pd.DataFrame): Historical price data with High, Low, Close
        period (int): ADX period (default 14)
        
    Returns:
        pd.Series: ADX values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate smoothed +DI and -DI
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx


def get_market_regime_vote(data):
    """
    Detect market regime and vote accordingly.
    
    Regime Detection:
    - Bull: Price > 50-day MA AND > 200-day MA, ADX > 20
    - Bear: Price < 50-day MA AND < 200-day MA, ADX > 20
    - Sideways: Mixed signals or ADX < 20
    
    Vote Logic:
    - Bull regime: +2 on dips (price below 20-day MA), 0 otherwise
    - Bear regime: -2 on rallies (price above 20-day MA), 0 otherwise
    - Sideways: Mean reversion (+1 if below 50-day MA, -1 if above)
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'regime': str,
            'adx': float,
            'explanation': str
        }
    """
    if len(data) < 200:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'regime': None,
            'adx': None,
            'explanation': 'Need at least 200 days of data for regime detection'
        }
    
    # Calculate moving averages
    ma_20 = data['Close'].rolling(window=20).mean()
    ma_50 = data['Close'].rolling(window=50).mean()
    ma_200 = data['Close'].rolling(window=200).mean()
    
    # Calculate ADX
    adx = calculate_adx(data)
    
    # Get current values
    current_price = data['Close'].iloc[-1]
    current_ma_20 = ma_20.iloc[-1]
    current_ma_50 = ma_50.iloc[-1]
    current_ma_200 = ma_200.iloc[-1]
    current_adx = adx.iloc[-1]
    
    # Check for NaN
    if pd.isna(current_ma_200) or pd.isna(current_adx):
        return {
            'vote': 0,
            'signal': 'Neutral',
            'regime': None,
            'adx': None,
            'explanation': 'Calculation returned NaN'
        }
    
    # Determine regime
    above_50 = current_price > current_ma_50
    above_200 = current_price > current_ma_200
    strong_trend = current_adx > 20
    
    if above_50 and above_200:
        if strong_trend:
            regime = 'Bull Market (Strong)'
        else:
            regime = 'Bull Market (Weak)'
        
        # In bull market, buy dips
        if current_price < current_ma_20:
            vote = 2
            signal = 'Bullish (Buy Dip)'
            explanation = f'{regime} - Buy dip below 20-day MA (ADX: {current_adx:.1f})'
        else:
            vote = 1
            signal = 'Bullish (Trending)'
            explanation = f'{regime} - Price above MAs (ADX: {current_adx:.1f})'
            
    elif not above_50 and not above_200:
        if strong_trend:
            regime = 'Bear Market (Strong)'
        else:
            regime = 'Bear Market (Weak)'
        
        # In bear market, sell rallies
        if current_price > current_ma_20:
            vote = -2
            signal = 'Bearish (Sell Rally)'
            explanation = f'{regime} - Sell rally above 20-day MA (ADX: {current_adx:.1f})'
        else:
            vote = -1
            signal = 'Bearish (Trending)'
            explanation = f'{regime} - Price below MAs (ADX: {current_adx:.1f})'
            
    else:
        # Mixed signals - sideways/transitional market
        regime = 'Sideways/Transitional'
        
        # Mean reversion in sideways market
        if current_price < current_ma_50:
            vote = 1
            signal = 'Mean Reversion Buy'
            explanation = f'{regime} - Price below 50-day MA, expect bounce (ADX: {current_adx:.1f})'
        elif current_price > current_ma_50:
            vote = -1
            signal = 'Mean Reversion Sell'
            explanation = f'{regime} - Price above 50-day MA, expect pullback (ADX: {current_adx:.1f})'
        else:
            vote = 0
            signal = 'Neutral'
            explanation = f'{regime} - No clear direction (ADX: {current_adx:.1f})'
    
    return {
        'vote': vote,
        'signal': signal,
        'regime': regime,
        'adx': round(current_adx, 1),
        'explanation': explanation
    }
