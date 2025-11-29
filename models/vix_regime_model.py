"""
VIX Regime Model
Uses VIX (volatility index) to determine market regime and generate contrarian/momentum signals.
"""

import pandas as pd
import numpy as np


def get_vix_regime_vote(spy_data, vix_data):
    """
    Calculate VIX Regime vote based on volatility levels and trends.
    
    Vote Logic:
    - High VIX (>25): Contrarian mode
      - +3 if VIX spiking (buy panic)
      - 0 if VIX stable high
    - Medium VIX (15-25): Neutral/Cautious
      - +1 if VIX decreasing (risk-on)
      - -1 if VIX increasing (risk-off)
    - Low VIX (<15): Momentum mode
      - +2 if SPY trending up (ride the trend)
      - -2 if SPY trending down
    
    Args:
        spy_data (pd.DataFrame): Historical SPY data
        vix_data (pd.DataFrame): Historical VIX data
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'vix_level': float,
            'vix_regime': str,
            'explanation': str
        }
    """
    if vix_data.empty or len(vix_data) < 20:
        return {
            'vote': 0,
            'signal': 'No VIX Data',
            'vix_level': None,
            'vix_regime': None,
            'explanation': 'VIX data not available'
        }
    
    if len(spy_data) < 20:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'vix_level': None,
            'vix_regime': None,
            'explanation': 'Need at least 20 days of SPY data'
        }
    
    try:
        # Align VIX data to SPY dates
        vix_aligned = vix_data.reindex(spy_data.index, method='ffill')
        
        # Get current VIX level
        current_vix = vix_aligned['Close'].iloc[-1]
        
        if pd.isna(current_vix):
            return {
                'vote': 0,
                'signal': 'VIX Data Missing',
                'vix_level': None,
                'vix_regime': None,
                'explanation': 'Current VIX data not available'
            }
        
        # Calculate VIX trend (5-day change)
        vix_5d_ago = vix_aligned['Close'].iloc[-6] if len(vix_aligned) >= 6 else current_vix
        vix_change = current_vix - vix_5d_ago
        vix_change_pct = (vix_change / vix_5d_ago) * 100 if vix_5d_ago > 0 else 0
        
        # Calculate 20-day VIX average
        vix_20d_avg = vix_aligned['Close'].iloc[-20:].mean()
        
        # Calculate SPY momentum (10-day)
        spy_momentum = ((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-11]) - 1) * 100 if len(spy_data) >= 11 else 0
        
        # Determine regime and vote
        vote = 0
        signal = 'Neutral'
        
        if current_vix > 25:
            # High VIX - Fear regime - Contrarian
            vix_regime = 'High Fear'
            if vix_change_pct > 10:
                # VIX spiking - extreme fear - buy signal
                vote = 3
                signal = 'Strong Bullish (Panic Buy)'
                explanation = f'VIX spiking to {current_vix:.1f} (+{vix_change_pct:.1f}%) - Contrarian buy'
            elif current_vix > vix_20d_avg * 1.2:
                # VIX elevated above recent avg
                vote = 2
                signal = 'Bullish (Fear Peak)'
                explanation = f'VIX elevated at {current_vix:.1f} (avg: {vix_20d_avg:.1f}) - Potential reversal'
            else:
                vote = 0
                signal = 'Neutral (High Fear)'
                explanation = f'VIX high at {current_vix:.1f} but stabilizing'
                
        elif current_vix >= 15:
            # Medium VIX - Normal regime - Trend-sensitive
            vix_regime = 'Normal'
            if vix_change_pct < -5:
                # VIX declining - risk-on
                vote = 1
                signal = 'Bullish (Risk-On)'
                explanation = f'VIX declining to {current_vix:.1f} ({vix_change_pct:.1f}%) - Risk appetite'
            elif vix_change_pct > 5:
                # VIX rising - risk-off
                vote = -1
                signal = 'Bearish (Risk-Off)'
                explanation = f'VIX rising to {current_vix:.1f} (+{vix_change_pct:.1f}%) - Fear building'
            else:
                vote = 0
                signal = 'Neutral'
                explanation = f'VIX stable at {current_vix:.1f}'
                
        else:
            # Low VIX - Complacency regime - Momentum
            vix_regime = 'Low/Complacent'
            if spy_momentum > 2:
                # SPY trending up with low vol - ride momentum
                vote = 2
                signal = 'Bullish (Momentum)'
                explanation = f'Low VIX {current_vix:.1f}, SPY momentum +{spy_momentum:.1f}% - Trend following'
            elif spy_momentum < -2:
                # SPY trending down with low vol
                vote = -2
                signal = 'Bearish (Weak Momentum)'
                explanation = f'Low VIX {current_vix:.1f}, SPY momentum {spy_momentum:.1f}% - Weak trend'
            else:
                vote = 1
                signal = 'Slightly Bullish'
                explanation = f'Low VIX {current_vix:.1f} - Calm market, slight bullish bias'
        
        return {
            'vote': vote,
            'signal': signal,
            'vix_level': round(current_vix, 2),
            'vix_regime': vix_regime,
            'explanation': explanation
        }
        
    except Exception as e:
        return {
            'vote': 0,
            'signal': 'Error',
            'vix_level': None,
            'vix_regime': None,
            'explanation': f'VIX model error: {str(e)[:50]}'
        }
