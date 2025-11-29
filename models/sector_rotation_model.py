"""
Sector Rotation Model
Analyzes momentum in key S&P sectors to predict SPY direction.
"""

import pandas as pd
import numpy as np


def get_sector_rotation_vote(spy_data, sector_data):
    """
    Calculate Sector Rotation vote based on sector momentum.
    
    SPY composition (approximate):
    - Technology (XLK): ~30%
    - Financials (XLF): ~13%
    - Energy (XLE): ~4%
    
    Vote Logic:
    - +1 if majority of sectors showing strength (10-day momentum > 0)
    - -1 if majority showing weakness
    - +2 if Tech (XLK) specifically strong (it's 30% of SPY)
    - -2 if Tech specifically weak
    
    Args:
        spy_data (pd.DataFrame): Historical SPY data
        sector_data (dict): Dictionary of sector dataframes {'XLK': df, 'XLF': df, 'XLE': df}
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'sector_strengths': dict,
            'explanation': str
        }
    """
    if not sector_data or len(sector_data) == 0:
        return {
            'vote': 0,
            'signal': 'No Sector Data',
            'sector_strengths': {},
            'explanation': 'Sector data not available'
        }
    
    if len(spy_data) < 15:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'sector_strengths': {},
            'explanation': 'Need at least 15 days of data'
        }
    
    try:
        sector_strengths = {}
        sector_momentum = {}
        
        # Calculate 10-day momentum for each sector
        for sector_name, sector_df in sector_data.items():
            if len(sector_df) >= 11:
                # Align to SPY dates
                sector_aligned = sector_df.reindex(spy_data.index, method='ffill')
                
                current_price = sector_aligned['Close'].iloc[-1]
                price_10d_ago = sector_aligned['Close'].iloc[-11]
                
                if not pd.isna(current_price) and not pd.isna(price_10d_ago) and price_10d_ago > 0:
                    momentum = ((current_price / price_10d_ago) - 1) * 100
                    sector_momentum[sector_name] = momentum
                    
                    if momentum > 0:
                        sector_strengths[sector_name] = 'Strong'
                    else:
                        sector_strengths[sector_name] = 'Weak'
        
        if not sector_momentum:
            return {
                'vote': 0,
                'signal': 'Data Issue',
                'sector_strengths': {},
                'explanation': 'Could not calculate sector momentum'
            }
        
        # Count strong vs weak sectors
        strong_count = sum(1 for s in sector_strengths.values() if s == 'Strong')
        weak_count = sum(1 for s in sector_strengths.values() if s == 'Weak')
        
        # Check XLK (Tech) specifically - it's the biggest sector
        xlk_momentum = sector_momentum.get('XLK', 0)
        xlk_strong = xlk_momentum > 0
        
        # Determine vote
        vote = 0
        signal = 'Neutral'
        
        # Tech sector dominance
        if xlk_momentum > 2:
            vote = 2
            signal = 'Strong Bullish (Tech Leading)'
            explanation = f'Tech leading with +{xlk_momentum:.1f}% momentum'
        elif xlk_momentum < -2:
            vote = -2
            signal = 'Strong Bearish (Tech Weak)'
            explanation = f'Tech weak with {xlk_momentum:.1f}% momentum'
        # Broad sector strength
        elif strong_count > weak_count:
            vote = 1
            signal = 'Bullish (Sector Rotation Positive)'
            sectors_list = ', '.join([f'{s}: {sector_strengths[s]}' for s in sector_strengths])
            explanation = f'{strong_count}/{len(sector_strengths)} sectors strong - {sectors_list}'
        elif weak_count > strong_count:
            vote = -1
            signal = 'Bearish (Sector Rotation Negative)'
            sectors_list = ', '.join([f'{s}: {sector_strengths[s]}' for s in sector_strengths])
            explanation = f'{weak_count}/{len(sector_strengths)} sectors weak - {sectors_list}'
        else:
            signal = 'Neutral'
            explanation = f'Mixed sector signals: {strong_count} strong, {weak_count} weak'
        
        return {
            'vote': vote,
            'signal': signal,
            'sector_strengths': sector_strengths,
            'explanation': explanation
        }
        
    except Exception as e:
        return {
            'vote': 0,
            'signal': 'Error',
            'sector_strengths': {},
            'explanation': f'Sector model error: {str(e)[:50]}'
        }
