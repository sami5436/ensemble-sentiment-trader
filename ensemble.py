"""
Ensemble Logic
Orchestrates all voting models and aggregates results.
"""

from models import (
    get_rsi_vote,
    get_mean_reversion_vote,
    get_garch_vote,
    get_ml_vote,
    get_factor_vote,
    get_technical_support_vote,
    get_macd_bb_vote,
    get_vix_regime_vote,
    get_market_regime_vote,
    get_sector_rotation_vote
)


def run_ensemble(data, vix_data=None, sector_data=None):
    """
    Run all 10 voting models and aggregate results.
    
    Args:
        data (pd.DataFrame): Historical SPY data sliced to target date (no look-ahead)
        vix_data (pd.DataFrame, optional): VIX data for regime analysis
        sector_data (dict, optional): Sector ETF data for rotation analysis
        
    Returns:
        dict: {
            'net_vote': float,
            'recommendation': str,
            'rec_color': str,
            'breakdown': list of dicts with individual model results,
            'active_models': int
        }
    """
    # Run all models
    rsi_result = get_rsi_vote(data)
    mean_rev_result = get_mean_reversion_vote(data)
    garch_result = get_garch_vote(data)
    ml_result = get_ml_vote(data, vix_data)
    factor_result = get_factor_vote(data)
    tech_support_result = get_technical_support_vote(data)
    macd_bb_result = get_macd_bb_vote(data)
    vix_regime_result = get_vix_regime_vote(data, vix_data) if vix_data is not None else {'vote': 0, 'signal': 'No Data', 'explanation': 'VIX data unavailable'}
    market_regime_result = get_market_regime_vote(data)
    sector_rotation_result = get_sector_rotation_vote(data, sector_data) if sector_data else {'vote': 0, 'signal': 'No Data', 'explanation': 'Sector data unavailable'}
    
    # Aggregate votes with weights
    # Note: Some models already have higher weight (±2 or ±3) built into their vote
    breakdown = [
        {
            'model': 'RSI Momentum',
            'vote': rsi_result['vote'],
            'weight': '±1',
            'signal': rsi_result['signal'],
            'explanation': rsi_result['explanation']
        },
        {
            'model': 'Mean Reversion',
            'vote': mean_rev_result['vote'],
            'weight': '±1',
            'signal': mean_rev_result['signal'],
            'explanation': mean_rev_result['explanation']
        },
        {
            'model': 'GARCH Volatility',
            'vote': garch_result['vote'],
            'weight': '±3',
            'signal': garch_result['signal'],
            'explanation': garch_result['explanation']
        },
        {
            'model': 'ML XGBoost',
            'vote': ml_result['vote'],
            'weight': '±1',
            'signal': ml_result['signal'],
            'explanation': ml_result['explanation']
        },
        {
            'model': 'Factor Model',
            'vote': factor_result['vote'],
            'weight': '±1',
            'signal': factor_result['signal'],
            'explanation': factor_result['explanation']
        },
        {
            'model': 'Technical Support',
            'vote': tech_support_result['vote'],
            'weight': '±3',
            'signal': tech_support_result['signal'],
            'explanation': tech_support_result['explanation']
        },
        {
            'model': 'MACD + Bollinger',
            'vote': macd_bb_result['vote'],
            'weight': '±1',
            'signal': macd_bb_result['signal'],
            'explanation': macd_bb_result['explanation']
        },
        {
            'model': 'VIX Regime',
            'vote': vix_regime_result['vote'],
            'weight': '±3',
            'signal': vix_regime_result['signal'],
            'explanation': vix_regime_result['explanation']
        },
        {
            'model': 'Market Regime',
            'vote': market_regime_result['vote'],
            'weight': '±2',
            'signal': market_regime_result['signal'],
            'explanation': market_regime_result['explanation']
        },
        {
            'model': 'Sector Rotation',
            'vote': sector_rotation_result['vote'],
            'weight': '±2',
            'signal': sector_rotation_result['signal'],
            'explanation': sector_rotation_result['explanation']
        }
    ]
    
    # Calculate net vote
    net_vote = sum(item['vote'] for item in breakdown)
    
    # Count active models (those that voted non-zero)
    active_models = sum(1 for item in breakdown if item['vote'] != 0)
    
    # Generate recommendation (adjusted thresholds for more models)
    if net_vote >= 6:
        recommendation = "🚀 STRONG BUY"
        rec_color = "green"
    elif net_vote >= 3:
        recommendation = "✅ BUY"
        rec_color = "green"
    elif net_vote >= -2:
        recommendation = "➡️ NEUTRAL / HOLD"
        rec_color = "gray"
    elif net_vote >= -5:
        recommendation = "⚠️ SELL"
        rec_color = "orange"
    else:
        recommendation = "🔴 STRONG SELL"
        rec_color = "red"
    
    return {
        'net_vote': net_vote,
        'recommendation': recommendation,
        'rec_color': rec_color,
        'breakdown': breakdown,
        'active_models': active_models
    }

