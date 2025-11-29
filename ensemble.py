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
    get_technical_support_vote
)


def run_ensemble(data):
    """
    Run all 6 voting models and aggregate results.
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date (no look-ahead)
        
    Returns:
        dict: {
            'net_vote': int,
            'recommendation': str,
            'breakdown': list of dicts with individual model results
        }
    """
    # Run all models
    rsi_result = get_rsi_vote(data)
    mean_rev_result = get_mean_reversion_vote(data)
    garch_result = get_garch_vote(data)
    ml_result = get_ml_vote(data)
    factor_result = get_factor_vote(data)
    tech_support_result = get_technical_support_vote(data)
    
    # Aggregate votes
    breakdown = [
        {
            'model': 'RSI Momentum',
            'vote': rsi_result['vote'],
            'signal': rsi_result['signal'],
            'explanation': rsi_result['explanation']
        },
        {
            'model': 'Mean Reversion',
            'vote': mean_rev_result['vote'],
            'signal': mean_rev_result['signal'],
            'explanation': mean_rev_result['explanation']
        },
        {
            'model': 'GARCH Volatility',
            'vote': garch_result['vote'],
            'signal': garch_result['signal'],
            'explanation': garch_result['explanation']
        },
        {
            'model': 'ML Random Forest',
            'vote': ml_result['vote'],
            'signal': ml_result['signal'],
            'explanation': ml_result['explanation']
        },
        {
            'model': 'Factor Model',
            'vote': factor_result['vote'],
            'signal': factor_result['signal'],
            'explanation': factor_result['explanation']
        },
        {
            'model': 'Technical Support',
            'vote': tech_support_result['vote'],
            'signal': tech_support_result['signal'],
            'explanation': tech_support_result['explanation']
        }
    ]
    
    # Calculate net vote
    net_vote = sum(item['vote'] for item in breakdown)
    
    # Generate recommendation
    if net_vote >= 5:
        recommendation = "🚀 STRONG BUY"
        rec_color = "green"
    elif net_vote >= 2:
        recommendation = "✅ BUY"
        rec_color = "green"
    elif net_vote >= -1:
        recommendation = "➡️ NEUTRAL / HOLD"
        rec_color = "gray"
    elif net_vote >= -4:
        recommendation = "⚠️ SELL"
        rec_color = "orange"
    else:
        recommendation = "🔴 STRONG SELL"
        rec_color = "red"
    
    return {
        'net_vote': net_vote,
        'recommendation': recommendation,
        'rec_color': rec_color,
        'breakdown': breakdown
    }
