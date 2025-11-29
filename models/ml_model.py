"""
Machine Learning Model (XGBoost)
Trains an XGBoost classifier with extensive technical indicators to predict next-day price movement.
"""

import pandas as pd
import numpy as np
import warnings


def create_ml_features(data, vix_data=None):
    """
    Create comprehensive features for ML model.
    
    Features:
    - Lagged returns: 1-day, 2-day, 5-day, 10-day, 20-day
    - Volume indicators: volume change, volume ratio
    - RSI (14-day)
    - MACD histogram
    - Bollinger Band width and position
    - Distance from 50-day and 200-day MA
    - VIX level (if available)
    - ADX trend strength
    
    Args:
        data (pd.DataFrame): Historical data
        vix_data (pd.DataFrame, optional): VIX data for additional feature
        
    Returns:
        pd.DataFrame: DataFrame with features and target
    """
    df = data.copy()
    
    # ===== Price-based features =====
    # Returns at different timeframes
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_2d'] = df['Close'].pct_change(2)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Distance from moving averages (normalized)
    df['Dist_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Dist_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    
    # ===== RSI =====
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ===== MACD =====
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = macd_line - signal_line
    
    # ===== Bollinger Bands =====
    bb_middle = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (bb_std * 4) / bb_middle  # Normalized width
    df['BB_Position'] = (df['Close'] - (bb_middle - 2*bb_std)) / (4*bb_std)  # 0-1 scale
    
    # ===== Volume features =====
    df['Volume_Change'] = df['Volume'].pct_change(1)
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # ===== Volatility =====
    df['Volatility_10d'] = df['Return_1d'].rolling(window=10).std()
    df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
    
    # ===== High-Low Range =====
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # ===== VIX feature (if available) =====
    if vix_data is not None and not vix_data.empty:
        vix_aligned = vix_data.reindex(df.index, method='ffill')
        df['VIX_Level'] = vix_aligned['Close']
        df['VIX_Change'] = vix_aligned['Close'].pct_change(1)
    else:
        df['VIX_Level'] = np.nan
        df['VIX_Change'] = np.nan
    
    # ===== Target: Next day return =====
    df['Next_Day_Return'] = df['Close'].shift(-1) - df['Close']
    df['Target'] = (df['Next_Day_Return'] > 0).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df


def get_ml_vote(data, vix_data=None):
    """
    Train XGBoost model and predict next day movement.
    
    Vote Logic:
    - +1 if model predicts bullish (green day) with confidence > 55%
    - -1 if model predicts bearish (red day) with confidence > 55%
    - 0 if insufficient data, model fails, or low confidence
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        vix_data (pd.DataFrame, optional): VIX data for feature enhancement
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'prediction_proba': float,
            'explanation': str
        }
    """
    # Need sufficient data for training
    if len(data) < 250:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'prediction_proba': None,
            'explanation': 'Need at least 250 days of data for enhanced ML model'
        }
    
    try:
        # Try to import XGBoost, fallback to RandomForest if not available
        try:
            from xgboost import XGBClassifier
            use_xgboost = True
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            use_xgboost = False
        
        # Create features
        df = create_ml_features(data, vix_data)
        
        if len(df) < 50:
            return {
                'vote': 0,
                'signal': 'Insufficient Data',
                'prediction_proba': None,
                'explanation': 'Insufficient data after feature engineering'
            }
        
        # Define feature columns
        feature_cols = [
            'Return_1d', 'Return_2d', 'Return_5d', 'Return_10d', 'Return_20d',
            'Dist_SMA20', 'Dist_SMA50', 'Dist_SMA200',
            'RSI', 'MACD_Histogram', 'BB_Width', 'BB_Position',
            'Volume_Change', 'Volume_Ratio',
            'Volatility_10d', 'Volatility_20d',
            'High_Low_Range'
        ]
        
        # Add VIX features if available
        if 'VIX_Level' in df.columns and not df['VIX_Level'].isna().all():
            feature_cols.extend(['VIX_Level', 'VIX_Change'])
        
        # Prepare training data
        X = df[feature_cols].iloc[:-1]  # Exclude last row (no target for it)
        y = df['Target'].iloc[:-1]
        
        # Get features for prediction (last available data point)
        X_pred = df[feature_cols].iloc[-1:].values
        
        # Train model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            if use_xgboost:
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
            
            model.fit(X, y)
        
        # Predict
        prediction = model.predict(X_pred)[0]
        prediction_proba = model.predict_proba(X_pred)[0]
        
        # Get probability of the predicted class
        prob = prediction_proba[prediction]
        
        # Only vote if confident (>55%)
        confidence_threshold = 0.55
        
        if prob < confidence_threshold:
            vote = 0
            signal = 'Low Confidence'
            explanation = f'ML uncertain (Confidence: {prob*100:.1f}%)'
        elif prediction == 1:
            vote = 1
            signal = 'Bullish'
            model_type = 'XGBoost' if use_xgboost else 'RandomForest'
            explanation = f'{model_type}: Bullish (Conf: {prob*100:.1f}%)'
        else:
            vote = -1
            signal = 'Bearish'
            model_type = 'XGBoost' if use_xgboost else 'RandomForest'
            explanation = f'{model_type}: Bearish (Conf: {prob*100:.1f}%)'
        
        return {
            'vote': vote,
            'signal': signal,
            'prediction_proba': round(prob, 3),
            'explanation': explanation
        }
        
    except Exception as e:
        return {
            'vote': 0,
            'signal': 'Model Failed',
            'prediction_proba': None,
            'explanation': f'ML model error: {str(e)[:50]}'
        }

