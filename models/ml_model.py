"""
Machine Learning Model (Random Forest)
Trains a RandomForestClassifier to predict next-day price movement.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings


def create_ml_features(data):
    """
    Create features for ML model.
    
    Features:
    - Lagged returns: 1-day, 2-day, 5-day
    - Volume change (1-day)
    
    Args:
        data (pd.DataFrame): Historical data
        
    Returns:
        pd.DataFrame: DataFrame with features and target
    """
    df = data.copy()
    
    # Calculate returns
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_2d'] = df['Close'].pct_change(2)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change(1)
    
    # Target: Is next day green? (1 = yes, 0 = no)
    df['Next_Day_Return'] = df['Close'].shift(-1) - df['Close']
    df['Target'] = (df['Next_Day_Return'] > 0).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df


def get_ml_vote(data):
    """
    Train Random Forest model and predict next day movement.
    
    Vote Logic:
    - +1 if model predicts bullish (green day)
    - -1 if model predicts bearish (red day)
    - 0 if insufficient data or model fails
    
    Args:
        data (pd.DataFrame): Historical data sliced to target date
        
    Returns:
        dict: {
            'vote': int,
            'signal': str,
            'prediction_proba': float,
            'explanation': str
        }
    """
    # Need sufficient data for training
    if len(data) < 30:
        return {
            'vote': 0,
            'signal': 'Insufficient Data',
            'prediction_proba': None,
            'explanation': 'Need at least 30 days of data for ML model'
        }
    
    try:
        # Create features
        df = create_ml_features(data)
        
        if len(df) < 20:
            return {
                'vote': 0,
                'signal': 'Insufficient Data',
                'prediction_proba': None,
                'explanation': 'Insufficient data after feature engineering'
            }
        
        # Features and target
        feature_cols = ['Return_1d', 'Return_2d', 'Return_5d', 'Volume_Change']
        X = df[feature_cols].iloc[:-1]  # Exclude last row (no target for it)
        y = df['Target'].iloc[:-1]
        
        # Get features for prediction (last available data point)
        X_pred = df[feature_cols].iloc[-1:].values
        
        # Train Random Forest
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
        
        # Predict
        prediction = rf_model.predict(X_pred)[0]
        prediction_proba = rf_model.predict_proba(X_pred)[0]
        
        # Get probability of the predicted class
        prob = prediction_proba[prediction]
        
        # Determine vote
        if prediction == 1:
            vote = 1
            signal = 'Bullish'
        else:
            vote = -1
            signal = 'Bearish'
        
        explanation = f"ML Prediction: {signal} (Confidence: {prob*100:.1f}%)"
        
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
