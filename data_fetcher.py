"""
Data Fetcher Module
Handles fetching SPY data and ensures no look-ahead bias with strict data slicing.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_spy_data():
    """
    Fetch SPY daily data for maximum available period.
    
    Returns:
        pd.DataFrame: Historical SPY data with OHLCV columns
    """
    spy = yf.Ticker("SPY")
    data = spy.history(period="max")
    
    if data.empty:
        raise ValueError("Failed to fetch SPY data from yfinance")
    
    return data


def slice_data_to_date(data, target_date):
    """
    Slice dataframe to only include data up to and including the target date.
    This ensures no look-ahead bias.
    
    Args:
        data (pd.DataFrame): Full historical data
        target_date (datetime or str): Target date to slice to
        
    Returns:
        pd.DataFrame: Sliced data up to target date
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # Ensure the index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Handle timezone mismatch - localize target_date if data index has timezone
    if data.index.tz is not None and target_date.tz is None:
        target_date = target_date.tz_localize(data.index.tz)
    elif data.index.tz is None and target_date.tz is not None:
        target_date = target_date.tz_localize(None)
    
    # Slice data up to and including target date
    sliced_data = data[data.index <= target_date].copy()
    
    return sliced_data


def get_next_day_return(data, target_date):
    """
    Get the actual return for the day after the target date.
    Used for backtesting validation.
    
    Args:
        data (pd.DataFrame): Full historical data
        target_date (datetime or str): Target date
        
    Returns:
        tuple: (next_day_return_pct, next_day_date) or (None, None) if not available
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Handle timezone mismatch
    if data.index.tz is not None and target_date.tz is None:
        target_date = target_date.tz_localize(data.index.tz)
    elif data.index.tz is None and target_date.tz is not None:
        target_date = target_date.tz_localize(None)
    
    # Find the target date in the data
    if target_date not in data.index:
        # Find the closest previous date
        available_dates = data.index[data.index <= target_date]
        if len(available_dates) == 0:
            return None, None
        target_date = available_dates[-1]
    
    # Find the next trading day
    future_dates = data.index[data.index > target_date]
    if len(future_dates) == 0:
        return None, None
    
    next_date = future_dates[0]
    target_close = data.loc[target_date, 'Close']
    next_close = data.loc[next_date, 'Close']
    
    return_pct = ((next_close - target_close) / target_close) * 100
    
    return return_pct, next_date


def get_latest_trading_date(data):
    """
    Get the most recent trading date from the data.
    
    Args:
        data (pd.DataFrame): Historical data
        
    Returns:
        datetime: Latest trading date
    """
    return data.index[-1]
