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


@st.cache_data(ttl=3600)
def fetch_vix_data():
    """
    Fetch CBOE VIX (Volatility Index) data.
    VIX measures market fear/volatility - key predictor for SPY movements.
    
    Returns:
        pd.DataFrame: Historical VIX data with OHLCV columns
    """
    try:
        vix = yf.Ticker("^VIX")
        data = vix.history(period="max")
        
        if data.empty:
            raise ValueError("Failed to fetch VIX data from yfinance")
        
        return data
    except Exception as e:
        st.warning(f"⚠️ Could not fetch VIX data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_treasury_data():
    """
    Fetch 10-Year Treasury Yield (^TNX) data.
    Treasury yields indicate institutional money flow and risk appetite.
    
    Returns:
        pd.DataFrame: Historical Treasury yield data
    """
    try:
        tnx = yf.Ticker("^TNX")
        data = tnx.history(period="max")
        
        if data.empty:
            raise ValueError("Failed to fetch Treasury data from yfinance")
        
        return data
    except Exception as e:
        st.warning(f"⚠️ Could not fetch Treasury data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_sector_etfs():
    """
    Fetch major sector ETF data for rotation analysis.
    - XLK: Technology Select Sector (~30% of SPY)
    - XLF: Financial Select Sector (~13% of SPY)
    - XLE: Energy Select Sector (~4% of SPY)
    
    Returns:
        dict: Dictionary of sector dataframes {'XLK': df, 'XLF': df, 'XLE': df}
    """
    sectors = ['XLK', 'XLF', 'XLE']
    sector_data = {}
    
    for sector in sectors:
        try:
            ticker = yf.Ticker(sector)
            data = ticker.history(period="max")
            
            if not data.empty:
                sector_data[sector] = data
            else:
                st.warning(f"⚠️ Could not fetch {sector} data")
        except Exception as e:
            st.warning(f"⚠️ Error fetching {sector}: {str(e)}")
    
    return sector_data


@st.cache_data(ttl=3600)
def fetch_all_data():
    """
    Fetch all required data sources and align them by date.
    
    Returns:
        dict: {
            'spy': pd.DataFrame,
            'vix': pd.DataFrame,
            'treasury': pd.DataFrame,
            'sectors': dict
        }
    """
    data = {}
    
    # Fetch SPY (primary data)
    data['spy'] = fetch_spy_data()
    
    # Fetch supplementary data
    data['vix'] = fetch_vix_data()
    data['treasury'] = fetch_treasury_data()
    data['sectors'] = fetch_sector_etfs()
    
    return data
