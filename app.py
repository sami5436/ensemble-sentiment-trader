"""
Ensemble Sentiment Trader - Streamlit Application
A voting-based algorithmic trading system for SPY with 6 models.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data_fetcher import (
    fetch_spy_data,
    slice_data_to_date,
    get_next_day_return,
    get_latest_trading_date
)
from ensemble import run_ensemble


# Page configuration
st.set_page_config(
    page_title="Ensemble Sentiment Trader",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Ensemble Sentiment Trader")
st.markdown("**A Voting-Based Algorithmic Trading System for SPY**")
st.markdown("---")

# Fetch data (cached)
try:
    with st.spinner("Fetching SPY data..."):
        full_data = fetch_spy_data()
    
    st.success(f"✅ Loaded {len(full_data)} days of SPY data")
    
except Exception as e:
    st.error(f"❌ Error fetching data: {str(e)}")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["🔮 Live Forecast", "⏰ Time Machine Backtest"])

# ========== TAB 1: LIVE FORECAST ==========
with tab1:
    st.header("Today's Live Forecast")
    st.markdown("Current ensemble prediction based on the latest available data.")
    
    # Get latest trading date
    latest_date = get_latest_trading_date(full_data)
    
    st.info(f"📅 Latest Trading Date: **{latest_date.strftime('%Y-%m-%d')}**")
    
    # Slice data to latest date (no look-ahead)
    current_data = slice_data_to_date(full_data, latest_date)
    
    # Run ensemble
    with st.spinner("Running ensemble models..."):
        results = run_ensemble(current_data)
    
    # Display Net Vote
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric(
            label="Net Vote",
            value=results['net_vote'],
            delta=None
        )
    
    with col2:
        # Color based on recommendation
        if results['rec_color'] == 'green':
            st.markdown(f"### :green[{results['recommendation']}]")
        elif results['rec_color'] == 'red':
            st.markdown(f"### :red[{results['recommendation']}]")
        elif results['rec_color'] == 'orange':
            st.markdown(f"### :orange[{results['recommendation']}]")
        else:
            st.markdown(f"### {results['recommendation']}")
    
    with col3:
        current_price = current_data['Close'].iloc[-1]
        st.metric(
            label="SPY Price",
            value=f"${current_price:.2f}"
        )
    
    # Display breakdown
    st.markdown("---")
    st.subheader("📊 Vote Breakdown")
    
    breakdown_df = pd.DataFrame(results['breakdown'])
    
    # Style the dataframe
    def color_vote(val):
        if val > 0:
            color = 'background-color: #90EE90'  # Light green
        elif val < 0:
            color = 'background-color: #FFB6C1'  # Light red
        else:
            color = 'background-color: #D3D3D3'  # Light gray
        return color
    
    styled_df = breakdown_df.style.applymap(color_vote, subset=['vote'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Legend
    st.markdown("""
    **Vote Weight Scale:**
    - RSI, Mean Reversion, ML, Factor: ±1 vote
    - GARCH, Technical Support: ±3 votes
    
    **Recommendation Thresholds:**
    - Strong Buy: ≥ 5 votes
    - Buy: 2-4 votes
    - Neutral/Hold: -1 to 1 votes
    - Sell: -4 to -2 votes
    - Strong Sell: ≤ -5 votes
    """)

# ========== TAB 2: TIME MACHINE BACKTEST ==========
with tab2:
    st.header("⏰ Time Machine Backtest")
    st.markdown("Travel back in time to test the ensemble on historical dates.")
    
    # Date picker
    min_date = full_data.index[100]  # Need at least 100 days for GARCH
    max_date = full_data.index[-2]  # Can't pick last day (need next day for reality check)
    
    selected_date = st.date_input(
        "Select a historical date:",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    # Convert to datetime
    selected_datetime = pd.to_datetime(selected_date)
    
    st.info(f"🕰️ Analyzing: **{selected_datetime.strftime('%Y-%m-%d')}**")
    
    # Slice data to selected date (STRICT NO LOOK-AHEAD)
    historical_data = slice_data_to_date(full_data, selected_datetime)
    
    st.warning(f"⚠️ Using only {len(historical_data)} days of data up to {selected_datetime.strftime('%Y-%m-%d')}")
    
    # Run ensemble on historical data
    with st.spinner("Running ensemble on historical data..."):
        historical_results = run_ensemble(historical_data)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗳️ Ensemble Prediction")
        st.metric("Net Vote", historical_results['net_vote'])
        
        if historical_results['rec_color'] == 'green':
            st.markdown(f"## :green[{historical_results['recommendation']}]")
        elif historical_results['rec_color'] == 'red':
            st.markdown(f"## :red[{historical_results['recommendation']}]")
        elif historical_results['rec_color'] == 'orange':
            st.markdown(f"## :orange[{historical_results['recommendation']}]")
        else:
            st.markdown(f"## {historical_results['recommendation']}")
    
    with col2:
        st.markdown("### ✅ Reality Check")
        
        # Get actual next day return
        actual_return, next_date = get_next_day_return(full_data, selected_datetime)
        
        if actual_return is not None:
            st.metric(
                "Actual Next Day Return",
                f"{actual_return:+.2f}%",
                delta=None
            )
            
            st.info(f"Next Trading Day: **{next_date.strftime('%Y-%m-%d')}**")
            
            # Check if prediction was correct
            predicted_bullish = historical_results['net_vote'] > 0
            actual_bullish = actual_return > 0
            
            if predicted_bullish == actual_bullish:
                st.success("✅ **CORRECT PREDICTION!**")
            else:
                st.error("❌ **INCORRECT PREDICTION**")
            
            # Show what actually happened
            if actual_bullish:
                st.markdown("📈 Market moved **UP** (Green Day)")
            else:
                st.markdown("📉 Market moved **DOWN** (Red Day)")
        else:
            st.warning("⚠️ Next day data not available")
    
    # Display breakdown
    st.markdown("---")
    st.subheader("📊 Historical Vote Breakdown")
    
    breakdown_df = pd.DataFrame(historical_results['breakdown'])
    
    styled_df = breakdown_df.style.applymap(color_vote, subset=['vote'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Ensemble Sentiment Trader</strong> | Built with Streamlit | Data: Yahoo Finance</p>
    <p><em>⚠️ For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)
