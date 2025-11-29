"""
Ensemble Sentiment Trader - Streamlit Application
A voting-based algorithmic trading system for SPY with 6 models.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data_fetcher import (
    fetch_spy_data,
    fetch_all_data,
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
    with st.spinner("Fetching SPY, VIX, Treasury, and Sector ETF data..."):
        all_data = fetch_all_data()
        full_data = all_data['spy']
        vix_data = all_data['vix']
        sector_data = all_data['sectors']
    
    data_sources = []
    data_sources.append(f"SPY: {len(full_data)} days")
    if not vix_data.empty:
        data_sources.append(f"VIX: {len(vix_data)} days")
    if sector_data:
        data_sources.append(f"Sectors: {', '.join(sector_data.keys())}")
    
    st.success(f"✅ Loaded data - {' | '.join(data_sources)}")
    
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
        results = run_ensemble(current_data, vix_data, sector_data)
    
    # Display Net Vote
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
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
    
    with col4:
        st.metric(
            label="Active Models",
            value=f"{results['active_models']}/10"
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
    - Light models (±1): RSI, Mean Reversion, ML, Factor, MACD+BB
    - Medium models (±2): Market Regime, Sector Rotation
    - Heavy models (±3): GARCH, Technical Support, VIX Regime
    
    **Recommendation Thresholds (10 models):**
    - Strong Buy: ≥ 6 votes
    - Buy: 3-5 votes
    - Neutral/Hold: -2 to 2 votes
    - Sell: -5 to -3 votes
    - Strong Sell: ≤ -6 votes
    """)

# ========== TAB 2: TIME MACHINE BACKTEST ==========
with tab2:
    st.header("⏰ Time Machine Backtest")
    st.markdown("Travel back in time to test the ensemble on historical dates.")
    
    # Add mode selector
    backtest_mode = st.radio(
        "Backtest Mode:",
        ["Single Date", "Date Range Analysis"],
        horizontal=True
    )
    
    # Date picker
    min_date = full_data.index[100]  # Need at least 100 days for GARCH
    max_date = full_data.index[-2]  # Can't pick last day (need next day for reality check)
    
    if backtest_mode == "Single Date":
        # ===== SINGLE DATE MODE =====
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
            historical_results = run_ensemble(historical_data, vix_data, sector_data)
        
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
    
    else:
        # ===== DATE RANGE ANALYSIS MODE =====
        st.markdown("### 📊 Backtest over Date Range")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=(max_date - pd.Timedelta(days=90)).date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="end_date"
            )
        
        if start_date > end_date:
            st.error("❌ Start date must be before end date!")
        else:
            # Run backtest
            if st.button("🚀 Run Backtest", type="primary"):
                start_datetime = pd.to_datetime(start_date)
                end_datetime = pd.to_datetime(end_date)
                
                # Handle timezone mismatch - localize datetimes if data index has timezone
                if full_data.index.tz is not None:
                    if start_datetime.tz is None:
                        start_datetime = start_datetime.tz_localize(full_data.index.tz)
                    if end_datetime.tz is None:
                        end_datetime = end_datetime.tz_localize(full_data.index.tz)
                
                # Get all trading dates in range
                date_range = full_data[(full_data.index >= start_datetime) & 
                                      (full_data.index <= end_datetime)].index
                
                if len(date_range) < 2:
                    st.warning("⚠️ Not enough trading days in selected range")
                else:
                    st.info(f"Backtesting {len(date_range)} trading days...")
                    
                    results_list = []
                    progress_bar = st.progress(0)
                    
                    for i, test_date in enumerate(date_range[:-1]):  # Exclude last date (need next day)
                        # Slice data to test date
                        test_data = slice_data_to_date(full_data, test_date)
                        
                        # Run ensemble
                        result = run_ensemble(test_data, vix_data, sector_data)
                        
                        # Get actual next day return
                        actual_return, next_date = get_next_day_return(full_data, test_date)
                        
                        if actual_return is not None:
                            predicted_bullish = result['net_vote'] > 0
                            actual_bullish = actual_return > 0
                            correct = predicted_bullish == actual_bullish
                            
                            results_list.append({
                                'date': test_date,
                                'net_vote': result['net_vote'],
                                'prediction': 'Bullish' if predicted_bullish else 'Bearish',
                                'actual_return': actual_return,
                                'actual_direction': 'Up' if actual_bullish else 'Down',
                                'correct': correct,
                                'breakdown': result['breakdown']
                            })
                        
                        progress_bar.progress((i + 1) / (len(date_range) - 1))
                    
                    progress_bar.empty()
                    
                    if len(results_list) == 0:
                        st.warning("No valid backtest results")
                    else:
                        # Calculate metrics
                        results_df = pd.DataFrame(results_list)
                        accuracy = (results_df['correct'].sum() / len(results_df)) * 100
                        total_trades = len(results_df)
                        correct_trades = results_df['correct'].sum()
                        incorrect_trades = total_trades - correct_trades
                        
                        # Display metrics
                        st.markdown("---")
                        st.subheader("📈 Performance Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.1f}%")
                        with col2:
                            st.metric("Total Predictions", total_trades)
                        with col3:
                            st.metric("✅ Correct", correct_trades)
                        with col4:
                            st.metric("❌ Incorrect", incorrect_trades)
                        
                        # Success/Failure Pie Chart
                        st.markdown("---")
                        st.subheader("🎯 Success vs Failure")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Create pie chart data
                            import plotly.graph_objects as go
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=['Correct', 'Incorrect'],
                                values=[correct_trades, incorrect_trades],
                                marker=dict(colors=['#90EE90', '#FFB6C1']),
                                hole=0.3
                            )])
                            fig.update_layout(
                                title="Prediction Accuracy",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Performance over time
                            import plotly.express as px
                            
                            results_df['cumulative_correct'] = results_df['correct'].cumsum()
                            results_df['cumulative_accuracy'] = (results_df['cumulative_correct'] / 
                                                                 (results_df.index + 1)) * 100
                            
                            fig = px.line(
                                results_df,
                                x='date',
                                y='cumulative_accuracy',
                                title='Accuracy Over Time',
                                labels={'cumulative_accuracy': 'Accuracy (%)', 'date': 'Date'}
                            )
                            fig.update_traces(line_color='#4CAF50')
                            fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                                         annotation_text="Random (50%)")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Model-level analysis
                        st.markdown("---")
                        st.subheader("🔬 Model Performance Analysis")
                        
                        # Calculate per-model accuracy
                        model_stats = []
                        model_names = ['RSI Momentum', 'Mean Reversion', 'GARCH Volatility', 
                                      'ML Random Forest', 'Factor Model', 'Technical Support']
                        
                        for model_name in model_names:
                            model_correct = 0
                            model_total = 0
                            
                            for result in results_list:
                                actual_bullish = result['actual_return'] > 0
                                
                                # Find model vote
                                for model_data in result['breakdown']:
                                    if model_data['model'] == model_name:
                                        model_vote = model_data['vote']
                                        
                                        # Only count if model made a prediction (non-zero vote)
                                        if model_vote != 0:
                                            model_bullish = model_vote > 0
                                            if model_bullish == actual_bullish:
                                                model_correct += 1
                                            model_total += 1
                            
                            if model_total > 0:
                                model_accuracy = (model_correct / model_total) * 100
                                model_stats.append({
                                    'Model': model_name,
                                    'Accuracy': model_accuracy,
                                    'Predictions': model_total,
                                    'Correct': model_correct
                                })
                        
                        model_stats_df = pd.DataFrame(model_stats)
                        
                        # Bar chart of model accuracies
                        fig = px.bar(
                            model_stats_df,
                            x='Model',
                            y='Accuracy',
                            title='Individual Model Accuracy (when making non-zero predictions)',
                            color='Accuracy',
                            color_continuous_scale=['red', 'yellow', 'green'],
                            range_color=[0, 100]
                        )
                        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                                     annotation_text="Random (50%)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed model stats table
                        st.dataframe(
                            model_stats_df.style.background_gradient(subset=['Accuracy'], cmap='RdYlGn', vmin=0, vmax=100),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Show recent predictions
                        st.markdown("---")
                        st.subheader("📋 Recent Predictions")
                        
                        recent_df = results_df[['date', 'prediction', 'net_vote', 'actual_direction', 
                                               'actual_return', 'correct']].tail(20).copy()
                        recent_df['date'] = recent_df['date'].dt.strftime('%Y-%m-%d')
                        recent_df['actual_return'] = recent_df['actual_return'].round(2)
                        
                        def highlight_correct(val):
                            return 'background-color: #90EE90' if val else 'background-color: #FFB6C1'
                        
                        st.dataframe(
                            recent_df.style.applymap(highlight_correct, subset=['correct']),
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
