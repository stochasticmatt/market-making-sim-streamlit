import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

OVER_ID = 'over_9.5'
UNDER_ID = 'under_9.5'
DATA_PATH = 'tests/results'

def load_data(sim_name):
    orders_path = f"{DATA_PATH}/{sim_name}.orders.csv"
    prices_path = f"{DATA_PATH}/{sim_name}.prices.csv"
    
    orders_df = pd.read_csv(orders_path).replace([
        f"STRIKEOUTS-PITCHER_{UNDER_ID}", f'STRIKEOUTS-PITCHER_{OVER_ID}'
    ], [UNDER_ID, OVER_ID])
    
    prices_df = pd.read_csv(prices_path)
    prices_df['under_true'] = 1.0 - prices_df['over_true']
    return orders_df, prices_df

# plot data using Altair
def plot_data(df, x, y, color):
    chart = alt.Chart(df).mark_point().encode(
        x=x,
        y=y,
        color=color,
        tooltip=list(df.columns)
    ).interactive().properties(
        width=800,
        height=400
    )
    return chart

def main():
    st.title("Market Simulation Analysis")

    # sidebar
    st.sidebar.header("Configuration")
    sim_name = st.sidebar.selectbox("Select Simulation", [
        'baseline_no_mm_low_vig', 'baseline_no_mm', 'baseline_no_mm_more_buys', 
        'baseline_no_mm_sharps', 'baseline_no_mm_sharps_high_vig'
    ])

    orders_df, prices_df = load_data(sim_name)

    if st.sidebar.checkbox("Show DataFrames"):
        st.subheader("Orders DataFrame")
        st.write(orders_df)
        st.subheader("Prices DataFrame")
        st.write(prices_df)

    st.sidebar.subheader("Plots")
    if st.sidebar.checkbox("Plot Bet Amounts"):
        bet_amount_df = orders_df.groupby('sim_num').bet_amount.sum().reset_index()
        st.altair_chart(plot_data(bet_amount_df, 'sim_num', 'bet_amount', 'sim_num'), use_container_width=True)

    if st.sidebar.checkbox("Plot P&L"):
        pnl_df = orders_df.groupby('sim_num').pnl.sum().reset_index()
        st.altair_chart(plot_data(pnl_df, 'sim_num', 'pnl', 'sim_num'), use_container_width=True)

    if st.sidebar.checkbox("Plot Holds"):
        holds_df = orders_df.groupby('sim_num').apply(
            lambda df: 100 * df.pnl.sum() / df.bet_amount.sum()
        ).reset_index(name='hold_percentage')
        st.altair_chart(plot_data(holds_df, 'sim_num', 'hold_percentage', 'sim_num'), use_container_width=True)

if __name__ == "__main__":
    main()
