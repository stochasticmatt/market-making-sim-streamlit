import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

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

    if 'time_step' not in prices_df.columns:
        raise ValueError("Missing 'time_step' in prices data.")
    
    close_prices = prices_df[prices_df.time_step == prices_df.time_step.max()] \
                    .set_index('sim_num')[['over_true', 'under_true']]
    
    close_prices = close_prices.reset_index().melt(id_vars=['sim_num'], 
        value_vars=['over_true', 'under_true'], var_name='stock_id', value_name='close_price') \
        .set_index('sim_num').replace({'over_true': OVER_ID, 'under_true': UNDER_ID})

    orders_df['bet_amount'] = (orders_df['price'] * orders_df['quantity']).astype(int)
    
    # Join with close prices to calculate returns and pnl
    orders_df = orders_df.join(
        close_prices.reset_index()[['sim_num', 'stock_id', 'close_price']].set_index(['sim_num', 'stock_id']),
        on=['sim_num', 'stock_id'], rsuffix='_p', how='left'
    )
    
    orders_df['returns'] = (orders_df['price'] - orders_df['close_price']) / orders_df['price']
    orders_df['pnl'] = orders_df['bet_amount'] * orders_df['returns']
    
    return orders_df, prices_df

# plot cumulative bet amount
def plot_cumulative_bets(df):
    plt.figure(figsize=(10, 5))
    grouped = df.groupby('sim_num')['bet_amount'].sum().cumsum()
    plt.plot(grouped)
    plt.title("Cumulative Bet Amounts by Simulation")
    plt.xlabel("Simulation Number")
    plt.ylabel("Cumulative Bet Amount")
    plt.grid(True)
    return plt

# plot P&L
def plot_pnl(df):
    plt.figure(figsize=(10, 5))
    pnl_data = df.groupby('sim_num')['pnl'].sum()
    plt.bar(pnl_data.index, pnl_data.values)
    plt.title("Profit and Loss by Simulation")
    plt.xlabel("Simulation Number")
    plt.ylabel("P&L")
    plt.grid(True)
    return plt

# plot Hold %
def plot_holds(df):
    plt.figure(figsize=(10, 5))
    holds = df.groupby('sim_num').apply(lambda x: 100 * x.pnl.sum() / x.bet_amount.sum())
    plt.plot(holds.index, holds.values)
    plt.title("Hold Percentage by Simulation")
    plt.xlabel("Simulation Number")
    plt.ylabel("Hold %")
    plt.grid(True)
    return plt

def main():
    st.title("Market Simulation Analysis")

    # Sidebar 
    st.sidebar.header("Configuration")
    sim_name = st.sidebar.selectbox("Select Simulation", [
        'baseline_no_mm_low_vig', 'baseline_no_mm', 'baseline_no_mm_more_buys', 
        'baseline_no_mm_sharps', 'baseline_no_mm_sharps_high_vig'
    ])

    orders_df, prices_df = load_data(sim_name)

    # display orders and prices DataFrames
    if st.sidebar.checkbox("Show DataFrames"):
        st.subheader("Orders DataFrame")
        st.write(orders_df)
        st.subheader("Prices DataFrame")
        st.write(prices_df)

    st.sidebar.subheader("Plots")
    if st.sidebar.checkbox("Plot Cumulative Bets"):
        fig = plot_cumulative_bets(orders_df)
        st.pyplot(fig)

    if st.sidebar.checkbox("Plot P&L"):
        fig = plot_pnl(orders_df)
        st.pyplot(fig)

    if st.sidebar.checkbox("Plot Holds"):
        fig = plot_holds(orders_df)
        st.pyplot(fig)

if __name__ == "__main__":
    main()