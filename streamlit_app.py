import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulations_config import simulations_config

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

def plot_trade_count(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    trade_counts = df.groupby('time_step').size()
    ax.plot(trade_counts.index, trade_counts.values)
    ax.set_title("Trade Count Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Trades")
    ax.grid(True)
    return fig


def plot_price_and_trades(prices_df, orders_df):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    if 'true_price' not in prices_df.columns and 'close_price' in orders_df.columns:
        prices_df['true_price'] = orders_df.groupby('time_step')['close_price'].mean()

    if 'true_price' in prices_df.columns:
        ax.plot(prices_df['time_step'], prices_df['true_price'], label='True Price')
    
    # Scatter for buys and sells
    buys = orders_df[orders_df['stock_id'] == OVER_ID]
    sells = orders_df[orders_df['stock_id'] == UNDER_ID]
    ax.scatter(buys['time_step'], buys['price'], color='green', label='Buys', marker='^')
    ax.scatter(sells['time_step'], sells['price'], color='red', label='Sells', marker='v')
    
    ax.set_title("Price and Trades Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig

def calculate_statistics(df):
    results = {
        'Avg Profit': df['pnl'].mean(),
        'Max Drawdown': df['pnl'].cumsum().min(),
        'Avg Win': df[df['pnl'] > 0]['pnl'].mean(),
        'Avg Lose': df[df['pnl'] < 0]['pnl'].mean(),
        'Sharpe Ratio': df['pnl'].mean() / df['pnl'].std() * np.sqrt(len(df)),
        'Hold %': 100 * df['pnl'].sum() / df['bet_amount'].sum()
    }
    return results

def plot_pnl_graphs(orders_data, title):
    all_paths = orders_summary('pnl', orders_data)
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    for path in all_paths:
        ax.plot(path, alpha=0.1, c='b')
    ax.plot(np.array(all_paths).mean(axis=0), c='r')
    ax.set_title(f"{title} PnL")
    return fig

def plot_volume_graphs(orders_data, title):
    all_paths = orders_summary('bet_amount', orders_data)
    fig, axs = plt.subplots(1, 2, figsize=(20, 3))
    for path in all_paths:
        axs[0].plot(path, alpha=0.1, c='b')
        axs[1].plot(np.abs(path), alpha=0.1, c='b')
    axs[0].plot(np.array(all_paths).mean(axis=0), c='r')
    axs[0].set_title(f"{title} Inventory")
    axs[1].plot(np.array(np.abs(all_paths)).mean(axis=0), c='r')
    axs[1].set_title(f"{title} Volume Imbalance")
    return fig

def plot_holds(df):
    plt.figure(figsize=(10, 5))
    holds = df.groupby('sim_num').apply(lambda x: 100 * x.pnl.sum() / x.bet_amount.sum())
    plt.plot(holds.index, holds.values)
    plt.title("Hold Percentage by Simulation")
    plt.xlabel("Simulation Number")
    plt.ylabel("Hold %")
    plt.grid(True)
    return plt

def orders_summary(field, orders_df):
    all_paths = []
    for sim_num in orders_df.sim_num.unique():
        buys_sells = pd.DataFrame([], index=orders_df.time_step.unique())
        buys = orders_df[(orders_df.stock_id == OVER_ID) & (orders_df.sim_num == sim_num)].set_index('time_step')[[field]]
        sells = orders_df[(orders_df.stock_id == UNDER_ID) & (orders_df.sim_num == sim_num)].set_index('time_step')[[field]]
        buys_sells = buys_sells.join(buys, how='outer').join(sells, how='outer', rsuffix='_s').fillna(0)
        if field in ['bet_amount', 'quantity']:
            buys_sells[f'{field}_s'] *= -1
        one_path = buys_sells.sum(axis=1).cumsum()
        all_paths.append(one_path.values)
    return all_paths

def main():
    st.title("Market Simulation Analysis")
    sim_selection = st.sidebar.selectbox("Choose Simulation", list(simulations_config.keys()))
    config = simulations_config[sim_selection]
    if st.sidebar.button("Show Configuration"):
        st.json(config)  # Displays the config as JSON in the main panel

    # Sidebar 
    st.sidebar.header("Configuration")
    sim_name = st.sidebar.selectbox("Select Simulation", [
        'baseline_no_mm_low_vig', 'baseline_no_mm', 'baseline_no_mm_more_buys', 
        'baseline_no_mm_sharps', 'baseline_no_mm_sharps_high_vig', 'baseline_test'
    ])

    orders_df, prices_df = load_data(sim_name)

    if st.sidebar.checkbox("Show DataFrames"):
        st.subheader("Orders DataFrame")
        st.write(orders_df)
        st.subheader("Prices DataFrame")
        st.write(prices_df)

    if st.sidebar.checkbox("Plot P&L Graph"):
        fig = plot_pnl_graphs(orders_df, sim_selection)
        st.pyplot(fig)

    if st.sidebar.checkbox("Plot Volume Graphs"):
        fig = plot_volume_graphs(orders_df, sim_selection)
        st.pyplot(fig)

    if st.sidebar.checkbox("Plot Holds"):
        fig = plot_holds(orders_df)
        st.pyplot(fig)

    if st.sidebar.checkbox("Plot Trade Count"):
        fig = plot_trade_count(orders_df)
        st.pyplot(fig)
    
    if st.sidebar.checkbox("Plot Price and Trades"):
        fig = plot_price_and_trades(prices_df, orders_df)
        st.pyplot(fig)
    
    if st.sidebar.checkbox("Show Statistics"):
        stats = calculate_statistics(orders_df)
        st.write(stats)
if __name__ == "__main__":
    main()
