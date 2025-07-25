import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest_and_plot(config):
    tickers = config['tickers']
    start_date = config['start_date']
    end_date = config['end_date']
    k = config['momentum']['k']
    x = config['momentum']['x']
    h = config['holding_period']

    # Download data
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[['Close']]
        df['Return'] = df['Close'].pct_change()
        data[ticker] = df

    for ticker, df in data.items():
        df['Momentum'] = df['Close'].pct_change(k)
        df['Signal'] = (df['Momentum'] > x).astype(int)
        data[ticker] = df

    # Print the first few rows for AAPL with new columns
    print('AAPL data with momentum and signal:')
    print(data['AAPL'].head(10))

    backtest_results = {}
    for ticker, df in data.items():
        returns = []
        i = 0
        while i < len(df) - h:
            if df['Signal'].iloc[i] == 1:
                entry = df['Close'].iloc[i+1]
                exit_ = df['Close'].iloc[i+1+h-1]
                ret = (exit_ - entry) / entry
                returns.append(ret)
                i += h
            else:
                i += 1
        backtest_results[ticker] = returns

    # Print summary stats for AAPL
    aapl_returns = backtest_results['AAPL']
    if aapl_returns:
        print(f"AAPL: {len(aapl_returns)} trades")
        print(f"Average return per trade: {np.mean(aapl_returns):.4f}")
        print(f"Win rate: {np.mean(np.array(aapl_returns) > 0) * 100:.2f}%")
    else:
        print("No trades for AAPL.")

    # Plot cumulative returns for all stocks
    fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 6 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]

    # Loop through each ticker and plot results
    for idx, ticker in enumerate(tickers):
        momentum_returns = np.array(backtest_results[ticker])
        if len(momentum_returns) > 0:
            df = data[ticker].copy()
            df['StrategyReturn'] = 0.0
            i = 0
            trade_idx = 0

            # Go through the DataFrame and assign strategy returns to entry days
            while i < len(df) - h and trade_idx < len(momentum_returns):
                if df['Signal'].iloc[i] == 1:
                    df.iloc[i+1, df.columns.get_loc('StrategyReturn')] = momentum_returns[trade_idx]
                    trade_idx += 1
                    # Skip ahead by the holding period to avoid overlapping trades
                    i += h
                else:
                    i += 1

            # Calculate cumulative returns for both strategies
            df['StrategyCumulative'] = (1 + df['StrategyReturn']).cumprod()
            df['BuyHoldCumulative'] = (1 + df['Return'].fillna(0)).cumprod()
            ax = axes[idx]

            # Plot buy-and-hold and momentum strategy on the same subplot
            ax.plot(df.index, df['BuyHoldCumulative'], label='Buy & Hold', color='gray')
            ax.plot(df.index, df['StrategyCumulative'], label='Momentum Strategy', color='blue')
            ax.set_title(f'{ticker}: Cumulative Return ({start_date[:4]}-{end_date[:4]})')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
        else:
            # If no trades were made, let the user know in the plot
            axes[idx].set_title(f'{ticker}: No trades, so no strategy plot.')
            axes[idx].set_ylabel('Cumulative Return')

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
    return data, backtest_results 