import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sharpe_ratio(returns, periods_per_year=252):
    """
    Calculate the annualized Sharpe ratio of a return series.
    """
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return np.nan
    return (mean / std) * np.sqrt(periods_per_year)


def max_drawdown(cum_returns):
    """
    Calculate the maximum drawdown of a cumulative return series.
    """
    roll_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - roll_max) / roll_max
    return drawdown.min()


def t_statistic(returns):
    """
    Calculate the t-statistic for the mean of a return series.
    """
    mean = np.mean(returns)
    std = np.std(returns)
    n = len(returns)
    if std == 0 or n == 0:
        return np.nan
    return mean / (std / np.sqrt(n))


def run_walkforward_backtest(df, k, x, h, train_window, test_window):
    """
    Perform walk-forward backtesting with rolling train/test windows.
    Args:
        df (pd.DataFrame): DataFrame with at least a 'Close' column (or similar).
        k (int): Lookback period for momentum calculation.
        x (float): Momentum threshold for signal.
        h (int): Holding period (in bars).
        train_window (int): Number of periods in each training window.
        test_window (int): Number of periods in each test window.
        verbose (bool): If True, print debug output (default False).
    Returns:
        np.ndarray: Array of all test-period returns from all windows.
    """
    df = df.reset_index(drop=True)

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
    close_col = 'Close'
    for col in df.columns:
        if 'Close' in col:
            close_col = col
            break

    n = len(df)
    test_returns = []
    start = 0
    window_count = 0

    while start + train_window + test_window <= n:
        train_idx = range(start, start + train_window)
        test_idx = range(start + train_window, start + train_window + test_window)
        # Prepend last k rows from train to test for momentum calculation
        if start + train_window - k >= 0:
            prepend_idx = range(start + train_window - k, start + train_window)
            full_idx = list(prepend_idx) + list(test_idx)
            test_df = df.iloc[full_idx].copy()
            test_df['Momentum'] = test_df[close_col].pct_change(k)
            test_df['Signal'] = (test_df['Momentum'] > x).astype(int)
            # Drop the prepended rows
            test_df = test_df.iloc[k:]
        else:
            test_df = df.iloc[list(test_idx)].copy()
            test_df['Momentum'] = test_df[close_col].pct_change(k)
            test_df['Signal'] = (test_df['Momentum'] > x).astype(int)
        i = 0
        trades_in_window = 0
        while i < len(test_df) - h:
            if test_df['Signal'].iloc[i] == 1:
                entry = test_df[close_col].iloc[i+1]
                exit_ = test_df[close_col].iloc[i+1+h-1]
                ret = (exit_ - entry) / entry
                test_returns.append(ret)
                trades_in_window += 1
                i += h
            else:
                i += 1
        window_count += 1
        start += test_window
    return np.array(test_returns)

def run_backtest_and_plot(config):
    """
    Run a simple momentum backtest for all tickers in config, plot results, and return data/results.
    
    Returns:
        tuple:
            - data (dict): Dict of DataFrames for each ticker, with calculated columns.
            - backtest_results (dict): Dict of lists of trade returns for each ticker.
    """
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

    # Add 'StrategyReturn' column to each DataFrame for downstream analysis (e.g., CAPM)
    for ticker in tickers:
        df = data[ticker]
        momentum_returns = np.array(backtest_results[ticker])
        df['StrategyReturn'] = 0.0
        i = 0
        trade_idx = 0
        while i < len(df) - h and trade_idx < len(momentum_returns):
            if df['Signal'].iloc[i] == 1:
                df.iloc[i+1, df.columns.get_loc('StrategyReturn')] = momentum_returns[trade_idx]
                trade_idx += 1
                i += h
            else:
                i += 1
        data[ticker] = df

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

if __name__ == "__main__":
    # Example usage for AAPL
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    df = yf.download('AAPL', start=config['start_date'], end=config['end_date'])
    df = df[['Close']]
    k = config['momentum']['k']
    x = config['momentum']['x']
    h = config['holding_period']
    train_window = config['train_window']
    test_window = config['test_window']
    wf_returns = run_walkforward_backtest(df, k, x, h, train_window, test_window)
    print(f"Walk-forward backtest: {len(wf_returns)} trades")
    print(f"Sharpe ratio: {sharpe_ratio(wf_returns):.3f}")
    print(f"Max drawdown: {max_drawdown(np.cumprod(1+wf_returns)):.3%}")
    print(f"t-statistic: {t_statistic(wf_returns):.3f}") 