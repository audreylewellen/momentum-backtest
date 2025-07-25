import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# List of tickers to download
tickers = ['AAPL', 'MSFT', 'SPY']
start_date = '2018-01-01'
end_date = '2024-01-01'

# Download data
data = {}
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']]
    df['Return'] = df['Close'].pct_change()
    data[ticker] = df

# Parameters for momentum strategy
k = 5  # lookback period (in days)
x = 0.02  # % threshold

for ticker, df in data.items():
    df['Momentum'] = df['Close'].pct_change(k)
    df['Signal'] = (df['Momentum'] > x).astype(int)
    data[ticker] = df

# Example: print the first few rows for AAPL with new columns
print('AAPL data with momentum and signal:')
print(data['AAPL'].head(10))

# Holding period (in days)
h = 5  

backtest_results = {}

# Prepare to backtest the momentum strategy for each ticker
for ticker, df in data.items():
    returns = []
    i = 0
    while i < len(df) - h:
        if df['Signal'].iloc[i] == 1:
            entry = df['Close'].iloc[i+1]
            exit_ = df['Close'].iloc[i+1+h-1]
            ret = (exit_ - entry) / entry
            returns.append(ret)

            # Skip ahead so no overlap
            i += h  
        else:
            i += 1
    backtest_results[ticker] = returns

# Example: print summary stats for AAPL
import numpy as np
aapl_returns = backtest_results['AAPL']
if aapl_returns:
    print(f"AAPL: {len(aapl_returns)} trades")
    print(f"Average return per trade: {np.mean(aapl_returns):.4f}")
    print(f"Win rate: {np.mean(np.array(aapl_returns) > 0) * 100:.2f}%")
else:
    print("No trades for AAPL.") 

# Plot cumulative returns for all stocks
fig, axes = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

# Loop through each ticker and plot its results
for idx, ticker in enumerate(['AAPL', 'MSFT', 'SPY']):
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
        ax.set_title(f'{ticker}: Cumulative Return (2018-2024)')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
    else:
        # If no trades were made, let user know in the plot
        axes[idx].set_title(f'{ticker}: No trades, so no strategy plot.')
        axes[idx].set_ylabel('Cumulative Return')

plt.xlabel('Date')
plt.tight_layout()
plt.show() 

def run_capm_analysis(strategy_returns, market_ticker, strategy_name):
    """
    Run CAPM regression and plot for a given strategy return series and market ticker.
    strategy_returns: pd.Series of strategy returns (indexed by date)
    market_ticker: str, e.g. 'SPY'
    strategy_name: str, for plot/report labeling
    """
    import yfinance as yf
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    if strategy_returns.empty:
        print(f"No strategy returns available for CAPM analysis: {strategy_name}")
        return
    # Download market data for the same period
    spy = yf.download(market_ticker, start=strategy_returns.index.min(), end=strategy_returns.index.max())
    spy['Market_Return'] = spy['Close'].pct_change()
    # Align dates and drop missing values
    returns_df = pd.DataFrame({
        'Strategy': strategy_returns,
        'Market': spy['Market_Return']
    }).dropna()
    if returns_df.empty:
        print(f"No overlapping dates for CAPM analysis: {strategy_name}")
        return
    # CAPM regression
    X = sm.add_constant(returns_df['Market'])
    y = returns_df['Strategy']
    capm_model = sm.OLS(y, X).fit()
    print(f"\nCAPM Regression Results ({strategy_name}):")
    print(capm_model.summary())
    # Plot actual vs. predicted
    returns_df['Predicted'] = capm_model.predict(X)
    plt.figure(figsize=(8,6))
    plt.scatter(returns_df['Market'], returns_df['Strategy'], alpha=0.5, label="Actual")
    plt.plot(returns_df['Market'], returns_df['Predicted'], color='red', label="CAPM Fit")
    plt.xlabel(f"Market Return ({market_ticker})")
    plt.ylabel(f"Strategy Return ({strategy_name})")
    plt.legend()
    plt.title(f"CAPM Alpha & Beta Regression: {strategy_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- CAPM Alpha & Beta Analysis for AAPL Momentum Strategy ---
# Step 0: Extract daily strategy returns as a pandas Series (aligned to trade entry days)
aapl_df = data['AAPL'].copy()
strategy_returns = aapl_df.loc[aapl_df['StrategyReturn'] != 0, 'StrategyReturn']
strategy_returns.name = 'Strategy'
run_capm_analysis(strategy_returns, 'SPY', 'AAPL Momentum Strategy') 