import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def run_capm_analysis(strategy_returns, market_ticker, strategy_name):
    """
    Run CAPM regression and plot for a given strategy return series and market ticker.
    strategy_returns: pd.Series of strategy returns (indexed by date)
    market_ticker: str, e.g. 'SPY'
    strategy_name: str, for plot/report labeling
    """
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