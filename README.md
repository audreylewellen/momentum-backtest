# Backtesting a Simple Momentum-Based Equity Strategy

## Objective

Backtest and analyze a simple momentum-based equity trading strategy on historical U.S. stock data, using walk-forward validation and performance metrics.

## Tools 
- Python 3
- pandas for data manipulation
- yfinance for historical stock data
- matplotlib for visualizations
- statsmodels for CAPM regression
- PyYAML for config file parsing

## Strategy 
Momentum strategy: 

“If return over past `k` trading days exceeds threshold `x`, buy and hold for `h` days.”

## Components

The workflow includes:

- Downloading historical price data for selected tickers using yfinance.
- Defining a momentum-based trading signal: if a stock's return over the past k days exceeds a threshold x, buy and hold for h days.
- Simulating trades and tracking returns, with logic to avoid overlapping trades.
- Analyzing results with summary statistics (average return per trade, win rate, etc.), and plotting cumulative returns vs. buy-and-hold.
- Walk-forward backtesting: The strategy is evaluated using rolling train/test windows (configurable via `train_window` and `test_window` in `config.yaml`), to better simulate out-of-sample performance.
- Risk and statistical metrics: For the walk-forward returns, the script computes Sharpe ratio, max drawdown, and t-statistic.
- CAPM regression analysis to decompose strategy returns into alpha and beta relative to the market. 

## Usage

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Edit `config.yaml` to set your tickers, date range, strategy parameters, and walk-forward window sizes. 

3. Run the backtest and analysis:
   ```bash
   python main.py --config config.yaml
   ```

4. The script will print:
   - Backtest summary stats
   - CAPM regression results
   - Walk-forward backtest results (number of trades, Sharpe, drawdown, t-stat)

---



