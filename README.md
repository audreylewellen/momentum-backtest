# momentum-backtest 

## Objective
Implement and evaluate a basic momentum trading strategy on historical U.S. stock data, comparing its performance to a buy-and-hold baseline. Analyze return behavior using Capital Asset Pricing Model (CAPM) regression to estimate alpha and beta.

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

## Usage

1. **Install requirements**
  ```bash
  pip install -r requirements.txt
  ```

2. **Configure your backtest**

   Edit `config.yaml` to set your tickers, date range, and strategy parameters:

   ```yaml
   tickers:
     - AAPL
     - MSFT
     - SPY
   start_date: '2018-01-01'
   end_date: '2024-01-01'
   momentum:
     k: 5
     x: 0.02
   holding_period: 5
   ```

3. **Run the backtest and CAPM analysis**

   ```bash
   python main.py --config config.yaml
   ```

   - This will run the momentum backtest, plot results, and perform CAPM alpha & beta analysis for the AAPL strategy.
   - Note: After the first plot window (showing the backtest results) appears, you need to close it to see the CAPM regression plot and output.



