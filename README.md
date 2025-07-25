# momentum-backtest 

## Objective
Implement and evaluate a basic momentum trading strategy on historical U.S. stock data, comparing its performance to a buy-and-hold baseline. Analyze return behavior using Capital Asset Pricing Model (CAPM) regression to estimate alpha and beta.

## Tools 
- Python 3
- pandas for data manipulation
- yfinance for historical stock data
- matplotlib for visualizations
- statsmodels for CAPM regression

## Strategy 
Momentum strategy: 

“If return over past `k` trading days exceeds threshold `x`, buy and hold for `h` days.”

## Usage

1. **Install requirements**
  ```bash
  pip install -r requirements.txt
  ```

2. **Run the backtest**
   
  Run the main script:
  ```bash
  python momentum_backtest.py
  ```

3. **View Summary Statistics**
   
  For each ticker (e.g., AAPL, MSFT), the script outputs:
  - Number of trades
  - Average return per trade
  - Win rate

4. **View Performance Plots**

  - Cumulative return curves for the **momentum strategy** vs. **buy-and-hold**.
  - Separate plots for each stock.

5. **CAPM Regression (AAPL Example)**
   
  The script performs a CAPM analysis for AAPL strategy returns:
  - Calculates **alpha**, **beta**, **p-values**, and **R²**
  - Visualizes the fit of strategy returns vs. market (SPY)

## Parameter Adjustments 
To adjust the backtest:
- Open `momentum_backtest.py`
- Edit these variables:
  ```python
  tickers = ['AAPL', 'MSFT', 'SPY']
  k = 5       # lookback period
  x = 0.02    # threshold
  h = 5       # holding period
  ```



