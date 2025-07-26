import argparse
import yaml
from backtest import run_backtest_and_plot
from capm import run_capm_analysis
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Momentum Backtest and CAPM Analysis")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run backtest and plot
    data, backtest_results = run_backtest_and_plot(config)

    # CAPM for AAPL as example
    strategy_returns = data['AAPL'].loc[data['AAPL']['StrategyReturn'] != 0, 'StrategyReturn']
    strategy_returns.name = 'Strategy'
    run_capm_analysis(strategy_returns, 'SPY', 'AAPL Momentum Strategy')

    # Walk-forward backtest for AAPL
    from backtest import run_walkforward_backtest, sharpe_ratio, max_drawdown, t_statistic
    df = data['AAPL']
    k = config['momentum']['k']
    x = config['momentum']['x']
    h = config['holding_period']
    train_window = config['train_window']
    test_window = config['test_window']
    wf_returns = run_walkforward_backtest(df, k, x, h, train_window, test_window, verbose=True)
    print(f"\nWalk-forward backtest (AAPL): {len(wf_returns)} trades")
    if len(wf_returns) == 0:
        print("No trades generated in walk-forward test windows. Adjust your parameters or window sizes.")
    else:
        print(f"Sharpe ratio: {sharpe_ratio(wf_returns):.3f}")
        print(f"Max drawdown: {max_drawdown(np.cumprod(1+wf_returns)):.3%}")
        print(f"t-statistic: {t_statistic(wf_returns):.3f}") 