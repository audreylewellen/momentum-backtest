import argparse
import yaml
from backtest import run_backtest_and_plot
from capm import run_capm_analysis

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