import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from True_Range_Prediction_Model import main

TICKER = 'SPY'
CAPITAL = 10000
TARGET_RISK_PCT = 0.01
RISK_FREE_RATE = 0.04
LEVERAGE = 1

def portfolio_dynamic_sizing(df, next_prediction, split, capital=CAPITAL, target_risk_pct=TARGET_RISK_PCT):
    df = df.iloc[split:].copy()
    target_dollar_risk = capital * target_risk_pct
    df['Target Capital'] = target_dollar_risk / next_prediction
    df['Target Capital'] = df['Target Capital'].clip(upper=CAPITAL)
    df['Target Shares'] = df['Target Capital'] / df['Close']

    print("--- POSITION SIZING FOR TOMORROW ---")
    print(f"Predicted True Range: {next_prediction.iloc[-1]:.4f}")
    print(f"Capital Allocated:    ${df['Target Capital'].iloc[-1]:.2f}")
    print(f"Shares to Buy:        {df['Target Shares'].iloc[-1]:.2f}")

    return df

def test_strategy(df, capital=CAPITAL, leverage=LEVERAGE):
    df['Close Change'] = df['Close'].pct_change()
    df['Weight'] = df['Target Capital'] / capital
    df['Weight'] = df['Weight'].shift(1)
    df['Weighted Position'] = df['Weight'] * df['Close Change'] * leverage
    df['Strategy Returns'] = capital * (1 + df['Weighted Position']).cumprod()
    df['Benchmark Returns'] = capital * (1 + df['Close Change']).cumprod()

    plt.plot(df['Benchmark Returns'])
    plt.plot(df['Strategy Returns'])
    plt.legend(['Benchmark Returns', 'Strategy Returns'])
    plt.ylabel('Growth of $10,000')
    plt.title('SPY vs. Dynamic Sizing Portfolio')
    plt.show()

    return df

def evaluate_strategy(df, risk_free_rate=RISK_FREE_RATE):
    # Sharpe Ratio
    annualized_portfolio_returns = df['Weighted Position'].mean() * 252
    p_std = df['Weighted Position'].std() * np.sqrt(252)
    p_sharpe_ratio = (annualized_portfolio_returns - risk_free_rate) / p_std
    annualized_benchmark_returns = df['Close Change'].mean() * 252
    b_std = df['Close Change'].std() * np.sqrt(252)
    b_sharpe_ratio = (annualized_benchmark_returns - risk_free_rate) / b_std

    # Sortino Ratio
    p_downside_variation = df['Weighted Position'].loc[df['Weighted Position'] < 0]
    p_downside_std = p_downside_variation.std() * np.sqrt(252)
    p_sortino_ratio = (annualized_portfolio_returns - risk_free_rate) / p_downside_std
    b_downside_variation = df['Close Change'].loc[df['Close Change'] < 0]
    b_downside_std = b_downside_variation.std() * np.sqrt(252)
    b_sortino_ratio = (annualized_benchmark_returns - risk_free_rate) / b_downside_std

    # Max Drawdown
    p_running_max = df['Strategy Returns'].cummax()
    p_max_drawdown = ((df['Strategy Returns'] / p_running_max) - 1).min()
    b_running_max = df['Benchmark Returns'].cummax()
    b_max_drawdown = ((df['Benchmark Returns'] / b_running_max) - 1).min()

    # Calmar Ratio
    p_calmar_ratio = annualized_portfolio_returns / abs(p_max_drawdown)
    b_calmar_ratio = annualized_benchmark_returns / abs(b_max_drawdown)

    # Beta
    portfolio_returns = df['Weighted Position'].dropna()
    benchmark_returns = df['Close Change'].dropna()

    common_index = benchmark_returns.index.intersection(portfolio_returns.index)
    benchmark_returns = benchmark_returns.loc[common_index]
    portfolio_returns = portfolio_returns.loc[common_index]

    covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
    covariance = covariance_matrix[0, 1]
    market_variance = covariance_matrix[1, 1]

    beta = covariance / market_variance

    # Alpha
    alpha = annualized_portfolio_returns - (risk_free_rate + beta * (annualized_benchmark_returns - risk_free_rate))

    # Daily VaR
    daily_p_std = df['Weighted Position'].std()
    daily_b_std = df['Close Change'].std()

    p_var = 2.33 * daily_p_std
    b_var = 2.33 * daily_b_std

    # Results Table
    metrics_data = {
        'Metric': [
            'Annualized Return',
            'Annualized Volatility',
            'Value at Risk (Daily)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown',
            'Calmar Ratio',
            'Beta',
            'Alpha'
        ],
        'DPS': [
            f'{annualized_portfolio_returns:.2%}',
            f'{p_std:.2%}',
            f'{p_var:.2%}',
            f'{p_sharpe_ratio:.2f}',
            f'{p_sortino_ratio:.2f}',
            f'{p_max_drawdown:.2%}',
            f'{p_calmar_ratio:.2f}',
            f'{beta:.2f}',
            f'{alpha:.2%}'
        ],
        '   Benchmark': [
            f'{annualized_benchmark_returns:.2%}',
            f'{b_std:.2%}',
            f'{b_var:.2%}',
            f'{b_sharpe_ratio:.2f}',
            f'{b_sortino_ratio:.2f}',
            f'{b_max_drawdown:.2%}',
            f'{b_calmar_ratio:.2f}',
            '1.00',  # Benchmark Beta is always 1
            '0.00%'  # Benchmark Alpha is always 0
        ]
    }

    comparison_df = pd.DataFrame(metrics_data)
    comparison_df.set_index('Metric', inplace=True)

    print('-' * 45)
    print('             DPS vs. Benchmark')
    print('-' * 45)
    print(comparison_df)
    print('-' * 45)

    return

def main_dps():
    df, next_prediction, split = main()
    df = portfolio_dynamic_sizing(df, next_prediction, split)
    df = test_strategy(df)
    evaluate_strategy(df)

    return df

if __name__ == '__main__':
    main_dps()