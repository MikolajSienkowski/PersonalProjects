import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Monday_Open_Prediction_Model import main

LEVERAGE = 1

def add_strategy(df_org, next_prediction, leverage=LEVERAGE):
    # Assign the Prediction column for easier use and slice the DataFrame accordingly to next_prediciton variable
    df = df_org
    start_date = next_prediction.index[0]
    df = df.loc[start_date:].copy()
    df['Prediction'] = next_prediction * 100
    df['Prediction'] = df['Prediction'].ffill()

    # Trade against the Gap Theory logic - the gap starts a weekly trend
    df['Strategy'] = np.where(df['Prediction'] > 0.25, leverage, np.where(df['Prediction'] < -0.25, -leverage, 0))
    df['Strategy'] = df['Strategy'].shift(1)

    return df.dropna(), start_date

def test_strategy(df):
    # Make a Daily Return column that assumes we hold the trade from Monday Open to Friday Close
    df['Daily Return'] = df['Close'].pct_change()
    monday_filter = df.index.dayofweek == 0
    df.loc[monday_filter, 'Daily Return'] = (df['Close'] - df['Open']) / df['Open']

    df['Strategy Returns'] = (1 + df['Daily Return'] * df['Strategy']).cumprod()
    df['Benchmark Returns'] = (1 + df['Close'].pct_change()).cumprod()

    plt.plot(df['Strategy Returns'])
    plt.plot(df['Benchmark Returns'])
    plt.title('Gap Theory Trading vs. Benchmark (SPY)')
    plt.legend(['Strategy Returns', 'Benchmark Returns'])
    plt.show()

    return df, monday_filter

def evaluate_performance(df, monday_filter, start_date):
    # Total Returns
    s_ret = df['Strategy Returns'].iloc[-1] - 1
    b_ret = df['Benchmark Returns'].iloc[-1] - 1

    # Max Drawdown
    s_running_max = df['Strategy Returns'].cummax()
    s_max_drawdown = ((df['Strategy Returns'] / s_running_max) - 1).min()
    b_running_max = df['Benchmark Returns'].cummax()
    b_max_drawdown = ((df['Benchmark Returns'] / b_running_max) - 1).min()

    # Win Rate
    df['Weekly Returns'] = (df['Close'].shift(-4) - df['Open']) / df['Open']
    df['Trade Returns'] = df['Weekly Returns'] * df['Strategy']
    trades = df[monday_filter & (df['Strategy'] != 0)]
    benchmark = df[monday_filter]

    n_trades = len(trades)
    s_w = len(trades[trades['Trade Returns'] > 0])
    s_wr = s_w / n_trades if n_trades > 0 else 0

    n_weeks = int(len(df) / 5)
    b_w = len(benchmark[benchmark['Weekly Returns'] > 0])
    b_wr = b_w / n_weeks if n_weeks > 0 else 0

    # Expected Value
    s_wEV = trades[trades['Trade Returns'] > 0]['Trade Returns'].mean()
    s_lEV = trades[trades['Trade Returns'] < 0]['Trade Returns'].mean()
    s_wEV = 0 if pd.isna(s_wEV) else s_wEV
    s_lEV = 0 if pd.isna(s_lEV) else s_lEV
    s_EV = s_wEV * s_wr + s_lEV * (1 - s_wr)

    b_wEV = benchmark[benchmark['Weekly Returns'] > 0]['Weekly Returns'].mean()
    b_lEV = benchmark[benchmark['Weekly Returns'] < 0]['Weekly Returns'].mean()
    b_wEV = 0 if pd.isna(b_wEV) else b_wEV
    b_lEV = 0 if pd.isna(b_lEV) else b_lEV
    b_EV = b_wEV * b_wr + b_lEV * (1 - b_wr)

    metrics_data = {
        'Metric': [
            'Total Returns',
            'Max Drawdown',
            'Win Rate',
            'N Trades',
            'Weeks of backtest',
            'Expected Value'
    ],
        'Strategy': [
            f'{s_ret:.2%}',
            f'{s_max_drawdown:.2%}',
            f'{s_wr:.2%}',
            f'{n_trades}',
            f'{n_weeks}',
            f'{s_EV:.2%}',
        ],
        'Benchmark': [
            f'{b_ret:.2%}',
            f'{b_max_drawdown:.2%}',
            f'{b_wr:.2%}',
            f'{n_weeks}',
            f'{n_weeks}',
            f'{b_EV:.2%}'
        ]
    }

    comparison_df = pd.DataFrame(metrics_data)
    comparison_df.set_index('Metric', inplace=True)

    print()
    print('-' * 45)
    print('             Strategy vs. Benchmark')
    print('-' * 45)
    print(comparison_df)
    print('-' * 45)

    return df

def main_gtt():
    next_prediction, df_org = main()
    df, start_date = add_strategy(df_org, next_prediction)
    df, monday_filter = test_strategy(df)
    df = evaluate_performance(df, monday_filter, start_date)

    return df

if __name__ == '__main__':
    main_gtt()