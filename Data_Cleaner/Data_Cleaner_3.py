import yfinance as yf
import numpy as np
import pandas as pd

INTERVAL = '1m'
TICKER = 'TSLA'

Z_THRESHOLD = 3
ITERATIONS = 1000
ITERATIONS2 = 50
ERROR = (0.01, 0.02, 0.03, 0.04, 0.05, 1.95, 1.96, 1.97, 1.98, 1.99)
WINDOW = range(1, 50, 1)
def get_data(ticker=TICKER, interval=INTERVAL):
    df = yf.download(tickers=ticker, interval=interval, period='max')
    df.columns = df.columns.get_level_values(0)
    return df

def infuse_error(df, error=ERROR):
    # We seperate first 100 rows to make sure they remain 'clean'. This will make calculations easier
    # and more realistic, we have some previous data that we can use for calculations
    df_start = df.iloc[:100].copy()
    df_rest = df.iloc[100:].copy()

    # Infusing our minutely data for TSLA with 'bad ticks'
    random_indices = np.random.choice(df_rest.index, 50)
    df_rest.loc[random_indices, 'Close'] = df_rest.loc[random_indices, 'Close'] * np.random.choice(error)
    df_dirty_data = pd.concat([df_start, df_rest])
    return df.dropna(), df_dirty_data.dropna()

def cleaning_data(df_dirty_data, window, z_threshold=Z_THRESHOLD):
    # I use Z-score to detect outliers and clean the data
    df_dirty_data['Price Change'] = df_dirty_data['Close'].pct_change()
    mu = df_dirty_data['Price Change'].rolling(window).mean()
    sigma = df_dirty_data['Price Change'].rolling(window).std()
    zscore = (df_dirty_data['Price Change'] - mu) / sigma
    df_dirty_data['Error'] = np.where(zscore.abs() > z_threshold, 1, 0)
    df_clean_data = df_dirty_data[df_dirty_data['Error'] == 0].copy()
    return df_dirty_data.dropna(), df_clean_data

def evaluate_model_performance(df, df_clean_data, df_dirty_data):
    # I created a Volume-Weighted Average Price Indicator to determine fair value of TSLA
    # ---Original Data---
    df['Volume x Price'] = df['Close'] * df['Volume']
    df['VxP Sum'] = df.groupby(df.index.normalize())['Volume x Price'].transform('sum')
    df['Volume Sum'] = df.groupby(df.index.normalize())['Volume'].transform('sum')
    df['Fair Value'] = df['VxP Sum'] / df['Volume Sum']
    org_vwap = df['Fair Value'].mean()

    # ---Dirty Data---
    df_dirty_data['Volume'] = df_dirty_data['Volume'].replace(0, np.nan).ffill()
    df_dirty_data['Volume x Price'] = df_dirty_data['Close'] * df_dirty_data['Volume']
    df_dirty_data['VxP Sum'] = df_dirty_data.groupby(df_dirty_data.index.normalize())['Volume x Price'].transform('sum')
    df_dirty_data['Volume Sum'] = df_dirty_data.groupby(df_dirty_data.index.normalize())['Volume'].transform('sum')
    df_dirty_data['Fair Value'] = df_dirty_data['VxP Sum'] / df_dirty_data['Volume Sum']
    dirty_vwap = df_dirty_data['Fair Value'].mean()

    # ---Clean Data---
    df_clean_data['Volume'] = df_clean_data['Volume'].replace(0, np.nan).ffill()
    df_clean_data['Volume x Price'] = df_clean_data['Close'] * df_clean_data['Volume']
    df_clean_data['VxP Sum'] = df_clean_data.groupby(df_clean_data.index.normalize())['Volume x Price'].transform('sum')
    df_clean_data['Volume Sum'] = df_clean_data.groupby(df_clean_data.index.normalize())['Volume'].transform('sum')
    df_clean_data['Fair Value'] = df_clean_data['VxP Sum'] / df_clean_data['Volume Sum']
    clean_vwap = df_clean_data['Fair Value'].mean()

    # How much does the model improve our data?
    error_dirty = abs((org_vwap - dirty_vwap) / org_vwap) * 100
    error_clean = abs((org_vwap - clean_vwap) / org_vwap) * 100
    improvement = error_dirty - error_clean
    return df, df_clean_data, df_dirty_data, error_clean, error_dirty, improvement


def main(n_iterations=ITERATIONS, n_iterations2=ITERATIONS2, window_options=WINDOW):
    # 1. Store the base dataframe and keep it PRISTINE.
    # Do not overwrite base_df inside the loops!
    base_df = get_data()

    results = {
        'dirty_errors': [],
        'clean_errors': [],
        'improvements': []
    }

    print(f'Running search over {len(window_options)} windows...')

    best_window = None
    # 2. We want the HIGHEST improvement, so we start at negative infinity
    best_improvement = float('-inf')

    # --- PHASE 1: GRID SEARCH ---
    for current_window in window_options:

        # Use copies so we don't contaminate the original dataset
        temp_df, temp_dirty_data = infuse_error(base_df.copy())
        temp_dirty_data, temp_clean_data = cleaning_data(temp_dirty_data, current_window)

        # Unpack the 6 returned values. We only care about the 6th one (improvement)
        _, _, _, _, _, improvement = evaluate_model_performance(temp_df, temp_clean_data, temp_dirty_data)

        # 3. Look for the MAXIMUM improvement
        if improvement > best_improvement:
            best_window = current_window
            best_improvement = improvement

    print(f'Winner found! Best Window: {best_window} (Improvement: {best_improvement:.4f}%)')

    # --- PHASE 2: VALIDATION ---
    print(f'\nRunning validation for {n_iterations} iterations with window {best_window}...')

    for n in range(n_iterations):
        # Again, use .copy() so every iteration starts fresh
        temp_df, temp_dirty_data = infuse_error(base_df.copy())
        temp_dirty_data, temp_clean_data = cleaning_data(temp_dirty_data, best_window)

        # Catch all the metrics we want to track
        _, _, _, error_clean, error_dirty, improvement = evaluate_model_performance(temp_df, temp_clean_data,
                                                                                    temp_dirty_data)

        results['dirty_errors'].append(error_dirty)
        results['clean_errors'].append(error_clean)
        results['improvements'].append(improvement)

        if (n + 1) % 100 == 0:
            print(f"Completed {n + 1}/{n_iterations} runs...")

    # --- RESULTS ---
    avg_dirty_error = np.mean(results['dirty_errors'])
    avg_clean_error = np.mean(results['clean_errors'])
    avg_improvement = np.mean(results['improvements'])

    print(f'\n--- RESULTS OVER {n_iterations} ITERATIONS ---')
    print(f'Average Dirty Data Error: {avg_dirty_error:.4f}%')
    print(f'Average Clean Data Error: {avg_clean_error:.4f}%')
    print(f'Average Improvement:      {avg_improvement:.4f} percentage points recovered')
    print(f'Worst Case Improvement:   {np.min(results["improvements"]):.4f}%')
    print(f'Best Case Improvement:    {np.max(results["improvements"]):.4f}%')

    # Returning the final iteration's data frames just in case you need to inspect them
    return temp_df, temp_clean_data, temp_dirty_data

main()