import yfinance as yf
import numpy as np
import pandas as pd
from tqdm import tqdm

INTERVAL = '1m'
TICKER = 'TSLA'

Z_THRESHOLD = 3
ITERATIONS = 1000
ERROR = (0.01, 0.02, 0.03, 0.04, 0.05, 1.95, 1.96, 1.97, 1.98, 1.99)

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

def cleaning_data(df_dirty_data, z_threshold=Z_THRESHOLD):
    # I use Z-score to detect outliers and clean the data
    df_dirty_data['Price Change'] = df_dirty_data['Close'].pct_change()
    mu = df_dirty_data['Price Change'].rolling(20).mean()
    sigma = df_dirty_data['Price Change'].rolling(20).std()
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

def main(n_iterations=ITERATIONS):
    df = get_data()

    results = {
        'dirty_errors': [],
        'clean_errors': [],
        'improvements': []
    }

    for n in tqdm(range(n_iterations), desc='Simulations', unit='simulations'):
        df, df_dirty_data = infuse_error(df.copy())
        df_dirty_data, df_clean_data = cleaning_data(df_dirty_data)
        df, df_clean_data, df_dirty_data, error_clean, error_dirty, improvement = evaluate_model_performance(df, df_clean_data, df_dirty_data)
        results['dirty_errors'].append(error_dirty)
        results['clean_errors'].append(error_clean)
        results['improvements'].append(improvement)


    avg_dirty_error = np.mean(results['dirty_errors'])
    avg_clean_error = np.mean(results['clean_errors'])
    avg_improvement = np.mean(results['improvements'])

    print(f'\n--- RESULTS OVER {n_iterations} ITERATIONS ---')
    print(f'Average Dirty Data Error: {avg_dirty_error:.4f}%')
    print(f'Average Clean Data Error: {avg_clean_error:.4f}%')
    print(f'Average Improvement:      {avg_improvement:.4f} percentage points recovered')
    print(f'Worst Case Improvement:   {np.min(results["improvements"]):.4f}%')
    print(f'Best Case Improvement:    {np.max(results["improvements"]):.4f}%')
    return df, df_clean_data, df_dirty_data

main()
