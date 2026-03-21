import yfinance as yf
import numpy as np
import pandas as pd

INTERVAL = '1m'
TICKER = 'TSLA'

def get_data(ticker=TICKER, interval=INTERVAL):
    df = yf.download(tickers=ticker, interval=interval, period='max')
    df.columns = df.columns.get_level_values(0)

    # We seperate first 100 rows to make sure they remain 'clean'. This will make calculations easier
    # and more realistic, we have some previous data that we can use for calculations
    df_start = df.iloc[:100].copy()
    df_rest = df.iloc[100:].copy()

    # Infusing our minutely data for TSLA with 'bad ticks'
    random_indices = np.random.choice(df_rest.index, 50)
    df_rest.loc[random_indices, 'Close'] = df_rest.loc[random_indices, 'Close'] * 0.01
    df_dirty_data = pd.concat([df_start, df_rest])
    return df.dropna(), df_dirty_data.dropna()

def cleaning_data(df_dirty_data):
    # I use Z-score to detect outliers and clean the data
    df_dirty_data['Price Change'] = df_dirty_data['Close'].pct_change()
    mu = df_dirty_data['Price Change'].mean()
    sigma = df_dirty_data['Price Change'].std()
    zscore = (df_dirty_data['Price Change'] - mu) / sigma
    df_dirty_data['Error'] = np.where(zscore.abs() > 3, 1, 0)
    print(f'--- Outlier Detection Report ---')
    print(f'Total Outliers Detected: {df_dirty_data['Error'].sum()}')
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
    print(f'The Fair Value of the original data: {org_vwap}')

    # ---Dirty Data---
    df_dirty_data['Volume'] = df_dirty_data['Volume'].replace(0, np.nan).ffill()
    df_dirty_data['Volume x Price'] = df_dirty_data['Close'] * df_dirty_data['Volume']
    df_dirty_data['VxP Sum'] = df_dirty_data.groupby(df_dirty_data.index.normalize())['Volume x Price'].transform('sum')
    df_dirty_data['Volume Sum'] = df_dirty_data.groupby(df_dirty_data.index.normalize())['Volume'].transform('sum')
    df_dirty_data['Fair Value'] = df_dirty_data['VxP Sum'] / df_dirty_data['Volume Sum']
    dirty_vwap = df_dirty_data['Fair Value'].mean()
    print(f'The Fair Value of the dirty data: {dirty_vwap}')

    # ---Clean Data---
    df_clean_data['Volume'] = df_clean_data['Volume'].replace(0, np.nan).ffill()
    df_clean_data['Volume x Price'] = df_clean_data['Close'] * df_clean_data['Volume']
    df_clean_data['VxP Sum'] = df_clean_data.groupby(df_clean_data.index.normalize())['Volume x Price'].transform('sum')
    df_clean_data['Volume Sum'] = df_clean_data.groupby(df_clean_data.index.normalize())['Volume'].transform('sum')
    df_clean_data['Fair Value'] = df_clean_data['VxP Sum'] / df_clean_data['Volume Sum']
    clean_vwap = df_clean_data['Fair Value'].mean()
    print(f'The Fair Value of the clean data: {clean_vwap}')

    # How much does the model improve our data?
    error_dirty = abs((org_vwap - dirty_vwap) / org_vwap) * 100
    error_clean = abs((org_vwap - clean_vwap) / org_vwap) * 100
    improvement = error_dirty - error_clean
    print('\n--- RESULTS ---')
    print(f'Original VWAP (Truth): {org_vwap:.4f}')
    print(f'Dirty VWAP:            {dirty_vwap:.4f} (Error: {error_dirty:.2f}%)')
    print(f'Cleaned VWAP:          {clean_vwap:.4f} (Error: {error_clean:.2f}%)')
    print(f'Improvement:           {improvement:.2f}% percentage points recovered.')
    return df, df_clean_data, df_dirty_data

def main():
    df, df_dirty_data = get_data()
    df_dirty_data, df_clean_data = cleaning_data(df_dirty_data)
    df, df_clean_data, df_dirty_data = evaluate_model_performance(df, df_clean_data, df_dirty_data)
    return df, df_clean_data, df_dirty_data

main()
