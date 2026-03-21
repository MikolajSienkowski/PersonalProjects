import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

TICKER = 'SPY'

def get_data(ticker=TICKER):
    start = dt.datetime(2024, 1, 1)
    end = dt.datetime(2026, 3, 13)

    df = yf.download(tickers=ticker, start=start, end=end, interval='1d', auto_adjust=True)
    btc = yf.download(tickers='BTC-USD', start=start, end=end, interval='1d', auto_adjust=True)

    df.columns = df.columns.get_level_values(0)
    btc.columns = btc.columns.get_level_values(0)

    return df.dropna(), btc.dropna()

def add_feature_and_target(df, btc):
    df['Target'] = (df['Open'].shift(-1) - df['Close']) / df['Close']
    btc['Feature'] = (btc['Close'].shift(-2) - btc['Close']) / btc['Close']

    df = df[df.index.dayofweek == 4].copy()
    btc = btc[btc.index.dayofweek == 4].copy()

    df['Mean of Target'] = df['Target'].mean()

    df = pd.concat([df, btc], axis=1).dropna()

    return df

def test_hypothesis(df):
    y = df['Target']
    y_guess = df[['Mean of Target']]
    X = df[['Feature']]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    print(model.summary())

    return X, y, y_guess

def train_and_test_model(X, y, y_guess):
    split = int(len(X) * 0.75)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test, y_guess_test = X[split:], y[split:], y_guess[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mse_guess = mean_squared_error(y_test, y_guess_test)
    r2 = r2_score(y_test, predictions)
    next_prediction = pd.Series(predictions, index=X_test.index)

    print('')
    print(f'--- MODEL REPORT ---')
    print(f'Training R²:        {model.score(X_train, y_train):.4f}')
    print(f'Testing R²:         {r2:.4f}')
    print(f'RMSE:               {np.sqrt(mse):.8f} (Average error)')
    print(f'Guessing (RMSE):    {np.sqrt(mse_guess):.8f} (Average error)')
    print(f'Next Prediction:    {next_prediction.iloc[-1]:.4f}')
    print('-' * 30)
    print('*guessing is described as assuming that the gap on Monday open')
    print('would be the size of an average gap from the period')

    return

def main():
    df, btc = get_data()
    df = add_feature_and_target(df, btc)
    X, y, y_guess = test_hypothesis(df)
    train_and_test_model(X, y, y_guess)

    return

if __name__ == '__main__':
    main()

# gap theory