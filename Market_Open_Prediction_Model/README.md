# Monday Open Prediction Model: Cross-Asset Momentum

## Overview
This quantitative research project explores cross-asset momentum by predicting the S&P 500 (SPY) Monday morning opening gap using Bitcoin (BTC) weekend price action. 

Because traditional equity markets are closed over the weekend, macroeconomic news and shifts in global sentiment often have no outlet for price discovery until Monday morning. By using a 24/7 crypto asset (BTC) as a proxy for weekend sentiment, this model successfully identifies a statistically significant leading indicator for traditional equity market gaps.

## Financial Logic & Methodology
1. **The Target:** The percentage gap between Friday's SPY Close and Monday's SPY Open.
2. **The Feature:** The percentage change in BTC over the weekend (aligned to traditional market hours).
3. **Data Alignment (Avoiding Look-Ahead Bias):** Vectorized index shifting ensures strict separation of timeframes. SPY data is shifted forward by 1 day, and BTC data is shifted forward by 2 days before filtering for Fridays (`dayofweek == 4`), guaranteeing that the model only uses data mathematically available *before* the Monday open.
4. **The Model:** An Ordinary Least Squares (OLS) Regression (`statsmodels`) and a standard Linear Regression (`scikit-learn`) trained on data from January 2024 to March 2026.

## Statistical Results & Edge (N = 111 weeks)
The model identified a highly significant linear relationship, proving that weekend crypto flows contain forward-looking information about traditional equities.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **R-Squared** | 0.202 | The model explains ~20% of the variance in SPY Monday gaps (exceptionally high for a single-feature financial model). |
| **P-Value** | < 0.001 | The relationship is statistically significant and not the result of random chance. |
| **Coefficient (Beta)** | 0.1469 | For every **1.00%** move in BTC over the weekend, SPY is expected to gap by **0.15%** in the same direction on Monday. |
| **RMSE (Model)** | 0.0057 | The model's average prediction error. |
| **RMSE (Baseline)**| 0.0064 | The error rate if simply guessing the historical average gap. The model successfully beats this baseline. |

*Conclusion:* In the modern post-ETF market environment where institutional capital bridges crypto and traditional equities, weekend BTC momentum is a viable predictive signal for SPY Monday open trajectories.

# Implementing the model into a trading strategy

## The Logic
* Assume that a Monday Open Gap dictates the trend for the upcoming week.
* Use the Model's predictions to define our weekly position.
* A  predicted gap larger than 0.25% triggers a long position, a gap smaller than -0.25% triggers a short position, and every prediction within that threshold results in no market exposure (staying in cash).

## The Out-Of-Sample Results (26 weeks)
| Metric               | Gap Trading (Model) | Buy & Hold SPY (Benchmark) |
|:---------------------|:--------------------|:---------------------------|
| **Total Returns**    | **5.77%**           | 6.24%                      |
| **Max Drawdown**     | **-3.88%**          | -5.07%                     |
| **Win Rate**         | **60%**             | 50%                        |
| **Number of Trades** | **10**              | 26                         |
| **Expected Value**   | **0.64%**           | -0.08%                     |

*Conclusion:* Although the backtest is conducted over a short period of time, the strategy captures roughly 92% of returns while cutting down on the Max Drawdown by roughly 23%. By remaining out of the market 62% of the time, the strategy acts as a highly selective momentum engine with a proven 60% win rate and a strictly positive Expected Value per trade.

## Tech Stack
* **Python** (pandas, numpy)
* **statsmodels** (for linear regression and statistical inference)
* **scikit-learn** (for model training and evaluation metrics)
* **yfinance** (for historical SPY and BTC market data)

## How to Run
1. Clone the repository.
2. Install the required packages: `pip install pandas numpy statsmodels yfinance scikit-learn`
3. Run the model: `python Monday_Open_Prediction_Model.py`
4. Run the strategy: `python Gap_Theory_Trading.py`