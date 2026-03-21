# True Range Prediction Model & Dynamic Position Sizing

## Overview
This project is a quantitative trading model that forecasts the daily True Range (volatility) of the S&P 500 (SPY) and dynamically scales capital exposure to maintain a constant daily risk target. 

By scaling down positions during periods of forecasted high volatility and scaling up during quiet periods, the model successfully **cuts the Maximum Drawdown in half** and generates **1.50% annualized Alpha** compared to a standard Buy & Hold strategy.

## Financial Logic & Methodology
1. **The Target:** We define risk as the Daily True Range (capturing overnight gaps, unlike standard High-Low range).
2. **The Features:** * *Autoregressive Volatility:* A 5-day rolling mean of historical True Range (capturing "volatility clustering").
   * *Exogenous Signal:* A 4-day percentage change in the `^VIX` (incorporating forward-looking options market sentiment).
3. **The Model:** An Ordinary Least Squares (OLS) Regression (`statsmodels`) trained on data from 1994-2018.
4. **The Sizer:** A Volatility Targeting algorithm. Instead of a fixed share count, the portfolio risks exactly 1% of total capital per day. $$Shares = \frac{Target\_Dollar\_Risk}{Price \times Predicted\_True\_Range}$$

## Out-of-Sample Results (2018 - 2026)
The model was tested on unseen, out-of-sample data covering major volatility events (e.g., the 2018 Volmageddon, the 2020 COVID Crash, and the 2022 Bear Market).

| Metric | Dynamic Sizing (Model) | Buy & Hold SPY (Benchmark) |
| :--- | :--- | :--- |
| **Annualized Return** | 11.10% | 15.17% |
| **Annualized Volatility** | 11.03% | 19.21% |
| **Max Drawdown** | **-17.27%** | -33.72% |
| **Sharpe Ratio** | **0.64** | 0.58 |
| **Sortino Ratio** | **0.87** | 0.71 |
| **Beta** | 0.50 | 1.00 |
| **Alpha** | **+1.50%** | 0.00% |

![Strategy Performance vs Benchmark](images\performance_chart.png)

*Conclusion:* The strategy sacrifices roughly 25% of absolute return to eliminate nearly **50% of the portfolio's risk (volatility and drawdown)**, resulting in superior risk-adjusted returns (Sharpe/Sortino) and positive Alpha. 

## Tech Stack
* **Python** (pandas, numpy)
* **statsmodels** (for linear regression and statistical inference)
* **scikit-learn** (for training the model)
* **yfinance** (for historical SPY and VIX market data)
* **matplotlib** (for equity curve and underwater drawdown visualizations)

## How to Run
1. Clone the repository.
2. Install the required packages: `pip install pandas numpy statsmodels yfinance matplotlib scikit-learn`
3. Run the main pipeline: `python main.py`