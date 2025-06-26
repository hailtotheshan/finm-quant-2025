import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

def get_hk_stock_daily_returns(hk_stocks, start_date, end_date):
    """
    Fetches daily adjusted-close for HK tickers,
    backfills/forwardfills missing prices, then returns pct-change.
    """
    prices = yf.download(
        tickers=hk_stocks,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False
    )["Close"]

    prices = prices.ffill(axis=0).bfill(axis=0)

    returns = prices.pct_change().dropna(how="all")
    returns = returns.ffill(axis=0).bfill(axis=0)

    return returns

def tangency_portfolio(returns, risk_free_rate=0.02):
    mean_returns = returns.mean() * 252  # annualized
    cov_matrix = returns.cov() * 252     # annualized

    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # No bounds ==> allow shorting (weights can be negative)
    bounds = None

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (port_return - risk_free_rate) / port_vol

    # Initial guess: equal weights
    init_guess = np.array([1. / num_assets] * num_assets)

    result = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

def main():
    hk_stocks = [
        '6862.HK', '2015.HK', '3690.HK', '3988.HK', '0388.HK', '1398.HK',
        '0941.HK', '1211.HK', '1299.HK', '9992.HK', '1357.HK', '0005.HK',
        '2331.HK', '2276.HK', '1810.HK', '9626.HK', '9633.HK', '9988.HK',
        '0700.HK', '0133.HK', '0806.HK', '1788.HK', '3037.HK', '1375.HK'
    ]
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    print("Downloading data...")
    etf_data = get_hk_stock_daily_returns(hk_stocks, start_date, end_date)

    risk_free_rate = 0.02

    print("Calculating tangency portfolio (with short selling allowed)...")
    weights = tangency_portfolio(etf_data, risk_free_rate)

    mean_returns = etf_data.mean() * 252
    cov_matrix = etf_data.cov() * 252

    expected_return = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    print("\nTangency Portfolio (Maximum Sharpe Ratio) Results:")
    print("---------------------------------------------------")
    print("Stock        Weight")
    print("-------------------")
    for stock, weight in zip(hk_stocks, weights):
        if abs(weight) > 1e-4:
            print(f"{stock:10s}  {weight:.4f}")
    print("-------------------")
    print(f"Expected annual return: {expected_return:.4%}")
    print(f"Annualized volatility:  {volatility:.4%}")
    print(f"Sharpe Ratio:           {sharpe_ratio:.4f}")

if __name__ == "__main__":
    main()