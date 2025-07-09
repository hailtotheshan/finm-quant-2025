import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


def get_hk_stock_daily_returns(hk_stocks, start_date, end_date):
    """
    Fetches daily adjusted-close for HK tickers,
    backfills/forwardfills missing prices, then returns pct-change.
    """
    # 1) Grab raw daily adjusted-closes
    prices = yf.download(
        tickers=hk_stocks,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False
    )["Close"]

    # 2) Fill holes in the price series
    #    .bfill() backfills NaNs from future valid data,
    #    .ffill() then forward fills any leading NaNs.
    prices = prices.ffill(axis=0).bfill(axis=0)

    # 3) Compute daily pct-change and drop any leftover all-NaN rows
    returns = prices.pct_change().dropna(how="all")

    # 4) (Optional) If you really hate any NaNs, you can fill returns too:
    returns = returns.ffill(axis=0).bfill(axis=0)

    return returns

def tangency_portfolio(returns, risk_free_rate=0.02):
    mean_returns = returns.mean() * 252  # annualized
    cov_matrix = returns.cov() * 252     # annualized

    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds: weights between 0 and 1 (long-only)
    bounds = tuple((0, 1) for _ in range(num_assets))

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


def draw_heatmap(df, plot_title=""):
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    print("Correlation matrix:\n", correlation_matrix)

    # Plot the correlation matrix as a heat‐map
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(plot_title)
    plt.show()


def main():
    """hk_stocks = [
        '6862.HK', '2015.HK', '3690.HK', '3988.HK', '0388.HK', '1398.HK',
        '0941.HK', '1211.HK', '1299.HK', '9992.HK', '1357.HK', '0005.HK',
        '2331.HK', '2276.HK', '1810.HK', '9626.HK', '9633.HK', '9988.HK',
        '0700.HK', '0133.HK', '0806.HK', '1788.HK', '3037.HK', '1375.HK'
    ]"""
    hk_stocks = ['1788.HK', '0806.HK', '2015.HK']
    start_date = "2025-01-01"
    end_date = "2026-01-01"

    print("Downloading data...")
    stock_data = get_hk_stock_daily_returns(hk_stocks, start_date, end_date)

    # Assume risk-free rate is 2% annualized (change as needed)
    risk_free_rate = 0.02

    print("Calculating tangency portfolio...")
    weights = tangency_portfolio(stock_data, risk_free_rate)

    # Portfolio metrics
    mean_returns = stock_data.mean() * 252
    cov_matrix = stock_data.cov() * 252

    expected_return = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    print("\nTangency Portfolio (Maximum Sharpe Ratio) Results:")
    print("---------------------------------------------------")
    print("Stock        Weight")
    print("-------------------")
    for stock, weight in zip(hk_stocks, weights):
        print(f"{stock:10s}  {weight:.4f}")
    print("-------------------")
    print(f"Expected annual return: {expected_return:.4%}")
    print(f"Annualized volatility:  {volatility:.4%}")
    print(f"Sharpe Ratio:           {sharpe_ratio:.4f}")

    # Testing period performance (using SAME weights but test data's mean/cov)
    test_start = '2025-01-01'
    test_end = '2026-01-01'
    testing_data = get_hk_stock_daily_returns(hk_stocks, test_start, test_end)
    test_mean_returns = testing_data.mean() * 252
    test_cov_matrix = testing_data.cov() * 252

    expected_return_test = np.dot(weights, test_mean_returns)
    volatility_test = np.sqrt(np.dot(weights.T, np.dot(test_cov_matrix, weights)))
    sharpe_ratio_test = (expected_return_test - risk_free_rate) / volatility_test

    print(f"\nTesting Period Performance ({test_start} to {test_end}):")
    print("------------------------------------------------------")
    print(f"Annualized mean return:        {expected_return_test:.4%}")
    print(f"Annualized volatility:         {volatility_test:.4%}")
    print(f"Sharpe Ratio:                  {sharpe_ratio_test:.4f}")

    draw_heatmap(stock_data)


if __name__ == "__main__":
    main()