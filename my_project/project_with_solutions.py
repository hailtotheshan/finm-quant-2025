import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt

def get_hk_stock_daily_returns(hk_stocks, start_date, end_date):
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
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.ffill(axis=0).bfill(axis=0)
    return returns

def tangency_portfolio(returns, risk_free_rate=0.02):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (port_return - risk_free_rate) / port_vol
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

def portfolio_metrics(weights, returns, risk_free_rate=0.02):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    expected_return = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility
    daily_port_ret = returns @ weights
    VaR = np.percentile(daily_port_ret, 5) * np.sqrt(252)
    CVaR = daily_port_ret[daily_port_ret <= VaR].mean() * np.sqrt(252)
    cumulative = (1 + daily_port_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak) - 1
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin() if not drawdown.isna().all() else None
    max_dd_start = cumulative[:max_dd_end].idxmax() if max_dd_end is not None and not cumulative[:max_dd_end].empty else None
    return {
        'mean': expected_return,
        'vol': volatility,
        'sharpe': sharpe_ratio,
        'VaR': VaR,
        'CVaR': CVaR,
        'max_drawdown': max_dd,
        'max_dd_start': max_dd_start,
        'max_dd_end': max_dd_end,
        'daily_returns': daily_port_ret,
    }

def rolling_backtest(hk_stocks, start_year, end_year, risk_free_rate=0.02, min_stocks=5):
    results = []
    all_dates = []
    all_cumrets = []
    for year in range(start_year, end_year):
        train_start = f"{year}-01-01"
        train_end = f"{year}-12-31"
        test_start = f"{year+1}-01-01"
        test_end = f"{year+1}-12-31"
        print(f"\nBacktest: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
        try:
            train_returns = get_hk_stock_daily_returns(hk_stocks, train_start, train_end)
            test_returns = get_hk_stock_daily_returns(hk_stocks, test_start, test_end)
            # Only keep stocks with no NaNs in either period
            common_stocks = set(train_returns.columns[train_returns.notna().all()].tolist())
            common_stocks &= set(test_returns.columns[test_returns.notna().all()].tolist())
            common_stocks = sorted(common_stocks)
            if len(common_stocks) < min_stocks:
                print(f"  Only {len(common_stocks)} stocks with full data, skipping year.")
                continue
            train_returns = train_returns[common_stocks]
            test_returns = test_returns[common_stocks]
            weights = tangency_portfolio(train_returns, risk_free_rate)
            metrics = portfolio_metrics(weights, test_returns, risk_free_rate)
            metrics['year'] = year + 1
            metrics['n_stocks'] = len(common_stocks)
            results.append(metrics)
            cumret = (1 + metrics['daily_returns']).cumprod()
            all_dates.extend(cumret.index)
            all_cumrets.extend(cumret.values)
        except Exception as e:
            print(f"  Error in year {year}: {e}")
            continue
    return pd.DataFrame(results), pd.Series(all_cumrets, index=all_dates)

def visualize_backtest(df, cumrets):
    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.plot(df['year'], df['mean'], label='Mean Return')
    plt.plot(df['year'], df['vol'], label='Volatility')
    plt.plot(df['year'], df['sharpe'], label='Sharpe Ratio')
    plt.xlabel("Year")
    plt.title("Annual Out-of-Sample Portfolio Stats")
    plt.legend()
    plt.subplot(122)
    cumrets = cumrets.sort_index()
    plt.plot(cumrets.index, cumrets.values)
    plt.title("Cumulative Portfolio Value (Backtest)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.show()

def print_overall_performance(df):
    print("\n==== Overall Out-of-Sample Performance ====")
    print(f"Mean annual return: {df['mean'].mean():.4%}")
    print(f"Annualized volatility: {df['vol'].mean():.4%}")
    print(f"Mean Sharpe ratio: {df['sharpe'].mean():.4f}")
    print(f"Mean annualized VaR (5%): {df['VaR'].mean():.4%}")
    print(f"Mean annualized CVaR (5%): {df['CVaR'].mean():.4%}")
    min_dd = df['max_drawdown'].min()
    dd_row = df[df['max_drawdown'] == min_dd].iloc[0]
    print(f"Max drawdown: {min_dd:.2%}")
    print(f"Drawdown from {dd_row['max_dd_start']} to {dd_row['max_dd_end']}")
    print("="*55)

def main():
    hk_stocks = [
        '6862.HK', '2015.HK', '3690.HK', '3988.HK', '0388.HK', '1398.HK', '0941.HK', '1211.HK', '1299.HK', '9992.HK',
        '1357.HK', '0005.HK', '2331.HK', '2276.HK', '1810.HK', '9626.HK', '9633.HK', '9988.HK', '0700.HK', '0133.HK',
        '0806.HK', '1788.HK', '3037.HK', '1375.HK', '0881.HK', '9901.HK', '2388.HK', '0066.HK', '3692.HK', '1378.HK',
        '6618.HK', '0386.HK', '0316.HK', '0857.HK', '0267.HK', '9618.HK', '2313.HK', '0762.HK', '0011.HK', '0300.HK',
        '1088.HK', '0288.HK', '9961.HK', '6690.HK', '2899.HK', '0669.HK', '2319.HK', '0291.HK', '0992.HK', '2628.HK',
        '0241.HK', '2318.HK'
    ]
    risk_free_rate = 0.02
    df, cumrets = rolling_backtest(hk_stocks, 2005, 2025, risk_free_rate)

    visualize_backtest(df, cumrets)
    print_overall_performance(df)
    print(df[['year','n_stocks','mean','vol','sharpe','VaR','CVaR','max_drawdown','max_dd_start','max_dd_end']].to_string(index=False))

if __name__ == "__main__":
    main()


"""
==== Overall Out-of-Sample Performance ====
Mean annual return: 33.4522%
Annualized volatility: 29.6835%
Mean Sharpe ratio: 1.0453
Mean annualized VaR (5%): -44.3668%
Mean annualized CVaR (5%): nan%
Max drawdown: -59.36%
Drawdown from 2008-01-07 00:00:00 to 2008-10-27 00:00:00
=======================================================
 year  n_stocks      mean      vol    sharpe       VaR  CVaR  max_drawdown max_dd_start max_dd_end
 2006        25  0.834104 0.233908  3.480450 -0.299513   NaN     -0.195888   2006-05-12 2006-06-14
 2007        27  0.534369 0.303355  1.695600 -0.508724   NaN     -0.239194   2007-10-30 2007-11-22
 2008        28 -0.506439 0.509577 -1.033089 -0.799714   NaN     -0.593607   2008-01-07 2008-10-27
 2009        28  1.914110 0.683077  2.772909 -0.897133   NaN     -0.265497   2009-10-23 2009-11-19
 2010        28 -0.012157 0.262380 -0.122558 -0.450278   NaN     -0.215819   2010-04-26 2010-05-25
 2011        31 -0.308493 0.326964 -1.004675 -0.560705   NaN     -0.518537   2011-01-06 2011-10-04
 2012        32  0.035650 0.273780  0.057163 -0.450123   NaN     -0.248851   2012-01-04 2012-06-28
 2013        32  0.269991 0.148732  1.680816 -0.251539   NaN     -0.136998   2013-05-28 2013-06-24
 2014        32  0.187174 0.217389  0.769008 -0.341927   NaN     -0.147126   2014-08-20 2014-12-17
 2015        33  0.004325 0.293804 -0.053353 -0.397448   NaN     -0.346921   2015-05-27 2015-08-26
 2016        34  0.116072 0.182232  0.527197 -0.281211   NaN     -0.129951   2016-09-06 2016-12-22
 2017        35  0.539960 0.179883  2.890540 -0.220329   NaN     -0.105328   2017-11-07 2017-12-06
 2018        35 -0.121376 0.204922 -0.689903 -0.346781   NaN     -0.249584   2018-06-07 2018-10-30
 2019        38  0.474407 0.282919  1.606139 -0.457423   NaN     -0.238389   2019-04-10 2019-06-04
 2020        40  0.697615 0.352737  1.921020 -0.538449   NaN     -0.299850   2020-01-14 2020-03-19
 2021        46 -0.026910 0.373919 -0.125454 -0.642661   NaN     -0.344693   2021-02-17 2021-07-27
 2022        51  0.099314 0.257002  0.308613 -0.377777   NaN     -0.235710   2022-06-08 2022-10-31
 2023        51  0.214382 0.241651  0.804390 -0.355812   NaN     -0.161126   2023-05-09 2023-06-29
 2024        51  0.526285 0.209106  2.421193 -0.262581   NaN     -0.137478   2024-07-04 2024-08-05
 2025        52  1.218051 0.399366  2.999882 -0.433225   NaN     -0.196843   2025-03-19 2025-04-07
 """