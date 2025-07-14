"""Haotian Lan
Copilot is used for only debugging in this assignment.
No code in this assignment is copied directly from copilot.
This is my answer to Homework 1 in FINM 25000"""

import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns


def main():
    excess_return = pd.read_excel("midterm_data.xlsx",
                                  sheet_name='excess returns', header=0, index_col=0)
    spy = pd.read_excel("midterm_data.xlsx",
                                  sheet_name='spy', header=0, index_col=0)

    summary = summary_statistics(excess_return, 52)
    print(summary)

    weight = tangency_weights(excess_return)
    print(weight)

    tangency_statistics(excess_return, 52)

    summary_with_tangency = summary.join(weight, how='inner')
    print(summary_with_tangency)

    risk = pd.DataFrame({
        # Compute and print out skewness and kurtosis
        'Skewness': excess_return.skew(),
        'Kurtosis': excess_return.kurt(),
        # Compute and print out VaR (.05) and CVaR (.05)
        # VaR (.05) = the fifth quantile of historic returns
        'VaR(.05)': excess_return.quantile(0.05),
        # CVaR (.05) = the mean of the returns at or below the fifth quantile
        'CVaR(.05)': excess_return[excess_return <= excess_return.quantile(0.05)].mean(),
        'Maximum Drawdown': cal_drawdown(excess_return),
    })
    print("\n", risk)

    capm_df = capm_ratios(excess_return[['TSLA']], spy[['SPY']])
    print(capm_df)

    forecasting = pd.read_excel("midterm_data.xlsx",
                                  sheet_name='forecasting', header=0, index_col=0)

    lagged_signals = forecasting[['Tnote rate', 'Tnote rate change']].shift(1)
    lagged_forecasting = lagged_signals.join(forecasting[['USO']]).dropna()

    """annualized_alpha, beta, r2, annualized_te, model = (
        regression_statistics(lagged_forecasting['Tnote rate'], lagged_forecasting['Tnote rate change']))
    print(annualized_alpha, beta, r2, annualized_te, model)"""

    X = lagged_forecasting[['Tnote rate', 'Tnote rate change']]
    Y = lagged_forecasting['USO']  # note the single brackets → Series
    alpha, beta, r2, te, model = regression_statistics(X, Y, scale=12)

    print("Alpha:", round(alpha,4))  # float
    print("Betas:", round(beta,4))  # Series with 2 entries
    print("R2", round(r2,4))

    # Rebuild each X with constant over the full combined_data
    X = sm.add_constant(forecasting[['Tnote rate', 'Tnote rate change']], has_constant='add')

    # Predict on these X’s (returns a pandas Series aligned to combined_data.index)
    pred = model.predict(X).shift(1)

    # Shift forward one so wt = f(X_t) lines up with r_{t+1}
    lagged_forecasting['Prediction'] = pred

    # Build this backtest
    lagged_forecasting['Strategy Returns'] = (0.5 + 50 * lagged_forecasting['Prediction']) * lagged_forecasting['USO']
    print(lagged_forecasting)


def regression_statistics(X, Y, scale=12):
    """
    Compute OLS stats for one or many dependent series.
    Output
    X : Series or DataFrame Regressor(s).  Can be one column or many.
    Y : Series or DataFrame Dependent var(s).  If DataFrame, runs one regression per column.
    scale : int Periods per year (12 for monthly, 252 for daily, etc.)
    Input
    alphas : float or Series Annualized intercept(s).
    betas : Series or DataFrame Beta coefficient(s).  If multivariate, rows=index of X, cols=index of Y.
    r2s : float or Series R² value(s).
    tes : float or Series Annualized tracking error(s).
    models : RegressionResults or dict of RegressionResults The fitted `statsmodels` object(s).
    """

    def _single_reg(x, y):
        xconst = sm.add_constant(x, has_constant='add')
        mod = sm.OLS(y, xconst).fit()
        α = mod.params['const'] * scale
        β = mod.params.drop('const')
        r2 = mod.rsquared
        te = mod.resid.std() * np.sqrt(scale)
        return α, β, r2, te, mod

    # If Y is a DataFrame, loop column–by–column
    if isinstance(Y, pd.DataFrame):
        alphas, betas, r2s, tes, models = {}, {}, {}, {}, {}
        for col in Y.columns:
            α, β, r2, te, m = _single_reg(X, Y[col])
            alphas[col] = α
            betas[col] = β
            r2s[col] = r2
            tes[col] = te
            models[col] = m

        alphas = pd.Series(alphas, name='alpha')
        betas = pd.DataFrame(betas)
        r2s = pd.Series(r2s, name='r2')
        tes = pd.Series(tes, name='tracking_error')
        return alphas, betas, r2s, tes, models

    # Otherwise, single-series regression
    return _single_reg(X, Y)


def capm_ratios(stocks_columns, market_columns, scale=12):
    """
    :param stocks_columns: dataframe with columns of stocks as X variables
    :param market_columns: dataframe with single column of market (SPY) as Y variable
    :param scale: monthly return * scale of 12 = annualized return
    :return: a dataframe containing alpha, beta, treynor ratio, and information ratio
    """
    # These dictionaries will be later merged into a single dataframe
    alpha_dict = {}
    beta_dict = {}
    treynor_ratio_dict = {}
    info_ratio_dict = {}

    for stocks in stocks_columns.columns:
        # pull out series of SPY and individual asset
        X = market_columns[market_columns.columns]
        y = stocks_columns[stocks]

        # add constant for intercept
        X_const = sm.add_constant(X)

        # run OLS: return_i = α + β·MKT + ε
        model = sm.OLS(y, X_const).fit()

        # extract the estimates
        alpha = model.params['const']
        beta = model.params[market_columns.columns]
        residuals = model.resid

        mean_return = y.mean()  # average monthly Agric return
        tracking_error = residuals.std()  # sd of εt

        # annualize mean and alpha by ×12, residual‐vol by √12
        annualized_mean = mean_return * scale
        annualized_alpha = alpha * scale
        annualized_te = tracking_error * (scale ** 0.5)

        # Calculate annualized treynor ratio and information ratio
        annualized_treynor = annualized_mean / beta
        annualized_info = annualized_alpha / annualized_te

        # Append the statistics into the dictionaries
        alpha_dict[stocks] = annualized_alpha
        beta_dict[stocks] = beta
        treynor_ratio_dict[stocks] = annualized_treynor
        info_ratio_dict[stocks] = annualized_info

    # Combine the dictionaries into one dataframe
    capm_regression = pd.DataFrame({
        'Alpha': pd.Series(alpha_dict),
        'Market Beta': pd.Series(beta_dict),
        'Treynor Ratio': pd.Series(treynor_ratio_dict),
        'Information Ratio': pd.Series(info_ratio_dict)})

    return capm_regression


def cal_drawdown(funds_data):
    drawdowns = {}

    # normalize the data by dividing all values by the first row
    normalized_data = funds_data / funds_data.iloc[0]

    # Outer loop iterates over each hedge fund
    for hedge_fund in normalized_data.columns:
        max_drawdown = 0
        running_maximum = 0

        # Inner loop iterates over each monthly_return in the current column
        for monthly_return in normalized_data[hedge_fund]:
            # Update the running maximum so far.
            running_maximum = max(running_maximum, monthly_return)
            percentage_drawdown = (monthly_return - running_maximum) / running_maximum

            # Update maximum drawdown
            max_drawdown = min(max_drawdown, percentage_drawdown)
        drawdowns[hedge_fund] = max_drawdown

    return drawdowns


def summary_statistics(df, scale=1):
    """
    :param df: a dataframe with rows of dates columns of stock returns
    :param scale: convert weekly return to annual return by a scale of 52.
    covert monthly return to annual return by a scale of 12
    :return: a dataframe with rows of mean, std, and sharpe ratio and columns of stock returns
    """

    # Mean, standard deviation, sharpe ratios, and VaR are annnualized
    df_statistics = pd.DataFrame({
        'Mean': df.mean() * scale,
        'Volatility': df.std() * np.sqrt(scale),
        'Sharpe Ratio': (df.mean() * scale) / (df.std() * np.sqrt(scale))
        # 'VaR(.05)': df.quantile(0.05) * np.sqrt(12)
    })
    # print("\n", df_statistics)

    return df_statistics


def tangency_weights(df):
    """
    Compute the tangency (mean‐variance) portfolio weights.
    Returns a one‐column DataFrame of weights indexed by the ETF tickers.
    """

    annual_means = df.mean()

    cov_matrix = df.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    unnorm_w = inv_cov_matrix.dot(annual_means)
    tangency_weights = unnorm_w / np.sum(unnorm_w)

    weights_df = (
        pd.Series(
            data=tangency_weights,
            index=df.columns,
            name="Tangency Weight"
        )
        .to_frame()
    )

    return weights_df


def tangency_statistics(df, scale=12):
    """
    Calculate and return annualized stats for the tangency portfolio.

    Inputs
    df : DataFrame periodic returns (monthly if freq=12, weekly if freq=52).
    freq : int number of periods per year (12 or 52).
    Outputs
    weights_df : One-column DataFrame of tangency weights.
    stats : dict annualized mean, std, and Sharpe ratio.
    """

    weights_df = tangency_weights(df)
    weights = weights_df["Tangency Weight"]  # convert to Series

    # 2) Compute portfolio returns as a Series
    port_ret = df.dot(weights)

    # 3) Annualize
    ann_mean = port_ret.mean() * scale
    ann_std = port_ret.std() * np.sqrt(scale)
    ann_sharpe = ann_mean / ann_std

    # 4) Print and return
    print("\nTangency portfolio results:")
    print(f"Annualized mean = {ann_mean:.4f}")
    print(f"Annualized std dev = {ann_std:.4f}")
    print(f"Annualized Sharpe = {ann_sharpe:.4f}")

    stats = {
        "annualized_mean": ann_mean,
        "annualized_std": ann_std,
        "annualized_sharpe": ann_sharpe,
    }
    return weights_df, stats


if __name__ == '__main__':
    main()
