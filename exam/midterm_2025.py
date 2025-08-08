import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():
    # show all rows
    pd.set_option('display.max_rows', None)
    # show all columns
    pd.set_option('display.max_columns', None)

    excess_returns = pd.read_excel("midterm_data.xlsx",
                                  sheet_name='excess returns', header=0, index_col=0)

    assets = excess_returns.drop(columns=['SPY'])

    summary = summary_statistics(excess_returns, 52)
    print("Question 1.1:")
    print(summary)

    print("""Question 1.2: The most attractive investment is NVDA because it has the highest sharpe ratio
and a moderate VAR and max drawdown.""")

    # These dictionaries will be later merged into a single dataframe
    alpha_dict = {}
    beta_dict = {}
    treynor_ratio_dict = {}
    info_ratio_dict = {}

    for asset in assets.columns:
        # pull out series of SPY and individual asset
        X = excess_returns['SPY']
        y = assets[asset]

        # add constant for intercept
        X_const = sm.add_constant(X)

        # run OLS: return_i = α + β·MKT + ε
        model = sm.OLS(y, X_const).fit()

        # extract the estimates
        alpha = model.params['const']
        beta = model.params['SPY']
        residuals = model.resid

        mean_return = y.mean()  # average monthly Agric return
        tracking_error = residuals.std()  # sd of εt

        # annualize mean and alpha by ×12, residual‐vol by √12
        annualized_mean = mean_return * 12
        annualized_alpha = alpha * 12
        annualized_te = tracking_error * (12 ** 0.5)

        # Calculate annualized treynor ratio and information ratio
        annualized_treynor = annualized_mean / beta
        annualized_info = annualized_alpha / annualized_te

        # Append the statistics into the dictionaries
        alpha_dict[asset] = annualized_alpha
        beta_dict[asset] = beta
        treynor_ratio_dict[asset] = annualized_treynor
        info_ratio_dict[asset] = annualized_info

    # Combine the dictionaries into one dataframe
    capm_regression = pd.DataFrame({
        'Alpha': pd.Series(alpha_dict),
        'Market Beta': pd.Series(beta_dict),
        'Treynor Ratio': pd.Series(treynor_ratio_dict),
        'Information Ratio': pd.Series(info_ratio_dict)})

    print("\nQuestion 1.3:")
    print(capm_regression)

    print("""Question 1.4: The most attractive investment to holding SPY is NVDA because
it has the highest alpha and information ratio, indicating NVDA is beating the market
without large amount of noise.""")

    print("Question 1.5: ")
    alphas, betas, r2s, tes, models = regression_statistics(excess_returns[['SPY', 'NVDA']], excess_returns[['AAPL']], scale=52)

    print("Alpha:\n", alphas, "\n")
    print("Betas:\n", betas, "\n")
    print("R²:\n", r2s, "\n")
    print("Tracking Error:\n", tes, "\n")

    beta_spy = betas.loc['SPY', 'AAPL']
    beta_nvda = betas.loc['NVDA', 'AAPL']

    size_spy = - beta_spy * 100
    size_nvda = - beta_nvda * 100

    print(f"For every $100 in AAPL, I will hedge by shorting:\n"
          f"SPY: ${size_spy:,.2f}\n"
          f"NVDA: ${size_nvda:,.2f}")

    print("""Question 1.6: The tracking error tells me how me whether the replication tracks well.
The smaller the tracking error, the closer match the replication is to AAPL.""")

    print("""Question 1.7: High correlation of 0.9 to ML benckmark and HFRI index is achieved.
The drawbacks of the replications are they are lagged behind the changes in hedge fund styles.
Furthermore, since T-bill is one of the largest factor in the replication, its heavy weights
introduces factor risks in T-bill.""")

    print("Question 2.1:")
    return_correlation = excess_returns.corr()
    print(return_correlation)
    print("""In this correlation matrix, LLY seems to have low correlation to other securities.
This makes LLY advantageous for hedging out the market risk and more likely to have higher weights
in the portfolio for diversification purposes.""")

    print("Question 2.2:")
    print(tangency_weights(excess_returns))

    print("Question 2.3:")
    weights_df, stats = tangency_statistics(excess_returns, 52)

    print("Question 2.4:")
    print("""In this portfolio, the biggest weight is BRK-B and the lowest weight is SPY.
The highest sharpe ratio is NVDA and the lowest sharpe ratio is BRK-B. The weights
do not align with the extreme sharpe ratio because sharpe ratio does not take
covariance between assets into considerations, which is crucial in portfolio diversification.""")

    print("Question 2.5:")
    print(tangency_weights(assets))

    print("Question 2.6:")
    print("""In our analysis of the multi-asset portfolio optimization, 
we found that a change in TIPS mean excess returns caused a large change in the...
performance of the tangency portfolio.  FALSE
weights of the tangency portfolio.     TRUE
correlation structure of the assets.    FALSE""")

    print("Question 2.7:")
    print("""The fully optimized portfolio is unrealistic because it is not robust:
a small shift in input returns can result in huge change in tangency weights. Additionally,
it does not take much constraints, such as trading cost and maximum weights of individual assets,
into considerations. 
Compared to fully optimized portfolio, Harvard imposed realistic bounds, such as maximum
weights for individual security. They also applied two level constraints, with first 
constraint on individual securities and second constraints on securities as a holistic class.
These constrained approaches make their portfolio optimization more realistic.
By first classifying securities into different assets, Harvard reduced the dimensionality
of portfolio, which makes it more robust to noise.
""")


    print("Question 3.1:")
    X = excess_returns[['SPY']]
    Y = excess_returns[['NVDA']]
    alpha, beta, r2, te, model = regression_statistics(X, Y, scale=12)

    print("Alpha:", round(alpha,4))
    print("Betas:", round(beta,4))
    print("R2", round(r2,4))

    print("Question 3.2:")
    print("""R-squared in this factor pricing model is as low as only 0.41. 
This signifies the factor pricing model cannot explain NVDA price well.""")

    print("Question 3.3:")
    nvda_ann_excess = excess_returns['NVDA'].mean() * 52
    spy_ann_excess = excess_returns['SPY'].mean() * 52
    beta_spy = beta.loc['SPY', 'NVDA']
    capm_explained = beta_spy * spy_ann_excess
    print(f"NVDA actual annualized excess return: {nvda_ann_excess}")
    print(f"CAPM explains: {capm_explained}")

    print("Question 3.4:")

    lagged_signals = excess_returns[['NVDA']].shift(-1)
    lagged_forecasting = lagged_signals.join(excess_returns[['SPY']]).dropna()

    alphas, betas, r2s, tes, models = (
        regression_statistics(lagged_forecasting[['SPY']], lagged_forecasting[['NVDA']], 52))

    print(f"alpha: ", round(alphas['NVDA'], 4))
    print(f"betas: ", round(betas.loc['SPY','NVDA'], 4))
    print(f"r2s: ", round(r2s['NVDA'], 4))

    print("Question 3.5:")
    print("""While the model has a large positive, it has a slightly negative beta.
More importantly, the variance of NVDA will not be well explained by the model because
the R-squared is very low. Thus the forecast of NVDA unlikely to be accurate.""")

    print("Question 3.6:")
    X_t = pd.DataFrame({'SPY': [excess_returns['SPY'].iloc[-1]]})
    X_t = sm.add_constant(X_t, has_constant='add')

    nvda_pred_weekly = models['NVDA'].predict(X_t).iloc[0]
    nvda_pred_annual = nvda_pred_weekly * 52

    print(f"Annualized return : {nvda_pred_annual}")

    print("Question 3.7:")
    print("""Creating style factors in the portfolio extract individual style premium
while filter out market risks and risks in other factors. Longing targeted style factors invests in assets 
with outperforming investment styles, which are more likely to have a higher returns in the future.
Shorting targeted style factors helps to hedge out the market risks correlate with
other factors. This hedging keeps style as a standalone factor without correlating 
with other factors.""")


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
        'Sharpe Ratio': (df.mean() * scale) / (df.std() * np.sqrt(scale)),
        # Compute and print out skewness and kurtosis
        'Skewness': df.skew(),
        'Kurtosis': df.kurt(),
        # Compute and print out VaR (.05) and CVaR (.05)
        # VaR (.05) = the fifth quantile of historic returns
        'VaR(.05)': df.quantile(0.05),
        # CVaR (.05) = the mean of the returns at or below the fifth quantile
        'CVaR(.05)': df[df <= df.quantile(0.05)].mean(),
        'Maximum Drawdown': cal_drawdown(df)
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
