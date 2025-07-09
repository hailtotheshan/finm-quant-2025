"""Haotian Lan
Copilot is used for only debugging in this assignment.
No code in this assignment is copied directly from copilot unless explicitly outlined in comments
This is my answer to Homework 3 in FINM 25000"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


def main():
    # Import the data from 'factors (excess returns)' sheet in the excel file given
    factors = pd.read_excel("factor_pricing_data.xlsx",
                            sheet_name='factors (excess returns)', header=0, index_col=0)
    # print(factors)

    print("\nPricing Factors and Assets\n")
    print("""\n1. The Factors. Calculate their univariate performance statistics:
 • mean
 • volatility
 • Sharpe
 • VaR(.05)
Does each factor have a premium (positive expected excess return)?""")

    # Mean, standard deviation, sharpe ratios, and VaR are annnualized
    factors_statistics = pd.DataFrame({
        'Mean': factors.mean() * 12,
        'Volatility': factors.std() * np.sqrt(12),
        'Sharpe ratio': (factors.mean() * 12) / (factors.std() * np.sqrt(12)),
        'VaR(.05)': factors.quantile(0.05) * np.sqrt(12)
    })
    print("\n", factors_statistics)
    print("Based on the statistics, all factors have a positive expected return.")

    print("""\nThe factors are constructed in such a way as to reduce correlation between them.
Report the correlation matrix across the three factors. Does the construction method succeed
in keeping correlations small?""")

    # Compute the correlation matrix
    correlation_matrix = factors.corr()
    print("Correlation matrix:\n", correlation_matrix)
    print("The correlation between market factor, value factor, and size factor range from -0.21 to 0.23."
          "\nThus, construction method has succeed in keeping correlations small.")

    print("""\n 3. Plot the cumulative returns of the factors.""")

    # Find cumulative return of each factor
    cumulative_return = (1 + factors).cumprod()

    plt.figure(figsize=(10, 6))
    cumulative_return.plot(linewidth=1.5, colormap='tab10')

    # Plot the cumulative return
    plt.title("Cumulative Return in Each Factor during 1980–2025")
    plt.xlabel("Date")
    plt.ylabel("Return in Percentage")
    plt.legend(title="Factors", loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("""\n 4. Test assets.
The “assets” tab provides monthly excess return data on various industry stock-portfolios.
Denote these as ri, for n = 1,...,12.
Calculate the (annualized) univariate statistics from 1.1.""")

    # Import the data from 'portfolios (excess returns)' sheet in the excel file given
    portfolios = pd.read_excel("factor_pricing_data.xlsx",
                               sheet_name='portfolios (excess returns)', header=0, index_col=0)

    # Mean, standard deviation, sharpe ratios, and VaR are annnualized
    portfolios_statistics = pd.DataFrame({
        'Mean': portfolios.mean() * 12,
        'Volatility': portfolios.std() * np.sqrt(12),
        'Sharpe ratio': (portfolios.mean() * 12) / (portfolios.std() * np.sqrt(12)),
        'VaR(.05)': portfolios.quantile(0.05) * np.sqrt(12)
    })
    print("\n", portfolios_statistics)

    print("""\n 5. Can the difference in mean excess returns of the portfolios be explained by differences in their
volatilities? Or by their VaR(.05) statistics?""")

    # mean → X, volatility → Y
    x = portfolios_statistics.iloc[:, 0].values
    y = portfolios_statistics.iloc[:, 1].values

    # Construct a linear regression between mean and volatility
    mean_volatility = LinearRegression()
    mean_volatility.fit(x.reshape(-1, 1), y)
    y_pred = mean_volatility.predict(x.reshape(-1, 1))
    # slope = mean_volatility.coef_[0], intercept = mean_volatility.intercept_

    # Plot the graph
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.6, label='data')
    plt.plot(x, y_pred, color='green', linestyle='--', label='regression line')

    plt.xlabel(portfolios_statistics.columns[0])
    plt.ylabel(portfolios_statistics.columns[1])
    plt.title("Mean-versus-Volatility Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2a. Use the model’s .score() method
    r_squared = mean_volatility.score(x.reshape(-1, 1), y)
    print(f"R² of linear regression between mean and volatility: {r_squared:.4f}")

    # mean → X, VaR → Y
    x = portfolios_statistics.iloc[:, 0].values
    y = portfolios_statistics.iloc[:, 2].values

    # Construct a linear regression between mean and VaR
    mean_var = LinearRegression()
    mean_var.fit(x.reshape(-1, 1), y)
    y_pred = mean_var.predict(x.reshape(-1, 1))
    # slope = mean_var.coef_[0], intercept = mean_var.intercept_

    # Plot the graph
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.6, label='data')
    plt.plot(x, y_pred, color='green', linestyle='--', label='regression line')

    plt.xlabel(portfolios_statistics.columns[0])
    plt.ylabel(portfolios_statistics.columns[1])
    plt.title("Mean-versus-VaR Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2a. Use the model’s .score() method
    r_squared = mean_var.score(x.reshape(-1, 1), y)
    print(f"R² linear regression between mean and VaR: {r_squared:.4f}")

    print("\nCAPM")
    print("""\n1. For each of the n = 12 test assets, run the CAPM time-series regression:
˜ ri t = αi +βi,mkt ˜ fmktt + ϵit(CAPM)
So you are running 12 separate regressions, each using the T-sized sample of time-series data.
2. Report the estimated βi,mkt, Treynor Ratio, αi, and Information Ratio for each of the n regres
sions.""")

    # Combine data from hedge_fund_series and merill_factors by date
    combined_data = portfolios.join(factors[['MKT']], how='inner')

    # These dictionaries will be later merged into a single dataframe
    alpha_dict = {}
    beta_dict = {}
    treynor_ratio_dict = {}
    info_ratio_dict = {}

    for asset in portfolios.columns:
        # pull out series of SPY and individual asset
        X = combined_data['MKT']
        y = combined_data[asset]

        # add constant for intercept
        X_const = sm.add_constant(X)

        # run OLS: return_i = α + β·MKT + ε
        model = sm.OLS(y, X_const).fit()

        # extract the estimates
        alpha = model.params['const']
        beta = model.params['MKT']
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

    print("\nCAPM Regression:")
    print(capm_regression)

    print("\n 3. If (CAPM)were true, what would be true of the Treynor Ratios, alphas, and Information Ratios?")
    print("""If CAPM (asset return = beta * market return (assume 0 risk free rate)) were true, 
alpha is 0, treynor ratio = asset return / beta = market return, and information ratio =
asset return / epsilon = 0 / epsilon = 0.""")

    # sum the absolute alphas, then divide by the number of rows
    annualized_mae = capm_regression['Alpha'].abs().sum() * 12 / len(capm_regression)
    print("\nMean absolute error:", round(annualized_mae, 4))

    print("""\nIf the pricing model worked, should these alpha estimates be large or small? Why? Based on
your MAE stat, does this seem to support the pricing model or not?
\nIf the pricing model worked, alpha estimates should be small because in an ideal CAPM all asset returns 
are only depending on beta and market return, not alpha. MAE stat does not support the pricing model
because MAE is large, meaning a discrepancy between the expected asset return as a proportion of
market return and the actual asset return.""")

    print("\n 4 Amultifactor model")
    print("""Let’s use regression methods to test whether the selected four pricing factors work.
For each equity security, estimate the following regression to test the 4-factor model.
For each regression, report the estimated α and r-squared.""")

    combined_data = portfolios.join(factors[['MKT', 'SMB', 'HML', 'UMD']], how='inner')

    # These dictionaries will be later merged into a single dataframe
    alpha_dict = {}
    r_square_dict = {}
    adj_r_square_dict = {}

    for asset in portfolios.columns:
        X = combined_data[['MKT', 'SMB', 'HML', 'UMD']]
        y = combined_data[asset]

        X = sm.add_constant(X)  # now columns = ['const','MKT','SMB','HML','UMD']

        # run OLS: return_i = α + β·MKT + ε
        model = sm.OLS(y, X_const).fit()

        # extract the estimates
        alpha = model.params['const']
        annualized_alpha = alpha * 12

        # Raw coefficient of determination
        r_squared = model.rsquared
        # Adjusted R-squared penalizes for number of regressors
        r_squared_adj = model.rsquared_adj

        # Append the statistics into the dictionaries
        alpha_dict[asset] = annualized_alpha
        r_square_dict[asset] = r_squared
        adj_r_square_dict[asset] = r_squared_adj

    # Combine the dictionaries into one dataframe
    multifactor_regression = pd.DataFrame({
        'Alpha': pd.Series(alpha_dict),
        'R-squared': pd.Series(r_square_dict),
        'Adjusted R-squared': pd.Series(adj_r_square_dict)})
    print(multifactor_regression)

    # sum the absolute alphas, then divide by the number of rows
    annualized_mae = multifactor_regression['Alpha'].abs().sum() * 12 / len(multifactor_regression)
    print("\nMean absolute error:", round(annualized_mae, 4))

    print("""\nIf the pricing model worked, should these alpha estimates be large or small? 
Why? Based on your MAE stat, does this seem to support the pricing model or not?
\nIf the pricing model worked, these alpha estimates should be small because asset returns
would be fully explained by the four factors and alphas would be negligible. The MAE stats
does not support this because a relatively large MAE indicates difference between returns 
estimated by the factor model and the actual return.""")


if __name__ == '__main__':
    main()
