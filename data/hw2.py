"""Haotian Lan
Copilot is used for only debugging in this assignment.
No code in this assignment is copied directly from copilot unless explicitly outlined in comments
This is my answer to Homework 1 in FINM 25000"""

import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Import the data from 'hedge_fund_series' sheet in the excel file given
    hedge_fund_series = pd.read_excel("proshares_analysis_data.xlsx",
                                      sheet_name='hedge_fund_series', header=0, index_col=0)
    # print(hedge_fund_series)

    print("\n1.For the series in the \"hedge fund series\" tab, "
          "report the following summary statistics:")
    print('''- mean 
- volatility
- Sharpe ratio
- Annualize these statistics.''')

    summary_statistics = pd.DataFrame({
        'Annualized Mean': hedge_fund_series.mean() * 12,
        'Annualized Volatility': hedge_fund_series.std() * np.sqrt(12),
        'Annualized Sharpe ratio': (hedge_fund_series.mean() * 12) / (hedge_fund_series.std() * np.sqrt(12))
    })
    print("\n", summary_statistics)

    print('''\n2. For the series in the "hedge fund series" tab, calculate the following statistics related to tail-risk.
- Skewness
- Excess Kurtosis (in excess of 3)
- VaR (.05) - the fifth quantile of historic returns
- CVaR (.05) - the mean of the returns at or below the fifth quantile
- Maximum drawdown - include the dates of the max/min/recovery within the max drawdown period.
There is no need to annualize any of these statistics.''')

    risk = pd.DataFrame({
        # Compute and print out skewness and kurtosis
        'Skewness': hedge_fund_series.skew(),
        'Kurtosis': hedge_fund_series.kurt(),
        # Compute and print out VaR (.05) and CVaR (.05)
        # VaR (.05) = the fifth quantile of historic returns
        'VaR(.05)': hedge_fund_series.quantile(0.05),
        # CVaR (.05) = the mean of the returns at or below the fifth quantile
        'CVaR(.05)': hedge_fund_series[hedge_fund_series <= hedge_fund_series.quantile(0.05)].mean(),
        'Maximum Drawdown': cal_drawdown(hedge_fund_series),
    })
    print("\n", risk)

    print("""\n3.
For the series in the "hedge fund series" tab, run a regression of each against SPY (found in the "merrill factors" tab.) 
Include an intercept. Report the following regression-based statistics:
- Market Beta
- Treynor Ratio
- Information ratio
Annualize these three statistics as appropriate.""")

    # Import the data from 'merrill_factors' sheet in the excel file given
    merrill_factors = pd.read_excel("proshares_analysis_data.xlsx",
                                    sheet_name='merrill_factors', header=0, index_col=0)
    # Combine data from hedge_fund_series and merill_factors by date
    combined_data = hedge_fund_series.join(merrill_factors[['SPY US Equity']], how='inner')

    # Compute betas of funds using calculate_beta function
    market_beta = calculate_beta(combined_data, 'SPY US Equity')

    treynor = {}
    # Treynor ratio = r_i / beta
    for hedge_fund, beta_value in market_beta.items():
        # Annualize return by a scale of 12
        treynor_ratio = combined_data[hedge_fund].mean() * 12 / beta_value
        # print(f"{hedge_fund}: {treynor_ratio:.4f}")
        treynor[hedge_fund] = treynor_ratio

    information = {}
    # This block about calculating information ratios is directly written by Copilot
    # print("\nInformation ratio:")
    for hedge_fund in hedge_fund_series:
        # For a given fund column 'FundX'
        y = combined_data[hedge_fund]
        X = combined_data['SPY US Equity']
        X = sm.add_constant(X)  # Add intercept

        # Run the OLS regression: r_fund = alpha + beta*r_market + error
        model = sm.OLS(y, X).fit()

        # Extract Regession Parameters
        alpha = model.params['const']
        beta = model.params['SPY US Equity']
        # print("Alpha:", round(alpha, 4), "Beta:", round(beta, 4))

        # Calculate residuals (tracking errors) from the regression
        residuals = model.resid

        # Compute the standard deviation (tracking error) of the residuals (monthly)
        tracking_error = np.std(residuals, ddof=1)

        # Annualize the monthly alpha and tracking error:
        annualized_alpha = alpha * 12
        annualized_tracking_error = tracking_error * np.sqrt(12)

        # Compute the annualized information ratio
        information_ratio = annualized_alpha / annualized_tracking_error

        # print(f"{hedge_fund}:", round(information_ratio, 4))
        information[hedge_fund] = information_ratio
    # Direct copy of Copilot ends in this line

    # Combine three dictionaries into one dataframe
    factor_decomposition = pd.DataFrame({
        'Market Beta': pd.Series(market_beta),
        'Treynor Ratio': pd.Series(treynor),
        'Information Ratio': pd.Series(information)})

    print("\nFactor Decomposition:")
    print(factor_decomposition)

    print("""\n4.
Discuss the previous statistics, and what they tell us about...

- the differences between SPY and the hedge-fund series?
- which performs better between HDG and QAI.
- whether HDG and the ML series capture the most notable properties of HFRI.""")
    print("""  Since the all four funds in the hedge_fund_series have a market beta less than 1,
the funds are less aligned to the equity market overall fluctuation and exposed to less systematic risks.
The Treynors ratios of hedge_fund series are below the ratio of SPY. This indicates either hedge_fund
did not earn enough return when bearing same amount of risk as the market or hedge funds are exposed to 
more risks when they earned the same amount of return as the market.
All funds have a negative information ratio, signifying a relatively large amount of noise in its return.
  QAI performed better than HDG because QAI has less market beta, higher treynor ratio, and higher information ratio.
This reveals QAI was less aligned with market fluctuation, earned a higher risk-adjusted return, and had less
noise in its alpha.
  While HDG and ML series have low beta, they still had large drawdown, 
low treynor ratios, and negative information ratios.""")

    print("""\n5.
Report the correlation matrix for these assets.

- Show the correlations as a heat map.
- Which series have the highest and lowest correlations?""")

    # Run the next line of code to generate heatmap
    # draw_heatmap(hedge_fund_series, "Hedge Funds Series vs. SPY")
    print("\nBased on the results shown on the heatmap, MLEIFCTR Index and MLEIFCTX Index"
          "have the highest correlation of 1."
          "\nMLEIFCTR Index and QAI US Equity have the lowest correlation of 0.89")

    print("""\n6.
Replicate HFRI with the six factors listed on the "merrill factors" tab. 
Include a constant, and run the unrestricted regression
a. Report the intercept and betas.

b. Are the betas realistic position sizes, or do they require huge long-short positions?

c. Report the R-squared.

d. Report the volatility of ϵ^merr, the tracking error.""")

    for factor in merrill_factors:
        print(f"\nRegressions with {factor} as the market benchmark:")
        print(hf_regression(hedge_fund_series, merrill_factors[[factor]]))

    print("""\nUSGG3M Index and EUO US Equity with approximately 0 beta does not need much hedging.
EUO US Equity with negative beta of -0.41 might require a long-hold position of SPY to mitigate market risk exposure.
However, the other equities, including EEM US Equity, EFA US Equity, and IWM US Equity 
have relatively high beta ranging from 0.8 to 1.2. This indicates a moderate long-short positions:
holding these equity might require long-short 80% to 120% of weights in SPY.""")

    print("\n7. Let's examine the replication out-of-sample (OOS).")

    window = 60
    dates_oos = []
    oos_pred = {f: [] for f in hedge_fund_series.columns}
    oos_true = {f: [] for f in hedge_fund_series.columns}

    for i in range(window, len(hedge_fund_series)):
        date_t = hedge_fund_series.index[i]
        dates_oos.append(date_t)

        for fund in hedge_fund_series.columns:
            y_train = hedge_fund_series[fund].iloc[i - window:i]
            X_train = sm.add_constant(merrill_factors['SPY US Equity'].iloc[i - window:i])
            model = sm.OLS(y_train, X_train).fit()

            alpha = model.params['const']
            beta = model.params['SPY US Equity']

            x_t = merrill_factors['SPY US Equity'].iloc[i]
            y_hat = alpha + beta * x_t

            oos_pred[fund].append(y_hat)
            oos_true[fund].append(hedge_fund_series[fund].iloc[i])

    # Calculate error DataFrames
    pred_df = pd.DataFrame(oos_pred, index=dates_oos)
    true_df = pd.DataFrame(oos_true, index=dates_oos)
    err_df = true_df - pred_df

    perf = pd.DataFrame(index=hedge_fund_series.columns)

    # Compute performance metrics:
    perf['OOS MSE'] = (err_df ** 2).mean()
    perf['OOS Corr'] = true_df.corrwith(pred_df)

    sse = (err_df ** 2).sum()
    sst = ((true_df - true_df.mean()) ** 2).sum()
    perf['OOS R2'] = 1 - sse / sst

    perf['Mean Error'] = err_df.mean()

    print(perf.round(4))


def hf_regression(hedge_fund_series, market_df):
    """The function "regression_statistics" is needed for this function.
    Input: hedge_fund_series is a df with each column of funds to be compared
    with the market by regression
    market_df is a single column df of market benchmark to be compared with.
    Output: A dataframe has four columns: alpha, beta, R-squared, tracking error.
    It has rows of funds corresponding to funds in hedge_fund_series."""

    # Initialize dictionaries to be compiled into a dataframe as the output
    hf_alpha = {}
    hf_beta = {}
    hf_r2 = {}
    hf_error = {}

    for hedge_fund in hedge_fund_series:
        # Combine hedge_fund_series with market_df
        combined_data = hedge_fund_series.join(market_df, how='inner')

        # Calculate regression statistics of a column of funds
        hedge_fund, annualized_alpha, beta, r2, annualized_tracking_error = (
            regression_statistics(combined_data, hedge_fund, market_df.columns))

        # Append the regression statistics to the dictionaries
        hf_alpha[hedge_fund] = annualized_alpha
        hf_beta[hedge_fund] = beta
        hf_r2[hedge_fund] = r2
        hf_error[hedge_fund] = annualized_tracking_error

    # Combine all dictionaries into a single dataframe
    hf_factors = pd.DataFrame({
        'Alpha': pd.Series(hf_alpha),
        'Beta': pd.Series(hf_beta),
        'R^2': pd.Series(hf_r2),
        'Tracking Error': pd.Series(hf_error)
    })

    return hf_factors


def regression_statistics(df, factor, market_column):
    """Inputs: df is a dataframe containing monthly return of factor and market_column
    factor is the columns in the df dataframe to be compared to the market.
    market_column is the string name of the market column in df.
    Outputs: alpha, beta, R-squared, and epsilon (tracking error) of factor
    with respect to the market."""

    # Y-axis = monthly returns of a factor
    y = df[factor]
    # X-axis = monthly returns of SPY
    X = df[market_column]
    X = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X).fit()

    # Calculate the regression statistics of alpha, beta, and R^2
    alpha = model.params['const']
    beta = model.params[market_column]
    r2 = model.rsquared

    # Compute tracking errors
    residuals = model.resid
    # Compute the standard deviation of tracking errors
    tracking_error = residuals.std()

    # Annualize alpha and tracking error by scale of 12 and sqrt(12)
    annualized_alpha = alpha * 12
    annualized_tracking_error = tracking_error * np.sqrt(12)

    return factor, annualized_alpha, beta, r2, annualized_tracking_error


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


def calculate_beta(df, benchmark):
    """df is a dataframe containing columns of funds and a market benchmark.
    df has row index in datetime with each row value as monthly return.
    benchmark is the column name (in strings) of market benchmark to be compared.
    The function returns betas of each fund in a dictionary"""
    fund_betas = {}

    # Compute the covariance matrix and variance of spy returns in combined data
    cov_matrix = np.cov(df[benchmark], df[benchmark])
    variance_spy = cov_matrix[0, 0]

    # Loop through each hedge fund column except the benchmark column
    for hedge_fund in df.columns:
        # Compute the covariance between the hedge fund and market benchmark returns.
        cov_matrix = np.cov(df[hedge_fund], df[benchmark])
        cov_value = cov_matrix[0, 1]

        # Market beta = covariance(r_i, r_m) / variance(r_m)
        beta = cov_value / variance_spy
        # print(f"{hedge_fund}: {beta:.4f}"
        fund_betas[hedge_fund] = beta

    return fund_betas


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


if __name__ == '__main__':
    main()
