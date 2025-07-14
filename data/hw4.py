"""Haotian Lan
Copilot is used for only debugging in this assignment.
No code in this assignment is copied directly from copilot unless explicitly outlined in comments
This is my answer to Homework 4 in FINM 25000"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    # show all rows
    pd.set_option('display.max_rows', None)
    # show all columns
    pd.set_option('display.max_columns', None)

    print("\nForecast Regressions\n")
    print("\n1. Consider the lagged regression, where the regressor, (X,) is a period behind the target, (rSPY)")
    print("""Estimate (1) and report the R2, as well as the OLS estimates for α and β. Do this for...
• X as a single regressor, the dividend-price ratio.
• X as a single regressor, the earnings-price ratio.
• X as three regressors, the dividend-price ratio, the earnings-price ratio, and the 10-year yield.
For each, report the r-squared""")

    # Import the data from 'signals' sheet in the Excel file gmo_analysis_data
    signals = pd.read_excel("gmo_analysis_data.xlsx",
                            sheet_name='signals', header=0, index_col=0)
    # Import the data from 'total returns' sheet in the Excel file gmo_analysis_data
    spy = pd.read_excel("gmo_analysis_data.xlsx",
                        sheet_name='total returns', header=0, index_col=0)

    # Shift the signals dataframe down by 1 row. Thus, the first row of signals is NaN.
    # The first row of combined data is dropped after joining two dataframes
    signals_lag = signals.shift(1)
    combined_data = spy.join(signals_lag, how='inner').dropna()
    # print(combined_data)

    print("\nX as a single regressor, the dividend-price ratio.\n")
    # Compute the alpha, beta, r-squared, and epsilon of dividend price ratio as X
    dp_alpha, dp_beta, dp_r2, dp_error, dp_model = regression_statistics(combined_data[['SPX D/P']],
                                                                         combined_data[['SPY']])

    # Convert beta data type from one element series to a float
    beta_value = dp_beta.item()
    print(f"Alpha: {round(dp_alpha, 4)}"
          f"\nBeta: {round(beta_value, 4)}"
          f"\nR-squared: {round(dp_r2, 4)}")

    print("\nX as a single regressor, the earnings-price ratio.\n")
    # Compute the alpha, beta, r-squared, and epsilon of earning price ratio as X
    ep_alpha, ep_beta, ep_r2, ep_error, ep_model = regression_statistics(combined_data[['SPX E/P']],
                                                                         combined_data[['SPY']])

    # Convert beta data type from one element series to a float
    beta_value = ep_beta.item()
    print(f"Alpha: {round(ep_alpha, 4)}"
          f"\nBeta: {round(beta_value, 4)}"
          f"\nR-squared: {round(ep_r2, 4)}")

    print("\nX as three regressors, the dividend-price ratio, the earnings-price ratio, and the 10-year yield.\n")
    # Compute the alpha, beta, r-squared, and epsilon of three signals as X
    trio_alpha, trio_beta, trio_r2, trio_error, trio_model \
        = regression_statistics(
        combined_data[['SPX D/P', 'SPX E/P', 'T-Note 10YR']],
        combined_data['SPY']
    )
    print(f"Alpha: {round(trio_alpha, 4)}"
          f"\nSPX D/P Beta: {round(trio_beta.iloc[0], 4)}"
          f"\nSPX E/P Beta: {round(trio_beta.iloc[1], 4)}"
          f"\nT-Note 10YR Beta: {round(trio_beta.iloc[2], 4)}"
          f"\nR-squared: {round(trio_r2, 4)}")

    print("\n 2. For each of the three regressions, let’s try to utilize the resulting forecast in a trading strategy.")
    print("""You should now have the trading strategy returns, rx for each of the forecasts. For each strategy, estimate
• mean, volatility, Sharpe,
• max-drawdown
• 5th quantile of returns
• market alpha
• market beta
• market Information ratio""")

    backtest = pd.DataFrame(combined_data[['SPY']], index=combined_data.index)

    # Rebuild each X with constant over the full combined_data
    X_dp = sm.add_constant(signals[['SPX D/P']], has_constant='add')
    X_ep = sm.add_constant(signals[['SPX E/P']], has_constant='add')
    X_trio = sm.add_constant(signals[['SPX D/P', 'SPX E/P', 'T-Note 10YR']],
                             has_constant='add')

    # Predict on these X’s (returns a pandas Series aligned to combined_data.index)
    dp_pred = dp_model.predict(X_dp).shift(1)
    ep_pred = ep_model.predict(X_ep).shift(1)
    trio_pred = trio_model.predict(X_trio).shift(1)

    # Shift forward one so wt = f(X_t) lines up with r_{t+1}
    backtest['SPX D/P Prediction'] = dp_pred
    backtest['SPX E/P Prediction'] = ep_pred
    backtest['Three-Signals Prediction'] = trio_pred

    # Build this backtest
    backtest['SPX D/P Returns'] = 100 * backtest['SPX D/P Prediction'] * backtest['SPY']
    backtest['SPX E/P Returns'] = 100 * backtest['SPX E/P Prediction'] * backtest['SPY']
    backtest['Three-Signals Returns'] = 100 * backtest['Three-Signals Prediction'] * backtest['SPY']

    # Create dictionaries to be later merged into a single dataframe
    mean = {}
    volatility = {}
    sharpe_ratio = {}
    alpha_dict = {}
    beta_dict = {}
    info_ratio = {}

    # Compute the following metrics and add to the dictionaries
    for strategy in backtest[['SPX D/P Returns', 'SPX E/P Returns', 'Three-Signals Returns']]:
        mean[strategy] = backtest[strategy].mean() * 12
        volatility[strategy] = backtest[strategy].std() * np.sqrt(12)
        sharpe_ratio[strategy] = mean[strategy] / volatility[strategy]

        alpha, beta, r2, error, model = regression_statistics(backtest[strategy], backtest['SPY'])
        alpha_dict[strategy] = alpha
        beta_dict[strategy] = beta
        info_ratio[strategy] = alpha / error

    # Combine the dictionaries into one dataframe
    trading_strategy = pd.DataFrame({
        'Mean': pd.Series(mean),
        'Volatility': pd.Series(volatility),
        'Sharpe Ratio': pd.Series(sharpe_ratio),
        'Maximum Drawdowns': cal_drawdown(backtest[['SPX D/P Returns', 'SPX E/P Returns', 'Three-Signals Returns']]),
        # VaR (.05) = the fifth quantile of historic returns
        'VaR(.05)': backtest[['SPX D/P Returns', 'SPX E/P Returns', 'Three-Signals Returns']].quantile(0.05),
        # CVaR (.05) = the mean of the returns at or below the fifth quantile
        'CVaR(.05)': backtest[['SPX D/P Returns', 'SPX E/P Returns', 'Three-Signals Returns']][
            backtest[['SPX D/P Returns', 'SPX E/P Returns', 'Three-Signals Returns']] <= backtest[
                ['SPX D/P Returns', 'SPX E/P Returns', 'Three-Signals Returns']].quantile(0.05)].mean(),
        'Alpha': pd.Series(alpha_dict),
        'Beta': pd.Series(beta_dict),
        'Information Ratio': pd.Series(info_ratio)})
    print(trading_strategy)

    """with pd.ExcelWriter("testing_data.xlsx",
                        engine="openpyxl") as writer:
        trading_strategy.to_excel(writer, sheet_name="trading_strategy")
        backtest.to_excel(writer, sheet_name="backtest")"""

    print("""\n3. The GMO case mentions that stocks under-performed short-term bonds from 2000-2011. Does
the dynamic portfolio above under-perform the risk-free rate over this time?\n""")

    # Import the data from 'risk-free rate' sheet in the Excel file gmo_analysis_data
    risk_free = pd.read_excel("gmo_analysis_data.xlsx",
                              sheet_name='risk-free rate', header=0, index_col=0)

    # Calculate the risk‐free annualized metrics
    rf = risk_free['TBill 3M']
    rf_mean = rf.mean() * 12
    rf_vol = rf.std() * np.sqrt(12)
    rf_sharpe = rf_mean / rf_vol

    # Put them in a Series
    rf_summary = pd.Series({
        'Mean': rf_mean,
        'Volatility': rf_vol,
        'Sharpe Ratio': rf_sharpe
    }, name='T-Bill 3M')

    # Select only the three columns for alignment
    strategies_metrics = trading_strategy[['Mean', 'Volatility', 'Sharpe Ratio']]
    # Append the risk‐free row
    rf_comparison = pd.concat([strategies_metrics, rf_summary.to_frame().T])

    print(rf_comparison)

    print("""\n4. Based on the regression estimates, in how many periods do we estimate a negative risk premium?
That is, in how many periods is our forecasted excess return negative?\n""")

    # Join the T-Bill series onto the backtest
    combined = backtest.join(risk_free, how="inner")

    # compute risk premia – subtract from the strategy returns columns
    combined['SPX D/P Risk Premium'] = combined['SPX D/P Prediction'] - combined['TBill 3M']
    combined['SPX E/P Risk Premium'] = combined['SPX E/P Prediction'] - combined['TBill 3M']
    combined['Three-Signals Risk Premium'] = combined['Three-Signals Prediction'] - combined['TBill 3M']

    # Count negative periods for each strategy
    negative_periods = {
        'SPX D/P Returns': (combined['SPX D/P Risk Premium'] < 0).sum(),
        'SPX E/P Returns': (combined['SPX E/P Risk Premium'] < 0).sum(),
        'Three-Signals Returns': (combined['Three-Signals Risk Premium'] < 0).sum()
    }

    # Count percentage of negative periods for each strategy
    negative_percentage = {
        'SPX D/P Returns': (combined['SPX D/P Risk Premium'] < 0).mean(),
        'SPX E/P Returns': (combined['SPX E/P Risk Premium'] < 0).mean(),
        'Three-Signals Returns': (combined['Three-Signals Risk Premium'] < 0).mean()
    }

    # Turn into a Series and assign to the trading_strategy
    trading_strategy['Negative Periods'] = pd.Series(negative_periods, name='Negative Periods')
    trading_strategy['Negative Periods in Percentage'] = pd.Series(negative_percentage,
                                                                   name='Negative Periods in Percentage')

    print(trading_strategy[['Negative Periods', 'Negative Periods in Percentage']])

    print("""\n5. Do you believe increased risk is behind the out-performance of ˜ rx and ˜ rgmo?""")

    out_performance = trading_strategy[['Mean', 'Volatility', 'Sharpe Ratio']].copy()

    for index in spy.columns:
        out_performance.loc[index] = [spy[index].mean() * 12, spy[index].std() * np.sqrt(12),
                                      spy[index].mean() * 12 / (spy[index].std() * np.sqrt(12))]
    print(out_performance)

    print("""For r_x of the three strategies, D/P and three-signals have higher mean and volatility than SPY, while
E/P has a lower mean and volatility than SPY. This supports that out-performance of SPY bears increased risk.
For r_gmo, GMWAX has a lower mean and volatility, which supports the statement. However, GMGEX has lower mean 
and higher volatility than SPY, indicating increased risk may occur even in the absence of out-performance.""")


def cal_drawdown(funds_data):
    """
    Input
    funds_data : a dataframe containing columns of returns
    Output
    drawdowns : a dictionary with keys for each column of funds_data and values of maximum drawdown
    """
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


def regression_statistics(X, Y):
    """
    Inputs
    X : Series of regressors
    Y : Series of dependent variable
    Outputs
    annualized_alpha : annualized alpha in float
    beta : betas in Series (indexed by X.columns)
    r2 : R-squared - float
    annualized_te : annualized tracking error in float
    model : the linear regression model
    """
    # Add intercept and fit OLS
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(Y, X).fit()

    # Extract raw parameters
    params = model.params
    intercept = params['const']
    beta = params.drop('const')  # Drop the intercept
    r2 = model.rsquared

    # Annualize tracking error and alpha
    te_monthly = model.resid.std()
    annualized_te = te_monthly * np.sqrt(12)
    annualized_alpha = intercept * 12

    return annualized_alpha, beta, r2, annualized_te, model


if __name__ == '__main__':
    main()
