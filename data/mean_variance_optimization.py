"""Haotian Lan
Copilot is used for only debugging in this assignment.
No code in this assignment is copied directly from copilot.
This is my answer to Homework 1 in FINM 25000"""

import pandas as pd
import os
import numpy as np


def main():
    # Import the data from 'excess return' sheet in the excel file given
    # The sheet name in the original excel file is changed from 'excess return' to 'excess_return so that I can import it.'
    etf_data = pd.read_excel("multi_asset_etf_data.xlsx",
                             sheet_name='excess_return', header=0, index_col=0)

    print("\n1. Summary Statistics"
          "\nCalculate and display the mean and volatility of each asset’s excess return. "
          "\n(Recall we use volatility to refer to standard deviation.)")

    # Calculate the mean and the standard deviation for each column in etf_data
    etf_means = etf_data.mean() * 12  # mean is scaled by 12
    etf_std = etf_data.std() * np.sqrt(12)  # vol is scaled by sqrt(12)

    # Merge the means and volatility into a single dataframe
    summary = pd.DataFrame([etf_means, etf_std], index=['Mean', 'Standard Deviation'])
    summary = summary.T

    # Sharpe ratio = mean/standard deviation
    summary['Sharpe'] = summary['Mean'] / summary['Standard Deviation']
    print(summary)

    print("\nWhich assets have the best and worst Sharpe ratios? "
          "\nRecall that the Sharpe Ratio is simply the ratio of the mean-to-volatility of excess returns:")

    # Find the maximum and minimum Sharpe ratio
    max_sharpe = summary['Sharpe'].max()
    min_sharpe = summary['Sharpe'].min()

    # Find the ETF with maximum Sharpe ratio and the ETF with minimum Sharpe ratio
    max_etf = summary['Sharpe'].idxmax()
    min_etf = summary['Sharpe'].idxmin()

    print(f"{max_etf} has the highest Sharpe ratio of {max_sharpe}")
    print(f"{min_etf} has the lowest Sharpe ratio of {min_sharpe}")

    print("\n2. Descriptive Analysis"
          "\nCalculate the correlation matrix of the returns. Which pair has the highest correlation? And the lowest?")

    # Compute the correlation matrix of the given etf data
    etf_covariance = etf_data.cov()
    print("\nThe etf correlation matrix is as follow:\n", etf_covariance)

    print("\nHow well have TIPS done in our sample? "
          "\nHave they outperformed domestic bonds? Foreign bonds?")

    # Get the mean, std, and sharpe of TIPS in the summary dataframe
    tip_mean = summary.loc["TIP", ['Mean']].values
    tip_sd = summary.loc["TIP", ['Standard Deviation']].values
    tip_sharpe = summary.loc["TIP", ['Sharpe']].values
    print(f"TIPS in our sample has a mean of {tip_mean}, standard deviaiton "
          f"of {tip_sd}, and a sharpe ratio of {tip_sharpe}.")

    # Allocate domestic bonds and foreign bonds into separate lists
    domestic_bonds = ['IEF', 'HYG']
    foreign_bonds = ['BWX']

    print("Compared to domestic bonds, ")
    # Compare sharpe ratio of "TIPS" to the sharpe ratios of domestic bonds
    for bond in domestic_bonds:
        if tip_sharpe > summary.loc[bond, ['Sharpe']].values:
            print(f"TIPS has a higher sharpe ratio than {bond}.")
        else:
            print(f"TIPS has a lower sharpe ratio than {bond}.")

    print("Compared to foreign bonds, ")
    # Compare sharpe ratio of "TIPS" to the sharpe ratios of foreign bonds
    for bond in foreign_bonds:
        if tip_sharpe > summary.loc[bond, ['Sharpe']].values:
            print(f"TIPS has a higher sharpe ratio than {bond}.")
        else:
            print(f"TIPS has a lower sharpe ratio than {bond}.")

    print("\n3. The MV frontier."
          "\nCompute and display the weights of the tangency portfolios: w^tan.")

    # Find the inverse of the covariance matrix
    inverse_covariance = np.linalg.inv(etf_covariance)
    # Slice the mean from the previous descriptive analysis
    excess_return = summary['Mean']

    # Compute the unnormalized tangency portfolio weights
    unnormalized_weights = inverse_covariance.dot(excess_return)

    # Normalize the weights to ensure weights add up to 1
    tangency_weights = unnormalized_weights / np.sum(unnormalized_weights)
    weights_df = pd.DataFrame(tangency_weights, index=summary.index, columns=["Weight"])
    print(weights_df)

    print("\nDoes the ranking of weights align with the ranking of Sharpe ratios?")

    # Rank the values in the 'Weight' column of the weight_df
    print("The ranking of weights is as follow:")
    weight_rank = pd.DataFrame(weights_df, index=weights_df.index)
    weight_rank['Rank'] = weights_df['Weight'].rank(method='min', ascending=False)

    # Create a new column 'ETF' from the existing index
    weight_rank['ETF'] = weight_rank.index
    # Set the 'Rank' column as the new index
    weight_rank = weight_rank.set_index('Rank')

    # Sort the dataframe by the row index and convert the row index to integers
    weight_rank = weight_rank.sort_index()
    weight_rank.index = weight_rank.index.astype(int)

    print(weight_rank)

    print("The ranking of sharpe ratio is as follow:")

    # Rank the values in the 'Weight' column of the summary
    sharpe = summary[['Sharpe']].copy()
    sharpe['Rank'] = sharpe['Sharpe'].rank(method='min', ascending=False)

    # Create a new column 'ETF' from the existing index
    sharpe['ETF'] = sharpe.index
    # Set the 'Rank' column as the new index
    sharpe = sharpe.set_index('Rank')

    # Sort the dataframe by the row index and convert the row index to integers
    sharpe = sharpe.sort_index()
    sharpe.index = sharpe.index.astype(int)

    print(sharpe)
    print("Based on the comparison of two rankings, "
          "the ranking of the weights is not perfectly aligned with the ranking of the sharpe ratio.")

    print("\nCompute the mean, volatility, and Sharpe ratio for the tangency portfolio corresponding to w^tan")

    # Performa matrix multiplication on etf_data and tangency_weights
    tangency_portfolio = etf_data.dot(tangency_weights)

    # Find mean and standard deviation of the tangency portfolio
    tangency_mean = tangency_portfolio.mean() * 12  # Annualized return by scaling of 12
    tangency_std = tangency_portfolio.std() * np.sqrt(12)  # Annualized standard deviation by scaling of sqrt(12)

    # Assume that sharpe ratio = mean / standard deviation
    tangency_sharpe = (tangency_mean / tangency_std)

    print(f"For the tangency portfolio,"
          f"\nannualized mean = {tangency_mean}"
          f"\nannualized standard deviation = {tangency_std}"
          f"\nannualized sharpe ratio = {tangency_sharpe}")

    print("\n4. TIPS")
    print("\nAssess how much the tangency portfolio (and performance) change "
          "if TIPS are dropped completely from the investment set.")

    # Drop the TIP column in the dataset.
    without_tips_data = etf_data.drop("TIP", axis=1)
    mv_portfolio(without_tips_data, "when TIPS are dropped from the investment set")

    # Create a copy of the original dataframe
    increased_tip = etf_data.copy()
    # Increase every value in the 'TIP' column by 0.0012
    increased_tip['TIP'] = increased_tip['TIP'] + 0.0012
    mv_portfolio(increased_tip, "when monthly expected return of TIPS increased by 0.0012")

    print("\n 3. Allocations")
    print("\nContinue with the same data file as the previous section."
          "\nSuppose the investor has a targeted mean excess return (per month) of ~μ^port = 0.01.")

    """Use equally-weighted allocation, risk-parity allocation,
    and mean-variance allocation to determine portfolio of etf data"""
    ew_portfolio(etf_data, 0.01)
    rp_portfolio(etf_data)
    mv_portfolio(etf_data)

    print("\nMean-variance portfolio has the highest sharpe ratio among the three portfolio. "
          "\nAt the same time, mean-variance portfolio bears the highest amount of risk."
          "\nWhile risk-parity has the lowest standard deviation, "
          "its sharpe ratio is low compared to mean-variance portfolio.")


def mv_portfolio(etf_data, description=""):  # mean-variance portfolio
    # Compute expected annual returns based on the input data
    annual_means = etf_data.mean() * 12  # Scale monthly mean to annual

    # Compute the covariance matrix and inverse matrix
    cov_matrix = etf_data.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Compute unnormalized tangency portfolio weights
    unnormalized_weights = inv_cov_matrix.dot(annual_means)
    # Normalize the weights so they sum to 1
    tangency_weights = unnormalized_weights / np.sum(unnormalized_weights)
    # Compute the portfolio monthly returns via matrix multiplication
    portfolio_returns = etf_data.dot(tangency_weights)

    # Annualize the portfolio statistics
    annualized_mean = portfolio_returns.mean() * 12
    annualized_std = portfolio_returns.std() * np.sqrt(12)
    annualized_sharpe = (annualized_mean / annualized_std)

    # Print the results with an optional description
    print(f"\nTangency portfolio results using mean-variance optimization {description}:")
    print(f"Annualized mean = {annualized_mean}")
    print(f"Annualized standard deviation = {annualized_std}")
    print(f"Annualized Sharpe ratio = {annualized_sharpe}")


def ew_portfolio(etf_data, target_mean, description=""):  # Equally-weighted portfolio
    n = etf_data.shape[1]  # number of assets
    ew_weights = np.ones(n) / n  # equally weighted portfolio

    # Determine the scaling factor to achieve the target monthly mean excess return:
    mu = etf_data.mean()  # a Series with asset expected returns (monthly)
    portfolio_returns = np.dot(ew_weights, mu)
    scale = target_mean / portfolio_returns

    # Scale the weights
    target_ew_weights = scale * ew_weights

    # Compute expected return by scaled weight
    target_return = np.dot(etf_data, target_ew_weights)

    # Annualize the portfolio statistics
    annualized_mean = target_return.mean() * 12
    annualized_std = target_return.std() * np.sqrt(12)
    annualized_sharpe = (annualized_mean / annualized_std)

    # Print the results with an optional description
    print(f"\nTangency portfolio results using equal-weight optimization with target mean of {target_mean} {description}:")
    print(f"Annualized mean = {annualized_mean}")
    print(f"Annualized standard deviation = {annualized_std}")
    print(f"Annualized Sharpe ratio = {annualized_sharpe}")


def rp_portfolio(etf_data):
    # Compute variances of each asset
    asset_variances = etf_data.var()
    rp_weights = 1 / asset_variances  # Weight = 1/variance
    # Normalize the weights so that they add up to 1
    rp_weights = rp_weights / rp_weights.sum()
    # Compute the portfolio monthly returns using the RP weights
    portfolio_returns = etf_data.dot(rp_weights)

    # Annualize the portfolio statistics:
    annualized_mean = portfolio_returns.mean() * 12
    annualized_std = portfolio_returns.std() * np.sqrt(12)
    annualized_sharpe = annualized_mean / annualized_std

    # Print out the results
    print("\nTangency portfolio results using equal-weight optimization:")
    print(f"Annualized Mean: {annualized_mean}")
    print(f"Annualized Standard Deviation: {annualized_std}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe}")


if __name__ == '__main__':
    main()
