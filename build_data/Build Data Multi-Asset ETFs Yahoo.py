import yfinance as yf
import pandas as pd


def get_hk_stock_monthly_returns(tickers, start_date, end_date):
    """
    Downloads historical data for the specified Hong Kong stocks, calculates monthly returns,
    and returns a DataFrame where each column is a stock ticker and the index is each month.

    Parameters:
        tickers (list): A list of stock ticker strings.
                        (E.g., ['0005.HK', '0939.HK', '0700.HK'])
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame: A DataFrame with the month-end dates as the index and each column representing
                   the monthly return (percentage change) of the given stock.
    """
    # Download monthly data for the tickers
    # Setting interval='1mo' gives end-of-month data
    prices = yf.download(tickers, start=start_date, end=end_date, interval='1mo', auto_adjust=True)

    # When auto_adjust=True, the 'Close' column is already adjusted for splits/dividends.
    # Check for multi-ticker data. If multiple tickers, the DataFrame's columns may be in a MultiIndex.
    if isinstance(prices.columns, pd.MultiIndex):
        # We assume the field 'Close' exists; adjust if needed
        adj_prices = prices['Close']
    else:
        # For a single ticker, prices will be a DataFrame with standard columns.
        adj_prices = prices[['Close']]

    # Compute monthly returns as percentage change.
    # The pct_change() function computes (current_value/previous_value - 1)
    monthly_returns = adj_prices.pct_change().dropna()

    # Optionally convert the index to a PeriodIndex reflecting months—for instance, to "2025-06"
    # Here, we convert it to monthly timestamps using Period and then back to timestamp.
    monthly_returns.index = monthly_returns.index.to_period('M').to_timestamp()

    return monthly_returns


# Example usage:
if __name__ == "__main__":
    # List of Hong Kong stocks (remember to include '.HK' if needed)
    hk_stocks = [
        '6862.HK', '2015.HK', '3690.HK', '3988.HK',
        '0388.HK', '1398.HK', '0941.HK', '1211.HK', '1299.HK',
        '9992.HK', '1357.HK', '0005.HK', '2331.HK', '2276.HK',
        '1810.HK', '9626.HK', '9633.HK', '9988.HK', '0700.HK',
        '0133.HK', '0806.HK', '1788.HK', '3037.HK', '1375.HK'
    ]

    start_date = "2020-01-01"
    end_date = "2023-01-01"
    monthly_returns_df = get_hk_stock_monthly_returns(hk_stocks, start_date, end_date)
    print(monthly_returns_df)
