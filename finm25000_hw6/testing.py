# copilot is used for this assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def main():
    # Load and prepare data
    df_spy = pd.read_excel("excel_dataframe.xlsx",
                           sheet_name='Sheet1', header=0, index_col=0).dropna()
    df = df_spy.drop(columns=['SPY'])
    log_returns = np.log(df).diff().iloc[1:]

    # Part 1: Equal-weighted strategy
    print("Part 1: Equal-weighted strategy")
    print("\n" + "=" * 80)
    print("1️⃣ EQUAL-WEIGHTED STRATEGY ANALYSIS")
    print("=" * 80)
    ew_returns, ew_values = simulate_equal_weight_portfolio(log_returns)

    print("Adjusted close price matrix:\n", df)
    print("Daily log returns:\n", log_returns)

    # Part 2: Signal-weighted strategy
    print("\n" + "=" * 80)
    print("2️⃣ SIGNAL-WEIGHTED STRATEGY ANALYSIS")
    print("=" * 80)
    sw_results = signal_weighted_portfolio(df, log_returns)
    sw_returns = sw_results['portfolio_returns']
    sw_values = sw_results['portfolio_values']

    print("Signal strength of signal-weighted portfolio:\n", sw_results['signal_strength'])

    portfolio = pd.DataFrame({
        'Equal-Weight Returns': ew_returns,
        'Equal-Weight Values': ew_values,
        'Signal-Weight Returns': sw_returns,
        'Signal-Weight Values': sw_values
    })
    print("Time series of portfolio returns and value:\n", portfolio)

    # Part 3: Market sensitivity (calculate separate betas for each portfolio)
    print("\n" + "=" * 80)
    print("3️⃣ MARKET SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Create DataFrames containing each portfolio's values along with SPY
    df_ew = pd.concat([pd.Series(ew_values, name='Portfolio'), df_spy['SPY']], axis=1)
    df_sw = pd.concat([pd.Series(sw_values, name='Portfolio'), df_spy['SPY']], axis=1)

    # Calculate betas for both portfolios
    print("\ni. Equal-Weighted Portfolio Beta Analysis")
    ew_beta_results = analyze_market_sensitivity(df_ew, 'Equal-Weighted Portfolio Beta Analysis: ')
    print("\nii. Signal-Weighted Portfolio Beta Analysis")
    sw_beta_results = analyze_market_sensitivity(df_sw, 'Signal-Weighted Portfolio Beta Analysis: ')

    # Part 4: Hedging implementations
    print("\n" + "=" * 80)
    print("4️⃣ HEDGING IMPLEMENTATION")
    print("=" * 80)

    # Equal-weighted hedging
    ew_hedging = implement_hedging_strategy(df_spy, ew_values, ew_beta_results)
    # Signal-weighted hedging (using its own beta)
    sw_hedging = implement_hedging_strategy(df_spy, sw_values, sw_beta_results)

    returns_values = pd.concat([ew_hedging['performance_comparison'], sw_hedging['performance_comparison']], axis=1)
    # Rename specific columns
    returns_values.columns = ['Equal-Weighted Unhedged Value', 'Equal-Weighted Hedged Value',
                              'Equal-Weighted Unhedged Return', 'Equal-Weighted Hedged Return',
                              'Signal-Weighted Unhedged Value', 'Signal-Weighted Hedged Value',
                              'Signal-Weighted Unhedged Return', 'Signal-Weighted Hedged Return']
    print("returns and values of hedged portfolios:\n", returns_values)

    # Part 5: Backtesting
    print("\n" + "=" * 80)
    print("5️⃣ BACKTESTING & PERFORMANCE EVALUATION")
    print("=" * 80)

    # Equal-weighted comparison
    print("\n" + "-" * 50)
    print("EQUAL-WEIGHTED vs HEDGED EQUAL-WEIGHTED")
    print("-" * 50)
    ew_backtest = backtest_performance(
        portfolio_values=ew_values,
        hedged_values=ew_hedging['hedged_values'],
        portfolio_returns=ew_returns,
        hedged_returns=ew_hedging['hedged_returns']
    )

    # Signal-weighted comparison
    print("\n" + "-" * 50)
    print("SIGNAL-WEIGHTED vs HEDGED SIGNAL-WEIGHTED")
    print("-" * 50)
    sw_backtest = backtest_performance(
        portfolio_values=sw_values,
        hedged_values=sw_hedging['hedged_values'],
        portfolio_returns=sw_returns,
        hedged_returns=sw_hedging['hedged_returns']
    )
    print_performance_comparison(ew_backtest, sw_backtest)

    # Plot all four strategies together
    plt.figure(figsize=(14, 8))
    ew_values.plot(label='Equal-Weighted')
    ew_hedging['hedged_values'].plot(label='Hedged Equal-Weighted')
    sw_values.plot(label='Signal-Weighted')
    sw_hedging['hedged_values'].plot(label='Hedged Signal-Weighted')
    plt.title('Comparison of All Strategies')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def backtest_performance(portfolio_values, hedged_values, portfolio_returns, hedged_returns,
                         transaction_cost_equity=0.0005, transaction_cost_spy=0.0003):
    """
    Comprehensive backtesting and performance evaluation for both unhedged and hedged portfolios

    Parameters:
        portfolio_values: Series of unhedged portfolio values
        hedged_values: Series of hedged portfolio values
        portfolio_returns: Series of unhedged daily returns
        hedged_returns: Series of hedged daily returns
        transaction_cost_equity: Cost per equity trade (5bps = 0.0005)
        transaction_cost_spy: Cost per SPY hedge trade (3bps = 0.0003)

    Returns:
        Dictionary containing:
        - metrics_df: DataFrame with all performance metrics
        - plots: Dictionary of matplotlib figures
    """

    """This function does not have take transaction costs correctly because in an unhedged equal weight portfolio all 
    the trades happen on the first day and no trade happen afterward. Thus, in this portfolio there is only one 
    transaction cost that happen on the first. However, in the transaction cost dataframe printed there are 
    transaction cost occur throughout the years. This is because you simply incorrectly  multiplied the net change in 
    portfolio value by transaction cost. This is clearly incorrect because change in portfolio value may not be 
    caused by trading: it also may be caused by stock price change in the portfolio."""


    # Calculate trade amounts for unhedged portfolio (entire portfolio value each day)
    unhedged_trade_amounts = portfolio_values.shift(1).fillna(portfolio_values.iloc[0])
    unhedged_daily_costs = unhedged_trade_amounts * transaction_cost_equity

    # Calculate net values for unhedged portfolio after transaction costs
    unhedged_values_net = [portfolio_values.iloc[0]]
    for i in range(1, len(portfolio_values)):
        cost = unhedged_trade_amounts.iloc[i] * transaction_cost_equity
        net_value = unhedged_values_net[-1] * (1 + portfolio_returns.iloc[i]) - cost
        unhedged_values_net.append(net_value)
    unhedged_values_net = pd.Series(unhedged_values_net, index=portfolio_values.index)
    unhedged_returns_net = unhedged_values_net.pct_change().iloc[1:]

    # Calculate trade amounts for hedged portfolio
    # 1. Equity trade: entire portfolio value each day
    hedged_equity_trade_amounts = hedged_values.shift(1).fillna(hedged_values.iloc[0])
    hedged_equity_costs = hedged_equity_trade_amounts * transaction_cost_equity

    # 2. SPY trade: absolute change in hedge position
    spy_prices = (1 + np.log(hedged_values).diff().fillna(0)).cumprod() * 100
    hedge_ratios = 1.0
    spy_positions = hedge_ratios * (hedged_values / spy_prices)
    spy_trade_amounts = spy_positions.diff().abs().fillna(spy_positions.iloc[0]) * spy_prices
    hedged_spy_costs = spy_trade_amounts * transaction_cost_spy

    # Calculate net values for hedged portfolio after transaction costs
    hedged_values_net = [hedged_values.iloc[0]]
    for i in range(1, len(hedged_values)):
        equity_cost = hedged_equity_trade_amounts.iloc[i] * transaction_cost_equity
        spy_cost = spy_trade_amounts.iloc[i] * transaction_cost_spy
        total_cost = equity_cost + spy_cost
        net_value = hedged_values_net[-1] * (1 + hedged_returns.iloc[i]) - total_cost
        hedged_values_net.append(net_value)
    hedged_values_net = pd.Series(hedged_values_net, index=hedged_values.index)
    hedged_returns_net = hedged_values_net.pct_change().iloc[1:]

    # Create transaction cost DataFrame
    cost_df = pd.DataFrame({
        'Date': portfolio_values.index,
        'Unhedged_Equity_Cost': unhedged_daily_costs,
        'Hedged_Equity_Cost': hedged_equity_costs.reindex(portfolio_values.index, fill_value=0),
        'Hedged_SPY_Cost': hedged_spy_costs.reindex(portfolio_values.index, fill_value=0)
    }).set_index('Date')

    # Add total costs
    cost_df['Hedged_Total_Cost'] = cost_df['Hedged_Equity_Cost'] + cost_df['Hedged_SPY_Cost']
    cost_df['Cumulative_Unhedged_Cost'] = cost_df['Unhedged_Equity_Cost'].cumsum()
    cost_df['Cumulative_Hedged_Cost'] = cost_df['Hedged_Total_Cost'].cumsum()

    # Print transaction cost DataFrame
    print("\n" + "=" * 80)
    print("TRANSACTION COSTS SUMMARY")
    print("=" * 80)
    print(f"Total Unhedged Equity Costs: ${cost_df['Unhedged_Equity_Cost'].sum():.2f}")
    print(f"Total Hedged Equity Costs: ${cost_df['Hedged_Equity_Cost'].sum():.2f}")
    print(f"Total Hedged SPY Costs: ${cost_df['Hedged_SPY_Cost'].sum():.2f}")
    print(f"Total Hedged Costs: ${cost_df['Hedged_Total_Cost'].sum():.2f}")
    print("\nDaily Transaction Costs (first 5 days and last 5 days):")
    print(pd.concat([cost_df.head(), cost_df.tail()]))

    # Calculate all metrics for both portfolios using net values and returns
    results = {}
    for name, returns, values in [('Unhedged', unhedged_returns_net, unhedged_values_net),
                                  ('Hedged', hedged_returns_net, hedged_values_net)]:
        # Basic return statistics
        total_return = values.iloc[-1] / values.iloc[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)

        # Drawdown calculations
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        time_in_drawdown = (drawdown < 0).mean()

        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / annualized_vol
        sortino_ratio = annualized_return / (returns[returns < 0].std() * np.sqrt(252))
        calmar_ratio = annualized_return / abs(max_drawdown)

        # Win/loss metrics
        hit_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean()
        avg_loss = returns[returns < 0].mean()
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        # Tail risk metrics
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Higher moment statistics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Rolling metrics
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)

        results[name] = {
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Hit Rate': hit_rate,
            'Win/Loss Ratio': win_loss_ratio,
            'Time in Drawdown (%)': time_in_drawdown * 100,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Rolling Sharpe': rolling_sharpe,
            'Drawdown': drawdown,
            'Returns': returns,
            'Values': values
        }

    # Create comparison DataFrame
    metrics_df = pd.DataFrame(results).T

    # Generate plots
    plots = {}

    # 1. Rolling Sharpe Ratio
    plt.figure(figsize=(12, 6))
    results['Unhedged']['Rolling Sharpe'].plot(label='Unhedged')
    results['Hedged']['Rolling Sharpe'].plot(label='Hedged')
    plt.title('60-Day Rolling Sharpe Ratio')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    plots['rolling_sharpe'] = plt.gcf()
    plt.show()

    # 2. Return histograms
    plt.figure(figsize=(12, 6))
    plt.hist(results['Unhedged']['Returns'], bins=50, alpha=0.5, label='Unhedged')
    plt.hist(results['Hedged']['Returns'], bins=50, alpha=0.5, label='Hedged')
    plt.title('Return Distributions')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')

    # Add skewness and kurtosis annotations
    for name in ['Unhedged', 'Hedged']:
        plt.annotate(f'{name}\nSkew: {results[name]["Skewness"]:.2f}\nKurtosis: {results[name]["Kurtosis"]:.2f}',
                     xy=(0.05, 0.85 - 0.1 * list(results.keys()).index(name)),
                     xycoords='axes fraction')
    plt.legend()
    plots['return_histograms'] = plt.gcf()
    plt.show()

    # 3. Time in drawdown
    plt.figure(figsize=(12, 6))
    results['Unhedged']['Drawdown'].plot(label='Unhedged')
    results['Hedged']['Drawdown'].plot(label='Hedged')
    plt.title('Portfolio Drawdown Over Time')
    plt.ylabel('Drawdown')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    plots['drawdown'] = plt.gcf()
    plt.show()

    # 4. VaR/CVaR timeline
    plt.figure(figsize=(12, 6))
    results['Unhedged']['Returns'].plot(label='Unhedged Returns')
    plt.axhline(results['Unhedged']['VaR (95%)'], color='red', linestyle='--',
                label='Unhedged VaR (95%)')
    plt.axhline(results['Hedged']['VaR (95%)'], color='blue', linestyle='--',
                label='Hedged VaR (95%)')
    plt.title('Daily Returns with VaR Thresholds')
    plt.ylabel('Daily Returns')
    plt.legend()
    plt.grid(True)
    plots['var_timeline'] = plt.gcf()
    plt.show()

    return {
        'metrics_df': metrics_df,
        'plots': plots
    }


def print_performance_comparison(ew_backtest, sw_backtest):
    """Prints a combined performance comparison of all four portfolio variants"""

    # Extract metrics from both backtests
    ew_metrics = ew_backtest['metrics_df']
    sw_metrics = sw_backtest['metrics_df']

    # Create a new DataFrame with all metrics
    combined_metrics = pd.DataFrame({
        'Equal-Weight Unhedged': ew_metrics.loc['Unhedged'],
        'Equal-Weight Hedged': ew_metrics.loc['Hedged'],
        'Signal-Weight Unhedged': sw_metrics.loc['Unhedged'],
        'Signal-Weight Hedged': sw_metrics.loc['Hedged']
    })

    # Select key metrics for display
    key_metrics = [
        'Annualized Return',
        'Annualized Volatility',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Calmar Ratio',
        'Max Drawdown',
        'Hit Rate',
        'Win/Loss Ratio',
        'Time in Drawdown (%)',
        'VaR (95%)',
        'CVaR (95%)',
        'Skewness',
        'Kurtosis'
    ]

    # Filter and transpose the DataFrame
    display_df = combined_metrics.loc[key_metrics]

    # Apply percentage formatting where appropriate
    percent_metrics = ['Annualized Return', 'Annualized Volatility', 'Hit Rate', 'Time in Drawdown (%)']
    for metric in percent_metrics:
        if metric in display_df.index:
            display_df.loc[metric] = display_df.loc[metric].apply(lambda x: f"{x:.2%}")

    # Format other numeric columns
    float_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown',
                     'Win/Loss Ratio', 'VaR (95%)', 'CVaR (95%)', 'Skewness', 'Kurtosis']
    for metric in float_metrics:
        if metric in display_df.index:
            display_df.loc[metric] = display_df.loc[metric].apply(lambda x: f"{x:.2f}")

    # Print the formatted DataFrame
    print("\nCOMBINED PERFORMANCE METRICS")
    print("=" * 80)
    print(display_df)

def implement_hedging_strategy(df_spy, portfolio_values, beta_results):
    """
    Implements a dynamic hedging strategy using rolling beta to short SPY.

    Parameters:
    df_spy (DataFrame): Contains SPY prices and portfolio asset prices
    portfolio_values (Series): Time series of unhedged portfolio values
    beta_results (DataFrame): Contains rolling beta coefficients from analyze_market_sensitivity()

    Returns:
    dict: Contains hedge ratios, short positions, hedged returns and values
    """
    # Prepare market returns (SPY)
    spy_prices = df_spy['SPY']
    spy_returns = np.log(spy_prices).diff().iloc[1:]

    # Align all time series
    common_index = portfolio_values.index.intersection(beta_results.index)
    portfolio_values = portfolio_values[common_index]
    spy_prices = spy_prices[common_index]
    spy_returns = spy_returns[common_index]
    betas = beta_results['Beta'][common_index]

    # Initialize containers
    hedge_ratios = pd.Series(index=common_index, dtype=float)
    short_positions = pd.Series(index=common_index, dtype=float)
    hedged_returns = pd.Series(index=common_index, dtype=float)
    hedged_values = pd.Series(index=common_index, dtype=float)

    # Initial values
    hedged_values.iloc[0] = portfolio_values.iloc[0]

    # Calculate daily hedge metrics
    for i in range(1, len(common_index)):
        current_date = common_index[i]
        prev_date = common_index[i - 1]

        # Current hedge ratio
        h_t = betas.loc[current_date] * (portfolio_values.loc[prev_date] / spy_prices.loc[prev_date])
        hedge_ratios.loc[current_date] = h_t

        # Short position in SPY (negative value)
        short_positions.loc[current_date] = -h_t * spy_prices.loc[current_date]

        # Portfolio and market returns
        r_p = (portfolio_values.loc[current_date] / portfolio_values.loc[prev_date]) - 1
        r_mkt = spy_returns.loc[current_date]

        # Hedged return
        r_hp = r_p - betas.loc[current_date] * r_mkt
        hedged_returns.loc[current_date] = r_hp

        # Update hedged portfolio value
        hedged_values.loc[current_date] = hedged_values.loc[prev_date] * (1 + r_hp)

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Unhedged_Value': portfolio_values,
        'Hedged_Value': hedged_values,
        'Unhedged_Return': portfolio_values.pct_change(),
        'Hedged_Return': hedged_returns
    })

    return {
        'hedge_ratios': hedge_ratios,
        'short_positions': short_positions,
        'hedged_returns': hedged_returns,
        'hedged_values': hedged_values,
        'performance_comparison': comparison
    }


def track_portfolio_beta(df_spy, window=63):  # 63 trading days ≈ 3 months
    """
    Perform rolling regression of portfolio returns against SPY returns
    and track the portfolio's beta over time.

    Parameters:
    df_spy (DataFrame): Contains both portfolio assets and SPY prices
    window (int): Rolling window size in trading days

    Returns:
    DataFrame: Contains alpha and beta coefficients
    """
    # Prepare returns
    portfolio_prices = df_spy.drop(columns=['SPY'])
    portfolio_returns = np.log(portfolio_prices).diff().iloc[1:]
    spy_returns = np.log(df_spy['SPY']).diff().iloc[1:]

    # Initialize storage for coefficients
    dates = []
    alphas = []
    betas = []

    # Perform rolling regression
    for i in range(window, len(portfolio_returns)):
        current_window = portfolio_returns.iloc[i - window:i]
        current_spy = spy_returns.iloc[i - window:i]

        # Add constant for intercept (alpha)
        X = sm.add_constant(current_spy)
        y = current_window.mean(axis=1)  # Equal-weighted portfolio return

        model = sm.OLS(y, X).fit()

        dates.append(portfolio_returns.index[i])
        alphas.append(model.params.iloc[0])  # Explicit position-based indexing
        betas.append(model.params.iloc[1])  # Explicit position-based indexing

    # Create results DataFrame
    results = pd.DataFrame({'Date': dates, 'Alpha': alphas, 'Beta': betas})
    results.set_index('Date', inplace=True)

    return results


def analyze_market_sensitivity(df_spy, description=''):
    """
    Full analysis of portfolio's market sensitivity including:
    - Rolling beta chart
    - Regression coefficients
    - Market sensitivity commentary
    """
    # Get rolling coefficients
    coefficients = track_portfolio_beta(df_spy)

    # 1. Plot beta over time
    plt.figure(figsize=(12, 6))
    coefficients['Beta'].plot(title=description + 'Rolling Portfolio Beta (3-month window)')
    plt.axhline(y=1, color='r', linestyle='--', label='Market Beta (SPY=1)')
    plt.ylabel('Beta')
    plt.xlabel('Date')
    plt.legend()
    plt.grid()
    plt.show()

    # 2. Print regression coefficients summary
    print("\nRegression Coefficients Summary:")
    print(f"Average Alpha: {coefficients['Alpha'].mean():.6f}")
    print(f"Average Beta: {coefficients['Beta'].mean():.4f}")
    print(f"Minimum Beta: {coefficients['Beta'].min():.4f}")
    print(f"Maximum Beta: {coefficients['Beta'].max():.4f}")

    # 3. Generate commentary
    print("\nMarket Sensitivity Commentary:")
    avg_beta = coefficients['Beta'].mean()

    if avg_beta > 1.2:
        print("-> The portfolio is highly sensitive to market movements (aggressive)")
    elif avg_beta > 0.8:
        print("-> The portfolio moves with the market (moderate sensitivity)")
    elif avg_beta > 0.5:
        print("-> The portfolio is less sensitive than the market (defensive)")
    else:
        print("-> The portfolio shows very low correlation with market movements")

    print(f"-> Beta ranged from {coefficients['Beta'].min():.2f} to {coefficients['Beta'].max():.2f} during the period")

    if (coefficients['Beta'].std() > 0.3):
        print("-> Note: Significant variation in beta over time indicates changing market sensitivity")

    return coefficients


def simulate_equal_weight_portfolio(log_returns, initial_investment=10000):
    """
    Simulates an equal-weight portfolio based on log returns and plots its growth over time.

    Parameters:
    - log_returns: DataFrame of log returns for each asset
    - initial_investment: Starting portfolio value (default $10,000)

    Returns:
    - Tuple of (portfolio_returns, portfolio_values)
    - Plots portfolio growth over time
    """
    # 1. Create equal weights (all assets weighted equally)
    n_assets = log_returns.shape[1]
    weights = np.array([1 / n_assets] * n_assets)

    # 2. Compute daily portfolio returns (simple returns, not log returns)
    # Convert log returns to simple returns first
    simple_returns = np.exp(log_returns) - 1
    portfolio_returns = simple_returns.dot(weights)

    # 3. Simulate portfolio value over time
    portfolio_values = [initial_investment]
    for ret in portfolio_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    # Convert to Series/DataFrame for easier handling
    portfolio_values = pd.Series(portfolio_values[1:], index=log_returns.index)
    portfolio_returns = pd.Series(portfolio_returns, index=log_returns.index)

    # 4. Plot portfolio growth
    plt.figure(figsize=(12, 6))
    portfolio_values.plot(linewidth=2)
    plt.title(f"Equal-Weight Portfolio Growth\nInitial Investment: ${initial_investment:,.0f}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)

    # Add final value annotation
    final_value = portfolio_values.iloc[-1]
    plt.annotate(f'Final Value: ${final_value:,.2f}',
                 xy=(portfolio_values.index[-1], final_value),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->'))

    plt.show()

    return portfolio_returns, portfolio_values


def signal_weighted_portfolio(df, log_returns, initial_investment=10000,
                              sma_window=20, ema_window=20, rsi_window=14,
                              macd_fast=12, macd_slow=26, macd_signal=9,
                              atr_window=14):
    """
    Computes technical indicators, generates signals, and simulates a signal-weighted portfolio.
    """
    # Initialize storage
    indicators = {}
    signals = pd.DataFrame(index=df.index, columns=df.columns)
    signal_strength = pd.DataFrame(0, index=df.index, columns=df.columns)
    base_weights = pd.DataFrame(1 / len(df.columns), index=df.index, columns=df.columns)

    for ticker in df.columns:
        # Compute indicators
        prices = df[ticker]

        # SMA
        sma = prices.rolling(window=sma_window).mean()

        # EMA
        ema = prices.ewm(span=ema_window, adjust=False).mean()

        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = prices.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal_line = macd.ewm(span=macd_signal, adjust=False).mean()

        # ATR
        high = df[ticker]  # Using close as proxy for high/low
        low = df[ticker]
        tr1 = high - low
        tr2 = (high - prices.shift(1)).abs()
        tr3 = (low - prices.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_window).mean()

        # Store indicators
        indicators[ticker] = {
            'SMA': sma,
            'EMA': ema,
            'RSI': rsi,
            'MACD': macd,
            'MACD_signal': macd_signal_line,
            'ATR': atr
        }

        # Generate signals
        long_condition = (rsi < 30) & (macd > macd_signal_line)
        short_condition = (rsi > 70) & (macd < macd_signal_line)

        signals[ticker] = np.where(long_condition, 'Buy',
                                   np.where(short_condition, 'Sell', 'Neutral'))

        # Calculate cumulative signal strength
        cumulative_signal = pd.Series(0, index=signals.index)
        current_value = 0

        for i in range(len(signals)):
            # Convert signal to numerical value
            signal_num = 0
            if signals[ticker].iloc[i] == 'Buy':
                signal_num = 1
            elif signals[ticker].iloc[i] == 'Sell':
                signal_num = -1

            if signal_num != 0:
                current_value += signal_num
            cumulative_signal.iloc[i] = current_value

        signal_strength[ticker] = cumulative_signal

    # Derive signal-based weights
    adjusted_weights = base_weights * (1 + signal_strength)
    # adjusted_weights = adjusted_weights.div(adjusted_weights.sum(axis=1), axis=0)

    # Compute portfolio returns and values
    simple_returns = np.exp(log_returns) - 1
    portfolio_returns = (simple_returns * adjusted_weights.shift(1)).sum(axis=1)

    portfolio_values = pd.Series(index=log_returns.index, dtype=float)
    if len(portfolio_returns) > 0:
        portfolio_values = initial_investment * (1 + portfolio_returns).cumprod()

    # Plot comparison
    plt.figure(figsize=(12, 6))

    # Equal-weight portfolio for comparison
    equal_weights = pd.DataFrame(1 / len(df.columns), index=df.index, columns=df.columns)
    equal_portfolio_returns = (simple_returns * equal_weights.shift(1)).sum(axis=1)

    if len(equal_portfolio_returns) > 0:
        equal_values = initial_investment * (1 + equal_portfolio_returns).cumprod()

        # Plot both strategies
        portfolio_values.plot(linewidth=2, label='Signal-Weighted Portfolio')
        equal_values.plot(linewidth=2, label='Equal-Weight Portfolio')

        plt.title(f"Portfolio Growth Comparison\nInitial Investment: ${initial_investment:,.0f}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.legend()

        # Add final value annotations
        final_signal = portfolio_values.iloc[-1]
        final_equal = equal_values.iloc[-1]

        plt.annotate(f'Signal: ${final_signal:,.2f}',
                     xy=(portfolio_values.index[-1], final_signal),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.annotate(f'Equal: ${final_equal:,.2f}',
                     xy=(equal_values.index[-1], final_equal),
                     xytext=(10, -20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))

        plt.show()

    return {
        'indicators': indicators,
        'signals': signals,
        'signal_strength': signal_strength,
        'adjusted_weights': adjusted_weights,
        'portfolio_returns': portfolio_returns,
        'portfolio_values': portfolio_values,
        'equal_values': equal_values if len(equal_portfolio_returns) > 0 else None
    }


if __name__ == "__main__":
    main()
