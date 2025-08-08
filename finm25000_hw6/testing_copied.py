# copilot is used for this assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


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

    # Part 5: Backtesting & Performance Evaluation
    print("\n" + "=" * 80)
    print("5️⃣ BACKTESTING & PERFORMANCE EVALUATION")
    print("=" * 80)

    # 1) Build weight matrices
    n_stocks = log_returns.shape[1]
    ew_weights = pd.DataFrame(1.0 / n_stocks,
                              index=log_returns.index,
                              columns=log_returns.columns)

    sw_weights = sw_results['adjusted_weights'] \
        .reindex_like(log_returns).fillna(0)

    # 2) Pull SPY prices and hedge ratios
    spy_prices = df_spy['SPY']  # series of SPY adj. closes
    ew_hedge_ratio = ew_hedging['hedge_ratios']  # daily h_t for EW
    sw_hedge_ratio = sw_hedging['hedge_ratios']  # daily h_t for SW

    # 3) Run backtests
    ew_bt = backtest(
        weights=ew_weights,
        gross_returns=ew_returns,
        spy_prices=spy_prices,
        hedge_ratios=ew_hedge_ratio,
        initial_capital=10_000
    )
    sw_bt = backtest(
        weights=sw_weights,
        gross_returns=sw_returns,
        spy_prices=spy_prices,
        hedge_ratios=sw_hedge_ratio,
        initial_capital=10_000
    )

    print("=== Equal-Weighted Backtest Metrics ===")
    print(ew_bt["metrics"])

    print("\n=== Signal-Weighted Backtest Metrics ===")
    print(sw_bt["metrics"])

    # (Optional) Plot post‐cost portfolio values
    plt.figure(figsize=(12, 6))
    ew_bt['net_values'].plot(label='EW (net)')
    sw_bt['net_values'].plot(label='SW (net)')
    plt.title("Net Portfolio Value (after costs & hedging)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_comparison(ew_bt, sw_bt)


def plot_comparison(ew_bt, sw_bt, hist_bins=50, rolling_window=60):
    # 1) Extract the four net‐return series
    series = {
        "EW Unhedged": ew_bt["net_returns"],
        "EW Hedged":   ew_bt["hedged_net_returns"],
        "SW Unhedged": sw_bt["net_returns"],
        "SW Hedged":   sw_bt["hedged_net_returns"]
    }
    # 2) Extract the four value series (for drawdown)
    values = {
        "EW Unhedged": ew_bt["net_values"],
        "EW Hedged":   ew_bt["hedged_net_values"],
        "SW Unhedged": sw_bt["net_values"],
        "SW Hedged":   sw_bt["hedged_net_values"]
    }

    # --- FIGURE 1: Rolling Sharpe Ratio ---
    plt.figure(figsize=(12, 5))
    for name, bt in [("EW", ew_bt), ("SW", sw_bt)]:
        df = bt["rolling_sharpe"]
        # plot the NET rolling sharpe
        plt.plot(df.index, df["Unhedged Net"], label=f"{name} Unhedged")
        plt.plot(df.index, df["Hedged Net"],   label=f"{name} Hedged", linestyle="--")
    plt.title(f"{rolling_window}‐Day Rolling Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- FIGURE 2: Return Histograms w/ Skew & Kurtosis ---
    plt.figure(figsize=(12, 6))
    for name, r in series.items():
        sns.histplot(r.dropna(), bins=hist_bins, stat="density",
                     label=f"{name} (skew={r.skew():+.2f}, kurt={r.kurt():.1f})",
                     alpha=0.4, element="step")
    plt.title("Return Distributions")
    plt.xlabel("Daily Net Return")
    plt.legend()
    plt.show()

    # --- FIGURE 3: Drawdown Over Time ---
    plt.figure(figsize=(12, 5))
    for name, vals in values.items():
        dd = vals / vals.cummax() - 1
        plt.plot(dd.index, dd, label=name)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.title("Drawdown Over Time")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- FIGURE 4: Rolling VaR & CVaR Timeline ---
    # compute rolling VaR(95%) & CVaR(95%) for each series
    def rolling_cvar(x, q=0.05):
        # for each window, take the mean of the worst q‐quantile
        return x.rolling(rolling_window).apply(
            lambda w: w[w < np.quantile(w, q)].mean(), raw=False
        )

    plt.figure(figsize=(12, 6))
    for name, r in series.items():
        var = r.rolling(rolling_window).quantile(0.05)
        cvar = rolling_cvar(r, 0.05)
        plt.plot(var.index, var, label=f"{name} VaR95", linestyle="-")
        plt.plot(cvar.index, cvar, label=f"{name} CVaR95", linestyle="--")

    plt.title(f"{rolling_window}‐Day Rolling VaR(95%) & CVaR(95%)")
    plt.ylabel("Loss Level")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()


def backtest(
    weights: pd.DataFrame,
    gross_returns: pd.Series,
    spy_prices: pd.Series,
    hedge_ratios: pd.Series,
    initial_capital: float = 10_000,
    transaction_cost_equity: float = 0.0005,
    transaction_cost_spy: float = 0.0003
) -> dict:
    # 0) turn your input log‐r into simple returns
    simple_r = np.exp(gross_returns.fillna(0)) - 1
    simple_r.iloc[0] = 0.0

    # 1) unhedged gross values
    gross_values = initial_capital * (1 + simple_r).cumprod()
    gross_values.iloc[0] = initial_capital

    # 2) equity tx costs (exactly as before)
    prev_uneq = gross_values.shift(1).fillna(initial_capital)
    dW        = weights.diff().abs().fillna(0)
    eq_notnl  = dW.multiply(prev_uneq, axis=0)
    eq_cost   = transaction_cost_equity * eq_notnl.sum(axis=1)

    # 3) SPY tx costs — but cap & lag your hedge ratio to [–1, +1]
    h_lag     = hedge_ratios.shift(1).fillna(0).clip(-1, 1)
    spy_prev  = spy_prices.shift(1).fillna(spy_prices.iloc[0])
    dH        = h_lag.diff().abs().fillna(0)
    spy_notnl = dH * spy_prev
    spy_cost  = transaction_cost_spy * spy_notnl

    tx = pd.DataFrame({
        "equity_costs": eq_cost,
        "spy_costs":    spy_cost
    }).fillna(0)
    tx["total_costs"] = tx.sum(axis=1)

    # 4) unhedged net returns & values
    net_r      = simple_r - tx["total_costs"].div(prev_uneq)
    net_r.iloc[0] = 0.0
    net_values = initial_capital * (1 + net_r).cumprod()
    net_values.iloc[0] = initial_capital

    # 5) build hedged‐gross returns
    spy_r        = spy_prices.pct_change().fillna(0)
    hgross_r     = simple_r - h_lag * spy_r

    # if you ever flip the value negative or ≤ –100%, clamp it just above –1
    hgross_r = hgross_r.clip(lower=-0.9999999)

    hgross_vals  = initial_capital * (1 + hgross_r).cumprod()
    hgross_vals.iloc[0] = initial_capital

    # 6) hedged net returns *only* when prior hedged value > 0
    prev_hedge = hgross_vals.shift(1).fillna(initial_capital)
    prev_hedge_safe = prev_hedge.where(prev_hedge > 0, np.nan)

    hnet_r = hgross_r - tx["total_costs"].div(prev_hedge_safe)
    hnet_r = hnet_r.fillna(hgross_r)   # fall back to gross if denom was zero/neg
    hnet_r.iloc[0] = 0.0

    hnet_vals = initial_capital * (1 + hnet_r).cumprod()
    hnet_vals.iloc[0] = initial_capital

    # 7) metrics helper that filters out any weird (–1 or Inf) days
    def performance_metrics(returns: pd.Series, values: pd.Series):
        mask = (
            returns.replace([np.inf, -np.inf], np.nan)
                   .gt(-1)  &
            values.replace([np.inf, -np.inf], np.nan)
                  .gt(0)
        )
        r = returns.loc[mask]
        v = values.loc[mask]

        ann_ret = (v.iloc[-1] / v.iloc[0]) ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        sr      = ann_ret / ann_vol

        neg     = r[r < 0]
        sortino = ann_ret / (neg.std() * np.sqrt(252)) if len(neg) else np.nan

        dd      = v / v.cummax() - 1
        max_dd  = dd.min()
        calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        roll_sr = (r.rolling(60).mean() * np.sqrt(252)) \
                  / r.rolling(60).std()

        return pd.Series({
            "Annualized Return":      ann_ret,
            "Annualized Volatility":  ann_vol,
            "Sharpe Ratio":           sr,
            "Sortino Ratio":          sortino,
            "Calmar Ratio":           calmar,
            "Max Drawdown":           max_dd,
            "Hit Rate":               (r > 0).mean(),
            "Win/Loss Ratio":         r[r>0].mean() / -r[r<0].mean(),
            "Time in Drawdown":       (dd < 0).mean(),
            "VaR (95%)":              r.quantile(0.05),
            "CVaR (95%)":             r[r < r.quantile(0.05)].mean(),
            "Skewness":               r.skew(),
            "Kurtosis":               r.kurt()
        }), roll_sr

    # 8) compute all four sets of metrics
    ug_m, ug_sr   = performance_metrics(simple_r,  gross_values)
    un_m, un_sr   = performance_metrics(net_r,     net_values)
    hg_m, hg_sr   = performance_metrics(hgross_r,  hgross_vals)
    hn_m, hn_sr   = performance_metrics(hnet_r,    hnet_vals)

    metrics = pd.DataFrame({
        "Unhedged (Gross)": ug_m,
        "Unhedged (Net)":   un_m,
        "Hedged (Gross)":   hg_m,
        "Hedged (Net)":     hn_m
    })

    rolling_sharpe = pd.DataFrame({
        "Unhedged Gross": ug_sr,
        "Unhedged Net":   un_sr,
        "Hedged Gross":   hg_sr,
        "Hedged Net":     hn_sr
    })

    return {
        "gross_returns":        simple_r,
        "gross_values":         gross_values,
        "net_returns":          net_r,
        "net_values":           net_values,
        "hedged_gross_returns": hgross_r,
        "hedged_gross_values":  hgross_vals,
        "hedged_net_returns":   hnet_r,
        "hedged_net_values":    hnet_vals,
        "transaction_costs":    tx,
        "metrics":              metrics,
        "rolling_sharpe":       rolling_sharpe
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
