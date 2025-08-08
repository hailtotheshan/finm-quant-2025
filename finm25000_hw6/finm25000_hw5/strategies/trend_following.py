import pandas as pd
from finm25000_hw5.order import Order
from finm25000_hw5.oms import OrderManagementSystem
from finm25000_hw5.order_book import LimitOrderBook
from finm25000_hw5.position_tracker import PositionTracker
from finm25000_hw5.market_data_loader import MarketDataLoader
import uuid
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def add_moving_average_signals(
    history: pd.DataFrame,
    short_win: int,
    long_win: int
) -> pd.DataFrame:
    """
    Generate crossover signals based on short- and long-term moving averages.

    Signals:
      +1  short MA crosses above long MA → enter long
      -1  short MA crosses below long MA → enter short
       0  otherwise (flat/exit)

    Returns a DataFrame (same index) with exactly these columns:
      ['last_price', 'mid', 'upper', 'lower', 'signal']
    ready to be passed into run_backtest().
    mid/upper/lower are simply placeholders.
    """
    df = history.copy()

    # 1) Compute MAs
    df["ma_short"] = df["last_price"].rolling(short_win).mean()
    df["ma_long"]  = df["last_price"].rolling(long_win).mean()

    # 2) Placeholders for compatibility
    df["mid"]   = df["ma_long"]
    df["upper"] = np.nan
    df["lower"] = np.nan

    # 3) Crossover logic
    prev_s = df["ma_short"].shift(1)
    prev_l = df["ma_long"].shift(1)

    buy_signals  = (df["ma_short"] > df["ma_long"]) & (prev_s <= prev_l)
    sell_signals = (df["ma_short"] < df["ma_long"]) & (prev_s >= prev_l)

    df["signal"] = 0
    df.loc[buy_signals,  "signal"] =  1
    df.loc[sell_signals, "signal"] = -1

    # 4) Return only the 5 columns run_backtest needs
    return df[["last_price", "mid", "upper", "lower", "signal"]]

def run_backtest_trend_following(
    history: pd.DataFrame,
    risk_params: dict
) -> (pd.DataFrame, list, dict):
    """
    Backtest any strategy that yields a 'signal' series in {-1,0,1}.

    history must have columns:
      ['last_price', 'mid', 'upper', 'lower', 'signal']
    Index must be a pandas DatetimeIndex.

    risk_params:
      {
        'symbol':        str,
        'starting_cash': float,
        'max_pos':       int
      }

    Returns:
      signals_df: DataFrame of bars where signal actually changed
      trades_list: list of execution dicts
      metrics_dict: {total_return, max_drawdown, sharpe_ratio}
    """
    # ---- unpack series ----
    price  = history["last_price"]
    mid    = history["mid"]
    upper  = history["upper"]
    lower  = history["lower"]
    signal = history["signal"].astype(int)

    # ---- build flat df & detect changes ----
    df = pd.DataFrame({
        "timestamp":  history.index.tz_localize(None),
        "last_price": price.values,
        "mid":        mid.values,
        "upper":      upper.values,
        "lower":      lower.values,
        "signal":     signal.values,
    })
    df["signal_prev"] = df["signal"].shift(1).fillna(0).astype(int)
    signals_df = df[df["signal"] != df["signal_prev"]].copy()

    # ---- init OMS / Book / Tracker ----
    symbol        = risk_params["symbol"]
    starting_cash = float(risk_params["starting_cash"])
    max_pos       = int(risk_params["max_pos"])

    oms     = OrderManagementSystem()
    book    = LimitOrderBook(symbol)
    tracker = PositionTracker(starting_cash=starting_cash)
    trades_list = []

    # ---- walk the signals and fire orders ----
    for _, row in signals_df.iterrows():
        s, prev = int(row["signal"]), int(row["signal_prev"])
        ts = row["timestamp"].to_pydatetime()

        # entry/exit logic
        if   (s == 1  and prev == 0):  side = "buy"   # enter long
        elif (s == 0  and prev == 1):  side = "sell"  # exit long
        elif (s == -1 and prev == 0):  side = "sell"  # enter short
        elif (s == 0  and prev == -1): side = "buy"   # exit short
        else:
            continue

        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=max_pos,
            type="limit",
            price=row["last_price"],
            timestamp=ts
        )

        oms.new_order(order)
        reports = book.add_order(order)

        # force‐fill if no match
        if not reports:
            reports = [{
                "order_id":  order.id,
                "symbol":    order.symbol,
                "side":      order.side,
                "filled_qty":order.quantity,
                "price":     row["last_price"],
                "timestamp": ts
            }]
            book.cancel_order(order.id)

        # unify timestamps & record
        for rpt in reports:
            rpt["timestamp"] = ts
            tracker.update(rpt)
            trades_list.append({
                "id":        rpt["order_id"],
                "symbol":    rpt["symbol"],
                "side":      rpt["side"],
                "quantity":  rpt["filled_qty"],
                "type":      rpt.get("type","limit"),
                "price":     rpt["price"],
                "timestamp": rpt["timestamp"]
            })

    # ---- performance stats ----
    blot = tracker.get_blotter()
    if "cash_flow" not in blot.columns:
        blot["cash_flow"] = np.where(
            blot["side"] == "buy",
            - blot["quantity"] * blot["price"],
            blot["quantity"] * blot["price"]
        )
    if "timestamp" in blot.columns:
        blot = blot.set_index("timestamp")
    blot = blot.sort_index()

    equity      = blot["cash_flow"].cumsum() + starting_cash
    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_dd      = drawdown.min()

    rets   = equity.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() else 0

    summary   = tracker.get_pnl_summary({symbol: price.iloc[-1]})
    total_ret = summary["total_pnl"] / starting_cash

    metrics_dict = {
        "total_return":  total_ret,
        "max_drawdown":  max_dd,
        "sharpe_ratio":  sharpe
    }

    return signals_df.drop(columns="signal_prev"), trades_list, metrics_dict



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def trend_following_performance(
    history: pd.DataFrame,
    signals_df: pd.DataFrame,
    trades_list: list,
    metrics: dict,
    starting_cash: float
):
    """
    Plots price, trades, equity curve, drawdown, and prints performance metrics.

    Parameters
    ----------
    history : pd.DataFrame
        Full history with columns ['last_price','mid','upper','lower','signal'].
        Index must be pd.DatetimeIndex.
    signals_df : pd.DataFrame
        Subset of bars where signal changed, as returned by run_backtest.
        Must include 'timestamp','signal'.
    trades_list : list of dict
        Execution reports returned by run_backtest().
        Each dict must have keys:
          ['order_id','symbol','side','quantity','price','timestamp']
    metrics : dict
        {'total_return', 'max_drawdown', 'sharpe_ratio'}
    starting_cash : float
        The same starting_cash you used for the backtest.
    """
    # 1) Prepare trades DataFrame
    trades = pd.DataFrame(trades_list).copy()
    # parse and tag as UTC so it lines up with history.index
    trades['timestamp'] = pd.to_datetime(trades['timestamp'], utc=True)
    trades = trades.set_index('timestamp').sort_index()

    # compute cash_flow per fill
    trades['cash_flow'] = np.where(
        trades['side'] == 'buy',
        -trades['quantity'] * trades['price'],
        trades['quantity'] * trades['price']
    )

    # 2) Build equity curve at fill times
    fill_equity = trades['cash_flow'].cumsum() + starting_cash

    # make fills tz-aware in UTC
    if fill_equity.index.tz is None:
        fill_equity = fill_equity.tz_localize('UTC')

    # 3) Re‐index equity curve to full history for smooth plotting
    equity = fill_equity.reindex(history.index, method='ffill').fillna(starting_cash)

    # 4) Compute drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    # 5) Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 5.1 Price + Bands + Signal arrows
    ax0 = axes[0]
    ax0.plot(history.index, history['last_price'], label='Price', color='black')
    if 'mid' in history.columns:
        ax0.plot(history.index, history['mid'], label='Mid MA', color='gray', linestyle='--')
    if 'upper' in history.columns:
        ax0.plot(history.index, history['upper'], label='Upper', color='red', alpha=0.5)
    if 'lower' in history.columns:
        ax0.plot(history.index, history['lower'], label='Lower', color='green', alpha=0.5)

    # Mark actual trade executions
    buys  = trades[trades['side']=='buy']
    sells = trades[trades['side']=='sell']
    ax0.scatter(buys.index,  buys['price'],  marker='^', color='green', s=100, label='Buy')
    ax0.scatter(sells.index, sells['price'], marker='v', color='red',   s=100, label='Sell')

    ax0.set_title('Price, Bands & Trades')
    ax0.legend(loc='upper left')

    # 5.2 Equity Curve
    ax1 = axes[1]
    ax1.plot(equity.index, equity.values, label='Equity', color='blue')
    ax1.axhline(starting_cash, color='gray', linestyle='--', label='Start Cash')
    ax1.set_title('Equity Curve')
    ax1.legend(loc='upper left')

    # 5.3 Drawdown
    ax2 = axes[2]
    ax2.plot(drawdown.index, drawdown.values, label='Drawdown', color='magenta')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # 6) Print metrics
    print(f"Total Return : {metrics['total_return']:.2%}")
    print(f"Max Drawdown : {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio : {metrics['sharpe_ratio']:.2f}")



'''loader = MarketDataLoader(interval="1d", period="3y")
# Pull the last month's daily bars
df = loader.get_history("9633.HK")

# 1) build your signals
sig_df = add_moving_average_signals(df, short_win=10, long_win=50)

# 2) backtest
signals, trades, metrics = run_backtest(
    history=sig_df,
    risk_params={
      "symbol": "AAPL",
      "starting_cash": 100_000,
      "max_pos": 10
    }
)

# 2) print results
print('signals:', signals)

# build DataFrame
trades_df = pd.DataFrame(trades)

# ensure timestamp is a datetime
trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

# pretty-print
print(trades_df.to_string(index=False,
                          columns=['id','symbol','side','quantity','price','timestamp']))

metrics_series = pd.Series(metrics, name='value')
print(metrics_series.to_frame())

# 3) Visualize
visualize_performance(
    history=df,
    signals_df=signals,
    trades_list=trades,
    metrics=metrics,
    starting_cash=100_000
)'''