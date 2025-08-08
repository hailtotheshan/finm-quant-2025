# strategies/mean_reversion.py

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


import pandas as pd
import numpy as np
import uuid
from datetime import datetime

# -------------------------------------------------------------------
# 1) Signal generator
# -------------------------------------------------------------------
def add_bollinger_and_signals(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2,
    price_col: str = "last_price"
) -> pd.DataFrame:
    """
    Compute Bollinger Bands over `price_col` and generate crossover signals.

    Signals:
      +1  price crosses below lower band → enter long
      -1  price crosses above upper band → enter short
       0  price crosses back to mid band → exit

    Returns a new DataFrame (same index as `df`) with exactly:
      ['last_price', 'mid', 'upper', 'lower', 'signal']
    """
    df = df.copy()
    price = df[price_col]

    # rolling mean & std
    mid   = price.rolling(window).mean()
    std   = price.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    # shift for crossover logic
    p0, m0, u0, l0 = price.shift(1), mid.shift(1), upper.shift(1), lower.shift(1)

    # init signal=0
    sig = pd.Series(0, index=df.index)

    # cross BELOW lower → +1
    sig[(p0 >= l0) & (price < lower)] = 1

    # cross ABOVE upper → -1
    sig[(p0 <= u0) & (price > upper)] = -1

    # cross BACK to mid → 0
    sig[
      ((p0 < m0) & (price >= mid)) |
      ((p0 > m0) & (price <= mid))
    ] = 0

    # assemble exactly the five columns run_backtest needs
    out = pd.DataFrame({
      "last_price": price,
      "mid":         mid,
      "upper":       upper,
      "lower":       lower,
      "signal":      sig.astype(int)
    }, index=df.index)

    return out


# -------------------------------------------------------------------
# 2) Backtester
# -------------------------------------------------------------------
def run_backtest_mean_reversion(
    history: pd.DataFrame,
    risk_params: dict
) -> (pd.DataFrame, list, dict):
    """
    Backtest a mean‐reversion Bollinger‐Band strategy.

    Parameters
    ----------
    history : DataFrame
        Must have columns ['last_price','mid','upper','lower','signal'].
        Index must be a DatetimeIndex.
    risk_params : dict
        {
          'symbol':       str,
          'starting_cash': float,
          'max_pos':      int
        }

    Returns
    -------
    signals_df : DataFrame
      Subset of bars where the signal actually changed
      (has columns timestamp, last_price, mid, upper, lower, signal).
    trades_list : list of dicts
      One dict per fill with keys ['id','symbol','side','quantity','type','price','timestamp'].
    metrics_dict : dict
      { 'total_return','max_drawdown','sharpe_ratio' }
    """
    # unpack
    price  = history["last_price"]
    mid    = history["mid"]
    upper  = history["upper"]
    lower  = history["lower"]
    signal = history["signal"].astype(int)

    # build a flat DataFrame with prev‐signal
    df = pd.DataFrame({
      "timestamp":   history.index.tz_localize(None),
      "last_price":  price.values,
      "mid":         mid.values,
      "upper":       upper.values,
      "lower":       lower.values,
      "signal":      signal.values,
    })
    df["signal_prev"] = df["signal"].shift(1).fillna(0).astype(int)

    # keep only actual changes
    signals_df = df[df["signal"] != df["signal_prev"]].copy()

    # risk parameters
    symbol        = risk_params["symbol"]
    starting_cash = float(risk_params["starting_cash"])
    max_pos       = int(risk_params["max_pos"])

    # initialize OMS / Book / Tracker
    oms     = OrderManagementSystem()
    book    = LimitOrderBook(symbol)
    tracker = PositionTracker(starting_cash=starting_cash)
    trades_list = []

    # walk signals → orders → executions → tracker
    for _, row in signals_df.iterrows():
        s, prev = int(row["signal"]), int(row["signal_prev"])
        ts = row["timestamp"].to_pydatetime()

        # map to buy/sell/exit
        if   (s == 1  and prev == 0):   side = "buy"   # enter long
        elif (s == 0  and prev == 1):   side = "sell"  # exit long
        elif (s == -1 and prev == 0):   side = "sell"  # enter short
        elif (s == 0  and prev == -1):  side = "buy"   # exit short
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

        # if no fills, force‐fill at bar price
        if not reports:
            reports = [{
              "order_id":  order.id,
              "symbol":    symbol,
              "side":      side,
              "filled_qty":order.quantity,
              "price":     row["last_price"],
              "timestamp": ts
            }]
            book.cancel_order(order.id)

        # record each execution
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

    # P&L and performance stats
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

    summary    = tracker.get_pnl_summary({symbol: price.iloc[-1]})
    total_ret  = summary["total_pnl"] / starting_cash

    metrics_dict = {
      "total_return":  total_ret,
      "max_drawdown":  max_dd,
      "sharpe_ratio":  sharpe
    }

    # strip the helper column
    return signals_df.drop(columns="signal_prev"), trades_list, metrics_dict



def mean_reversion_performance(
        history: pd.DataFrame,
        trades_list: list,
        starting_cash: float
):
    """
    history: DataFrame indexed by timestamps, must contain
             ['last_price','mid','upper','lower'] columns.
    trades_list: list of dicts with keys
             ['timestamp','side','quantity','price'] (and others).
    starting_cash: float, cash at t=0.
    """

    # 1) Build trades DataFrame
    df_trades = pd.DataFrame(trades_list)
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
    df_trades = df_trades.set_index('timestamp').sort_index()

    # cash_flow per fill: buys are negative, sells positive
    df_trades['cash_flow'] = np.where(
        df_trades['side'] == 'buy',
        -df_trades['quantity'] * df_trades['price'],
        df_trades['quantity'] * df_trades['price']
    )

    # 2) Equity curve & drawdown
    equity = df_trades['cash_flow'].cumsum() + starting_cash
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    # 3) Plotting
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # --- Panel 1: Price & Bollinger Bands + Trade Markers
    hist = history.copy()
    hist.index = pd.to_datetime(hist.index)

    ax0.plot(hist['last_price'], label='Price', color='black')
    ax0.plot(hist['upper'], '--', label='Upper Band', color='gray')
    ax0.plot(hist['mid'], '-', label='Mid Band', color='blue')
    ax0.plot(hist['lower'], '--', label='Lower Band', color='gray')

    buys = df_trades[df_trades['side'] == 'buy']
    sells = df_trades[df_trades['side'] == 'sell']

    ax0.scatter(
        buys.index, buys['price'],
        marker='^', color='green', label='Buys', s=60
    )
    ax0.scatter(
        sells.index, sells['price'],
        marker='v', color='red', label='Sells', s=60
    )

    ax0.set_ylabel('Price')
    ax0.legend(loc='best')
    ax0.set_title('Bollinger Bands & Executions')

    # --- Panel 2: Equity Curve
    ax1.plot(equity, color='blue', lw=1.5)
    ax1.set_ylabel('Equity')
    ax1.grid(True)

    # --- Panel 3: Drawdown
    ax2.plot(drawdown, color='red', lw=1.5)
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Time')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


loader = MarketDataLoader(interval="1d", period="3y")
# Pull the last month's daily bars
df = loader.get_history("9633.HK")

df_with_signals = add_bollinger_and_signals(
    df,
    window=20,
    num_std=2,
)

'''raw = loader.get_history("9633.HK")
df = add_bollinger_and_signals(raw, window=20, num_std=2)

# 1) run backtest
signals_df, trades_list, metrics = run_backtest(df, {
    'symbol': '9633.HK',
    'starting_cash': 100000,
    'max_pos': 10
})

# 2) print results
print('signals:', signals_df)

# build DataFrame
trades_df = pd.DataFrame(trades_list)

# ensure timestamp is a datetime
trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

# pretty-print
print(trades_df.to_string(index=False,
                          columns=['id','symbol','side','quantity','price','timestamp']))

metrics_series = pd.Series(metrics, name='value')
print(metrics_series.to_frame())

# 3) visualize
plot_performance(
    history    = df.set_index(df.index.tz_localize(None)),
    trades_list= trades_list,
    starting_cash= 100000
)
'''