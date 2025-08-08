import pandas as pd
import numpy as np
from finm25000_hw5.order import Order
from finm25000_hw5.oms import OrderManagementSystem
from finm25000_hw5.order_book import LimitOrderBook
from finm25000_hw5.position_tracker import PositionTracker
from finm25000_hw5.market_data_loader import MarketDataLoader
import uuid
from datetime import datetime
import matplotlib.pyplot as plt


def generate_arbitrage_signals(hist1, hist2, threshold):
    df = pd.DataFrame({
        "price1": hist1["last_price"],
        "price2": hist2["last_price"]
    }).dropna()

    # Hedge ratio
    beta = np.polyfit(df["price2"], df["price1"], 1)[0]
    df["spread"] = df["price1"] - beta * df["price2"]

    # Z‐score it
    mu, sigma = df["spread"].mean(), df["spread"].std()
    df["z"]      = (df["spread"] - mu) / sigma
    df["z_prev"] = df["z"].shift(1).fillna(0)

    # init
    df["signal"] = 0

    # entry
    df.loc[(df["z_prev"] >= -threshold) & (df["z"] < -threshold), "signal"] = 1
    df.loc[(df["z_prev"] <=  threshold) & (df["z"] >  threshold), "signal"] = -1

    # exit
    df.loc[
      ((df["signal"].shift(1) ==  1) & (df["z"] >= -threshold)) |
      ((df["signal"].shift(1) == -1) & (df["z"] <=  threshold)),
      "signal"
    ] = 0

    return df[["price1","price2","spread","z","signal"]]


# -------------------------------------------------------------------
# 2) Backtester
# -------------------------------------------------------------------
def run_backtest_arbitrage(
    hist1: pd.DataFrame,
    hist2: pd.DataFrame,
    signals_df: pd.DataFrame,
    risk_params: dict,
    tx_cost: float = 0.0
):
    """
    Backtest cross‐asset arbitrage given:
      hist1, hist2   – price histories (must have 'last_price', same index)
      signals_df     – output of generate_arbitrage_signals (price1, price2, spread, signal)
      risk_params = {
        'symbol1': str,
        'symbol2': str,
        'starting_cash': float,
        'max_pos': int
      }
      tx_cost        – per‐fill transaction cost (absolute)
    Returns:
      signals_traded – subset of signals where signal actually changed
      trades_list    – list of all execution dicts (both legs)
      metrics_dict   – total_return, max_drawdown, sharpe_ratio
    """

    # 1) Build flat DF with prev‐signal
    df = signals_df.copy()
    df["signal_prev"] = df["signal"].shift(1).fillna(0).astype(int)

    # 2) Keep only actual changes
    signals_traded = df[df["signal"] != df["signal_prev"]].copy()

    # 3) Unpack risk params
    s1, s2        = risk_params["symbol1"], risk_params["symbol2"]
    starting_cash = float(risk_params["starting_cash"])
    max_pos       = int(risk_params["max_pos"])

    # 4) Initialize OMS / 2 Books / Tracker
    oms     = OrderManagementSystem()
    book1   = LimitOrderBook(s1)
    book2   = LimitOrderBook(s2)
    tracker = PositionTracker(starting_cash=starting_cash)
    trades  = []

    # 5) Loop through each signal change
    for ts, row in signals_traded.iterrows():
        sig, prev = int(row["signal"]), int(row["signal_prev"])
        tstamp    = ts.to_pydatetime()

        # Determine trade legs:
        #  +1 → buy asset1, sell asset2
        #  -1 → sell asset1, buy asset2
        #   0 → exit prior position
        if   sig== 1 and prev==0:  side1, side2 = "buy",  "sell"
        elif sig== 0 and prev==1:  side1, side2 = "sell", "buy"
        elif sig==-1 and prev==0:  side1, side2 = "sell", "buy"
        elif sig== 0 and prev==-1: side1, side2 = "buy",  "sell"
        else:
            continue

        # Fetch fill prices from the histories
        price1 = hist1.loc[ts, "last_price"]
        price2 = hist2.loc[ts, "last_price"]

        # Create two Orders
        order1 = Order(
            id=str(uuid.uuid4()), symbol=s1, side=side1,
            quantity=max_pos, type="limit", price=price1, timestamp=tstamp
        )
        order2 = Order(
            id=str(uuid.uuid4()), symbol=s2, side=side2,
            quantity=max_pos, type="limit", price=price2, timestamp=tstamp
        )

        # Submit & match each leg
        for order, book in ((order1, book1), (order2, book2)):
            oms.new_order(order)
            reports = book.add_order(order)

            # Force‐fill if no native fills
            if not reports:
                reports = [{
                  "order_id":   order.id,
                  "symbol":     order.symbol,
                  "side":       order.side,
                  "filled_qty": order.quantity,
                  "price":      order.price,
                  "timestamp":  tstamp
                }]
                book.cancel_order(order.id)

            # Apply transaction cost, update tracker, record trades
            for rpt in reports:
                rpt["price"] = rpt["price"] + (tx_cost if rpt["side"]=="buy" else -tx_cost)
                tracker.update(rpt)

                trades.append({
                  "id":        rpt["order_id"],
                  "symbol":    rpt["symbol"],
                  "side":      rpt["side"],
                  "quantity":  rpt["filled_qty"],
                  "type":      rpt.get("type","limit"),
                  "price":     rpt["price"],
                  "timestamp": rpt["timestamp"]
                })

    # 6) Compute equity curve & metrics

    # get the raw blotter
    blot_df = tracker.get_blotter()

    # only re‐index if a timestamp column exists
    if "timestamp" in blot_df.columns:
        blot = blot_df.set_index("timestamp")
    else:
        blot = blot_df.copy()

    blot = blot.sort_index()

    # now build the equity curve
    equity = blot["cash_flow"].cumsum() + starting_cash

    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max

    # stats
    rets   = equity.pct_change().dropna()
    max_dd = drawdown.min()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() else 0

    # total P&L across both legs
    pnl_summary = tracker.get_pnl_summary({
        s1: hist1["last_price"].iloc[-1],
        s2: hist2["last_price"].iloc[-1]
    })
    total_ret = pnl_summary["total_pnl"] / starting_cash

    metrics = {
      "total_return":  total_ret,
      "max_drawdown":  max_dd,
      "sharpe_ratio":  sharpe
    }

    return signals_traded, trades, metrics


def arbitrage_performance(
    history: pd.DataFrame,
    trades_list: list,
    metrics: dict,
    starting_cash: float,
    symbol: str
):
    """
    Plot price + trades (for one symbol), equity curve, and drawdown.

    Parameters
    ----------
    history : DataFrame
        Must have a DatetimeIndex and a 'last_price' column for `symbol`.
    trades_list : list of dicts
        Each dict must have keys:
          ['symbol','side','quantity','price','timestamp'].
    metrics : dict
        Contains 'total_return', 'max_drawdown', 'sharpe_ratio'.
    starting_cash : float
        Cash balance at t0.
    symbol : str
        The ticker to filter trades and to label the price chart.
    """
    # ----------------------------------------
    # 1) Build a DataFrame of only this symbol’s trades
    # ----------------------------------------
    blot = pd.DataFrame(trades_list)
    if blot.empty:
        print(f"No trades to plot for {symbol}.")
        return

    blot['timestamp'] = pd.to_datetime(blot['timestamp'])
    blot = blot.set_index('timestamp').sort_index()

    # filter out other‐symbol trades
    blot = blot[blot['symbol'] == symbol]
    if blot.empty:
        print(f"No trades for {symbol} found in trades_list.")
        return

    # ensure cash_flow column exists
    if 'cash_flow' not in blot.columns:
        blot['cash_flow'] = np.where(
            blot['side'] == 'buy',
            - blot['quantity'] * blot['price'],
            + blot['quantity'] * blot['price']
        )

    # ----------------------------------------
    # 2) Build equity curve (aggregate multiple trades at same timestamp)
    # ----------------------------------------
    # sum all cash flows per timestamp
    cf_by_ts = blot['cash_flow'].groupby(blot.index).sum()

    # reindex to the full history, forward‐fill zero for no‐trade bars
    cf_aligned = cf_by_ts.reindex(history.index, method='ffill').fillna(0)

    equity = cf_aligned.cumsum() + starting_cash
    running_max = equity.cummax()
    drawdown    = equity - running_max

    # ----------------------------------------
    # 3) Set up the figure
    # ----------------------------------------
    fig, (ax_price, ax_perf) = plt.subplots(
        nrows=2, ncols=1,
        sharex=True,
        figsize=(12, 8),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # ----------------------------------------
    # 4) Price chart + trade markers
    # ----------------------------------------
    ax_price.plot(
        history.index,
        history['last_price'],
        color='black',
        linewidth=1.2,
        label=f'{symbol} Price'
    )

    buys  = blot[blot['side'] == 'buy']
    sells = blot[blot['side'] == 'sell']

    ax_price.scatter(
        buys.index, buys['price'],
        marker='^', s=80, c='green', label='Buys'
    )
    ax_price.scatter(
        sells.index, sells['price'],
        marker='v', s=80, c='red',   label='Sells'
    )

    ax_price.set_ylabel('Price')
    ax_price.set_title(f'{symbol} Price & Trade Markers')
    ax_price.legend(loc='upper left')

    # ----------------------------------------
    # 5) Equity curve & drawdown
    # ----------------------------------------
    ax_perf.plot(equity, color='blue', label='Equity')
    ax_perf.fill_between(
        drawdown.index,
        drawdown,
        0,
        color='red',
        alpha=0.3,
        label='Drawdown'
    )

    ax_perf.set_ylabel('Portfolio Value')
    ax_perf.set_xlabel('Time')
    ax_perf.legend(loc='upper left')

    # ----------------------------------------
    # 6) Annotate performance metrics
    # ----------------------------------------
    title = (
        f"Total Return: {metrics['total_return']:.2%}    "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}    "
        f"Max Drawdown: {metrics['max_drawdown']:.2%}"
    )
    fig.suptitle(title, fontsize=14, y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

'''loader    = MarketDataLoader(interval="1d", period="1y")
hist1     = loader.get_history("9633.HK")
hist2     = loader.get_history("1788.HK")

signals = generate_arbitrage_signals(hist1, hist2, threshold=0.1)

# Debug prints
print(signals.head())
print("Signal counts:\n", signals["signal"].value_counts())
print("Spread stats:\n", signals["spread"].describe())

sig_df, trades, metrics = run_backtest_arbitrage(
    hist1, hist2, signals,
    {"symbol1": "9633.HK", "symbol2": "1788.HK", "starting_cash": 100000, "max_pos": 10},
    tx_cost=0.05
)

print("Signals traded:\n", sig_df)
print("Trades:\n", pd.DataFrame(trades))
print("Metrics:\n", metrics)

raw = hist1

arbitrage_performance(
    history=raw,            # hist1 with a last_price column
    trades_list=trades,
    metrics=metrics,
    starting_cash=100000,
    symbol="9633.HK"        # <— add the symbol you’re plotting!
)
'''