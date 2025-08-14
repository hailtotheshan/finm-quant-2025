import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = os.getenv("APCA_API_KEY_ID", "PKX2KNF58GYEGE6Q8BAG")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "oVtD5qrWqQbfSderahrYjN7hddKsHOYvdbmoWAeS")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def fetch_data(symbol: str,
               start: str = None,
               end: str = None,
               days: int = None
               ) -> pd.DataFrame:
    """
    Fetch daily bars for `symbol` via Alpaca and return a tidy DataFrame.

    You MUST supply either:
      • start/end as "YYYY-MM-DD" strings, or
      • days as an integer (it will compute start/end for you).
    """
    # 1) build start/end strings if user passed `days`
    if days is not None:
        now_utc = datetime.now(timezone.utc)  # ← timezone-aware UTC
        end = now_utc.date().isoformat()  # "YYYY-MM-DD"
        start = (now_utc - timedelta(days=days)) \
            .date() \
            .isoformat()

    if not (start and end):
        raise ValueError("Provide either days or both start/end dates.")

    # 2) construct the exact same StockBarsRequest as in your example
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start,
        end=end
    )

    # 3) fetch, extract .df, drop the symbol‐level
    bars_response = data_client.get_stock_bars(request_params)
    raw_df = bars_response.df
    df = raw_df.xs(symbol, level="symbol").copy()

    # 4) make sure timestamp is a DatetimeIndex
    df.index = pd.to_datetime(df.index)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns ['open','high','low','close','volume'],
    return a new DataFrame with a wide array of technical indicators.
    """
    f = df.copy()

    # 1) Simple & Exponential Moving Averages
    ma_windows = [5, 10, 20, 50, 200]
    for w in ma_windows:
        f[f"ma_{w}"] = f["close"].rolling(w).mean()
        f[f"ema_{w}"] = f["close"].ewm(span=w, adjust=False).mean()

    # 2) Bollinger Bands (20-day, 2σ)
    bb_w = 20
    f["bb_mid"] = f["close"].rolling(bb_w).mean()
    f["bb_std"] = f["close"].rolling(bb_w).std()
    f["bb_upper"] = f["bb_mid"] + 2 * f["bb_std"]
    f["bb_lower"] = f["bb_mid"] - 2 * f["bb_std"]

    # 3) Relative Strength Index (14)
    delta = f["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    f["rsi_14"] = 100 - (100 / (1 + rs))

    # 4) Moving Average Convergence Divergence (MACD)
    ema12 = f["close"].ewm(span=12, adjust=False).mean()
    ema26 = f["close"].ewm(span=26, adjust=False).mean()
    f["macd_line"] = ema12 - ema26
    f["macd_signal"] = f["macd_line"].ewm(span=9, adjust=False).mean()
    f["macd_hist"] = f["macd_line"] - f["macd_signal"]

    # 5) Stochastic Oscillator & KDJ
    low9 = f["low"].rolling(9).min()
    high9 = f["high"].rolling(9).max()
    f["stoch_%K"] = (f["close"] - low9) / (high9 - low9) * 100
    f["stoch_%D"] = f["stoch_%K"].rolling(3).mean()
    # KDJ J-line: 3*K - 2*D
    f["kdj_j"] = 3 * f["stoch_%K"] - 2 * f["stoch_%D"]

    # 6) Average True Range (ATR, 14)
    tr1 = f["high"] - f["low"]
    tr2 = (f["high"] - f["close"].shift()).abs()
    tr3 = (f["low"] - f["close"].shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f["atr_14"] = true_range.rolling(14).mean()

    # 7) On-Balance Volume (OBV)
    obv = (np.sign(f["close"].diff()) * f["volume"]).fillna(0).cumsum()
    f["obv"] = obv

    # 8) Momentum & Rate of Change
    f["momentum_10"] = f["close"] - f["close"].shift(10)
    f["roc_10"] = f["close"].pct_change(10)

    # 9) Commodity Channel Index (CCI, 20)
    tp = (f["high"] + f["low"] + f["close"]) / 3
    ma_tp = tp.rolling(20).mean()
    md_tp = tp.rolling(20).apply(lambda x: np.fabs(x - x.mean()).mean())
    f["cci_20"] = (tp - ma_tp) / (0.015 * md_tp)

    # 10) Williams %R (14)
    f["williams_%R"] = (high9 - f["close"]) / (high9 - low9) * -100

    # 11) Average Directional Index (ADX, 14)
    #   Simplified; for full ADX you need +DI, -DI, etc.
    plus_dm = f["high"].diff().clip(lower=0)
    minus_dm = -f["low"].diff().clip(upper=0)
    tr14 = true_range.rolling(14).sum()
    f["+DI_14"] = 100 * plus_dm.rolling(14).sum() / tr14
    f["-DI_14"] = 100 * minus_dm.rolling(14).sum() / tr14
    dx = (np.abs(f["+DI_14"] - f["-DI_14"]) / (f["+DI_14"] + f["-DI_14"])) * 100
    f["adx_14"] = dx.rolling(14).mean()

    # 12) Average Directional Index (ADX, 14)
    dx = (np.abs(f["+DI_14"] - f["-DI_14"]) / (f["+DI_14"] + f["-DI_14"])) * 100
    f["adx_14"] = dx.rolling(14).mean()

    # 13) your custom model features
    # 13.1 simple daily return
    f["return"] = f["close"].pct_change()

    # 13.2 MA5 – MA20 spread
    f["ma5_minus_ma20"] = f["ma_5"] - f["ma_20"]

    # 13.3 5-day volume moving average
    f["vol_ma5"] = f["volume"].rolling(5).mean()

    # 14) Drop rows with NaNs from the longest lookback
    # only drop rows missing the core signals
    needed = ["ma_20", "ma_50", "rsi_14", "vol_ma5"]
    f = f.dropna(subset=needed)
    return f


def make_labels(df: pd.DataFrame) -> pd.Series:
    return (df["close"].shift(-1) > df["close"]).astype(int).iloc[:-1]


def train_model(X: pd.DataFrame, y: pd.Series):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model, scaler


def place_market_order(symbol: str, side: OrderSide, qty: int = 1):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC
    )
    resp = trading_client.submit_order(order)
    print(f"Order placed: {side} {qty} {symbol}", resp)


def backtest_trades(
        df: pd.DataFrame,
        signal_col: str = "signal",
        open_col: str = "open",
        close_col: str = "close"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Convert a series of daily signals (1, 0, -1) into trade PnL and summary metrics.

    Parameters
    ----------
    df : pd.DataFrame
      Must include columns [signal_col, open_col, close_col], indexed by date.
    signal_col : str
      Daily position signal: 1=long, -1=short, 0=flat.
    open_col : str
      Column name for the next‐period fill price (we use open price for entry/exit).
    close_col : str
      Column name for close price (for cum‐equity).

    Returns
    -------
    df_bt : pd.DataFrame
      Original df + columns:
        position      : shifted signal
        daily_return  : PnL for each bar
        cum_return    : cumulative product of (1 + daily_return)
    trades_df : pd.DataFrame
      One row per completed trade, columns:
        entry, exit, direction, return, holding_days
    metrics : dict
      Aggregate trade statistics:
        total_return, num_trades, win_rate,
        avg_trade_return, avg_holding_days
    """
    df_bt = df.copy()
    # build today’s position from yesterday’s signal
    df_bt["position"] = df_bt[signal_col].shift(1).fillna(0).astype(int)

    # mark every time the position level changes
    df_bt["chg"] = df_bt["position"] != df_bt["position"].shift(1).fillna(0)
    events = df_bt[df_bt["chg"]].index

    trades = []
    for i in range(len(events) - 1):
        entry_date = events[i]
        exit_date = events[i + 1]
        entry_pos = df_bt.at[entry_date, "position"]
        if entry_pos == 0:
            continue  # skip flat→flat
        entry_price = df_bt.at[entry_date, open_col]
        exit_price = df_bt.at[exit_date, open_col]
        ret = (exit_price / entry_price - 1) * entry_pos
        days_held = (exit_date - entry_date).days

        trades.append({
            "entry": entry_date,
            "exit": exit_date,
            "direction": entry_pos,
            "return": ret,
            "holding_days": days_held
        })

    trades_df = pd.DataFrame(trades)

    # 6) Aggregate metrics
    total_return = df_bt["cum_return"].iloc[-1] - 1
    num_trades = len(trades_df)
    win_rate = (trades_df["return"] > 0).mean() if num_trades else 0.0
    avg_trade_return = trades_df["return"].mean() if num_trades else 0.0
    avg_holding_days = trades_df["holding_days"].mean() if num_trades else 0.0

    metrics = {
        "Total Return": total_return,
        "Number of Trades": num_trades,
        "Win Rate": win_rate,
        "Avg Trade Return": avg_trade_return,
        "Avg Holding Days": avg_holding_days
    }

    return df_bt, trades_df, metrics


def main():
    symbol = "AAPL"
    days = 500

    # 1) Fetch & engineer
    now_utc = datetime.now(timezone.utc)
    end = now_utc.date().isoformat()
    start = (now_utc - timedelta(days=days)).date().isoformat()

    df_raw = fetch_data(symbol, start=start, end=end)
    df_feats = engineer_features(df_raw)

    # 2) Build signal (example: MA crossover)
    short_ma = df_feats["close"].rolling(20).mean()
    long_ma = df_feats["close"].rolling(50).mean()
    df_feats["signal"] = 0
    df_feats.loc[short_ma > long_ma, "signal"] = 1
    df_feats.loc[short_ma < long_ma, "signal"] = -1

    df_feats["ma_short"] = df_feats["close"].rolling(10).mean()
    df_feats["ma_long"] = df_feats["close"].rolling(20).mean()

    df_feats["signal"] = 0
    df_feats.loc[df_feats["ma_short"] > df_feats["ma_long"], "signal"] = 1
    df_feats.loc[df_feats["ma_short"] < df_feats["ma_long"], "signal"] = -1

    print(df_feats['signal'].value_counts())
    print(df_feats[['close']].assign(
        short_ma=lambda d: d['close'].rolling(20).mean(),
        long_ma=lambda d: d['close'].rolling(50).mean(),
        signal=df_feats['signal']
    ).tail(10))

    # 3) Backtest trades
    df_bt, trades_df, perf = backtest_trades(
        df_feats,
        signal_col="signal",
        open_col="open",
        close_col="close"
    )

    print(trades_df)  # should list entry/exit dates
    print(df_bt['position'].unique())

    # 4) Show summary
    print("\n=== Backtest Performance ===")
    for k, v in perf.items():
        # format percentages
        if k in ("Total Return", "Win Rate", "Avg Trade Return"):
            print(f"{k:20s}: {v:.2%}")
        else:
            print(f"{k:20s}: {v:.2f}")



if __name__ == "__main__":
    main()
