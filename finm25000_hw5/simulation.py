from order import Order
from order_book import LimitOrderBook
from market_data_loader import MarketDataLoader
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timezone
from strategies.arbitrage import generate_arbitrage_signals, run_backtest_arbitrage, arbitrage_performance
from strategies.mean_reversion import add_bollinger_and_signals, run_backtest_mean_reversion ,mean_reversion_performance
from strategies.trend_following import add_moving_average_signals, run_backtest_trend_following, trend_following_performance


def main():
    # Arbitrage strategy example
    loader = MarketDataLoader(interval="1d", period="1y")
    hist1 = loader.get_history("9633.HK")
    hist2 = loader.get_history("1788.HK")

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
        history=raw,  # hist1 with a last_price column
        trades_list=trades,
        metrics=metrics,
        starting_cash=100000,
        symbol="9633.HK"  # <— add the symbol you’re plotting!
    )

    # Mean reversion strategy example
    raw = loader.get_history("9633.HK")
    df = add_bollinger_and_signals(raw, window=20, num_std=2)

    # 1) run backtest
    signals_df, trades_list, metrics = run_backtest_mean_reversion(df, {
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
                              columns=['id', 'symbol', 'side', 'quantity', 'price', 'timestamp']))

    metrics_series = pd.Series(metrics, name='value')
    print(metrics_series.to_frame())

    # 3) visualize
    mean_reversion_performance(
        history=df.set_index(df.index.tz_localize(None)),
        trades_list=trades_list,
        starting_cash=100000
    )

    # Moving average crossover strategy example
    loader = MarketDataLoader(interval="1d", period="3y")
    # Pull the last month's daily bars
    df = loader.get_history("9633.HK")

    # 1) build your signals
    sig_df = add_moving_average_signals(df, short_win=10, long_win=50)

    # 2) backtest
    signals, trades, metrics = run_backtest_trend_following(
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
                              columns=['id', 'symbol', 'side', 'quantity', 'price', 'timestamp']))

    metrics_series = pd.Series(metrics, name='value')
    print(metrics_series.to_frame())

    # 3) Visualize
    trend_following_performance(
        history=df,
        signals_df=signals,
        trades_list=trades,
        metrics=metrics,
        starting_cash=100_000
    )


if __name__ == "__main__":
    main()
