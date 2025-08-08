from market_data_loader import MarketDataLoader
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np


# 5-minute bars over the last month
loader = MarketDataLoader(interval="1d", period="3y")

# Fetch a DataFrame of UTC-indexed OHLCV for AAPL
df = loader.get_history("1299.HK")

# rename columns to match backtesting.py
df_bt = (
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "last_price": "Close",
        "volume": "Volume"
    })
    # backtesting.py requires a monotonic index
    .sort_index()
)


class SmaCross(Strategy):
    fast_window = 50
    slow_window = 200

    def init(self):
        close = self.data.Close

        # wrap the raw array in a pd.Series so .rolling() works
        self.fast_sma = self.I(self._sma, close, self.fast_window)
        self.slow_sma = self.I(self._sma, close, self.slow_window)

    def _sma(self, x, n):
        """
        x comes in as a NumPy array.  We convert it to a Series,
        apply rolling, then return the .values back to backtesting.py.
        """
        return pd.Series(x).rolling(n).mean().values

    def next(self):
        if crossover(self.fast_sma, self.slow_sma):
            self.buy()
        elif crossover(self.slow_sma, self.fast_sma):
            self.position.close()

# initialize backtest with $100,000 capital, 0.1% commission
bt = Backtest(
    df_bt,
    SmaCross,
    cash=100_000,
    commission=0.001,    # 0.1%
    trade_on_close=True  # execute orders at bar close
)

stats = bt.run()

print("Total Return [%]:", stats["Return [%]"])
print("Sharpe Ratio:",    stats["Sharpe Ratio"])

# show an interactive plot with price, SMAs, markers, and equity curve
bt.plot()
