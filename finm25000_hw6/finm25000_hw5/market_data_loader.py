# market_data_loader.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict
import pytz


class MarketDataLoader:
    """
    Fetches OHLCV for equities, ETFs, FX, crypto, bonds/futures and option chains,
    either by fixed period or explicit start/end.
    """

    def __init__(self, interval: str, period: str):
        # frequency (e.g. "1m", "5m", "1d") and default lookback (e.g. "1mo", "1y")
        self.interval = interval
        self.period = period
        # caches: { symbol -> DataFrame } and { (symbol, start, end) -> DataFrame }
        self._period_cache: Dict[str, pd.DataFrame] = {}
        self._range_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    def _rename_and_tz(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # ── Step 0: flatten MultiIndex columns if present ──
        if isinstance(df.columns, pd.MultiIndex):
            # drop the Ticker level (level=1), keep only the Price level
            df.columns = df.columns.droplevel(1)

        # ── Step 1: rename to your standard open/high/low/last_price/volume ──
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "last_price",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)[list(col_map.values())]

        # ── Step 2: ensure UTC tz-aware index ──
        idx = df.index
        if idx.tz is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")

        return df

    def _load_period(self, symbol: str) -> pd.DataFrame:
        """
        Download and cache fixed-period history for symbol.
        """
        if symbol in self._period_cache:
            return self._period_cache[symbol]

        df = yf.download(
            symbol,
            period=self.period,
            interval=self.interval,
            auto_adjust=True,
            progress=False
        )
        df = self._rename_and_tz(df)
        self._period_cache[symbol] = df
        return df

    def get_history(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV for symbol. If start & end given, download that slice;
        otherwise use fixed-period cache.
        """
        if start is not None and end is not None:
            key = (symbol, start, end)
            if key in self._range_cache:
                return self._range_cache[key]

            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=self.interval,
                auto_adjust=True,
                progress=False
            )
            df = self._rename_and_tz(df)
            self._range_cache[key] = df
            return df

        return self._load_period(symbol)

    def _locate_timestamp(self, df: pd.DataFrame, ts: datetime) -> pd.Timestamp:
        """
        Align a timestamp to the nearest prior bar in df.index.
        """
        if df.empty:
            raise KeyError("No data to locate timestamp")

        # ensure UTC
        ts = pd.to_datetime(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=pytz.UTC)
        else:
            ts = ts.astimezone(pytz.UTC)

        # find indexer
        pos = df.index.get_indexer([ts], method="ffill")[0]
        if pos < 0:
            raise KeyError(f"No bar at or before {ts}")
        return df.index[pos]

    def _scalar_to_float(self, x) -> float:
        # if it’s a one-element Series, grab the lone value
        if isinstance(x, pd.Series):
            if len(x) == 1:
                x = x.iloc[0]
            else:
                raise ValueError(f"Expected a scalar or 1-element Series, got Series of length {len(x)}")
        return float(x)

    def _scalar_to_int(self, x) -> int:
        if isinstance(x, pd.Series):
            if len(x) == 1:
                x = x.iloc[0]
            else:
                raise ValueError(f"Expected a scalar or 1-element Series, got Series of length {len(x)}")
        return int(x)

    def get_price(self, symbol: str, timestamp: datetime) -> float:
        """
        Return last_price at or immediately before timestamp.
        """
        df = self.get_history(symbol)
        ts0 = self._locate_timestamp(df, timestamp)
        # label-based lookup is more robust than .at here
        val = df.loc[ts0, "last_price"]
        return self._scalar_to_float(val)

    def get_bid_ask(self, symbol: str, timestamp: datetime) -> Tuple[float, float]:
        """
        Approximate bid/ask around the mid-price using a
        tiny spread based on asset class.
        """
        mid = self.get_price(symbol, timestamp)

        # identify asset type by symbol suffix
        st = symbol.upper()
        if st.endswith("=X"):
            spread = 0.00002          # FX
        elif st.endswith("-USD"):
            spread = 0.002            # crypto
        elif st.endswith("=F"):
            spread = 0.0005           # futures
        else:
            spread = 0.0001           # equities, ETFs

        half = mid * spread / 2
        bid = mid - half
        ask = mid + half
        return self._scalar_to_float(bid), self._scalar_to_float(ask)

    def get_volume(
        self,
        symbol: str,
        start: str,
        end: str
    ) -> int:
        """
        Sum traded volume between start and end (inclusive).
        """
        df = self.get_history(symbol, start=start, end=end)
        total = df["volume"].sum()
        return self._scalar_to_int(total)

    def get_option_chain(
        self,
        symbol: str,
        expiry: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns {'calls': DataFrame, 'puts': DataFrame} for the given expiry.
        If expiry is None, picks the nearest available.
        """
        tkr = yf.Ticker(symbol)
        all_exps = tkr.options
        if not all_exps:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}

        chosen = expiry or all_exps[0]
        if expiry and expiry not in all_exps:
            # fallback to nearest
            chosen = min(all_exps, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(expiry)))

        chain = tkr.option_chain(chosen)
        calls = self._rename_and_tz(chain.calls.set_index("contractSymbol"))
        puts = self._rename_and_tz(chain.puts.set_index("contractSymbol"))

        return {"calls": calls, "puts": puts}

'''loader = MarketDataLoader(interval="1d", period="5y")
# Pull the last month's daily bars
df = loader.get_history("9633.HK")
print(df.columns)'''

def print_section(title: str):
    """Prints a section header surrounded by underlines."""
    print(f"\n{title}")
    print("-" * len(title))


def pretty_table(
    df: pd.DataFrame,
    caption: str = None,
    float_format: str = "{:.6f}"
):
    """
    Prints DataFrame as an aligned ASCII table.
    - df: must have reset_index() applied if you want the index shown as a column.
    - caption: optional text printed above the table.
    - float_format: Python format string for floats.
    """
    if caption:
        print(f"\n{caption}")
    fmtters = {
        col: (lambda x, fmt=float_format: fmt.format(x))
        for col in df.select_dtypes(include=["float", "float64"]).columns
    }
    print(
        df.to_string(
            index=False,
            formatters=fmtters,
            na_rep="",
            justify="left"
        )
    )


def main():
    loader = MarketDataLoader(interval="5m", period="1mo")

    # 1) Fixed-period history (last 1 month) for AAPL
    print_section("1) Fixed-period History: AAPL (Last 1 Month)")
    hist = loader.get_history("AAPL")
    df1 = hist.head(5).reset_index().rename(columns={"index": "timestamp"})
    df2 = hist.tail(5).reset_index().rename(columns={"index": "timestamp"})
    pretty_table(df1, caption="First 5 rows of AAPL history")
    pretty_table(df2, caption="Last 5 rows of AAPL history")

    # 2) Explicit date range for AAPL
    print_section("2) Explicit Date Range: 2025-06-15 → 2025-07-01 for AAPL")
    start, end = "2025-06-15", "2025-07-01"
    hist_range = loader.get_history("AAPL", start=start, end=end)
    print(f"\nRange Start: {hist_range.index.min()}  Range End: {hist_range.index.max()}\n")
    df3 = hist_range.head(5).reset_index().rename(columns={"index": "timestamp"})
    df4 = hist_range.tail(5).reset_index().rename(columns={"index": "timestamp"})
    pretty_table(df3, caption="First 5 rows of range")
    pretty_table(df4, caption="Last 5 rows of range")

    # 3) EURUSD Price & Bid/Ask lookup
    print_section("3) EURUSD=X Price & Bid/Ask @ 2025-07-01 15:30 UTC")
    ts = datetime(2025, 7, 1, 15, 30)
    price = loader.get_price("EURUSD=X", ts)
    bid, ask = loader.get_bid_ask("EURUSD=X", ts)
    print(f"\nPrice:  {price:.6f}")
    print(f"Bid:    {bid:.6f}")
    print(f"Ask:    {ask:.6f}\n")

    # 4) BTC-USD Volume over custom window
    print_section("4) BTC-USD Volume: 2025-06-30 → 2025-07-01")
    vol = loader.get_volume("BTC-USD", start="2025-06-30", end="2025-07-01")
    print(f"\nTotal Volume (UTC): {vol:,}\n")

    '''# 5) AAPL Option Chain (nearest expiry)
    print_section("5) AAPL Option Chain (nearest available expiry)")
    opts = loader.get_option_chain("AAPL", expiry="2025-07-17")
    calls = opts["calls"].sort_values("strike").head(5).reset_index(drop=True)
    puts  = opts["puts"].sort_values("strike").head(5).reset_index(drop=True)
    pretty_table(calls, caption="Top 5 AAPL Calls")
    pretty_table(puts,  caption="Top 5 AAPL Puts")'''


if __name__ == "__main__":
    main()