import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# import our loader
from market_data_loader import MarketDataLoader
import pandas as pd


# ------------------------
# STRATEGY DEFINITIONS
# ------------------------

def sma_signals(df: pd.DataFrame, short_w: int = 50, long_w: int = 200) -> pd.DataFrame:
    data = df.copy()
    data['SMA_Short'] = data['Close'].rolling(short_w).mean()
    data['SMA_Long']  = data['Close'].rolling(long_w).mean()
    data['Signal']    = 0
    data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1
    data['Position']  = data['Signal'].shift().fillna(0)
    return data

def rsi_signals(df: pd.DataFrame, period: int = 14,
                lower: int = 30, upper: int = 70) -> pd.DataFrame:
    data = df.copy()
    delta = data['Close'].diff()
    up    = delta.clip(lower=0).rolling(period).mean()
    down  = -delta.clip(upper=0).rolling(period).mean()
    rs    = up / down
    data['RSI']    = 100 - 100 / (1 + rs)
    data['Signal'] = 0
    data.loc[data['RSI'] < lower, 'Signal'] = 1
    data.loc[data['RSI'] > upper, 'Signal'] = 0
    data['Position'] = data['Signal'].shift().fillna(0)
    return data


# ------------------------
# BACKTEST ENGINE
# ------------------------

def backtest(data: pd.DataFrame) -> pd.DataFrame:
    initial_capital = 100_000.0
    df = data.copy()
    df['Holdings'] = df['Position'] * df['Close']
    df['Cash']     = initial_capital - (df['Position'].diff().fillna(df['Position']) * df['Close']).cumsum()
    df['Total']    = df['Holdings'] + df['Cash']
    df['Returns']  = df['Total'].pct_change().fillna(0)
    return df

def compute_metrics(df: pd.DataFrame) -> tuple:
    total_return = df['Total'].iloc[-1] / df['Total'].iloc[0] - 1
    daily_ret     = df['Returns']
    ann_return    = daily_ret.mean() * 252
    ann_vol       = daily_ret.std() * np.sqrt(252)
    sharpe        = ann_return / ann_vol if ann_vol else np.nan
    dd            = (df['Total'] - df['Total'].cummax()) / df['Total'].cummax()
    max_dd        = dd.min()
    return total_return, sharpe, max_dd


# ------------------------
# GUI APPLICATION
# ------------------------

class BacktestGUI:
    def __init__(self, master):
        master.title("Technical Indicator Backtester")

        # initialize loader: 1d bars up to 5y
        self.loader = MarketDataLoader(interval="1d", period="5y")

        # Ticker
        ttk.Label(master, text="Ticker:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.ticker_var = tk.StringVar(value="9633.HK")
        ttk.Entry(master, textvariable=self.ticker_var, width=12).grid(row=0, column=1)

        # Start Date
        ttk.Label(master, text="Start (YYYY-MM-DD):").grid(row=1, column=0, sticky="e")
        self.start_var = tk.StringVar(value="2022-01-01")
        ttk.Entry(master, textvariable=self.start_var, width=12).grid(row=1, column=1)

        # End Date
        ttk.Label(master, text="End (YYYY-MM-DD):").grid(row=2, column=0, sticky="e")
        self.end_var = tk.StringVar(value=datetime.utcnow().strftime("%Y-%m-%d"))
        ttk.Entry(master, textvariable=self.end_var, width=12).grid(row=2, column=1)

        # Strategy
        ttk.Label(master, text="Strategy:").grid(row=3, column=0, sticky="e", pady=5)
        self.strategy_var = tk.StringVar(value="SMA Crossover")
        ttk.Combobox(
            master,
            textvariable=self.strategy_var,
            values=["SMA Crossover", "RSI"],
            state="readonly",
            width=14
        ).grid(row=3, column=1)

        # Run Button
        ttk.Button(master, text="Run Backtest", command=self.run_backtest) \
            .grid(row=4, column=0, columnspan=2, pady=10)

    def run_backtest(self):
        ticker = self.ticker_var.get().strip()
        start  = self.start_var.get().strip()
        end    = self.end_var.get().strip()
        strat  = self.strategy_var.get()

        # 1) Download via MarketDataLoader
        try:
            df = self.loader.get_history(ticker, start=start, end=end)
            if df.empty:
                raise ValueError("No data returned for that range.")
        except Exception as e:
            messagebox.showerror("Download Error", str(e))
            return

        # 2) rename last_price → Close for strategy funcs
        df = df.rename(columns={"last_price": "Close"})

        # 3) Generate signals
        if strat == "SMA Crossover":
            signals = sma_signals(df)
        else:
            signals = rsi_signals(df)

        # 4) Backtest & metrics
        df_bt = backtest(signals)
        total_ret, sharpe, max_dd = compute_metrics(df_bt)

        # 5) Console output
        print(f"\n=== {strat} on {ticker} ===")
        print(f"Total Return:  {total_ret:.2%}")
        print(f"Sharpe Ratio:  {sharpe:.2f}")
        print(f"Max Drawdown:  {max_dd:.2%}")

        # 6) Equity curve
        plt.figure(figsize=(10, 6))
        plt.plot(df_bt.index, df_bt['Total'], label="Equity Curve", color="navy")
        plt.title(f"{ticker} — {strat} Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ------------------------
# ENTRY POINT
# ------------------------

if __name__ == "__main__":
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    app = BacktestGUI(root)
    root.mainloop()
