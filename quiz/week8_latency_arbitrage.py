import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def simulate_exchange(name, lag_ms=0, seed=0, n_ticks=1000):
    np.random.seed(seed)
    timestamps = pd.date_range("2025-08-01", periods=n_ticks, freq="ms") + pd.Timedelta(milliseconds=lag_ms)
    prices = 100 + np.cumsum(np.random.normal(0, 0.03, size=n_ticks))
    sizes = np.random.randint(50, 200, size=n_ticks)
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "size": sizes,
        "exchange": name
    })

class LatencyArbAgent:
    def __init__(self, threshold=0.25):
        self.threshold = threshold
        self.pnl = 0.0
        self.trades = []
        self.feed_buffer = []

    def observe(self, tick):
        self.feed_buffer.append(tick)
        if len(self.feed_buffer) < 2:
            return
        latest = self.feed_buffer[-1]
        prior = self.feed_buffer[-2]
        if latest["exchange"] != prior["exchange"]:
            price_diff = latest["price"] - prior["price"]
            if abs(price_diff) >= self.threshold:
                self.execute_arbitrage(buy=prior if price_diff > 0 else latest,
                                       sell=latest if price_diff > 0 else prior)

    def execute_arbitrage(self, buy, sell):
        size = min(buy["size"], sell["size"])
        profit = (sell["price"] - buy["price"]) * size
        self.pnl += profit
        self.trades.append({
            "timestamp": sell["timestamp"],
            "buy_price": buy["price"],
            "sell_price": sell["price"],
            "size": size,
            "pnl": profit
        })

    def display_metrics(agent):
        df = pd.DataFrame(agent.trades)
        if df.empty:
            print("No trades executed.")
            return

        win_rate = sum(df["pnl"] > 0) / len(df)
        sharpe = df["pnl"].mean() / df["pnl"].std() if df["pnl"].std() != 0 else 0

        print(f"\n📊 Strategy Metrics")
        print(f"Total Trades: {len(df)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total PnL: ${agent.pnl:.2f}")
        print(f"Average Trade PnL: ${df['pnl'].mean():.2f}")
        print(f"Sharpe Ratio (approx): {sharpe:.2f}")

    def plot_price_streams(fast, slow):
        plt.figure(figsize=(12, 5))
        plt.plot(fast["timestamp"], fast["price"], label="Fast Exchange")
        plt.plot(slow["timestamp"], slow["price"], label="Slow Exchange")
        plt.legend()
        plt.title("Exchange Price Divergence")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trades(agent):
        df = pd.DataFrame(agent.trades)
        if df.empty:
            return
        plt.figure(figsize=(12, 5))
        plt.scatter(df["timestamp"], df["buy_price"], color="green", label="Buy", s=40)
        plt.scatter(df["timestamp"], df["sell_price"], color="red", label="Sell", s=40)
        plt.legend()
        plt.title("Arbitrage Trade Points")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_pnl(agent):
        df = pd.DataFrame(agent.trades)
        if df.empty:
            return
        df["cumulative_pnl"] = df["pnl"].cumsum()
        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["cumulative_pnl"], label="Cumulative PnL", color="blue")
        plt.legend()
        plt.title("Cumulative Profit Over Time")
        plt.xlabel("Time")
        plt.ylabel("PnL ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# 1. Simulate two asynchronous feeds
fast_feed = simulate_exchange("Fast", lag_ms=0, seed=1, n_ticks=1000)
slow_feed = simulate_exchange("Slow", lag_ms=3, seed=2, n_ticks=1000)
combined_feed = pd.concat([fast_feed, slow_feed]) \
                  .sort_values("timestamp") \
                  .reset_index(drop=True)

# 2. Instantiate the latency-arb agent
agent = LatencyArbAgent(threshold=0.25)

# 3. Feed ticks into the agent
for _, tick in combined_feed.iterrows():
    agent.observe(tick)

LatencyArbAgent.display_metrics(agent)
LatencyArbAgent.plot_price_streams(fast_feed, slow_feed)
LatencyArbAgent.plot_trades(agent)
LatencyArbAgent.plot_pnl(agent)
