import pandas as pd
from typing import List, Dict
from datetime import datetime, timezone


class PositionTracker:
    def __init__(self, starting_cash: float = 0.0):
        self.positions: Dict[str, int] = {}
        self.cash: float = starting_cash
        self.blotter: List[Dict] = []

    def update(self, report: Dict) -> None:
        symbol = report["symbol"]
        qty = report["filled_qty"]
        price = report["price"]
        side = report["side"]
        timestamp = report["timestamp"]

        # Update position
        delta = qty if side == "buy" else -qty
        self.positions[symbol] = self.positions.get(symbol, 0) + delta

        # Update cash
        cash_flow = -qty * price if side == "buy" else qty * price
        self.cash += cash_flow

        # Record blotter entry
        self.blotter.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "cash_flow": cash_flow
        })

    def get_blotter(self) -> pd.DataFrame:
        # Always produce a DataFrame with these columns,
        # so downstream code can .set_index("timestamp")
        # and index ["cash_flow"] without KeyErrors.
        cols = ["timestamp", "symbol", "side", "quantity", "price", "cash_flow"]
        return pd.DataFrame(self.blotter, columns=cols)

    def get_pnl_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        blotter_df = self.get_blotter()
        realized_pnl = float(blotter_df["cash_flow"].sum())

        unrealized_pnl = 0.0
        if current_prices:
            for sym, pos in self.positions.items():
                price = current_prices.get(sym, 0.0)
                unrealized_pnl += pos * price

        total_pnl = realized_pnl + unrealized_pnl

        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "current_cash": self.cash,
            "positions": dict(self.positions)
        }


'''pt = PositionTracker(starting_cash=10000)
rpt = {
    "order_id": "1",
    "symbol": "AAPL",
    "side": "buy",
    "filled_qty": 10,
    "price": 150.0,
    "timestamp": datetime.now(timezone.utc)
}
pt.update(rpt)
print(pt.cash)  # 10000 - 1500 = 8500
print(pt.positions)  # {'AAPL': 10}

rpt2 = rpt.copy()
rpt2["side"] = "sell"
pt.update(rpt2)
print(pt.cash)  # 8500 + 1500 = 10000
print(pt.positions)  # {'AAPL': 0}

summary = pt.get_pnl_summary(current_prices={'AAPL': 155.0})
print(summary)'''
