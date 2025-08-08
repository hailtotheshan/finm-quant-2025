from typing import List, Dict
from finm25000_hw5.order import Order
from datetime import datetime, timezone


class LimitOrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        self.buy_stops = []
        self.sell_stops = []
        self.last_trade_price: float = None

    def add_order(self, order: Order) -> List[Dict]:
        reports = []
        if order.type == "market":
            # 1) Try to execute immediately
            reports += self._execute_market(order)

            # 2) If there's still unfilled qty, make it *resting* liquidity
            if order.quantity > 0:
                self._insert_resting(order)

        elif order.type == "limit":
            reports += self._match_limit(order)
            if order.quantity > 0:
                self._insert_resting(order)

        elif order.type == "stop":
            if order.side == "buy":
                self.buy_stops.append(order)
            else:
                self.sell_stops.append(order)
            return []
        return reports

    def _match_limit(self, order: Order) -> List[Dict]:
        reports = []
        opposite = self.asks if order.side == "buy" else self.bids

        while order.quantity > 0 and opposite:
            best = opposite[0]

            # If the resting order is itself a market order,
            # we always match (no price check), at the incoming limit price.
            if best.type == "market":
                trade_price = order.price
            else:
                # Regular price check
                if order.side == "buy" and best.price > order.price:
                    break
                if order.side == "sell" and best.price < order.price:
                    break
                trade_price = best.price

            fill_qty = min(order.quantity, best.quantity)
            ts = datetime.now(timezone.utc)
            self.last_trade_price = trade_price

            # build reports (aggressor & resting)…
            reports += [
                {
                    "order_id": order.id, "symbol": order.symbol,
                    "side": order.side, "filled_qty": fill_qty,
                    "price": trade_price, "timestamp": ts,
                    "status": ("filled" if fill_qty == order.quantity else "partial_fill")
                },
                {
                    "order_id": best.id, "symbol": best.symbol,
                    "side": best.side, "filled_qty": fill_qty,
                    "price": trade_price, "timestamp": ts,
                    "status": ("filled" if fill_qty == best.quantity else "partial_fill")
                }
            ]

            order.quantity -= fill_qty
            best.quantity -= fill_qty
            if best.quantity == 0:
                opposite.pop(0)

        if self.last_trade_price is not None:
            self._trigger_stops(self.last_trade_price)

        return reports

    def _execute_market(self, order: Order) -> List[Dict]:
        reports = []
        opposite = self.asks if order.side == "buy" else self.bids

        while order.quantity > 0 and opposite:
            best = opposite[0]

            # Determine price:
            if best.type == "market":
                # No price on either side—use last_trade_price
                trade_price = self.last_trade_price
            else:
                trade_price = best.price

            fill_qty = min(order.quantity, best.quantity)
            ts = datetime.now(timezone.utc)
            self.last_trade_price = trade_price

            # build reports…
            reports += [
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "filled_qty": fill_qty,
                    "price": trade_price,
                    "timestamp": ts,
                    "status": ("filled" if fill_qty == order.quantity else "partial_fill")
                },
                {
                    "order_id": best.id,
                    "symbol": best.symbol,
                    "side": best.side,
                    "filled_qty": fill_qty,
                    "price": trade_price,
                    "timestamp": ts,
                    "status": ("filled" if fill_qty == best.quantity else "partial_fill")
                }
            ]

            order.quantity -= fill_qty
            best.quantity -= fill_qty
            if best.quantity == 0:
                opposite.pop(0)

        if self.last_trade_price is not None:
            self._trigger_stops(self.last_trade_price)

        return reports

    def _trigger_stops(self, last_price: float):
        # buy-stops
        triggered = [o for o in self.buy_stops if last_price >= o.price]
        for o in sorted(triggered, key=lambda o: o.timestamp):
            self.buy_stops.remove(o)
            o.type = "market"
            self.add_order(o)

        # sell-stops
        triggered = [o for o in self.sell_stops if last_price <= o.price]
        for o in sorted(triggered, key=lambda o: o.timestamp):
            self.sell_stops.remove(o)
            o.type = "market"
            self.add_order(o)

    def _insert_resting(self, order: Order):
        # Separate the two sides
        book = self.bids if order.side == "buy" else self.asks

        # Market orders: immediate priority (price irrelevant)
        if order.type == "market":
            book.insert(0, order)
            return

        # Existing logic for limit orders
        idx = 0
        while idx < len(book):
            if order.side == "buy" and book[idx].price > order.price:
                idx += 1
                continue
            if order.side == "sell" and book[idx].price < order.price:
                idx += 1
                continue
            break

        book.insert(idx, order)

    def cancel_order(self, order_id: str) -> bool:
        """
        Remove a resting order (by id) from bids or asks.
        Returns True if found & removed, False otherwise.
        """
        for book_side in (self.bids, self.asks):
            for idx, resting in enumerate(book_side):
                if resting.id == order_id:
                    book_side.pop(idx)
                    return True
        return False


"""lob = LimitOrderBook("AAPL")

buy1 = Order("buy1", "AAPL", "buy", 10, "limit", 150.0)
sell1 = Order("sell1", "AAPL", "sell", 15, "limit", 149.0)
sell2 = Order("sell2", "AAPL", "sell", 15, "limit", 149.0)
sell3 = Order("sell2", "AAPL", "sell", 15, "market")

print("sell1", lob.add_order(sell1))
print("sell2", lob.add_order(sell2))
print("sell3", lob.add_order(sell3))
print("buy1", lob.add_order(buy1))

print("lob.bids", lob.bids)
print("lob.asks", lob.asks)"""
'''
print("\ntesting:\n")
print("""-----------------------------------------------------------------------------
1) LIMIT ORDERS: insertion + matching
-----------------------------------------------------------------------------""")

lob = LimitOrderBook("AAPL")

# 1a) no book to match against → resting
o1 = Order("L1", "AAPL", "buy", 10, "limit", 150.0)
print("add_order(L1):", lob.add_order(o1))
print("  bids:", lob.bids)
print("  asks:", lob.asks, "\n")

# 1b) incoming sell @ 149 → matches 10@149, leaves no resting sell, removes buy
o2 = Order("L2", "AAPL", "sell", 10, "limit", 149.0)
print("add_order(L2):", lob.add_order(o2))
print("  bids:", lob.bids)
print("  asks:", lob.asks, "\n")

# 1c) partial match: buy 5 @ 148 into empty book → resting buy
o3 = Order("L3", "AAPL", "buy", 5, "limit", 148.0)
print("add_order(L3):", lob.add_order(o3))
print("  bids:", lob.bids)
print("  asks:", lob.asks)

print("""-----------------------------------------------------------------------------
2) MARKET ORDERS: eat the book until filled or empty
-----------------------------------------------------------------------------""")
lob = LimitOrderBook("AAPL")

# prep book with two asks
lob.add_order(Order("A1", "AAPL", "sell", 5, "limit", 150.0))
lob.add_order(Order("A2", "AAPL", "sell", 5, "limit", 151.0))
print("initial asks:", lob.asks)

# market buy for 7 → fills 5@150 + 2@151, removes both asks
m1 = Order("M1", "AAPL", "buy", 7, "market")
print("add_order(M1):", lob.add_order(m1))
print("asks after M1:", lob.asks)

# market sell into empty book → nothing happens
m2 = Order("M2", "AAPL", "sell", 3, "market")
print("add_order(M2):", lob.add_order(m2))
print("bids after M2:", lob.bids)

print("""-----------------------------------------------------------------------------
3) STOP ORDERS: sit in stop‐book until a trade hits the trigger price
-----------------------------------------------------------------------------""")
lob = LimitOrderBook("AAPL")

# seed one resting ask @100 so trades can happen
lob.add_order(Order("A1", "AAPL", "sell", 10, "limit", 100.0))

# place a buy‐stop at 105 (won’t fire yet)
stop_buy = Order("S1", "AAPL", "buy", 5, "stop", 105.0)
print("add_order(S1):", lob.add_order(stop_buy))
print("  buy_stops:", lob.buy_stops)
print("  last_trade_price:", lob.last_trade_price, "\n")

# now create a resting ask @105 then hit it with a market buy → last_trade_price=105
lob.add_order(Order("A2", "AAPL", "sell", 1, "limit", 105.0))
reports = lob.add_order(Order("M3", "AAPL", "buy", 1, "market"))
print("triggering trade reports:", reports)
print("  buy_stops after trigger:", lob.buy_stops)
print("  bids:", lob.bids)
print("  asks:", lob.asks, "\n")

"""# symmetrical: a sell‐stop at 95 into a resting bid
lob = LimitOrderBook("AAPL")
lob.add_order(Order("B1", "AAPL", "buy", 10, "limit", 100.0))
stop_sell = Order("S2", "AAPL", "sell", 5, "stop", 95.0)
print("add_order(S2):", lob.add_order(stop_sell))
print("  sell_stops:", lob.sell_stops)

# seed bid @95 & hit with market sell → last_trade_price=95 triggers S2
lob.add_order(Order("B2", "AAPL", "buy", 1, "limit", 95.0))
reports = lob.add_order(Order("M4", "AAPL", "sell", 1, "market"))
print("triggering trade reports:", reports)
print("  sell_stops after trigger:", lob.sell_stops)
print("  bids:", lob.bids)
print("  asks:", lob.asks)"""

"""# 1) Reset book
lob = LimitOrderBook("AAPL")
lob.add_order(Order("A1", "AAPL", "sell", 10, "limit", 100.0))
stop_buy = Order("S1", "AAPL", "buy", 5, "stop", 105.0)
lob.add_order(stop_buy)
lob.add_order(Order("A2", "AAPL", "sell", 1, "limit", 105.0))

# 2) Trigger both the big market buy and the stop
reports = lob.add_order(Order("M_trigger", "AAPL", "buy", 11, "market"))
print("all reports:", reports)
print("remaining book bids:", lob.bids)
print("remaining book asks:", lob.asks)
print("stops left:", lob.buy_stops)"""
'''