import pandas as pd
import yfinance as yf
import random
import matplotlib.pyplot as plt


start_date = '2001-01-01'
end_date   = '2018-01-01'

# Fetch daily data for GOOG from Yahoo Finance
df = yf.download('GOOG', start=start_date, end=end_date)

# Store in goog_data
goog_data = df.copy()

random.seed(6666)

# Prepare signals DataFrame with the same dates
signals = pd.DataFrame(index=goog_data.index)

# Randomly assign 0 (no position) or 1 (long one share)
signals['signal'] = [random.choice([0, 1]) for _ in range(len(signals))]

# Compute orders: +1 when we go from 0→1 (buy), –1 for 1→0 (sell)
signals['orders'] = signals['signal'].diff()

plt.figure(figsize=(14, 6))
plt.plot(goog_data['Close'], label='GOOG Close Price', color='black', lw=1)

# Plot buy signals (orders == +1)
buy = signals[signals['orders'] == 1.0]
plt.scatter(buy.index, goog_data.loc[buy.index, 'Close'],
            marker='^', color='green', label='Buy', s=100)

# Plot sell signals (orders == –1)
sell = signals[signals['orders'] == -1.0]
plt.scatter(sell.index, goog_data.loc[sell.index, 'Close'],
            marker='v', color='red', label='Sell', s=100)

plt.title('GOOG Price with Buy/Sell Signals')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Initialize portfolio DataFrame
portfolio = pd.DataFrame(index=signals.index)

# Number of shares held each day (0 or 1)
portfolio['positions'] = signals['signal']

# Daily price for valuation
portfolio['price'] = goog_data['Close']

# Holding value = shares × price
portfolio['holdings'] = portfolio['positions'] * portfolio['price']

# Initial cash
initial_capital = 10_000.0

# Cash: start with initial_capital and subtract cost of trades
portfolio['cash'] = (initial_capital
                     - (signals['orders'] * portfolio['price']).cumsum())

# Total portfolio value = cash + holdings
portfolio['total'] = portfolio['cash'] + portfolio['holdings']


plt.figure(figsize=(14, 6))

plt.plot(portfolio['total'],   label='Total Value',   color='blue',  lw=2)
plt.plot(portfolio['cash'],    label='Cash',          color='orange', lw=1)
plt.plot(portfolio['holdings'],label='Holdings Value',color='green',  lw=1)

plt.title('Portfolio Value Over Time')
plt.ylabel('Value ($)')
plt.legend()
plt.show()



# 1) Fetch GOOG data (same as before)
start_date = '2001-01-01'
end_date   = '2018-01-01'
goog_data  = yf.download('GOOG', start=start_date, end=end_date).copy()

# 2) Construct signals: always 1 share in hand
signals_one = pd.DataFrame(index=goog_data.index)
signals_one['signal'] = 1                # long 1 share for every date
# Mark buy on first date, sell on last date
signals_one['orders'] = signals_one['signal'].diff().fillna(1)
signals_one.loc[signals_one.index[-1], 'orders'] = -1

# Initialize portfolio
portfolio_one = pd.DataFrame(index=signals_one.index)
portfolio_one['positions'] = signals_one['signal']
portfolio_one['price']     = goog_data['Close']
portfolio_one['holdings']  = portfolio_one['positions'] * portfolio_one['price']

initial_capital = 10_000.0
# cash: subtract cost of buy/sell trades
portfolio_one['cash'] = (
    initial_capital
    - (signals_one['orders'] * portfolio_one['price']).cumsum()
)
portfolio_one['total'] = portfolio_one['cash'] + portfolio_one['holdings']

# Plot the one-position portfolio breakdown
plt.figure(figsize=(14, 6))

plt.plot(portfolio_one.index, portfolio_one['total'],
         label='Total Portfolio Value', color='blue', lw=2)
plt.plot(portfolio_one.index, portfolio_one['cash'],
         label='Cash', color='orange', lw=1)
plt.plot(portfolio_one.index, portfolio_one['holdings'],
         label='Holdings Value', color='green', lw=1)

plt.title('One-Position Strategy Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

# Random strategy (from previous run)
plt.plot(portfolio.index,       portfolio['total'],
         label='Random Strategy', color='gray',  lw=1.5, alpha=0.7)

# One-position (buy-and-hold)
plt.plot(portfolio_one.index, portfolio_one['total'],
         label='One-Position Strategy', color='blue', lw=2)

plt.title('Equity Curve: Random vs. One-Position Strategy')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()

