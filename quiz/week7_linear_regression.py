# backtest_linear_regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from backtesting import Backtest, Strategy
from market_data_loader import MarketDataLoader
import matplotlib.pyplot as plt


class LRStrategy(Strategy):
    """
    A backtesting.py strategy that goes long if predicted next-close > current close,
    else goes short.
    """

    def init(self):
        # self.data.Pred is the column we’ll inject into the DataFrame
        self.pred = self.data.Pred

    def next(self):
        price = self.data.Close[-1]
        prediction = self.pred[-1]

        if prediction > price:
            if not self.position.is_long:
                self.buy()
        else:
            if not self.position.is_short:
                self.sell()


def prepare_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """
    Add lagged closes, a rolling average, and daily return to df.
    """
    df = df.copy()
    # Lagged closes
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    # Rolling mean of last n_lags
    df['ma'] = df['Close'].rolling(window=n_lags).mean()

    # Daily return
    df['ret'] = df['Close'].pct_change()

    return df.dropna()


def main():
    # 1) Load 5 years of daily data for AAPL
    loader = MarketDataLoader(interval="1d", period="5y")
    raw = loader.get_history("AAPL")

    # Rename to OHLCV as expected by backtesting.py
    df = raw.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'last_price': 'Close',
        'volume': 'Volume'
    })[['Open', 'High', 'Low', 'Close', 'Volume']]

    # 2) Feature engineering
    feat = prepare_features(df, n_lags=5)

    # 3) Create target: next day's Close
    feat['Target'] = feat['Close'].shift(-1)
    feat = feat.dropna()

    # 4) Split into train (80%) and test (20%)
    split = int(len(feat) * 0.8)
    train, test = feat.iloc[:split], feat.iloc[split:]

    feature_cols = [c for c in feat.columns if c.startswith('lag_')] + ['ma', 'ret']
    X_train = train[feature_cols]
    y_train = train['Target']
    X_test = test[feature_cols]

    # 5) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6) Train linear regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 7) Predict next-day close on test set
    test = test.copy()
    test['Pred'] = model.predict(X_test_scaled)

    # 8) Prepare DataFrame for backtesting (only test period)
    bt_df = test[['Open', 'High', 'Low', 'Close', 'Volume', 'Pred']]

    # 9) Run backtest
    bt = Backtest(
        bt_df,
        LRStrategy,
        cash=10_000,
        commission=0.0005,
        trade_on_close=True,
        hedging=True  # allow both long and short
    )
    stats = bt.run()

    # 10) Output results
    print(stats[['Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]']])
    bt.plot()


if __name__ == "__main__":
    main()
