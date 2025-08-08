# Copilot is used to complete this code.

import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# 1) Load headlines from Google News RSS
rss_url = "https://news.google.com/rss/search?q=AAPL&hl=en-US&gl=US&ceid=US:en"
feed = feedparser.parse(rss_url)

rows = []
for e in feed.entries:
    dt = pd.to_datetime(e.published).date()
    rows.append({"date": dt, "headline": e.title})

df_news = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

# 2) Score sentiment with VADER
sid = SentimentIntensityAnalyzer()
sent = df_news['headline'].apply(sid.polarity_scores).apply(pd.Series)
df_news = pd.concat([df_news, sent], axis=1)

# 3) Daily mean compound score
daily_sentiment = (
    df_news.groupby('date')['compound']
           .mean()
           .reset_index()
)

# 4) Download 6 months of raw AAPL OHLC
end_date   = pd.Timestamp.today().date()
start_date = end_date - pd.DateOffset(months=6)

raw = yf.download(
    "AAPL",
    start=start_date.strftime("%Y-%m-%d"),
    end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    auto_adjust=False
)

# 5) Flatten columns if they come in as a MultiIndex
if isinstance(raw.columns, pd.MultiIndex):
    # level 0 holds the fields: Open, High, Low, Close, Adj Close, Volume
    raw.columns = raw.columns.get_level_values(0)

# 6) Pull out Adj Close (or fall back to Close)
if "Adj Close" in raw.columns:
    series = raw["Adj Close"].copy()
elif "Close" in raw.columns:
    series = raw["Close"].copy()
else:
    raise KeyError(f"No 'Adj Close' or 'Close' in columns: {list(raw.columns)}")

series.name = "adj_close"

# 7) Reset index → two‐column DF and force column names
df_prices = series.reset_index()
df_prices.columns = ["date", "adj_close"]
df_prices["date"] = pd.to_datetime(df_prices["date"]).dt.date

# 8) Merge with sentiment
df_events = pd.merge(df_prices, daily_sentiment, on="date", how="left")

# safe fill – no chained‐assignment warning
df_events["compound"] = df_events["compound"].fillna(0)


# 1) Ensure df_events is sorted by date
df_events = df_events.sort_values('date').reset_index(drop=True)

# 2) Compute 2-day forward return via pct_change
#    (return_t = (price_{t+2} - price_t) / price_t)
df_events['2d_return'] = (
    df_events['adj_close']
             .pct_change(periods=2)   # (p_t - p_{t-2})/p_{t-2} at index t
             .shift(-2)               # move that value back to index t-2 → index t
)

# 3) Filter for high-sentiment “event” days
threshold = 0.3
event_days = (
    df_events
      .loc[df_events['compound'].abs() > threshold,
           ['date','compound','2d_return']]
      .dropna(subset=['2d_return'])
      .reset_index(drop=True)
)

print(f"Found {len(event_days)} event‐day reactions:\n")
print(event_days)

# 4) Bar chart of 2-day returns
plt.figure(figsize=(12, 6))
palette = event_days['compound'].apply(lambda c: 'green' if c > 0 else 'red')
sns.barplot(
    x='date',
    y='2d_return',
    data=event_days,
    palette=palette.values
)
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, ha='right')
plt.title('2-Day Returns After High-Sentiment AAPL News Events')
plt.xlabel('Event Date')
plt.ylabel('2-Day Return')
plt.tight_layout()
plt.show()

# 5) Summary statistics by sentiment sign
event_days['sign'] = np.where(event_days['compound'] > 0, 'positive', 'negative')
summary = (
    event_days
      .groupby('sign')['2d_return']
      .agg(count       = 'size',
           avg_return  = 'mean',
           median_ret  = 'median')
)
print("\nSummary by Sentiment Sign:")
print(summary)
