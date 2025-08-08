# copilot is used for this assignment

import pandas as pd
import numpy as np
from market_data_loader import MarketDataLoader
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # 1. Download & Clean Raw OHLCV
    loader = MarketDataLoader(interval="1d", period="3y")
    hist = loader.get_history("1810.HK")

    hist.ffill(inplace=True)
    hist.dropna(how="any", inplace=True)
    hist = remove_outliers_roll_z(hist, window=20, threshold=3.0)

    # 2. Compute Standard Technical Indicators
    hist = technical_indicators(hist)

    # 3–6. Feature Engineering (returns, vol, momentum, label, dropna)
    hist = engineer_features(hist)

    # 7. Select & Scale Features
    feature_cols = hist.select_dtypes(include=[np.number]).columns.tolist()
    hist_scaled = scale_features(hist[feature_cols], method="minmax")

    feature_cols = [c for c in hist_scaled.columns if c != 'label_5d']
    X = hist_scaled[feature_cols].copy()
    y = hist_scaled['label_5d'].copy()

    # 7a. Compute Correlation & Drop Collinear Features
    corr_abs = X.corr().abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
    X_reduced = X.drop(columns=to_drop)

    # 7b. Low-variance filter
    vt = VarianceThreshold(threshold=0.01)
    X_final = pd.DataFrame(
        vt.fit_transform(X_reduced),
        index=X_reduced.index,
        columns=X_reduced.columns[vt.get_support()]
    )

    # 8. PCA & auto‐naming report
    pca = PCA()
    pca.fit(X_final)
    evr = pca.explained_variance_ratio_

    top_components = np.argmax(np.cumsum(evr) >= 0.90) + 1
    print(f"\nKeeping {top_components} components to capture ≥90% variance")

    # This builds X_pca_df with columns like ['PC1_open', 'PC2_RSI', …]
    X_pca_df = pca_feature_report(X_final, pca, evr, top_components)

    # 9. Walk‐forward using renamed PCA features
    data = X_pca_df.copy()
    data['label'] = y.loc[data.index]

    wf_results = walk_forward_lr(
        df=data,
        feature_cols=X_pca_df.columns.tolist(),
        target_col='label',
        initial_train=200,
        test_size=50,
        step_size=50
    )

    if wf_results.empty:
        raise RuntimeError(
            f"No walk-forward blocks. n={len(data)}, "
            f"initial_train=200, test_size=50"
        )

    # 10. Metrics & plots (unchanged)
    wf_results['mse'] = (wf_results['y_true'] - wf_results['y_pred']) ** 2
    rmse_by_block = np.sqrt(wf_results.groupby('block')['mse'].mean())
    overall_rmse = np.sqrt(wf_results['mse'].mean())
    print(f"\nOverall out-of-sample RMSE: {overall_rmse:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(wf_results['date'], wf_results['y_true'], label='Actual', alpha=0.7)
    plt.plot(wf_results['date'], wf_results['y_pred'], label='Pred', alpha=0.7)
    plt.legend()
    plt.title("Walk-Forward Linear Regression: Predicted vs Actual")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    rmse_by_block.plot(marker='o')
    plt.xlabel("Block")
    plt.ylabel("RMSE")
    plt.title("RMSE per Test Block")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(
        "\nCheck block-wise RMSE above. "
        "If you see RMSE steadily rising, your model may be overfitting early data "
        "and failing to generalize to newer regimes.\n"
    )

    # 8. Assemble classification DataFrame
    data = X_final.copy()
    data['label_bin_5d'] = y.loc[data.index]

    # 9. Walk‐forward classification
    wf_clf = walk_forward_clf(
        df=data,
        feature_cols=X_final.columns.tolist(),
        target_col='label_bin_5d',
        initial_train=200,
        test_size=50,
        step_size=50)

    # Plotting probability vs actual up/down
    plt.figure(figsize=(12, 4))
    plt.plot(wf_clf['date'], wf_clf['y_proba'], label='Prob Up')
    plt.scatter(
        wf_clf['date'],
        wf_clf['y_true'],
        c=wf_clf['y_true'],
        cmap='coolwarm',
        marker='x',
        alpha=0.6
    )
    plt.legend()
    plt.show()

    # Block‐level classification metrics
    summary = wf_clf.groupby('block').apply(
        lambda df: pd.Series({
            'Accuracy': accuracy_score(df.y_true, df.y_pred),
            'Precision': precision_score(df.y_true, df.y_pred, zero_division=0),
            'Recall': recall_score(df.y_true, df.y_pred, zero_division=0),
            'F1': f1_score(df.y_true, df.y_pred, zero_division=0)
        }),
        include_groups=False
    )
    print(summary)


def walk_forward_clf(df, feature_cols, target_col,
                     initial_train=200, test_size=50, step_size=50):
    df = df.sort_index()
    n = len(df)

    # Fallback to single 70/30 split if too small
    if n < initial_train + test_size:
        initial_train = int(n * 0.7)
        test_size = n - initial_train
        step_size = test_size

    records = []
    block = 0

    for train_end in range(initial_train, n - test_size + 1, step_size):
        block += 1
        train_idx = df.index[:train_end]
        test_idx = df.index[train_end:train_end + test_size]

        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, feature_cols]
        y_test = df.loc[test_idx, target_col]

        clf = LogisticRegression(solver='liblinear', class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        for dt, true, pred, proba in zip(test_idx, y_test, y_pred, y_proba):
            records.append({
                'block': block,
                'date': dt,
                'y_true': true,
                'y_pred': pred,
                'y_proba': proba
            })

        print(f"Block {block}: {metrics}")

    return pd.DataFrame(records)


def pca_feature_report(X, pca, evr, top_k):
    """
    Build PCA‐transformed DataFrame, auto‐name each PC by its top original feature,
    and print:
      • Explained variance per PC
      • Original features fed into PCA
      • Top 5 original contributors to each PC
      • Final (renamed) PCA feature names

    Returns
    -------
    X_pca_df : pd.DataFrame
        PCA‐transformed feature matrix with renamed columns,
        e.g. ['PC1_Open', 'PC2_RSI', …]
    """
    # 1. Raw PC names
    raw_pcnames = [f"PC{i + 1}" for i in range(top_k)]

    # 2. Loading matrix: rows=PCs, cols=original features
    loadings = pd.DataFrame(
        data=pca.components_[:top_k, :],
        index=raw_pcnames,
        columns=X.columns
    )

    # 3. Determine top single feature for each PC
    top_feature = loadings.abs().idxmax(axis=1)
    # e.g. Series({'PC1':'open', 'PC2':'RSI', …})

    # 4. Build final PC names by appending the top feature
    final_pcnames = [
        f"{pc}_{top_feature[pc]}"
        for pc in raw_pcnames
    ]

    # 5. Wrap the transformed array in a DataFrame using final names
    X_pca_df = pd.DataFrame(
        data=pca.transform(X)[:, :top_k],
        index=X.index,
        columns=final_pcnames
    )

    # 6. Print explained variance
    print("\nExplained variance per principal component:")
    for i, var in enumerate(evr[:top_k], start=1):
        print(f"  {final_pcnames[i - 1]}: {var:.2%}")

    # 7. List original features
    print("\nOriginal features used in PCA:")
    print(" ", X.columns.tolist())

    # 8. Top 5 contributors
    print("\nTop 5 contributing originals to each PC:")
    for pc in raw_pcnames:
        top5 = (
            loadings
            .loc[pc]
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .index
            .tolist()
        )
        print(f"  {pc}_{top_feature[pc]} ← {top5}")

    # 9. Final PC names
    print("\nFinal PCA features for modeling:")
    print(" ", final_pcnames, "\n")

    return X_pca_df


def walk_forward_lr(df, feature_cols, target_col,
                    initial_train=200, test_size=50, step_size=50):
    """
    Expanding-window walk-forward Validation for LinearRegression.
    Dynamically adjusts initial_train/test_size if n is too small.
    Returns DataFrame with [block, date, y_true, y_pred].
    """
    # ensure date order
    df = df.sort_index()
    n = len(df)

    # if even one block cannot be formed, shrink to 70/30 single split
    if n < initial_train + test_size:
        initial_train = int(n * 0.7)
        test_size = n - initial_train
        step_size = test_size  # only one block
        if test_size <= 0:
            raise ValueError(
                f"Not enough data for walk-forward. n={n}, "
                f"70/30 fallback gave test_size={test_size}"
            )
        print(f">>> Adjusted to single-block split: "
              f"initial_train={initial_train}, test_size={test_size}")

    records = []
    block = 0

    # build non-overlapping expanding windows
    for train_end in range(initial_train, n - test_size + 1, step_size):
        block += 1

        train_idx = df.index[:train_end]
        test_idx = df.index[train_end: train_end + test_size]

        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, feature_cols]
        y_test = df.loc[test_idx, target_col]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for dt, true, pred in zip(test_idx, y_test, y_pred):
            records.append({
                'block': block,
                'date': dt,
                'y_true': true,
                'y_pred': pred
            })

    return pd.DataFrame(records)


def engineer_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute returns, rolling volatility, momentum features,
    5-day forward binary label, and drop any rows with NaNs.
    """
    # Add Returns & Rolling Volatility
    hist["return"] = hist["last_price"].pct_change()
    hist["vol"] = hist["return"].rolling(20).std()

    # Add Derived Momentum Features
    for lag in [1, 5, 10]:
        hist[f"mom_{lag}d"] = hist["last_price"] - hist["last_price"].shift(lag)

    # Create 5-Day Forward Binary Label
    hist["label_5d"] = np.where(
        hist["last_price"].shift(-5) > hist["last_price"],
        1,
        0
    )

    # Drop all NaNs introduced by shifts/rolling
    hist.dropna(how="any", inplace=True)

    return hist


def technical_indicators(hist):
    """    Add technical indicators to the historical data DataFrame."""
    # Calculate moving averages
    hist['SMA50'] = hist['last_price'].rolling(window=50).mean()
    hist['SMA200'] = hist['last_price'].rolling(window=200).mean()

    # Calculate the short-term and long-term EMAs and MACD
    short_window = 12
    long_window = 26
    hist['EMA12'] = hist['last_price'].ewm(span=short_window, adjust=False).mean()
    hist['EMA26'] = hist['last_price'].ewm(span=long_window, adjust=False).mean()
    hist['MACD'] = hist['EMA12'] - hist['EMA26']

    # Calculate RSI
    rsi_period = 14  # Common period for RSI
    delta = hist['last_price'].diff()

    # compute gain/loss arrays
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # wrap with the same index as hist
    gain_s = pd.Series(gain, index=hist.index)
    loss_s = pd.Series(loss, index=hist.index)

    # rolling averages now align properly
    avg_gain = gain_s.rolling(window=rsi_period).mean()
    avg_loss = loss_s.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    # Calculate the moving average and standard deviation
    window = 20
    no_of_std = 2

    hist['MA20'] = hist['last_price'].rolling(window).mean()
    hist['STD'] = hist['last_price'].rolling(window).std()
    hist['Upper Band'] = hist['MA20'] + (hist['STD'] * no_of_std)
    hist['Lower Band'] = hist['MA20'] - (hist['STD'] * no_of_std)

    return hist


def scale_features(df, method='minmax'):
    """
    Scale numeric columns of df using either min-max normalization or standard scaling.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with feature columns.
    method : {'minmax', 'standard'}
        'minmax'    → scales each feature to [0, 1]
        'standard'  → zero mean and unit variance

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the same index/columns, but numeric columns scaled.
    """
    df_scaled = df.copy()
    num_cols = df_scaled.select_dtypes(include=[np.number]).columns

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("`method` must be either 'minmax' or 'standard'")

    # Fit & transform only on numeric columns
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    return df_scaled


def remove_outliers_roll_z(df, window=20, threshold=3.0):
    num_cols = df.select_dtypes(include=[np.number]).columns
    rm = df[num_cols].rolling(window, min_periods=1).mean()
    rs = df[num_cols].rolling(window, min_periods=1).std()
    zs = (df[num_cols] - rm) / rs
    outlier_rows = zs.abs().gt(threshold).any(axis=1)
    return df.loc[~outlier_rows].copy()


if __name__ == "__main__":
    main()
