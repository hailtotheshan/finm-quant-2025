# Copilot is used in this file to implement a walk-forward validation framework

import os
import backtesting
os.environ["OMP_NUM_THREADS"] = "3"
from IPython.display import display
from sklearn.base import clone
import pandas as pd
import numpy as np
from market_data_loader import MarketDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import seaborn as sns
from sklearn.metrics import roc_curve, auc

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score


def main():
    raw = load_and_clean("1810.HK")
    X_final, y_reg = engineer_and_scale(raw)

    # PCA for regression
    X_pca = apply_pca(X_final, variance_threshold=0.90)

    # Walk‐forward regression + summary
    wf_reg, reg_summary = run_regression_flow(
        X=X_pca,
        y=y_reg,
        initial_train=200,
        test_size=50,
        step_size=50,
        model=LinearRegression()
    )

    # Binary label & classification
    y_bin = (
            raw['last_price'].shift(-5)
            > raw['last_price']
    ).astype(int).loc[X_final.index]

    wf_clf, clf_summary = run_classification_flow(
        X=X_final,
        y_bin=y_bin,
        initial_train=200,
        test_size=50,
        step_size=50
    )

    # Now you have:
    #   wf_reg       -> walk-forward predictions for regression
    #   reg_summary  -> MSE, RMSE, MAE, R2 per block
    #   wf_clf       -> walk-forward predictions for classification
    #   clf_summary  -> Accuracy, Precision, Recall, F1 per block

    plot_walk_forward_performance(wf_reg, wf_clf)

    # (Optional) print or log your summaries
    print("Regression Summary:")
    print(reg_summary.to_markdown(index=False))

    print("Classification Summary:")
    print(clf_summary.to_markdown(index=False))

    # Assume LinearRegression & LogisticRegression were imported
    final_reg_model = LinearRegression()
    final_reg_model.fit(X_pca, y_reg)

    final_clf_model = LogisticRegression(solver='liblinear')
    final_clf_model.fit(X_final, y_bin)

    evaluate_and_interpret(
        wf_reg=wf_reg,
        reg_summary=reg_summary,
        wf_clf=wf_clf,
        clf_summary=clf_summary,
        model_reg=final_reg_model,
        model_clf=final_clf_model,
        X_reg=X_pca,
        X_clf=X_final,
        top_n=10
    )

    # ── Part 5: Unsupervised Exploration ──
    print("\n\n=== Part 5: Unsupervised Exploration ===")
    # Use your final feature matrix X_final (or X_pca if you prefer clustering PCs)
    kmeans_labels, hier_labels, linkage_mat = run_unsupervised_exploration(X_final, final_k=4)

    # wf_clf has columns ['block','date','y_true','y_pred','y_proba']
    # Make y_pred into a Series indexed by date:
    signal_series = (
        wf_clf
        .set_index('date')['y_pred']  # 0 or 1
        .sort_index()
        .rename('signal')
    )

    # 1) Build OHLC from 'last_price'
    # rename columns to match backtesting.py
    df_bt = (
        raw.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "last_price": "Close",
            "volume": "Volume"
        })
        # backtesting.py requires a monotonic index
        .sort_index()
    )

    # 2) Merge in your model ‘signal’
    df_bt = df_bt.join(signal_series, how='left')

    # 3) Fill any gaps (no signal → flat)
    df_bt['signal'].fillna(0, inplace=True)

    print(df_bt)



def evaluate_and_interpret(
        wf_reg: pd.DataFrame,
        reg_summary: pd.DataFrame,
        wf_clf: pd.DataFrame,
        clf_summary: pd.DataFrame,
        model_reg,
        model_clf,
        X_reg: pd.DataFrame,
        X_clf: pd.DataFrame,
        top_n: int = 10
):
    # right at the top of evaluate_and_interpret:
    wf_reg = wf_reg.copy()
    wf_reg['date'] = X_reg.index[-len(wf_reg):]  # last N dates

    wf_clf = wf_clf.copy()
    wf_clf['date'] = X_clf.index[-len(wf_clf):]  # same idea

    # ── 0. Coerce 'block' column into both summaries ──
    for df in (reg_summary, clf_summary):
        if 'block' not in df.columns:
            df.reset_index(inplace=True)  # moves index into column
            df.rename(columns={df.columns[0]: 'block'}, inplace=True)

    # ── 1. Combined Evaluation Table ──
    eval_df = (
        pd.concat(
            [
                reg_summary.set_index('block'),
                clf_summary.set_index('block')
            ],
            axis=1
        )
        .rename_axis('Block')
    )
    print("\n=== Block-Level Evaluation Metrics ===")
    print(eval_df.to_markdown())

    # 2. Confusion Matrix (Classification)
    plt.figure(figsize=(4, 3))
    cm = confusion_matrix(wf_clf['y_true'], wf_clf['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # 3. ROC Curve (if probabilities available)
    if 'y_proba' in wf_clf.columns:
        fpr, tpr, _ = roc_curve(wf_clf['y_true'], wf_clf['y_proba'])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 4. Prediction Curves
    #   4a. Regression
    plt.figure(figsize=(10, 3))
    plt.plot(wf_reg['date'], wf_reg['y_true'], label='Actual', alpha=0.7)
    plt.plot(wf_reg['date'], wf_reg['y_pred'], label='Predicted', alpha=0.7)
    plt.title('Regression: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    #   4b. Classification
    plt.figure(figsize=(10, 2))
    plt.step(wf_clf['date'], wf_clf['y_true'], where='post',
             label='Actual', alpha=0.7)
    plt.step(wf_clf['date'], wf_clf['y_pred'], where='post',
             label='Predicted', alpha=0.7)
    plt.title('Classification: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. Feature Interpretation Helper
    def _interpret(model, features, X=None):
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            df = pd.DataFrame({
                'feature': features,
                'importance': imps
            })
        elif hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            df = pd.DataFrame({
                'feature': features,
                'coefficient': coefs,
                'importance': np.abs(coefs)
            })
        elif _SHAP_AVAILABLE and X is not None:
            explainer = shap.Explainer(model, X)
            shap_vals = explainer(X)
            mean_abs = np.mean(np.abs(shap_vals.values), axis=0)
            df = pd.DataFrame({
                'feature': features,
                'importance': mean_abs
            })
        else:
            raise ValueError("Model has no importances/coefficients and SHAP not available.")
        return df.sort_values('importance', ascending=False).head(top_n)

    # 6. Print Top Features
    print("\n=== Top Regression Features ===")
    display(_interpret(model_reg, X_reg.columns.tolist(), X_reg))

    print("\n=== Top Classification Features ===")
    display(_interpret(model_clf, X_clf.columns.tolist(), X_clf))


def get_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    Returns a pandas Series with MSE, RMSE, MAE, and R².
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.Series({
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })


def get_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute classification metrics.
    Always returns Accuracy, Precision, Recall, F1.
    If y_proba is provided, also returns ROC AUC.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_proba is not None:
        try:
            metrics['ROC AUC'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['ROC AUC'] = np.nan
    return pd.Series(metrics)


def plot_walk_forward_performance(wf_reg: pd.DataFrame,
                                  wf_clf: pd.DataFrame):
    """
    Visualize walk-forward metrics over time for regression and classification,
    and compare each model’s predicted vs. actual values.

    Parameters
    ----------
    wf_reg : pd.DataFrame
        Walk-forward output of regression flow. Must contain columns
        ['block', 'y_true', 'y_pred'] and a datetime or integer index
        that reflects time order.
    wf_clf : pd.DataFrame
        Walk-forward output of classification flow. Must contain columns
        ['block', 'y_true', 'y_pred'] and a datetime or integer index
        that reflects time order.

    Returns
    -------
    None
    """

    # -----------------------------
    # 1. AGGREGATE METRICS PER BLOCK
    # -----------------------------
    reg_summary = (
        wf_reg
        .groupby('block')[['y_true', 'y_pred']]
        .apply(lambda df: pd.Series({
            'MSE': mean_squared_error(df.y_true, df.y_pred),
            'RMSE': np.sqrt(mean_squared_error(df.y_true, df.y_pred)),
            'MAE': mean_absolute_error(df.y_true, df.y_pred),
            'R2': r2_score(df.y_true, df.y_pred)
        }))
    )

    clf_summary = (
        wf_clf
        .groupby('block')[['y_true', 'y_pred']]
        .apply(lambda df: pd.Series({
            'Accuracy': accuracy_score(df.y_true, df.y_pred),
            'Precision': precision_score(df.y_true, df.y_pred, zero_division=0),
            'Recall': recall_score(df.y_true, df.y_pred, zero_division=0),
            'F1': f1_score(df.y_true, df.y_pred, zero_division=0)
        }))
    )

    # -----------------------------
    # 2. PLOT METRICS TIME-SERIES
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    reg_summary.plot(ax=axes[0], marker='o', fontsize=10)
    axes[0].set_title('Regression Metrics Over Blocks')
    axes[0].set_ylabel('Metric Value')
    axes[0].grid(alpha=0.3)
    axes[0].legend(title='Reg Metric')

    clf_summary.plot(ax=axes[1], marker='s', fontsize=10)
    axes[1].set_title('Classification Metrics Over Blocks')
    axes[1].set_xlabel('Block Number')
    axes[1].set_ylabel('Metric Value')
    axes[1].grid(alpha=0.3)
    axes[1].legend(title='Clf Metric')

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # 3. PLOT PREDICTED VS ACTUAL FOR EACH MODEL OVER TIME
    # ----------------------------------------------------
    fig, (ax_reg, ax_clf) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # 3a. Regression: continuous predictions
    ax_reg.plot(wf_reg.index, wf_reg['y_true'], label='Actual', color='tab:blue', lw=1.5)
    ax_reg.plot(wf_reg.index, wf_reg['y_pred'], label='Predicted', color='tab:orange', lw=1.2, alpha=0.8)
    ax_reg.set_title('Regression: Actual vs Predicted Over Time')
    ax_reg.set_ylabel('Target Value')
    ax_reg.legend()
    ax_reg.grid(alpha=0.3)

    # 3b. Classification: discrete predictions
    ax_clf.step(wf_clf.index, wf_clf['y_true'], where='post',
                label='Actual (0/1)', color='tab:green', lw=1.5)
    ax_clf.step(wf_clf.index, wf_clf['y_pred'], where='post',
                label='Predicted (0/1)', color='tab:red', lw=1.2, alpha=0.8)
    ax_clf.set_title('Classification: Actual vs Predicted Over Time')
    ax_clf.set_ylabel('Class Label')
    ax_clf.set_xlabel('Time Index')
    ax_clf.legend()
    ax_clf.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def remove_outliers_roll_z(df, window=20, threshold=3.0):
    num_cols = df.select_dtypes(include=[np.number]).columns
    rm = df[num_cols].rolling(window, min_periods=1).mean()
    rs = df[num_cols].rolling(window, min_periods=1).std()
    zs = (df[num_cols] - rm) / rs
    outlier_rows = zs.abs().gt(threshold).any(axis=1)
    return df.loc[~outlier_rows].copy()


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


def load_and_clean(ticker, interval="1d", period="5y"):
    loader = MarketDataLoader(interval=interval, period=period)
    df = loader.get_history(ticker)
    df.ffill(inplace=True)
    df.dropna(how="any", inplace=True)
    return remove_outliers_roll_z(df, window=20, threshold=3.0)


def engineer_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns, rolling volatility, momentum,
    continuous 5-day return, 5-day up/down label, then drop NaNs.
    """
    # 1) Daily pct return & 20-day vol
    hist["return"] = hist["last_price"].pct_change()
    hist["vol"] = hist["return"].rolling(20).std()

    # 2) Momentum features
    for lag in [1, 5, 10]:
        hist[f"mom_{lag}d"] = hist["last_price"] - hist["last_price"].shift(lag)

    # ── NEW: continuous 5-day forward return
    hist["ret_5d"] = hist["last_price"].shift(-5) / hist["last_price"] - 1

    # ── binary up/down label over next 5 days
    hist["label_5d"] = np.where(
        hist["last_price"].shift(-5) > hist["last_price"],
        1,
        0
    )

    # 3) drop any rows with NaNs introduced by shifts/rolling
    hist.dropna(how="any", inplace=True)
    return hist


def engineer_and_scale(df: pd.DataFrame):
    """
    1) Add technical indicators
    2) Engineered features (incl. ret_5d & label_5d)
    3) z-score scale all continuous numeric cols (exclude targets/labels/binary/discrete)
    4) Split into X (drop labels/targets) and y_reg = ret_5d (unscaled)
    5) Drop collinear & low-variance
    """
    # Step 1–2: build features
    df = technical_indicators(df)
    df = engineer_features(df)

    # Step 3: select numeric cols, exclude label_5d & ret_5d and any binary/discrete columns from scaling
    # Only scale continuous-valued predictors
    num = df.select_dtypes(include=[np.number]).copy()
    # Find columns to exclude from scaling (targets and binary/discrete labels)
    exclude_cols = ["label_5d", "ret_5d"]
    # Also exclude any columns that are binary/discrete (nunique <= 2)
    exclude_cols += [col for col in num.columns if num[col].nunique() <= 2 and col not in exclude_cols]
    feature_cols = [col for col in num.columns if col not in exclude_cols]

    # Scale only continuous features
    scaled_features = scale_features(num[feature_cols], method="zscore")
    # Keep excluded columns as-is, append to scaled features
    for col in exclude_cols:
        if col in num.columns:
            scaled_features[col] = num[col]

    # Step 4: separate X/y_reg
    X = scaled_features.drop(columns=["label_5d", "ret_5d"], errors="ignore")
    y_reg = num["ret_5d"]  # Use unscaled target for regression

    # Step 5a: drop collinear (corr > 0.90)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.90)]
    X = X.drop(columns=to_drop)

    # Step 5b: drop low-variance (<0.01)
    vt = VarianceThreshold(0.01)
    X_final = pd.DataFrame(
        vt.fit_transform(X),
        index=X.index,
        columns=X.columns[vt.get_support()]
    )

    return X_final, y_reg


def apply_pca(X, variance_threshold=0.90):
    pca = PCA()
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    k = np.argmax(np.cumsum(evr) >= variance_threshold) + 1
    X_pca = pca_feature_report(X, pca, evr, k)
    return X_pca


def run_regression_flow(X, y, initial_train, test_size, step_size, model=None):
    """
    Walk-forward regression.
    Returns:
      - wf_reg: DataFrame with columns ['block','y_true','y_pred', ...]
      - reg_summary: DataFrame indexed by block with metrics MSE, RMSE, MAE, R2
    """
    if model is None:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    wf_records = []
    n_samples = X.shape[0]
    start = initial_train
    block = 0
    # expanding‐window walk‐forward, block numbers start at 1
    while start + test_size <= n_samples:
        block += 1
        train_idx = list(range(start))
        test_idx = list(range(start, start + test_size))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # tag every sample in this block with the same block number
        for true, pred in zip(y_test, y_pred):
            wf_records.append({
                'block': block,
                'y_true': true,
                'y_pred': pred})
        start += step_size

    # Assemble walk‐forward DataFrame
    wf_reg = pd.DataFrame(wf_records)

    # Compute regression metrics per block
    # after you’ve built wf_reg
    reg_summary = (
        wf_reg
        .groupby('block')[['y_true', 'y_pred']]  # pick just the cols you need
        .apply(lambda df: get_regression_metrics(
            df['y_true'], df['y_pred'])
               )
        .reset_index()
    )

    return wf_reg, reg_summary


def run_classification_flow(X, y_bin, **wf_kwargs):
    data = X.copy()
    data['label_bin_5d'] = y_bin
    wf = walk_forward_clf(
        df=data,
        feature_cols=X.columns.tolist(),
        target_col='label_bin_5d',
        **wf_kwargs
    )

    # only grab y_true and y_pred before applying
    summary = (
        wf
        .groupby('block')[['y_true', 'y_pred']]
        .apply(lambda df: pd.Series({
            'Accuracy': accuracy_score(df.y_true, df.y_pred),
            'Precision': precision_score(df.y_true, df.y_pred, zero_division=0),
            'Recall': recall_score(df.y_true, df.y_pred, zero_division=0),
            'F1': f1_score(df.y_true, df.y_pred, zero_division=0)
        }))
    )

    print(summary)
    return wf, summary


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


def scale_features(df, method="zscore"):
    # Defensive: don't scale discrete/binary columns
    discrete_cols = [col for col in df.columns if df[col].nunique() <= 2]
    continuous_cols = [col for col in df.columns if col not in discrete_cols]
    if method == "zscore":
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df[continuous_cols])
        # Construct scaled DataFrame for continuous, then reattach discrete columns
        scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=continuous_cols)
        for col in discrete_cols:
            scaled_df[col] = df[col]
        return scaled_df
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")


def plot_silhouette_scores(X, ks=range(2, 11)):
    """Compute & plot silhouette score vs. number of clusters."""
    X_scaled = StandardScaler().fit_transform(X)
    sil_scores = []
    for k in ks:
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
    plt.figure(figsize=(6, 3))
    plt.plot(list(ks), sil_scores, 'o-', color='tab:blue')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Average silhouette')
    plt.title('Silhouette Analysis')
    plt.grid(alpha=0.3)
    plt.show()


def plot_dendrogram(X, truncate_level=5):
    """Compute Ward linkage and plot truncated dendrogram."""
    X_scaled = StandardScaler().fit_transform(X)
    Z = linkage(X_scaled, method='ward')
    plt.figure(figsize=(8, 4))
    dendrogram(Z, truncate_mode='level', p=truncate_level, leaf_rotation=90)
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Ward distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.tight_layout()
    plt.show()
    return Z


def assign_and_plot_time_clusters(X, labels, title="Cluster Assignments Over Time"):
    """
    Simple 1×N heatmap of cluster IDs over time using imshow.
    X      : DataFrame indexed by datetime (or anything monotonic)
    labels : array‐like of length==len(X), integer cluster IDs
    """
    arr = np.array(labels)[np.newaxis, :]  # shape (1, N)
    fig, ax = plt.subplots(figsize=(12, 1.5))

    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="tab20",
        origin="lower",
    )
    ax.set_yticks([])  # hide the single row axis
    # show only a few date‐ticks so it stays readable:
    N = len(X)
    step = max(1, N // 10)
    ax.set_xticks(np.arange(0, N, step))
    ax.set_xticklabels(
        [X.index[i].strftime("%Y-%m-%d") for i in range(0, N, step)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation="vertical", label="Cluster ID")
    plt.tight_layout()
    plt.show()


def run_unsupervised_exploration(X, final_k=4):
    # 1) Silhouette
    plot_silhouette_scores(X)

    # 2) Dendrogram + linkage
    Z = plot_dendrogram(X)

    # 3) K‐Means final
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=final_k, random_state=42).fit(X_scaled)
    k_labels = km.labels_

    # 4) Print centroids
    centroids = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    print("\nCluster centroids (scaled space):\n", centroids)

    # 5) Plot K-Means regimes
    assign_and_plot_time_clusters(X, k_labels, title=f"KMeans (k={final_k}) Over Time")

    # 6) Optional: hierarchy flat clusters
    h_labels = fcluster(Z, t=final_k, criterion="maxclust")
    assign_and_plot_time_clusters(X, h_labels, title=f"Hierarchical (k={final_k}) Over Time")

    return k_labels, h_labels, Z


if __name__ == "__main__":
    main()
