"""
Trains a lightweight per-crop price forecasting model using lag/rolling
features + XGBoost (falls back to GradientBoosting if xgboost unavailable).
Saves all trained models + the full price history to price_model.pkl for
use in the Streamlit app's "6-month price forecast" feature.
"""

import pandas as pd
import numpy as np
import pickle

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv("price_data.csv", parse_dates=["date"])

FORECAST_HORIZON = 6  # months ahead


def make_features(series: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"price": series.values})
    for lag in [1, 2, 3, 6, 12]:
        d[f"lag_{lag}"] = d["price"].shift(lag)
    d["rolling_mean_3"] = d["price"].shift(1).rolling(3).mean()
    d["rolling_std_3"] = d["price"].shift(1).rolling(3).std()
    d["month"] = pd.date_range(end="2100-01-01", periods=len(d), freq="MS").month  # placeholder, replaced below
    return d


models = {}
histories = {}
metrics = {}

for (state, crop), g in df.groupby(["state", "crop"]):
    g = g.sort_values("date").reset_index(drop=True)
    series = g["modal_price_rs_per_quintal"]

    feat = pd.DataFrame({"price": series.values})
    for lag in [1, 2, 3, 6, 12]:
        feat[f"lag_{lag}"] = feat["price"].shift(lag)
    feat["rolling_mean_3"] = feat["price"].shift(1).rolling(3).mean()
    feat["rolling_std_3"] = feat["price"].shift(1).rolling(3).std().fillna(0)
    feat["month"] = g["date"].dt.month.values

    feat = feat.dropna().reset_index(drop=True)
    X = feat.drop(columns=["price"])
    y = feat["price"]

    if len(X) < 20:
        continue

    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if HAS_XGB:
        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, pred)
        metrics[(state, crop)] = mape

    # refit on full data for best forward forecast
    model.fit(X, y)
    models[(state, crop)] = model
    histories[(state, crop)] = g[["date", "modal_price_rs_per_quintal"]].reset_index(drop=True)

avg_mape = np.mean(list(metrics.values())) if metrics else None
print(f"Trained {len(models)} per-(state,crop) price models")
print(f"Average MAPE on holdout: {avg_mape:.3%}" if avg_mape else "no holdout")

with open("price_model.pkl", "wb") as f:
    pickle.dump({
        "models": models,
        "histories": histories,
        "metrics": metrics,
        "avg_mape": avg_mape,
        "has_xgb": HAS_XGB,
        "horizon": FORECAST_HORIZON,
    }, f)

print("Saved price_model.pkl")
