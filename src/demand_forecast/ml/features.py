from __future__ import annotations

from datetime import date as date_type

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "dayofweek",
    "month",
    "is_month_start",
    "store",
    "item",
]


def add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["store", "item", "date"]).reset_index(drop=True)

    grouped_sales = frame.groupby(["store", "item"], sort=False)["sales"]
    frame["lag_7"] = grouped_sales.shift(7)
    frame["lag_14"] = grouped_sales.shift(14)
    frame["lag_28"] = grouped_sales.shift(28)

    # Use shifted windows so rolling aggregates only use past observations.
    frame["rolling_mean_7"] = grouped_sales.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).mean()
    )
    frame["rolling_mean_28"] = grouped_sales.transform(
        lambda values: values.shift(1).rolling(window=28, min_periods=28).mean()
    )
    frame["rolling_std_7"] = grouped_sales.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).std()
    )

    frame["dayofweek"] = frame["date"].dt.dayofweek.astype("int16")
    frame["month"] = frame["date"].dt.month.astype("int16")
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype("int16")
    frame["store"] = frame["store"].astype("int16")
    frame["item"] = frame["item"].astype("int16")
    return frame


def build_inference_features(
    store: int,
    item: int,
    date: date_type,
    historical_sales: list[float],
) -> pd.DataFrame:
    if len(historical_sales) != 28:
        raise ValueError("historical_sales must contain exactly 28 values")

    target_date = pd.Timestamp(date)
    history = np.asarray(historical_sales, dtype=float)
    rolling_std_7 = float(np.std(history[-7:], ddof=1))
    if np.isnan(rolling_std_7):
        rolling_std_7 = 0.0

    row = {
        "lag_7": float(history[-7]),
        "lag_14": float(history[-14]),
        "lag_28": float(history[-28]),
        "rolling_mean_7": float(np.mean(history[-7:])),
        "rolling_mean_28": float(np.mean(history)),
        "rolling_std_7": rolling_std_7,
        "dayofweek": int(target_date.dayofweek),
        "month": int(target_date.month),
        "is_month_start": int(target_date.is_month_start),
        "store": int(store),
        "item": int(item),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
