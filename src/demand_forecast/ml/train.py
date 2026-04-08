from __future__ import annotations

# Allow running as a script: python src/demand_forecast/ml/train.py
if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from demand_forecast.ml.features import FEATURE_COLUMNS, add_training_features

MODEL_VERSION = "1.0.0"


def load_training_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {"date", "store", "item", "sales"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_date = df["date"].max() - pd.Timedelta(days=29)
    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]
    return train_df, test_df


def generate_synthetic_training_data(data_path: Path) -> None:
    # Mirror scripts/download_data.py by sourcing rows from California Housing.
    from sklearn.datasets import fetch_california_housing

    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame

    # Convert source rows into the schema expected by this training pipeline.
    synthetic_df = pd.DataFrame(
        {
            "date": pd.date_range(start="2013-01-01", periods=len(frame), freq="D"),
            "store": 1,
            "item": 1,
            "sales": (frame["MedHouseVal"] * 100).round().clip(lower=0).astype("int32"),
        }
    )

    data_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(data_path, index=False)
    print(f"Generated synthetic dataset at {data_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data_path = root / "data" / "raw" / "train.csv"
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        generate_synthetic_training_data(data_path)

    raw_df = load_training_data(data_path)
    feature_df = add_training_features(raw_df)
    feature_df = feature_df.dropna(subset=FEATURE_COLUMNS + ["sales"])

    train_df, test_df = split_train_test(feature_df)
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty partition")

    model = LGBMRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["sales"])

    predictions = model.predict(test_df[FEATURE_COLUMNS])
    mae = mean_absolute_error(test_df["sales"], predictions)
    print(f"Test MAE: {mae:.4f}")

    out_path = model_dir / "model.pkl"
    joblib.dump({"model": model, "version": MODEL_VERSION}, out_path)
    print(f"Saved model bundle to {out_path}")


if __name__ == "__main__":
    main()
