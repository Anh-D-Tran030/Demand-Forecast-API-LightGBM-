from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from demand_forecast.ml.features import REQUIRED_FEATURES


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data_path = root / "data" / "raw" / "california_housing.csv"
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run scripts/download_data.py first."
        )

    df = pd.read_csv(data_path)
    X = df[REQUIRED_FEATURES]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Validation MAE: {mae:.4f}")

    out_path = model_dir / "model.pkl"
    joblib.dump({"model": model, "features": REQUIRED_FEATURES}, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
