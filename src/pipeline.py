from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.preprocessing import run_pipeline
from src.feature_engineering import build_feature_dataset
from src.train import train_model
from src.forecast import forecast_future
from src.evaluate import evaluate_forecast_results


def main():
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) preprocess raw data
    artifacts = run_pipeline()
    print('1. Finished Data preprocessing.')

    # 2) build engineered features from merged training data
    train_features = build_feature_dataset(artifacts["merged"])
    dates = pd.to_datetime(
        {
            "year": train_features["Year"],
            "month": train_features["Month"],
            "day": train_features["Day"],
        }
    )
    min_date = dates.min()
    max_date = dates.max()
    print(f'2. Finished Feature Engineering. Training Data range: {min_date} to {max_date}')

    # 3) train model
    model, search_results, val_metrics = train_model(train_features)
    print("3. Finished Model Training.")

    # # Load Model
    # model = pd.read_pickle("src/rossmann_lightgbm.pkl")

    # 4) Test with already pre-merged splitted test & kaggle_test.csv (with store metadata)
    test_df = pd.read_csv("data/processed/test.csv")
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    print(f'4. Loaded Test data. Test Data range: {test_df["Date"].min()} to {test_df["Date"].max()}')

    # 5) forecast
    feature_cols = [
        c for c in train_features.columns if c not in {"Sales", "Open"}
    ]
    print("5. Started Forecasting on Test Data...")
    predictions = forecast_future(
        history_df=train_features,
        future_df=test_df,
        model=model,
        feature_cols=feature_cols,
    )
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    print(predictions.head())

    test_results = evaluate_forecast_results(test_df, predictions)
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
