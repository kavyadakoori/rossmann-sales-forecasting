import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import TARGET_COLUMN


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2 for a forecast result."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Public wrapper used by the notebook-style evaluation flow."""
    return compute_metrics(y_true, y_pred)


# def evaluate_model_on_validation(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
#     """Run model evaluation on a validation split."""
#     preds = model.predict(X_val)
#     return evaluate_predictions(y_val, preds)


def evaluate_forecast_results(test_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict[str, float]:
    """Compare held-out Sales with forecast output."""
    merged = test_df[["Store", "Date", TARGET_COLUMN]].merge(
        forecast_df[["Store", "Date", "Sales"]].rename(columns={"Sales": "Predicted_Sales"}),
        on=["Store", "Date"],
        how="inner",
    )

    return evaluate_predictions(merged[TARGET_COLUMN], merged["Predicted_Sales"])
