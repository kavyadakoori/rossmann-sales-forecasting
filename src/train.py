import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from .config import MODEL_PATH, TARGET_COLUMN, TRAIN_END_DATE, FEATURE_SPLIT_DATE
from .data_loader import load_processed_data


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_validation_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split engineered data into train and validation using the notebook cutoff."""
    df = df.copy()

    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    else:
        df["Date"] = pd.to_datetime(df["Date"])

    df = df[df["Open"] == 1].copy()
    df = df.drop(columns=["Open"], errors="ignore")

    train_df = df[df["Date"] <= TRAIN_END_DATE].copy()
    val_df = df[(df["Date"] > TRAIN_END_DATE) & (df["Date"] <= FEATURE_SPLIT_DATE)].copy()

    return train_df, val_df


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model feature columns, excluding the target and date."""
    return [column for column in df.columns if column not in {TARGET_COLUMN, "Date"}]


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """Run RandomizedSearchCV on train data using time-series cross-validation."""
    lgbm = LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    param_grid = {
        "n_estimators": [300, 500, 700, 1000],
        "learning_rate": [0.02, 0.05, 0.08, 0.1],
        "num_leaves": [15, 31, 50, 70],
        "max_depth": [-1, 5, 8, 12],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_grid,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(X_train, y_train)
    return random_search


def train_model(df: pd.DataFrame) -> tuple[LGBMRegressor, dict[str, object], dict[str, float]]:
    """Correct notebook flow:
    1) tune on train data
    2) evaluate on validation data
    3) fit final model on all available training data
    4) save the final model
    """
    train_df, val_df = train_validation_split(df)

    feature_cols = build_feature_columns(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN]

    random_search = tune_lightgbm(X_train, y_train)
    best_model = random_search.best_estimator_

    val_pred = best_model.predict(X_val)
    val_metrics = compute_metrics(y_val, val_pred)
    print(f"Validation metrics: {val_metrics}")

    search_results = {
        "best_params": random_search.best_params_,
        "best_cv_rmse": -random_search.best_score_,
    }

    # Fit the final model on the full available training history (train + validation)
    X_final = df.copy()
    if "Open" in X_final.columns:
        X_final = X_final[X_final["Open"] == 1].copy()
        X_final = X_final.drop(columns=["Open"], errors="ignore")
    y_final = X_final[TARGET_COLUMN]
    X_final = X_final[feature_cols]

    final_model = LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **random_search.best_params_,
    )
    final_model.fit(X_final, y_final)

    save_model(final_model)

    return final_model, search_results, val_metrics


def save_model(model: LGBMRegressor) -> str:
    """Persist the trained model artifact."""
    joblib.dump(model, MODEL_PATH)
    return str(MODEL_PATH)


def main() -> tuple[LGBMRegressor, dict[str, object], dict[str, float]]:
    """Load processed features, tune/evaluate the LightGBM model, refit on all data, and save the artifact."""
    feature_data = load_processed_data()["train_features"].copy()
    model, search_results, val_metrics = train_model(feature_data)
    return model, search_results, val_metrics
