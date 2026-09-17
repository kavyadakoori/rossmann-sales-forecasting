import pandas as pd

from .config import (
    FEATURE_SPLIT_DATE,
    KAGGLE_TEST_PATH,
    MERGED_DATA_PATH,
    PROCESSED_DATA_DIR,
    TEST_DATA_PATH,
)
from .data_loader import load_forecast_test_data, load_store_data, load_training_data


def clean_store_data(store_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing store metadata values using the same rules from the notebooks."""
    store = store_df.copy()

    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(
        store["CompetitionDistance"].median()
    )
    store["CompetitionOpenSinceMonth"] = store["CompetitionOpenSinceMonth"].fillna(0)
    store["CompetitionOpenSinceYear"] = store["CompetitionOpenSinceYear"].fillna(0)
    store["Promo2SinceWeek"] = store["Promo2SinceWeek"].fillna(0)
    store["Promo2SinceYear"] = store["Promo2SinceYear"].fillna(0)
    store["PromoInterval"] = store["PromoInterval"].fillna("None")

    return store


def prepare_merged_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the merged training dataset and save the cleaned store metadata."""
    train_df = load_training_data().copy()
    store_df = clean_store_data(load_store_data())

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    store_df.to_csv(PROCESSED_DATA_DIR / "store.csv", index=False)

    train_df["Date"] = pd.to_datetime(train_df["Date"])

    merged_df = train_df.merge(store_df, on="Store", how="left")
    merged_df = merged_df.sort_values(["Store", "Date"]).reset_index(drop=True)

    merged_df.to_csv(MERGED_DATA_PATH, index=False)

    return merged_df


def prepare_kaggle_test_dataset() -> pd.DataFrame:
    """Save the competition test dataset enriched with store metadata."""
    future_df = load_forecast_test_data().copy()
    store_df = clean_store_data(load_store_data())
    future_df["Date"] = pd.to_datetime(future_df["Date"])
    future_df["Open"] = future_df["Open"].fillna(1)

    future_df = future_df.merge(store_df, on="Store", how="left")
    future_df = future_df.sort_values(["Store", "Date"]).reset_index(drop=True)

    future_df.to_csv(KAGGLE_TEST_PATH, index=False)
    return future_df


def prepare_test_split(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create the hold-out test dataset used to validate forecasting locally."""
    test_df = merged_df.copy()
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df = test_df[test_df["Date"] > FEATURE_SPLIT_DATE].copy()
    test_df = test_df.sort_values(["Store", "Date"]).reset_index(drop=True)

    test_df.to_csv(TEST_DATA_PATH, index=False)
    return test_df


def run_pipeline() -> dict[str, pd.DataFrame]:
    """Run the full preprocessing pipeline once."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    merged_df = prepare_merged_datasets()
    kaggle_test_df = prepare_kaggle_test_dataset()
    test_df = prepare_test_split(merged_df)

    return {
        "merged": merged_df,
        "kaggle_test": kaggle_test_df,
        "test": test_df,
    }
