import pandas as pd

from .config import (
    FORECAST_TEST_RAW_PATH,
    KAGGLE_TEST_PATH,
    MERGED_DATA_PATH,
    STORE_RAW_PATH,
    TEST_DATA_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_RAW_PATH,
)


def load_training_data() -> pd.DataFrame:
    """Load the raw Rossmann training dataset."""
    return pd.read_csv(TRAIN_RAW_PATH)


def load_store_data() -> pd.DataFrame:
    """Load the raw store metadata dataset."""
    return pd.read_csv(STORE_RAW_PATH)


def load_forecast_test_data() -> pd.DataFrame:
    """Load the Kaggle competition test dataset without sales values."""
    return pd.read_csv(FORECAST_TEST_RAW_PATH)


def load_processed_data() -> dict[str, pd.DataFrame]:
    """Load already-processed project datasets."""
    return {
        "merged": pd.read_csv(MERGED_DATA_PATH),
        "kaggle_test": pd.read_csv(KAGGLE_TEST_PATH),
        "test": pd.read_csv(TEST_DATA_PATH),
        "train_features": pd.read_csv(TRAIN_FEATURES_PATH),
    }
