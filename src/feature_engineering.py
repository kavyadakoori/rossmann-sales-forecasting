import numpy as np
import pandas as pd

from .config import CAT_MAPPINGS, TRAIN_FEATURES_PATH, FEATURE_SPLIT_DATE


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features from the Date column."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    df["Season"] = np.select(
        [
            df["Month"].isin([12, 1, 2]),
            df["Month"].isin([3, 4, 5]),
            df["Month"].isin([6, 7, 8]),
        ],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode string categorical columns into the integer representation used by the model."""
    df = df.copy()

    for column, mapping in CAT_MAPPINGS.items():
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna(0).astype(int)

    return df


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 7, 14, 21, 28)) -> pd.DataFrame:
    """Create lag features for store-level historical sales."""
    df = df.sort_values(["Store", "Date"]).copy()

    for lag in lags:
        df[f"Lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Create rolling mean/std features using only prior observations."""
    df = df.sort_values(["Store", "Date"]).copy()

    grouped_sales = df.groupby("Store")["Sales"]

    for window in windows:
        df[f"Rolling_Mean_{window}"] = grouped_sales.transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
        df[f"Rolling_Std_{window}"] = grouped_sales.transform(
            lambda x: x.shift(1).rolling(window).std()
        )

    return df


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook feature-engineering pipeline and persist the result."""
    df = df.copy()
    TRAIN_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = df[df['Date'] <= FEATURE_SPLIT_DATE].copy()  # Only use training data for feature engineering
    df = add_date_features(df)
    df = encode_categorical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    lag_roll_cols = [col for col in df.columns if "Lag_" in col or "Rolling_" in col]
    df = df.dropna(subset=lag_roll_cols).copy()

    feature_cols = [
        "Sales",
        "Store",
        "Open",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        "Year",
        "Month",
        "Day",
        "Week",
        "DayOfWeek",
        "DayOfYear",
        "Quarter",
        "Season",
        "IsWeekend",
        "IsMonthStart",
        "IsMonthEnd",
        "StateHoliday",
        "SchoolHoliday",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_21",
        "Lag_28",
        "Rolling_Mean_7",
        "Rolling_Mean_14",
        "Rolling_Mean_28",
        "Rolling_Std_7",
        "Rolling_Std_14",
        "Rolling_Std_28",
    ]

    df_final = df[feature_cols].copy()
    df_final.to_csv(TRAIN_FEATURES_PATH, index=False)

    return df_final
