from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_RAW_PATH = RAW_DATA_DIR / "train.csv"
STORE_RAW_PATH = RAW_DATA_DIR / "store.csv"
FORECAST_TEST_RAW_PATH = RAW_DATA_DIR / "forecast_test.csv"
MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged.csv"
KAGGLE_TEST_PATH = PROCESSED_DATA_DIR / "kaggle_test.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / "train_features.csv"
MODEL_PATH = PROJECT_ROOT / "src" / "rossmann_lightgbm.pkl"

TARGET_COLUMN = "Sales"
DATE_COLUMN = "Date"
FEATURE_SPLIT_DATE = "2015-06-19"
TRAIN_END_DATE = "2015-03-31"

CAT_MAPPINGS = {
    "Season": {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3},
    "StoreType": {"a": 0, "b": 1, "c": 2, "d": 3},
    "Assortment": {"a": 0, "b": 1, "c": 2},
    "StateHoliday": {"0": 0, "a": 1, "b": 2, "c": 3},
    "PromoInterval": {
        "None": 0,
        "Jan,Apr,Jul,Oct": 1,
        "Feb,May,Aug,Nov": 2,
        "Mar,Jun,Sept,Dec": 3,
    },
}

FEATURE_COLS = ['Store', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval', 'Year', 'Month', 'Day', 'Week',
       'DayOfWeek', 'DayOfYear', 'Quarter', 'Season', 'IsWeekend',
       'IsMonthStart', 'IsMonthEnd', 'StateHoliday', 'SchoolHoliday', 'Lag_1',
       'Lag_7', 'Lag_14', 'Lag_21', 'Lag_28', 'Rolling_Mean_7',
       'Rolling_Mean_14', 'Rolling_Mean_28', 'Rolling_Std_7', 'Rolling_Std_14',
       'Rolling_Std_28']