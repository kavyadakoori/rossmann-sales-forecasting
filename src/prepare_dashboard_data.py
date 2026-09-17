# ── Imports ───────────────────────────────────────────────────────
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.feature_engineering import (
    add_date_features,
    add_lag_features,
    add_rolling_features,
    encode_categorical_features,
)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DASH_DIR = BASE_DIR / "dashboard" / "data"
RESULTS_DIR = DATA_DIR / "results"

DASH_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def prepare_dashboard_data() -> None:
    print("📦 Loading processed data...")
    merged_df = safe_read_csv(PROCESSED_DATA_DIR / "merged.csv")
    if merged_df.empty:
        raise FileNotFoundError(
            "Missing merged.csv. Run the preprocessing pipeline first."
        )

    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    df_open = merged_df[merged_df["Open"] == 1].copy()
    df_open["Year"] = df_open["Date"].dt.year
    df_open["Month"] = df_open["Date"].dt.month
    df_open["Week"] = df_open["Date"].dt.isocalendar().week.astype(int)

    test_df = safe_read_csv(PROCESSED_DATA_DIR / "test.csv")
    test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce")
    # ── 1. KPIs ───────────────────────────────────────────────────────
    print("📊 Computing KPIs...")
    store_totals = df_open.groupby("Store")["Sales"].sum()
    best_store = int(store_totals.idxmax())
    best_store_sales = float(store_totals.max())

    kpis = {
        "total_sales": float(df_open["Sales"].sum()),
        "avg_daily_sales": float(df_open["Sales"].mean()),
        "total_customers": float(df_open["Customers"].sum()),
        "total_stores": int(df_open["Store"].nunique()),
        "best_store": best_store,
        "best_store_sales": best_store_sales,
    }

    with (DASH_DIR / "kpis.json").open("w", encoding="utf-8") as f:
        json.dump(kpis, f)
    print("   ✅ kpis.json saved")

    # ── 2. Monthly Sales ──────────────────────────────────────────────
    print("📊 Computing monthly sales...")
    monthly = df_open.groupby(["Year", "Month"])["Sales"].sum().reset_index()
    monthly["Date"] = pd.to_datetime(
        monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str)
    )
    monthly = monthly.sort_values("Date").copy()
    monthly["Date"] = monthly["Date"].astype(str)
    monthly.to_csv(DASH_DIR / "monthly_sales.csv", index=False)
    print(f"   ✅ monthly_sales.csv saved — {len(monthly)} rows")

    # ── 3. Store Performance ──────────────────────────────────────────
    print("📊 Computing store performance...")
    store_perf = (
        df_open.groupby("Store")
        .agg(
            Total_Sales=("Sales", "sum"),
            Avg_Sales=("Sales", "mean"),
            Total_Customers=("Customers", "sum"),
        )
        .reset_index()
        .sort_values("Total_Sales", ascending=False)
    )
    store_perf.to_csv(DASH_DIR / "store_perf.csv", index=False)
    print(f"   ✅ store_perf.csv saved — {len(store_perf)} rows")

    # ── 4. Promo Sales ────────────────────────────────────────────────
    print("📊 Computing promo impact...")
    promo = df_open.groupby("Promo")["Sales"].mean().reset_index()
    promo["Promo"] = promo["Promo"].map({0: "No Promo", 1: "Promo"})
    promo.to_csv(DASH_DIR / "promo_sales.csv", index=False)
    print(f"   ✅ promo_sales.csv saved — {len(promo)} rows")

    # ── 5. Day of Week ────────────────────────────────────────────────
    print("📊 Computing day of week sales...")
    dow = df_open.groupby("DayOfWeek")["Sales"].mean().reset_index()
    dow["DayName"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow.to_csv(DASH_DIR / "dow_sales.csv", index=False)
    print(f"   ✅ dow_sales.csv saved — {len(dow)} rows")

    # ── 6. Test predictions (Actual vs Predicted) ─────────────────────
    print("📊 Computing test predictions...")
    test_predictions = pd.DataFrame(
        columns=["Store", "Date", "Actual_Sales", "Predicted_Sales"]
    )

    predictions_path = RESULTS_DIR / "predictions.csv"
    if predictions_path.exists():
        predictions = safe_read_csv(predictions_path)
        if not predictions.empty:
            predictions = predictions[["Store", "Date", "Sales"]].copy()
            predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
            predictions = predictions.rename(columns={"Sales": "Predicted_Sales"})

            actual = test_df[["Store", "Date", "Sales"]].copy()
            actual = actual.rename(columns={"Sales": "Actual_Sales"})
            actual["Date"] = pd.to_datetime(actual["Date"], errors="coerce")

            test_predictions = actual.merge(
                predictions, on=["Store", "Date"], how="inner"
            )

    test_predictions["Date"] = test_predictions["Date"].astype(str)
    test_predictions.to_csv(DASH_DIR / "test_predictions.csv", index=False)
    print(f"   ✅ test_predictions.csv saved — {len(test_predictions)} rows")

    # ── 7. Features lite (recent engineered features for forecasting) ─────
    print("📊 Saving features lite...")
    if not merged_df.empty:
        features_lite = merged_df.copy()
        features_lite["Date"] = pd.to_datetime(features_lite["Date"], errors="coerce")

        features_lite = add_date_features(features_lite)
        features_lite = encode_categorical_features(features_lite)
        features_lite = add_lag_features(features_lite)
        features_lite = add_rolling_features(features_lite)

        lag_roll_cols = [
            col for col in features_lite.columns if col.startswith("Lag_") or col.startswith("Rolling_")
        ]
        features_lite = features_lite.dropna(subset=lag_roll_cols).copy()
        features_lite = features_lite.groupby("Store").tail(60).copy()

        features_lite["Date"] = features_lite["Date"].astype(str)
        features_lite.to_csv(DASH_DIR / "features_lite.csv", index=False)
        print(f"   ✅ features_lite.csv saved — {len(features_lite)} rows")
    else:
        print("   ⚠️ merged_df is empty; skipping features_lite.csv")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("✅ All dashboard data prepared!")
    print("📁 Saved to: dashboard/data/")
    print()
    for path in sorted(DASH_DIR.iterdir()):
        size = path.stat().st_size / 1024
        print(f"   {path.name:<35} : {size:.1f} KB")
    print("=" * 50)


if __name__ == "__main__":
    prepare_dashboard_data()
