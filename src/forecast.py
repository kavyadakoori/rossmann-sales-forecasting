import numpy as np
import pandas as pd

from .config import CAT_MAPPINGS


def forecast_future(history_df, future_df, model, feature_cols):
    history = history_df.copy()
    future = future_df.copy()

    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"], errors="raise")
    else:
        history["Date"] = pd.to_datetime(
            {
                "year": history["Year"],
                "month": history["Month"],
                "day": history["Day"],
            },
            errors="raise",
        )
    future["Date"] = pd.to_datetime(future["Date"])

    # keep store histories in memory as dicts of rows
    store_history = {
        store: group.sort_values("Date").reset_index(drop=True)
        for store, group in history.groupby("Store", sort=False)
    }

    # sort future once
    future = future.sort_values(["Store", "Date"]).reset_index(drop=True)

    predictions = []

    for current_date in sorted(future["Date"].unique()):
        current_date = pd.Timestamp(current_date)
        today = future[future["Date"] == current_date]

        for row in today.itertuples(index=False):
            store = row.Store
            if store not in store_history:
                raise ValueError(f"Missing history for Store={store}")
            hist = store_history[store].copy()

            # create a lightweight dict for the current row
            row_dict = row._asdict()

            row_dict["Year"] = current_date.year
            row_dict["Month"] = current_date.month
            row_dict["Day"] = current_date.day
            row_dict["Week"] = current_date.isocalendar().week
            row_dict["DayOfWeek"] = current_date.dayofweek
            row_dict["DayOfYear"] = current_date.dayofyear
            row_dict["Quarter"] = (current_date.month - 1) // 3 + 1
            row_dict["IsWeekend"] = int(current_date.dayofweek >= 5)
            row_dict["IsMonthStart"] = int(current_date.is_month_start)
            row_dict["IsMonthEnd"] = int(current_date.is_month_end)

            month = current_date.month
            if month in [12, 1, 2]:
                row_dict["Season"] = "Winter"
            elif month in [3, 4, 5]:
                row_dict["Season"] = "Spring"
            elif month in [6, 7, 8]:
                row_dict["Season"] = "Summer"
            else:
                row_dict["Season"] = "Autumn"

            # encode categories
            for column, mapping in CAT_MAPPINGS.items():
                if column in row_dict and row_dict[column] is not None:
                    row_dict[column] = mapping.get(str(row_dict[column]), 0)

            sales = hist["Sales"].astype(float).to_numpy()

            row_dict["Lag_1"] = sales[-1]
            row_dict["Lag_7"] = sales[-7]
            row_dict["Lag_14"] = sales[-14]
            row_dict["Lag_21"] = sales[-21]
            row_dict["Lag_28"] = sales[-28]

            row_dict["Rolling_Mean_7"] = np.mean(sales[-7:])
            row_dict["Rolling_Mean_14"] = np.mean(sales[-14:])
            row_dict["Rolling_Mean_28"] = np.mean(sales[-28:])
            row_dict["Rolling_Std_7"] = np.std(sales[-7:])
            row_dict["Rolling_Std_14"] = np.std(sales[-14:])
            row_dict["Rolling_Std_28"] = np.std(sales[-28:])

            if row_dict.get("Open", 1) == 0:
                pred = 0.0
            else:
                feature_frame = pd.DataFrame([row_dict])[feature_cols]
                if feature_frame.isnull().any().any():
                    raise ValueError(
                        f"Missing features for Store={store}, Date={current_date.date()}"
                    )
                pred = float(model.predict(feature_frame)[0])
                pred = max(0.0, pred)

            row_dict["Sales"] = pred
            predictions.append(row_dict)

            # append prediction to the store history
            new_row = pd.DataFrame([row_dict])
            new_row["Date"] = pd.to_datetime(new_row["Date"])
            hist = pd.concat([hist, new_row], ignore_index=True)
            hist = hist.sort_values("Date").tail(28).reset_index(drop=True)
            store_history[store] = hist

    return pd.DataFrame(predictions)


def forecast(history_df: pd.DataFrame, test_df: pd.DataFrame, store_df: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    """Merge store metadata into the future query frame and forecast each store/date."""
    # future_df = test_df.merge(store_df, on="Store", how="inner")
    return forecast_future(history_df, test_df, model, feature_cols)
