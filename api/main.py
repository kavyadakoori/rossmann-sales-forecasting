import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.forecast import forecast_future as forecast_with_history

app = FastAPI(title="Rossmann Sales Forecast API", version="1.0.0")


class ForecastRequest(BaseModel):
    store_id: int = Field(default=1, ge=1)
    weeks_ahead: int = Field(default=4, ge=1, le=52)
    open_override: int = Field(default=1, ge=0, le=1)
    promo_override: int = Field(default=0, ge=0, le=1)
    state_holiday: str = "0"
    school_holiday: int = Field(default=0, ge=0, le=1)


class ForecastPoint(BaseModel):
    date: str
    sales: float


class ForecastResponse(BaseModel):
    store_id: int
    forecast: list[ForecastPoint]

MODEL_PATH = PROJECT_ROOT / "src" / "rossmann_lightgbm.pkl"
STORE_PATH = PROCESSED_DATA_DIR / "store.csv"
if not STORE_PATH.exists():
    STORE_PATH = RAW_DATA_DIR / "store.csv"

FEATURES_LITE_PATH = PROJECT_ROOT / "dashboard" / "data" / "features_lite.csv"

model = joblib.load(MODEL_PATH)
store_df = pd.read_csv(STORE_PATH, low_memory=False)
features_lite = pd.read_csv(FEATURES_LITE_PATH, low_memory=False)
if "Date" in features_lite.columns:
    features_lite["Date"] = pd.to_datetime(features_lite["Date"])
else:
    features_lite["Date"] = pd.to_datetime(
        features_lite[["Year", "Month", "Day"]], errors="coerce"
    )


def build_future_query_frame(
    store_id: int,
    weeks_ahead: int,
    open_override: int = 1,
    promo_override: int = 0,
    state_holiday: str = "0",
    school_holiday: int = 0,
) -> pd.DataFrame:
    history = features_lite[features_lite["Store"] == store_id].sort_values("Date").copy()
    if history.empty:
        return pd.DataFrame()

    store_meta = store_df[store_df["Store"] == store_id].iloc[0].to_dict()
    store_meta = {key: (0 if pd.isna(value) else value) for key, value in store_meta.items()}

    last_date = pd.Timestamp(history["Date"].max())
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=weeks_ahead * 7,
        freq="D",
    )

    rows = []
    for current_date in future_dates:
        row = {
            "Store": int(store_id),
            "Date": current_date,
            "Open": int(open_override),
            "Promo": int(promo_override),
            "StateHoliday": str(state_holiday),
            "SchoolHoliday": int(school_holiday),
        }

        for column in [
            "StoreType",
            "Assortment",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval",
        ]:
            row[column] = store_meta.get(column, 0)

        rows.append(row)

    future_df = pd.DataFrame(rows)
    future_df["Date"] = pd.to_datetime(future_df["Date"])
    return future_df


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Rossmann Sales Forecast API is running"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(payload: ForecastRequest):
    try:
        store_id = payload.store_id
        weeks_ahead = payload.weeks_ahead
        open_override = payload.open_override
        promo_override = payload.promo_override
        state_holiday = payload.state_holiday
        school_holiday = payload.school_holiday

        history = features_lite[features_lite["Store"] == store_id].sort_values("Date").copy()
        if history.empty:
            return JSONResponse(
                status_code=404,
                content={"detail": f"No history found for Store {store_id}"},
            )

        future_df = build_future_query_frame(
            store_id=store_id,
            weeks_ahead=weeks_ahead,
            open_override=open_override,
            promo_override=promo_override,
            state_holiday=state_holiday,
            school_holiday=school_holiday,
        )

        if future_df.empty:
            return JSONResponse(
                status_code=404,
                content={"detail": f"No future frame generated for Store {store_id}"},
            )

        predictions = forecast_with_history(
            history_df=history,
            future_df=future_df,
            model=model,
            feature_cols=FEATURE_COLS,
        )

        result = []
        for _, row in predictions.iterrows():
            result.append(
                {
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "sales": float(max(0.0, row["Sales"])),
                }
            )

        return {"store_id": store_id, "forecast": result}

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )
