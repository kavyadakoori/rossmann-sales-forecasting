# Rossmann Sales Forecasting

An end-to-end Rossmann store sales forecasting project with a modular Python pipeline, a LightGBM model, a FastAPI inference service, and a Dash analytics dashboard.

## Architecture

The project has two runtime processes:

```text
Raw data
  -> src.preprocessing
  -> src.feature_engineering
  -> src.train
  -> src/rossmann_lightgbm.pkl

Dashboard analytics: dashboard/app.py -> dashboard/data/*.csv and *.json

Interactive forecast:
User -> Dash dashboard -> HTTP POST /forecast
      -> FastAPI api/main.py -> src/forecast.py -> LightGBM
      -> JSON response -> Dash chart and summary
```

The dashboard reads prepared analytical datasets directly. Only interactive future forecasting goes through FastAPI.

## Repository structure

```text
api/main.py                       FastAPI inference service
dashboard/app.py                  Dash dashboard
dashboard/data/                   Generated dashboard datasets
src/config.py                     Shared paths, mappings, and model features
src/preprocessing.py              Raw-data cleaning and dataset preparation
src/feature_engineering.py        Calendar, categorical, lag, and rolling features
src/train.py                      Tuning, validation, final training, model save
src/forecast.py                   Recursive inference logic used by API and pipeline
src/evaluate.py                   Regression and forecast metrics
src/pipeline.py                   End-to-end preprocessing, training, and evaluation
src/prepare_dashboard_data.py     Dashboard artifact generation
src/rossmann_lightgbm.pkl          Trained LightGBM model artifact
data/raw/                         Required raw Rossmann CSV files
data/processed/                   Generated processed datasets
notebooks/                        Original exploratory workflow
reports/                          Generated analysis images and reports
Dockerfile                        Canonical FastAPI container
requirements.txt                  Python dependencies
```

## Requirements

- Python 3.11 or 3.12
- The Rossmann CSV files in `data/raw/`
- Docker, only if using the container

Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Build the project from raw data

Run these commands from the repository root:

```powershell
python -m src.pipeline
python src/prepare_dashboard_data.py
```

The pipeline creates or refreshes:

- `data/processed/merged.csv`
- `data/processed/store.csv`
- `data/processed/test.csv`
- `data/processed/train_features.csv`
- `data/results/predictions.csv`
- `src/rossmann_lightgbm.pkl`

The dashboard preparation script creates the files under `dashboard/data/`.

## Run locally

Start the API in one terminal:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

FastAPI documentation is available at:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

Start the dashboard in a second terminal:

```powershell
python dashboard/app.py
```

Open `http://localhost:7860`.

The dashboard defaults to `http://localhost:8000` for inference. Configure another API location with:

```powershell
$env:API_BASE_URL = "http://your-api-host:8000"
python dashboard/app.py
```

The `/forecast` request accepts `store_id`, `weeks_ahead`, `open_override`, `promo_override`, `state_holiday`, and `school_holiday`. The response contains one predicted sales value per future date.

## Docker

The root [Dockerfile](Dockerfile) is the only Dockerfile in the project. It builds the application image. The [docker-compose.yml](docker-compose.yml) file runs the API and dashboard as separate services:

```powershell
docker compose up --build
```

Then use:

- API documentation: `http://localhost:8000/docs`
- Dashboard: `http://localhost:7860`

The Compose dashboard uses `API_BASE_URL=http://api:8000`, which is the internal service name. For an API-only container:

```powershell
docker build -t rossmann-forecast-api .
docker run --rm -p 8000:8000 rossmann-forecast-api
```

Then use `http://localhost:8000/docs`.

The dashboard remains a separate process. Run it locally with `API_BASE_URL=http://localhost:8000`, or deploy it as a separate web service using the existing `dashboard/Procfile` command from the repository root:

```text
gunicorn dashboard.app:server --timeout 120 --workers 1 --threads 2
```

When the dashboard is deployed in a separate container or service, set `API_BASE_URL` to the reachable API service hostname. Do not use `localhost` unless both processes share the same network namespace.

## Validation

Basic smoke checks:

```powershell
python -m py_compile api/main.py dashboard/app.py src/*.py
python -c "import requests; print(requests.get('http://localhost:8000/').json())"
```

The API must be running for the second check. The interactive `/docs` page can be used to submit a forecast request and inspect the JSON response.

## Notes

- Raw and processed datasets are local runtime inputs and are ignored by Git because of their size.
- `dashboard/data/` is generated output and is retained for the dashboard to start without rebuilding analytics immediately.
- The model artifact is currently stored at `src/rossmann_lightgbm.pkl`; the old `src/model.pkl` name is obsolete.
- The notebooks and reports document the original analysis but are not required to run the API or dashboard.