# Rossmann Sales Forecasting

A sales forecasting project for the Rossmann store dataset, combining data preparation, exploratory analysis, machine learning modeling, and an interactive Dash dashboard.

## Overview

This repository includes:
- `notebooks/` for data understanding, EDA, feature engineering, model training, and dashboard development
- `data/` raw and processed datasets
- `src/prepare_dashboard_data.py` to generate dashboard-ready CSV/JSON files
- `dashboard/` containing the Dash app, dashboard data, and deployment configuration
- `src/model.pkl` serialized LightGBM model used by the dashboard

The dashboard provides:
- KPI summaries
- Monthly sales trends
- Store performance, promo impact, and day-of-week analysis
- Machine learning forecast visualization for future sales

## Repository structure

- `README.md` - project overview and setup
- `requirements.txt` - Python dependencies
- `data/raw/` - raw Rossmann train/test/store data
- `data/processed/` - processed feature and merged datasets
- `dashboard/`
  - `app.py` - Dash application
  - `Dockerfile` - container image build
  - `Procfile` - Heroku-style deployment command
  - `data/` - generated dashboard data files
- `src/prepare_dashboard_data.py` - script to create dashboard data from processed datasets
- `src/model.pkl` - trained LightGBM model artifact
- `notebooks/` - analysis and modeling notebooks
- `reports/` - project reports and deliverables

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt