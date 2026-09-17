# ── Imports ───────────────────────────────────────────────────────
import json
import os
from pathlib import Path

import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# ── Load pre-computed data ────────────────────────────────────────
with (DATA_DIR / "kpis.json").open(encoding="utf-8") as f:
    kpis = json.load(f)

monthly_sales = pd.read_csv(DATA_DIR / "monthly_sales.csv")
monthly_sales["Date"] = pd.to_datetime(monthly_sales["Date"])

store_perf = pd.read_csv(DATA_DIR / "store_perf.csv")
promo_sales = pd.read_csv(DATA_DIR / "promo_sales.csv")
dow_sales = pd.read_csv(DATA_DIR / "dow_sales.csv")

test_preds = pd.read_csv(DATA_DIR / "test_predictions.csv")
if "Actual Sales" in test_preds.columns:
    test_preds = test_preds.rename(
        columns={"Actual Sales": "Actual_Sales", "Predicted Sales": "Predicted_Sales"}
    )
if "Date" in test_preds.columns:
    test_preds["Date"] = pd.to_datetime(test_preds["Date"])

features_lite = pd.read_csv(DATA_DIR / "features_lite.csv")
if "Date" in features_lite.columns:
    features_lite["Date"] = pd.to_datetime(features_lite["Date"])
else:
    features_lite["Date"] = pd.to_datetime(
        features_lite[["Year", "Month", "Day"]], errors="coerce"
    )

# ── KPI values ────────────────────────────────────────────────────
total_sales = kpis["total_sales"]
avg_daily_sales = kpis["avg_daily_sales"]
total_customers = kpis["total_customers"]
best_store = kpis["best_store"]
best_store_sales = kpis["best_store_sales"]


# ── Colors ────────────────────────────────────────────────────────
COLORS = {
    'primary' : '#2196F3',
    'success' : '#4CAF50',
    'warning' : '#FF9800',
    'danger'  : '#F44336',
    'dark'    : '#263238',
    'light'   : '#F5F5F5'
}

# ── KPI Card ──────────────────────────────────────────────────────
def kpi_card(title, value, subtitle, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, style={'color':'#666','fontSize':'13px'}),
            html.H3(value, style={'color':color,'fontWeight':'bold'}),
            html.P(subtitle, style={'color':'#999','fontSize':'12px','margin':'0'})
        ])
    ], style={'borderLeft':f'4px solid {color}','borderRadius':'8px'})

# ── App ───────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# ── Layout ────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("Rossmann Sales Forecasting Dashboard",
                    style={'color':COLORS['dark'],'fontWeight':'bold'}),
            html.P("Interactive analytics and ML-powered sales forecasting",
                   style={'color':'#666'})
        ])
    ], style={'padding':'20px 0 10px 0'}),

    html.Hr(),

    # KPIs
    dbc.Row([
        dbc.Col(kpi_card("Total Sales", f"€{total_sales/1e6:.0f}M",
                         "Jan 2013 — Jul 2015", COLORS['primary']), width=3),
        dbc.Col(kpi_card("Avg Daily Sales", f"€{avg_daily_sales:,.0f}",
                         "Per store per day", COLORS['success']), width=3),
        dbc.Col(kpi_card("Total Customers", f"{total_customers/1e6:.0f}M",
                         "Total visits", COLORS['warning']), width=3),
        dbc.Col(kpi_card("Best Store", f"Store {best_store}",
                         f"€{best_store_sales/1e6:.1f}M total",
                         COLORS['danger']), width=3),
    ], style={'marginBottom':'20px'}),

    # Trend + DOW
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Monthly Sales Trend"),
                dbc.CardBody([
                    dcc.Graph(figure=px.line(
                        monthly_sales, x='Date', y='Sales',
                        color='Year',
                        labels={'Sales':'Total Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10,b=10)
                    ))
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Avg Sales by Day"),
                dbc.CardBody([
                    dcc.Graph(figure=px.bar(
                        dow_sales, x='DayName', y='Sales',
                        color='Sales',
                        color_continuous_scale='Blues',
                        labels={'Sales':'Avg Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10,b=10),
                        showlegend=False
                    ))
                ])
            ])
        ], width=4),
    ], style={'marginBottom':'20px'}),

    # Store Performance + Promo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 15 Stores by Total Sales"),
                dbc.CardBody([
                    dcc.Graph(figure=px.bar(
                        store_perf.head(15),
                        x='Store', y='Total_Sales',
                        color='Total_Sales',
                        color_continuous_scale='Greens',
                        labels={'Total_Sales':'Total Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10,b=10)
                    ))
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Promo vs No Promo"),
                dbc.CardBody([
                    dcc.Graph(figure=px.bar(
                        promo_sales, x='Promo', y='Sales',
                        color='Promo',
                        color_discrete_map={
                            'No Promo':COLORS['danger'],
                            'Promo'   :COLORS['success']},
                        labels={'Sales':'Avg Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10,b=10),
                        showlegend=False
                    ))
                ])
            ])
        ], width=4),
    ], style={'marginBottom':'20px'}),

    # Actual vs Predicted
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sales Forecast — Select Store"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='store-dropdown',
                                options=[{'label':f'Store {i}','value':i}
                                         for i in sorted(test_preds['Store'].unique())],
                                value=1,
                                clearable=False,
                                style={'marginBottom':'10px'}
                            )
                        ], width=3),
                        dbc.Col([
                            html.P("Select any store to see actual vs predicted sales",
                                   style={'color':'#666','marginTop':'8px'})
                        ], width=9)
                    ]),
                    dcc.Graph(id='forecast-chart')
                ])
            ])
        ], width=12)
    ], style={'marginBottom':'20px'}),

    # Future Forecast
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Future Sales Forecast — ML Powered"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Store:",
                                       style={'fontWeight':'bold'}),
                            dcc.Dropdown(
                                id='future-store-dropdown',
                                options=[{'label':f'Store {i}','value':i}
                                         for i in sorted(features_lite['Store'].unique())],
                                value=1,
                                clearable=False,
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Weeks Ahead:",
                                       style={'fontWeight':'bold'}),
                            dcc.Slider(
                                id='weeks-slider',
                                min=4, max=16, step=4,
                                value=8,
                                marks={4:'4 weeks',8:'8 weeks',
                                       12:'12 weeks',16:'16 weeks'},
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Open:", style={'fontWeight':'bold'}),
                            dcc.Dropdown(
                                id='future-open-dropdown',
                                options=[{'label':'Open', 'value':1}, {'label':'Closed', 'value':0}],
                                value=1,
                                clearable=False,
                            )
                        ], width=2),
                        dbc.Col([
                            html.Label("Promo:", style={'fontWeight':'bold'}),
                            dcc.Dropdown(
                                id='future-promo-dropdown',
                                options=[{'label':'No Promo', 'value':0}, {'label':'Promo', 'value':1}],
                                value=0,
                                clearable=False,
                            )
                        ], width=2),
                        dbc.Col([
                            html.Label("State Holiday:", style={'fontWeight':'bold'}),
                            dcc.Dropdown(
                                id='future-stateholiday-dropdown',
                                options=[{'label':'0', 'value':'0'}, {'label':'a', 'value':'a'}, {'label':'b', 'value':'b'}, {'label':'c', 'value':'c'}],
                                value='0',
                                clearable=False,
                            )
                        ], width=2),
                        dbc.Col([
                            html.Label("School Holiday:", style={'fontWeight':'bold'}),
                            dcc.Dropdown(
                                id='future-schoolholiday-dropdown',
                                options=[{'label':'No', 'value':0}, {'label':'Yes', 'value':1}],
                                value=0,
                                clearable=False,
                            )
                        ], width=2),
                        dbc.Col([
                            html.Br(),
                            dbc.Button(
                                "Generate Forecast",
                                id='forecast-btn',
                                color='primary',
                                className='mt-1',
                            )
                        ], width=1),
                    ], style={'marginBottom':'15px'}),
                    dcc.Loading(
                        id='loading-forecast',
                        type='circle',
                        children=[dcc.Graph(id='future-forecast-chart')]
                    ),
                    html.Div(id='forecast-summary',
                             style={'marginTop':'10px'})
                ])
            ])
        ], width=12)
    ]),

    html.Br()

], fluid=True, style={'backgroundColor':COLORS['light']})


# ── Callback — Actual vs Predicted ───────────────────────────────
@app.callback(
    Output('forecast-chart','figure'),
    Input('store-dropdown','value')
)
def update_forecast(store_id):
    store_data = test_preds[test_preds['Store'] == store_id].copy()

    if len(store_data) == 0:
        return go.Figure()

    actual_col = 'Actual_Sales' if 'Actual_Sales' in store_data.columns else 'Actual Sales'
    pred_col = 'Predicted_Sales' if 'Predicted_Sales' in store_data.columns else 'Predicted Sales'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=store_data['Date'], y=store_data[actual_col],
        name='Actual',
        line=dict(color=COLORS['primary'], width=2),
        mode='lines+markers', marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=store_data['Date'], y=store_data[pred_col],
        name='Predicted',
        line=dict(color=COLORS['danger'], width=2, dash='dash'),
        mode='lines+markers', marker=dict(size=4)
    ))
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h'),
        xaxis_title='Date',
        yaxis_title='Sales (€)',
        margin=dict(t=10)
    )
    return fig


# ── Callback — Future Forecast ────────────────────────────────────
@app.callback(
    Output('future-forecast-chart','figure'),
    Output('forecast-summary','children'),
    Input('forecast-btn','n_clicks'),
    Input('future-store-dropdown','value'),
    Input('weeks-slider','value'),
    Input('future-open-dropdown','value'),
    Input('future-promo-dropdown','value'),
    Input('future-stateholiday-dropdown','value'),
    Input('future-schoolholiday-dropdown','value'),
)
def update_future_forecast(
    n_clicks,
    store_id,
    weeks_ahead,
    open_override,
    promo_override,
    state_holiday,
    school_holiday,
):
    if n_clicks is None:
        return go.Figure(), html.Div()

    try:
        historical = features_lite[
            features_lite['Store'] == store_id
        ].tail(30)[['Date','Sales']].copy()
        historical['Date'] = pd.to_datetime(historical['Date'])

        response = requests.post(
            f"{API_BASE_URL}/forecast",
            json={
                "store_id": int(store_id),
                "weeks_ahead": int(weeks_ahead),
                "open_override": int(open_override),
                "promo_override": int(promo_override),
                "state_holiday": str(state_holiday),
                "school_holiday": int(school_holiday),
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        forecast_rows = payload.get('forecast', [])
        if not forecast_rows:
            return go.Figure(), "No forecast data returned"

        forecast_df = pd.DataFrame(forecast_rows)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df['sales'] = pd.to_numeric(forecast_df['sales'], errors='coerce').fillna(0)
        forecast_df = forecast_df.rename(columns={'date': 'Date', 'sales': 'Predicted_Sales'})

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=historical['Date'], y=historical['Sales'],
            name='Historical Sales',
            line=dict(color=COLORS['primary'], width=2),
            mode='lines+markers', marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Predicted_Sales'],
            name='Future Forecast',
            line=dict(color=COLORS['danger'], width=2, dash='dash'),
            mode='lines+markers', marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Sales'] * 1.10,
            line=dict(color=COLORS['danger'], width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Sales'] * 0.90,
            fill='tonexty',
            fillcolor='rgba(244,67,54,0.15)',
            line=dict(color=COLORS['danger'], width=0),
            showlegend=False
        ))

        forecast_start = str(forecast_df['Date'].min().date())
        fig.add_shape(
            type='line',
            x0=forecast_start, x1=forecast_start,
            y0=0, y1=1, yref='paper',
            line=dict(color='green', width=2, dash='dot')
        )
        fig.add_annotation(
            x=forecast_start, y=1, yref='paper',
            text='Forecast Start',
            showarrow=False, yshift=10,
            font=dict(color='green')
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h'),
            xaxis_title='Date',
            yaxis_title='Sales (€)',
            margin=dict(t=30),
            height=450
        )

        avg_pred = forecast_df['Predicted_Sales'].mean()
        max_pred = forecast_df['Predicted_Sales'].max()
        min_pred = forecast_df['Predicted_Sales'].min()
        total_pred = forecast_df['Predicted_Sales'].sum()

        summary = dbc.Row([
            dbc.Col(kpi_card("Avg Daily Forecast", f"€{avg_pred:,.0f}",
                             f"Over {weeks_ahead} weeks",
                             COLORS['primary']), width=3),
            dbc.Col(kpi_card("Peak Day Forecast", f"€{max_pred:,.0f}",
                             "Highest predicted day",
                             COLORS['success']), width=3),
            dbc.Col(kpi_card("Lowest Day Forecast", f"€{min_pred:,.0f}",
                             "Lowest predicted day",
                             COLORS['warning']), width=3),
            dbc.Col(kpi_card("Total Forecast Revenue", f"€{total_pred:,.0f}",
                             f"Next {weeks_ahead} weeks total",
                             COLORS['danger']), width=3),
        ])

        return fig, summary

    except requests.exceptions.ConnectionError:
        return go.Figure(), html.P(
            f"Forecast API is unavailable at {API_BASE_URL}. "
            "Start it with: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000",
            style={'color': 'red'}
        )
    except requests.exceptions.Timeout:
        return go.Figure(), html.P(
            "The forecast API did not respond within the timeout period.",
            style={'color': 'red'}
        )
    except requests.exceptions.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        return go.Figure(), html.P(
            f"Forecast API error: {detail}",
            style={'color': 'red'}
        )
    except Exception as e:
        import traceback
        print("❌ ERROR:", traceback.format_exc())
        return go.Figure(), html.P(f"Error: {str(e)}",
                                    style={'color':'red'})


# ── Run ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)