# ── Imports ───────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import json
import os

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data', 'processed')
DASH_DIR   = os.path.join(BASE_DIR, '..', 'dashboard', 'data')

# ── Create dashboard data folder ──────────────────────────────────
os.makedirs(DASH_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
print("📦 Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, 'train_merged.csv'),
                 low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df_open = df[df['Open'] == 1].copy()
df_open['Year']  = df_open['Date'].dt.year
df_open['Month'] = df_open['Date'].dt.month
df_open['Week']  = df_open['Date'].dt.isocalendar().week.astype(int)

df_features = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'),
                           low_memory=False)
df_features['Date'] = pd.to_datetime(df_features[['Year','Month','Day']])

# ── 1. KPIs ───────────────────────────────────────────────────────
print("📊 Computing KPIs...")
store_totals     = df_open.groupby('Store')['Sales'].sum()
best_store       = int(store_totals.idxmax())
best_store_sales = float(store_totals.max())

kpis = {
    'total_sales'      : float(df_open['Sales'].sum()),
    'avg_daily_sales'  : float(df_open['Sales'].mean()),
    'total_customers'  : float(df_open['Customers'].sum()),
    'total_stores'     : int(df_open['Store'].nunique()),
    'best_store'       : best_store,
    'best_store_sales' : best_store_sales
}
with open(os.path.join(DASH_DIR, 'kpis.json'), 'w') as f:
    json.dump(kpis, f)
print(f"   ✅ kpis.json saved")

# ── 2. Monthly Sales ──────────────────────────────────────────────
print("📊 Computing monthly sales...")
monthly = df_open.groupby(['Year','Month'])['Sales'].sum().reset_index()
monthly['Date'] = pd.to_datetime(
    monthly['Year'].astype(str) + '-' +
    monthly['Month'].astype(str))
monthly = monthly.sort_values('Date')
monthly['Date'] = monthly['Date'].astype(str)
monthly.to_csv(os.path.join(DASH_DIR, 'monthly_sales.csv'), index=False)
print(f"   ✅ monthly_sales.csv saved — {len(monthly)} rows")

# ── 3. Store Performance ──────────────────────────────────────────
print("📊 Computing store performance...")
store_perf = df_open.groupby('Store').agg(
    Total_Sales    = ('Sales','sum'),
    Avg_Sales      = ('Sales','mean'),
    Total_Customers= ('Customers','sum')
).reset_index().sort_values('Total_Sales', ascending=False)
store_perf.to_csv(os.path.join(DASH_DIR, 'store_perf.csv'), index=False)
print(f"   ✅ store_perf.csv saved — {len(store_perf)} rows")

# ── 4. Promo Sales ────────────────────────────────────────────────
print("📊 Computing promo impact...")
promo = df_open.groupby('Promo')['Sales'].mean().reset_index()
promo['Promo'] = promo['Promo'].map({0:'No Promo', 1:'Promo'})
promo.to_csv(os.path.join(DASH_DIR, 'promo_sales.csv'), index=False)
print(f"   ✅ promo_sales.csv saved — {len(promo)} rows")

# ── 5. Day of Week ────────────────────────────────────────────────
print("📊 Computing day of week sales...")
dow = df_open.groupby('DayOfWeek')['Sales'].mean().reset_index()
dow['DayName'] = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow.to_csv(os.path.join(DASH_DIR, 'dow_sales.csv'), index=False)
print(f"   ✅ dow_sales.csv saved — {len(dow)} rows")

# ── 6. Test predictions (Actual vs Predicted) ─────────────────────
print("📊 Computing test predictions...")
import pickle
model_path = os.path.join(BASE_DIR, 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

feature_cols = [c for c in df_features.columns
                if c not in ['Sales','Date']]

test_data = df_features[df_features['Date'] >= '2015-06-01'].copy()
preds = model.predict(test_data[feature_cols],
                      num_iteration=model.best_iteration)
preds = np.clip(preds, 0, None)
test_data = test_data[['Store','Date','Sales']].copy()
test_data['Predicted'] = preds
test_data['Date'] = test_data['Date'].astype(str)
test_data.to_csv(os.path.join(DASH_DIR, 'test_predictions.csv'), index=False)
print(f"   ✅ test_predictions.csv saved — {len(test_data)} rows")

# ── 7. Features lite (last 60 days per store for forecasting) ─────
print("📊 Saving features lite...")
features_lite = df_features.groupby('Store').tail(30).copy()
features_lite['Date'] = features_lite['Date'].astype(str)
features_lite.to_csv(os.path.join(DASH_DIR, 'features_lite.csv'), index=False)
print(f"   ✅ features_lite.csv saved — {len(features_lite)} rows")

# ── Summary ───────────────────────────────────────────────────────
print()
print("=" * 50)
print("✅ All dashboard data prepared!")
print(f"📁 Saved to: dashboard/data/")
print()
for f in os.listdir(DASH_DIR):
    size = os.path.getsize(os.path.join(DASH_DIR, f)) / 1024
    print(f"   {f:<35} : {size:.1f} KB")
print("=" * 50)