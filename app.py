# %% ============================================================
# Retail Demand Forecast (India-local) — UI Polished Edition
# Keep all original logic; improve layout/clarity/visual polish only
# Added Marketing Strategy feature (NOW POWERED BY GOOGLE GEMINI)
# ================================================================
import os
import io
import math
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from rapidfuzz import process, fuzz
# New import for the marketing feature using Google Gemini
import google.generativeai as genai


# =========================
# App Config & Light Theme
# =========================
st.set_page_config(page_title="Retail Demand Forecast (India-local)", page_icon="🛒", layout="wide")

# Small, tasteful CSS polish (cards, buttons, spacing, tables)
st.markdown("""
<style>
/* Overall container spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Headings */
h1, h2, h3 { letter-spacing: .2px; }
h1 { margin-bottom: .4rem; }

/* Cards */
.card {
  padding: 1rem 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
  margin-bottom: .6rem;
}

/* KPI tweaks */
div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e5e7eb;
  border-radius: 10px; padding: .6rem .8rem; }
div[data-testid="stMetricValue"] { font-size: 1.4rem; }

/* Buttons */
.stButton>button {
  border-radius: 8px !important;
  border: 1px solid #1d4ed8 !important;
  background: #1d4ed8 !important;
  color: #fff !important;
}
.stDownloadButton>button {
  border-radius: 8px !important;
}

/* Tables */
thead tr th { background: #f8fafc !important; }

/* Small gray text */
.note { color: #6b7280; font-size: .92rem; }

/* Tiny tags */
.tag {
  display: inline-block; padding: .15rem .5rem; border-radius: 999px;
  background: #eef2ff; color: #3730a3; font-size: .78rem; margin-right: .25rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Data / Model Paths
# =========================
DATA_PATH = "inventory_with_india_features.csv"
MODEL_PATH = "inventory_gbr_india_local.pkl"
ITEMS_MASTER_PATH = "items_master.csv"

DEFAULT_ITEMS = ['Soap', 'Shampoo', 'Toothpaste', 'Rice', 'Sugar', 'Salt', 'Biscuits', 'Juice', 'Milk', 'Bread']
DEFAULT_CATEGORIES = {
    'Soap': 'Personal Care', 'Shampoo': 'Personal Care', 'Toothpaste': 'Personal Care',
    'Rice': 'Grocery', 'Sugar': 'Grocery', 'Salt': 'Grocery',
    'Biscuits': 'Snacks', 'Juice': 'Beverages', 'Milk': 'Dairy', 'Bread': 'Bakery'
}
ALL_CATEGORIES = ['Grocery','Personal Care','Snacks','Beverages','Dairy','Bakery']

FEATURE_COLS = [
    'Price', 'DayOfWeek', 'Month', 'Day',
    'IsMonsoon','IsPaydayWindow','IsLocalHoliday','IsDiwaliSeason','IsHoliSeason','IsEidSeason',
    'IsStockout','Category','MRPBand'
]
TARGET_COL = 'Quantity'

# =========================
# India-local Calendar & Helpers
# =========================
def is_monsoon(dt: date) -> bool:
    return dt.month in [6, 7, 8, 9]

def is_payday_window(dt: date) -> bool:
    return dt.day in {1, 2, 3, 28, 29, 30, 31}

def fixed_holidays(include_karnataka=True):
    holidays = {(1, 26), (5, 1), (8, 15), (10, 2), (12, 25)}
    if include_karnataka:
        holidays.add((11, 1))  # Karnataka Rajyotsava
    return holidays

def is_local_holiday(dt: date, fixed_hdays) -> bool:
    return (dt.month, dt.day) in fixed_hdays

def is_diwali_season(dt: date) -> bool:
    start = date(dt.year, 10, 15); end = date(dt.year, 11, 25)
    return start <= dt <= end

def is_holi_season(dt: date) -> bool:
    start = date(dt.year, 2, 25); end = date(dt.year, 3, 20)
    return start <= dt <= end

def is_eid_season(dt: date) -> bool:
    start = date(dt.year, 4, 10); end = date(dt.year, 5, 25)
    return start <= dt <= end

def mrp_band(price: float) -> str:
    if price <= 20: return "≤20"
    if price <= 50: return "21–50"
    if price <= 80: return "51–80"
    return "81–120"

def derive_features(df: pd.DataFrame, include_karnataka=True) -> pd.DataFrame:
    """Ensure Date is datetime and derive time + India-local flags, MRPBand. Does NOT change Quantity."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    hdays = fixed_holidays(include_karnataka=include_karnataka)
    df['DayOfWeek'] = df['Date'].dt.weekday
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsMonsoon'] = df['Date'].dt.date.map(lambda d: int(is_monsoon(d)))
    df['IsPaydayWindow'] = df['Date'].dt.date.map(lambda d: int(is_payday_window(d)))
    df['IsLocalHoliday'] = df['Date'].dt.date.map(lambda d: int(is_local_holiday(d, hdays)))
    df['IsDiwaliSeason'] = df['Date'].dt.date.map(lambda d: int(is_diwali_season(d)))
    df['IsHoliSeason'] = df['Date'].dt.date.map(lambda d: int(is_holi_season(d)))
    df['IsEidSeason'] = df['Date'].dt.date.map(lambda d: int(is_eid_season(d)))
    df['MRPBand'] = df['Price'].apply(mrp_band)
    if 'IsStockout' not in df.columns:
        df['IsStockout'] = (df['Quantity'] == 0).astype(int)
    else:
        df['IsStockout'] = df['IsStockout'].fillna(0).astype(int)
    df['Category'] = df['Category'].astype(str)
    df['Item Name'] = df['Item Name'].astype(str)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

# =========================
# Data I/O
# =========================
def load_items_master() -> pd.DataFrame:
    if os.path.exists(ITEMS_MASTER_PATH):
        m = pd.read_csv(ITEMS_MASTER_PATH)
        m['Item Name'] = m['Item Name'].astype(str)
        m['Category'] = m['Category'].astype(str)
        return m
    # Build from defaults
    rows = [{'Item Name': it, 'Category': DEFAULT_CATEGORIES[it]} for it in DEFAULT_ITEMS]
    m = pd.DataFrame(rows)
    m.to_csv(ITEMS_MASTER_PATH, index=False)
    return m

def save_items_master(dfm: pd.DataFrame):
    dfm.to_csv(ITEMS_MASTER_PATH, index=False)

def load_or_create_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df = derive_features(df)
        return df
    st.info("No data file found — creating a starter dataset with India-local signals.")
    items_master = load_items_master()
    N = 2000
    rows = []
    rng = np.random.default_rng(42)
    today = datetime.today().date()
    for i in range(N):
        dt = today - timedelta(days=N - i)
        row = items_master.sample(1, random_state=int(rng.integers(0, 1e6))).iloc[0]
        item = row['Item Name']; category = row['Category']
        price = float(np.round(rng.uniform(10, 100), 2))
        flags = {
            'IsMonsoon': int(is_monsoon(dt)),
            'IsPaydayWindow': int(is_payday_window(dt)),
            'IsLocalHoliday': int(is_local_holiday(dt, fixed_holidays(True))),
            'IsDiwaliSeason': int(is_diwali_season(dt)),
            'IsHoliSeason': int(is_holi_season(dt)),
            'IsEidSeason': int(is_eid_season(dt)),
        }
        base_by_category = {'Grocery':80,'Personal Care':40,'Snacks':60,'Beverages':50,'Dairy':70,'Bakery':65}
        base_q = base_by_category.get(category, 50)
        m = 1.0
        if flags['IsDiwaliSeason']:
            m *= 1.30 if category in ['Snacks','Beverages','Grocery','Bakery'] else 1.10
        if flags['IsHoliSeason']:
            m *= 1.20 if category in ['Beverages','Snacks','Dairy'] else 1.05
        if flags['IsEidSeason']:
            m *= 1.15 if category in ['Grocery','Bakery','Dairy'] else 1.05
        if flags['IsMonsoon']:
            m *= 1.08 if category in ['Grocery','Snacks','Dairy'] else 1.03
        if flags['IsPaydayWindow']:
            m *= 1.10
        if flags['IsLocalHoliday']:
            m *= 1.10
        latent = base_q * m - 0.30 * price + rng.normal(0, 5)
        isStockout = int(rng.random() < 0.03)
        qty = 0 if isStockout else max(0, int(latent))
        rows.append({
            'Item Name': item, 'Category': category, 'Price': price, 'Date': str(dt),
            **flags,
            'MRPBand': mrp_band(price),
            'IsStockout': isStockout, 'Quantity': qty
        })
    df = pd.DataFrame(rows)
    df = derive_features(df)
    df.to_csv(DATA_PATH, index=False)
    return df

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)

# =========================
# Model
# =========================
def build_pipeline(random_state=42, n_estimators=600, learning_rate=0.05, max_depth=3, subsample=0.9):
    preprocess = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), ['Category','MRPBand'])
        ],
        remainder='passthrough'
    )
    gbr = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state
    )
    pipe = Pipeline(steps=[('prep', preprocess), ('gbr', gbr)])
    return pipe

def train_and_eval(df: pd.DataFrame, train_frac: float = 0.8):
    df = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(train_frac * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(float)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].astype(float)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    wape = np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test)) + 1e-9)
    metrics = {'r2': r2, 'mae': mae, 'wape': wape, 'test_preview': test_df.assign(Predicted=np.round(y_pred,1))}
    return pipe, metrics

def predict_one(pipe: Pipeline, row: dict) -> float:
    df = pd.DataFrame([row], columns=FEATURE_COLS)
    pred = pipe.predict(df)[0]
    return float(pred)

def rebuild_row_for_prediction(predict_date: date, item_name: str, category: str, price: float, is_stockout=0, include_karnataka=True):
    d = pd.Timestamp(predict_date)
    row = {
        'Price': float(price),
        'DayOfWeek': int(d.weekday()),
        'Month': int(d.month),
        'Day': int(d.day),
        'IsMonsoon': int(is_monsoon(predict_date)),
        'IsPaydayWindow': int(is_payday_window(predict_date)),
        'IsLocalHoliday': int(is_local_holiday(predict_date, fixed_holidays(include_karnataka))),
        'IsDiwaliSeason': int(is_diwali_season(predict_date)),
        'IsHoliSeason': int(is_holi_season(predict_date)),
        'IsEidSeason': int(is_eid_season(predict_date)),
        'IsStockout': int(is_stockout),
        'Category': category,
        'MRPBand': mrp_band(price)
    }
    return row

# =========================
# Session State
# =========================
if 'data' not in st.session_state:
    st.session_state.data = load_or_create_data()
if 'items_master' not in st.session_state:
    st.session_state.items_master = load_items_master()
if 'model' not in st.session_state:
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state.model = joblib.load(MODEL_PATH)
        except Exception:
            st.session_state.model = None
    else:
        st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

data = st.session_state.data
items_master = st.session_state.items_master
model = st.session_state.model
metrics = st.session_state.metrics

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="card" style="background: linear-gradient(90deg,#0ea5e9 0%, #2563eb 55%); color: white; margin-top: -10px;">
      <div style="display:flex;align-items:center;gap:.75rem;">
        <div style="font-size: 1.7rem;">🛒</div>
        <div>
          <div style="font-size:1.2rem;font-weight:700;letter-spacing:.3px;">Retail Demand Forecast (India‑local)</div>
          <div style="opacity:.95; font-size:.95rem;">Diwali/Holi/Eid • Monsoon • Payday • Local Holidays • MRP bands • Stockouts</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True
)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "🧰 Data Manager", "🤖 Train / Evaluate", "🔮 Predict", "🗓️ 30‑day Forecast", "💡 Marketing Strategy"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Rows: **{len(data):,}** |  Items: **{data['Item Name'].nunique()}**")

# =========================
# 📊 Overview
# =========================
if page == "📊 Overview":
    st.subheader("Dataset Overview")

    # Filters section in a card
    with st.container():
        c1, c2, c3 = st.columns([2, 1.5, 1.5])
        with c1:
            dr = st.date_input(
                "Date range",
                value=(pd.to_datetime(data['Date']).min().date(), pd.to_datetime(data['Date']).max().date())
            )
            if isinstance(dr, (tuple, list)) and len(dr) == 2:
                start_d, end_d = dr
            else:
                start_d, end_d = pd.to_datetime(data['Date']).min().date(), pd.to_datetime(data['Date']).max().date()
        with c2:
            cat_options = sorted(data['Category'].unique().tolist())
            cat_sel = st.multiselect("Category", options=cat_options, default=cat_options)
        with c3:
            item_options = sorted(data['Item Name'].unique().tolist())
            item_sel = st.multiselect("Item", options=item_options, default=item_options)

    # Apply filters for overview visuals
    dfv = data.copy()
    dfv['Date'] = pd.to_datetime(dfv['Date'])
    dfv['Revenue'] = dfv['Price'] * dfv['Quantity']
    mask = (
        (dfv['Date'].dt.date >= start_d) &
        (dfv['Date'].dt.date <= end_d) &
        (dfv['Category'].isin(cat_sel)) &
        (dfv['Item Name'].isin(item_sel))
    )
    dfv = dfv.loc[mask]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dfv):,}")
    c2.metric("Items", int(dfv['Item Name'].nunique()))
    c3.metric("Categories", int(dfv['Category'].nunique()))
    days = max(1, (dfv['Date'].max() - dfv['Date'].min()).days + 1) if len(dfv) else 1
    c4.metric("Units / Day", f"{(dfv['Quantity'].sum() / days) if len(dfv) else 0:.1f}")

    
    # 📊 Enhanced Visual Insights
    import matplotlib.pyplot as plt
    dfv['Month'] = dfv['Date'].dt.to_period('M')

    st.markdown("#### 📈 Category-wise Sales Trends")
    cat_trend = dfv.groupby(['Date', 'Category'])['Quantity'].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for cat in cat_trend['Category'].unique():
        subset = cat_trend[cat_trend['Category'] == cat]
        ax1.plot(subset['Date'], subset['Quantity'], label=cat)
    ax1.set_title("Quantity Sold Over Time by Category")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Quantity")
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("#### 📦 Item-wise Quantity Distribution")
    item_dist = dfv.groupby('Item Name')['Quantity'].sum().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    item_dist.plot(kind='barh', ax=ax2, color='skyblue')
    ax2.set_title("Total Quantity Sold per Item")
    ax2.set_xlabel("Quantity")
    st.pyplot(fig2)

    st.markdown("#### 💰 Monthly Revenue Comparison")
    monthly_rev = dfv.groupby('Month')['Revenue'].sum().reset_index()
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar(monthly_rev['Month'].astype(str), monthly_rev['Revenue'], color='orange')
    ax3.set_title("Monthly Revenue")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Revenue (₹)")
    st.pyplot(fig3)

# Trend charts
    if len(dfv):
        daily = (dfv.groupby('Date', as_index=False)[['Quantity','Revenue']].sum()
                     .sort_values('Date'))
        st.line_chart(daily, x='Date', y=['Quantity','Revenue'], height=260)
    else:
        st.info("No data in the selected range/filters.")

    st.markdown("#### Data snapshot")
    st.dataframe(dfv.head(250), use_container_width=True, hide_index=True)

    # Download current full data
    st.markdown("#### Download current data")
    csv_buf = io.StringIO()
    data.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download CSV", csv_buf.getvalue(), file_name="inventory_with_india_features.csv", mime="text/csv")

# =========================
# 🧰 Data Manager
# =========================
elif page == "🧰 Data Manager":
    st.subheader("Upload or Manage Data")

    # Upload block (card)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Upload CSV (optional)** — Expected columns: `Item Name, Category, Price, Date, Quantity` (`IsStockout` optional).")
    up = st.file_uploader("Upload your sales CSV", type=['csv'])
    colu1, colu2 = st.columns([1,1])
    with colu1:
        mode = st.radio("Upload mode", ["Replace current data", "Append to current data"], horizontal=True, index=0)
    with colu2:
        commit = st.checkbox("Write to file after load", value=True)
    if up is not None and st.button("Process upload"):
        try:
            up_df = pd.read_csv(up)
            req_cols = {'Item Name','Category','Price','Date','Quantity'}
            if not req_cols.issubset(set(up_df.columns)):
                st.error(f"CSV must contain columns: {req_cols}")
            else:
                up_df = derive_features(up_df)
                if mode == "Replace current data":
                    st.session_state.data = up_df
                    data = up_df
                else:
                    data = pd.concat([data, up_df], ignore_index=True).sort_values('Date').reset_index(drop=True)
                    st.session_state.data = data
                if commit:
                    save_data(data)
                st.success(f"Loaded {len(up_df):,} rows from uploaded CSV.")
        except Exception as e:
            st.exception(e)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Items Master (avoid spelling mistakes)")
    st.caption("Use this section to manage **Item Name ↔ Category**. The helper suggests near‑duplicates to avoid typos.")

    st.dataframe(items_master, use_container_width=True, hide_index=True)
    with st.expander("➕ Add/Update Item in Items Master"):
        new_item = st.text_input("New item name")
        if new_item:
            choices = items_master['Item Name'].tolist()
            matches = process.extract(new_item, choices, scorer=fuzz.WRatio, limit=3)
            similar = [f"{m[0]} (match {m[1]}%)" for m in matches if m[1] >= 85]
            if similar:
                st.info("Similar existing items (to avoid duplicates): " + ", ".join(similar))
        new_cat = st.selectbox("Category", ALL_CATEGORIES, index=ALL_CATEGORIES.index('Grocery'))
        if st.button("Add / Update Item"):
            if not new_item.strip():
                st.warning("Enter a valid item name.")
            else:
                idx = items_master.index[items_master['Item Name'].str.lower()==new_item.strip().lower()].tolist()
                if idx:
                    items_master.loc[idx[0], 'Category'] = new_cat
                    st.success(f"Updated category for **{new_item}** → {new_cat}")
                else:
                    items_master = pd.concat([
                        items_master,
                        pd.DataFrame([{'Item Name': new_item.strip(), 'Category': new_cat}])
                    ], ignore_index=True)
                    st.success(f"Added item **{new_item}** in category **{new_cat}**.")
                st.session_state.items_master = items_master
                save_items_master(items_master)

    st.markdown("---")
    st.markdown("### ➕ Append a New Row to the Data CSV")
    st.caption("Add a daily record (Date, Item, Category, Price, Quantity). Features are derived automatically.")
    c1, c2 = st.columns(2)
    with c1:
        add_date = st.date_input(
            "Date",
            value=min(date.today(), pd.to_datetime(data['Date']).max().date() + timedelta(days=1))
        )
        item_options = sorted(items_master['Item Name'].unique().tolist())
        selected_item = st.selectbox("Item Name", options=item_options + ["➕ Add new item..."], index=0)
        if selected_item == "➕ Add new item...":
            new_item_name = st.text_input("Enter new item name")
            if new_item_name:
                matches = process.extract(new_item_name, item_options, scorer=fuzz.WRatio, limit=3)
                similar = [f"{m[0]} (match {m[1]}%)" for m in matches if m[1] >= 85]
                if similar:
                    st.info("Did you mean: " + ", ".join(similar))
            new_item_cat = st.selectbox("Category for new item", ALL_CATEGORIES, key="new_item_cat_addrow")
        price_val = st.number_input("Price (₹)", min_value=0.0, max_value=99999.0, value=50.0, step=1.0)
    with c2:
        qty_val = st.number_input("Quantity (units)", min_value=0, max_value=100000, value=0, step=1)
        is_stockout = st.checkbox("Is Stockout? (zero sales due to no stock)", value=(qty_val==0))
        include_karnataka = st.checkbox("Include Karnataka Rajyotsava in Local Holidays", value=True)
    if st.button("Append Row to CSV"):
        if selected_item == "➕ Add new item...":
            if not new_item_name.strip():
                st.warning("Please provide a valid new item name.")
                st.stop()
            if not (items_master['Item Name'].str.lower() == new_item_name.strip().lower()).any():
                items_master = pd.concat([
                    items_master,
                    pd.DataFrame([{'Item Name': new_item_name.strip(), 'Category': new_item_cat}])
                ], ignore_index=True)
                save_items_master(items_master)
            item_name_final = new_item_name.strip()
            category_final = items_master.loc[
                items_master['Item Name'].str.lower()==item_name_final.lower(), 'Category'
            ].iloc[0]
        else:
            item_name_final = selected_item
            category_final = items_master.loc[
                items_master['Item Name']==item_name_final, 'Category'
            ].iloc[0]
        new_row = {
            'Item Name': item_name_final,
            'Category': category_final,
            'Price': float(price_val),
            'Date': pd.to_datetime(add_date),
            'Quantity': int(qty_val),
            'IsStockout': int(is_stockout)
        }
        new_row_df = pd.DataFrame([new_row])
        new_row_df = derive_features(new_row_df, include_karnataka=include_karnataka)
        data = pd.concat([data, new_row_df], ignore_index=True).sort_values('Date').reset_index(drop=True)
        st.session_state.data = data
        save_data(data)
        st.success("Row appended and CSV updated ✅")

# =========================
# 🤖 Train / Evaluate
# =========================
elif page == "🤖 Train / Evaluate":
    st.subheader("Train Gradient Boosting Model")
    st.caption("Chronological split to avoid leakage. Targets Quantity using India‑local features + price.")

    train_frac = st.slider("Train fraction", min_value=0.6, max_value=0.95, value=0.8, step=0.05)
    if st.button("Train / Retrain Model"):
        with st.spinner("Training..."):
            m, met = train_and_eval(data, train_frac=train_frac)
        st.session_state.model = m
        st.session_state.metrics = met
        joblib.dump(m, MODEL_PATH)
        st.success("Model trained and saved ✅")

    if st.session_state.metrics:
        met = st.session_state.metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{met['r2']:.3f}")
        c2.metric("MAE (units)", f"{met['mae']:.2f}")
        c3.metric("WAPE", f"{100*met['wape']:.2f}%")
        st.markdown("**Test preview (top 15)**")
        st.dataframe(
            met['test_preview'][['Date','Item Name','Category','Price','Quantity','Predicted']].head(15),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Train the model to see evaluation metrics.")

# =========================
# 🔮 Predict (Single)
# =========================
elif page == "🔮 Predict":
    st.subheader("Single‑day Prediction")
    if model is None:
        st.warning("No trained model found. Train it in **Train / Evaluate**.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            pred_date = st.date_input("Prediction date", value=date.today() + timedelta(days=1))
            item_options = sorted(items_master['Item Name'].tolist())
            item_sel = st.selectbox("Item Name", options=item_options)
            category_sel = items_master.loc[items_master['Item Name']==item_sel, 'Category'].iloc[0]
        with c2:
            median_price = float(data.loc[data['Item Name']==item_sel, 'Price'].median()) if (data['Item Name']==item_sel).any() else 50.0
            price_in = st.number_input("Price (₹)", min_value=0.0, max_value=99999.0, value=float(median_price), step=1.0)
            assume_no_stockout = st.checkbox("Assume available (no stockout)", value=True)
            include_karnataka = st.checkbox("Include Karnataka Rajyotsava", value=True)

        row = rebuild_row_for_prediction(
            pred_date, item_sel, category_sel, price_in,
            is_stockout=0 if assume_no_stockout else 1,
            include_karnataka=include_karnataka
        )
        pred = predict_one(model, row)

        # Pretty result card
        st.markdown(
            f"""
            <div class="card" style="border-left: 6px solid #1d4ed8;">
              <div style="font-size:.95rem;color:#6b7280;">Predicted quantity for</div>
              <div style="font-size:1.05rem;"><b>{item_sel}</b> on <b>{pred_date}</b> at <b>₹{price_in:.0f}</b></div>
              <div style="font-size:2rem;margin-top:.2rem;"><b>{pred:.1f} units</b></div>
            </div>
            """, unsafe_allow_html=True
        )

# =========================
# 🗓️ 30‑day Forecast
# =========================
elif page == "🗓️ 30‑day Forecast":
    st.subheader("30‑day Purchase Plan Forecast")
    if model is None:
        st.warning("No trained model found. Train it in **Train / Evaluate**.")
    else:
        with st.form("forecast_form"):
            start_date = st.date_input("Start date", value=date.today() + timedelta(days=1))
            horizon = st.slider("Horizon (days)", 7, 60, 30, step=1)
            items_selected = st.multiselect(
                "Items to forecast",
                options=sorted(items_master['Item Name'].tolist()),
                default=sorted(items_master['Item Name'].unique().tolist())
            )
            price_mode = st.selectbox("Price source", ["Median historical price per item", "Single price for all"])
            if price_mode == "Single price for all":
                base_price = st.number_input("Price for all (₹)", min_value=0.0, max_value=99999.0, value=50.0, step=1.0)
            include_karnataka = st.checkbox("Include Karnataka Rajyotsava", value=True)
            submitted = st.form_submit_button("Generate Forecast")

        if submitted:
            rows = []
            for it in items_selected:
                cat = items_master.loc[items_master['Item Name']==it, 'Category'].iloc[0]
                if price_mode == "Median historical price per item":
                    p = float(data.loc[data['Item Name']==it, 'Price'].median()) if (data['Item Name']==it).any() else 50.0
                else:
                    p = float(base_price)
                for d in range(horizon):
                    dt = start_date + timedelta(days=d)
                    r = rebuild_row_for_prediction(dt, it, cat, p, is_stockout=0, include_karnataka=include_karnataka)
                    yhat = predict_one(model, r)
                    rows.append({'Date': dt, 'Item Name': it, 'Category': cat, 'Price': p, 'Predicted Quantity': round(yhat,1)})

            fc = pd.DataFrame(rows).sort_values(['Date','Item Name']).reset_index(drop=True)

            # Mini trend (total predicted units / day)
            daily_fc = fc.groupby('Date', as_index=False)['Predicted Quantity'].sum()
            st.line_chart(daily_fc, x='Date', y='Predicted Quantity', height=220)

            st.dataframe(fc.head(300), use_container_width=True, hide_index=True)

            out = io.StringIO(); fc.to_csv(out, index=False)
            st.download_button("⬇️ Download forecast CSV", data=out.getvalue(), file_name="purchase_plan_30d.csv", mime="text/csv")


# =========================
# 💡 Marketing Strategy
# =========================
elif page == "💡 Marketing Strategy":
    st.subheader("💡 Marketing Strategy Suggestions (AI)")
    st.markdown("Select a period of past sales and items to get actionable marketing advice powered by a generative AI model.")

    with st.container(border=True):
        period_days = st.selectbox("Analyze past period:", [7, 14, 28, 60], index=1, format_func=lambda x: f"Last {x} days")
        
        all_items = sorted(items_master['Item Name'].tolist())
        default_items = all_items[:10] if len(all_items) > 10 else all_items

        items_selected = st.multiselect(
            "Items to analyze",
            options=all_items,
            default=default_items
        )

    if st.button("Generate Strategy"):
        if not items_selected:
            st.warning("Please select at least one item to analyze.")
        else:
            with st.spinner("Analyzing data and asking Gemini for suggestions..."):
                try:
                    # Filter past sales
                    df_copy = data.copy()
                    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                    end_date = df_copy['Date'].max()
                    start_date = end_date - pd.Timedelta(days=period_days)
                    
                    df_period = df_copy[
                        (df_copy['Date'] >= start_date) &
                        (df_copy['Date'] <= end_date) &
                        (df_copy['Item Name'].isin(items_selected))
                    ]
                    
                    if df_period.empty:
                        st.error(f"No sales data found for the selected items in the last {period_days} days.")
                    else:
                        # --- NEW GEMINI API LOGIC (INSECURE METHOD) ---
                        # WARNING: This is not a safe way to handle API keys.
                        # Your key is exposed and can be stolen if this code is shared.
                        
                        # 1. Configure the API with the hardcoded secret key
                        api_key = "AIzaSyD_P3niRhEywaMQP-l0oJafMASePegvImM"
                        genai.configure(api_key=api_key)

                        # 2. Initialize the Gemini Pro model
                        model = genai.GenerativeModel('gemini-2.5-flash')

                        # 3. Create the summary and the prompt (this is the same as before)
                        summary = df_period.groupby(['Item Name', 'Category'])['Quantity'].agg(['sum', 'mean']).reset_index()
                        summary['sum'] = summary['sum'].astype(int)
                        summary['mean'] = summary['mean'].round(1)
                        summary.rename(columns={'sum': 'Total Units Sold', 'mean': 'Avg Daily Units'}, inplace=True)
                        summary_text = summary.to_string(index=False)
                        
                        # The prompt creation and model call follows...
                        prompt = (
                            f"You are a marketing manager for a local Kirana store in Coimbatore, India. It's early October, so Diwali festival preparations are starting. "
                            f"Analyze the following sales data from the last {period_days} days:\n\n"
                            f"--- SALES DATA ---\n{summary_text}\n---\n\n"
                            f"Your task is to create a list of specific, actionable discount offers and combo deals to increase sales for the upcoming festive season. "
                            f"Do not just repeat the data or the instructions. "
                            f"Focus on practical ideas like percentage discounts, 'Buy One Get One' (BOGO) offers, and product bundles that would appeal to local customers. "
                            f"Structure your response as a bulleted list. Start each suggestion with the item or category name. Here are some examples of the format I want:\n"
                            f"- **Rice & Grocery:** Offer a 'Diwali Pantry Bundle' with Rice, Sugar, and Salt at a 5% total discount.\n"
                            f"- **Personal Care:** Create a 'Buy 2 Soaps, Get 1 small Shampoo free' combo.\n\n"
                            f"Now, generate the marketing offers based on the data provided."
                        )

                        response = model.generate_content(prompt)
                        suggestions = response.text

                        st.markdown("### Suggested Marketing Strategies from Google Gemini")
                        st.markdown(suggestions)

                except Exception as e:
                    st.error(f"An error occurred while contacting the Gemini API: {e}")
                    st.info("Please ensure your API key is correctly configured in .streamlit/secrets.toml and that it is active.")


# =========================
# Footer
# =========================
st.markdown(
    """
    <hr style="margin-top:1.2rem;margin-bottom:.6rem;"/>
    <div class="note">
      Built for Indian retail context. UI polished only; all modeling & data logic preserved.
    </div>
    """, unsafe_allow_html=True
)