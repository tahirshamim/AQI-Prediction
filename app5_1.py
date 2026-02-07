# =====================================================
# KARACHI AQI FORECAST DASHBOARD (DARK MODE ENHANCED)
# =====================================================

import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import timedelta,datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
import os


# ---------------- CONFIG ----------------
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "aqi_mlops"

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS (DARK MODE OPTIMIZED) ----------------
st.markdown("""
    <style>
        /* Global Font & Spacing */
        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 3rem;
            font-family: 'Inter', sans-serif;
        }
        
        /* Glassmorphism Card Style */
        .glass-card {
            background-color: rgba(255, 255, 255, 0.05); /* Subtle transparency */
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }

        /* Metric Typography */
        .metric-label {
            color: var(--text-color);
            opacity: 0.7;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-value {
            color: var(--text-color);
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
        }

        /* Tabs container */
        .stTabs [data-baseweb="tab"] {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-radius: 10px;
            height: 48px;
            padding: 8px 16px;
            font-weight: 500;
            border: 1px solid rgba(0,0,0,0.05);
        }

        /* Selected tab (works in light & dark) */
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: var(--background-color) !important;
            border: none;
        }

        /* Hover (theme-aware) */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: color-mix(
                in srgb,
                var(--primary-color) 15%,
                var(--secondary-background-color)
            );
        }

        /* Clean up standard Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
    """, unsafe_allow_html=True)

# ---------------- MongoDB ----------------
# Use st.cache_resource for connection to prevent reconnection on every rerun
@st.cache_resource
def init_connection():
    return MongoClient(MONGO_URI)

client = init_connection()
db = client[DB_NAME]


# ---------------- AQI HELPER FUNCTIONS ----------------
def get_aqi_color(aqi):
    # Returning Tuple: (Category Name, Hex Color, Text Color for contrast)
    if aqi <= 50: return "Good", "#00b894", "#ffffff"       # Mint Green
    elif aqi <= 100: return "Moderate", "#fdcb6e", "#000000" # Mustard Yellow
    elif aqi <= 150: return "Unhealthy (Sen.)", "#e17055", "#ffffff" # Burnt Orange
    elif aqi <= 200: return "Unhealthy", "#d63031", "#ffffff"    # Red
    elif aqi <= 300: return "Very Unhealthy", "#6c5ce7", "#ffffff" # Purple
    return "Hazardous", "#2d3436", "#ffffff"                     # Dark Charcoal

# =====================================================
# LOAD DATA
# =====================================================
# ---------------- LATEST HOURLY AQI (TODAY) ----------------
latest_hour_doc = db.raw_aqi_hourly.find_one(
    sort=[("timestamp", -1)]
)

if latest_hour_doc is None:
    st.error("No hourly AQI data found.")
    st.stop()

latest_hour_aqi = int(latest_hour_doc["aqi"])
latest_hour_time = latest_hour_doc["timestamp"]
# ----------------  Yesterday AQI ----------------
try:
    features_cursor = db.aqi_features.find({}, {"_id": 0}).sort("date", 1)
    features_df = pd.DataFrame(list(features_cursor))
    
    if features_df.empty:
        st.error("No AQI feature data found.")
        st.stop()
        
    features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")
    features_df = features_df.dropna(subset=["date"])
    
    pred_doc = db.aqi_predictions.find_one(sort=[("prediction_date", -1)])
    if pred_doc is None:
        st.error("No prediction data found.")
        st.stop()
        
    predictions = pred_doc["predictions"]

except Exception as e:
    st.error(f"Database Error: {e}")
    st.stop()

# =====================================================
# CALCULATIONS
# =====================================================
latest_date = features_df.iloc[-1]["date"]
latest_aqi = float(features_df.iloc[-1]["AQI"])

forecast_dates = pd.date_range(start=latest_date + timedelta(days=1), periods=3, freq="D")
forecast_values = [float(predictions["AQI_t+1"]), float(predictions["AQI_t+2"]), float(predictions["AQI_t+3"])]

# =====================================================
# DASHBOARD HEADER
# =====================================================
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🌙 Karachi AQI Watch")
    st.markdown(
    f"<span style='color: #a0a0a0;'>Live AQI • Updated: {latest_hour_time.strftime('%B %d, %Y %I:%M %p')}</span>",unsafe_allow_html=True
    )

with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# =====================================================
# ROW 1: KEY METRICS & GAUGE
# =====================================================
cat, color, txt_color = get_aqi_color(latest_hour_aqi)

col_main, col_gauge = st.columns([1.5, 1])

with col_main:
    st.markdown(f"""
    <div class="glass-card" style="background: linear-gradient(135deg, {color}aa 0%, {color}44 100%); border-left: 6px solid {color}; display: flex; align-items: center; justify-content: space-between; padding: 30px;">
        <div>
            <div style="color: {txt_color}; opacity: 0.9; font-size: 1.1rem; font-weight: 500;">TODAY • LIVE AQI</div>
            <div style="color: {txt_color}; font-size: 4rem; font-weight: 800; line-height: 1;">{int(latest_hour_aqi)}</div>
            <div style="color: {txt_color}; font-size: 1.5rem; font-weight: 600; margin-top: 5px;">{cat}</div>
        </div>
        <div style="font-size: 4rem; opacity: 0.8;">
            {'😷' if latest_hour_aqi > 100 else '🙂'}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Context note
    st.caption("ℹ️ *Note: AQI above 150 is considered unhealthy for the general public.*")
    st.markdown(f"""
    <div class="glass-card" style="padding: 12px; opacity: 0.85;">
        <div class="metric-label">Yesterday Average AQI</div>
        <div style="font-size: 1.4rem; font-weight: 600;">
            {int(latest_aqi)} • {get_aqi_color(latest_aqi)[0]}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    # Transparent Gauge Chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_hour_aqi,
        number={'font': {'size': 35,'color': 'var(--text-color)'}}, # White text for dark mode
        gauge={
            "axis": {"range": [0, 400], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": "rgba(255,255,255,0.8)", "thickness": 0.25}, # White glow bar
            "bgcolor": "rgba(0,0,0,0)", # Transparent
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#2ecc71"},
                {"range": [50, 100], "color": "#f1c40f"},
                {"range": [100, 150], "color": "#e67e22"},
                {"range": [150, 200], "color": "#e74c3c"},
                {"range": [200, 300], "color": "#8e44ad"},
                {"range": [300, 400], "color": "#2c3e50"},
            ]
        }
    ))
    gauge_fig.update_layout(
        margin=dict(l=30, r=30, t=10, b=35), 
        height=230,
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'var(--text-color)'}
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

# =====================================================
# ROW 2: 3-DAY FORECAST CARDS
# =====================================================
st.subheader("📅 3-Day Outlook")

cols = st.columns(3)
for i in range(3):
    value = forecast_values[i]
    date_str = forecast_dates[i].strftime("%a, %b %d")
    cat_f, color_f, _ = get_aqi_color(value)
    
    with cols[i]:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-bottom: 4px solid {color_f};">
            <div class="metric-label">{date_str}</div>
            <div class="metric-value">{int(value)}</div>
            <div style="background-color: {color_f}33; color: {color_f}; display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
                {cat_f}
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# ROW 3: ADVANCED ANALYTICS
# =====================================================
hist_part = features_df[["date", "AQI"]].copy()
hist_part["Type"] = "Historical"
forecast_df = pd.DataFrame({"date": forecast_dates, "AQI": forecast_values})
forecast_df["Type"] = "Forecast"

# Bridge the line gap
last_hist_row = hist_part.iloc[[-1]].copy()
last_hist_row["Type"] = "Forecast" 
forecast_connected = pd.concat([last_hist_row, forecast_df])

tab1, tab2, tab3 = st.tabs(["📉 Trend Analysis", "📜 History", "🧠 Model Metrics"])

with tab1:
    fig_combined = go.Figure()
    
    # Gradient Area for History
    fig_combined.add_trace(go.Scatter(
        x=hist_part['date'], y=hist_part['AQI'],
        mode='lines', name='Observed',
        line=dict(color='#00cec9', width=2),
        fill='tozeroy', fillcolor='rgba(0, 206, 201, 0.1)'
    ))

    # Dashed line for Forecast
    fig_combined.add_trace(go.Scatter(
        x=forecast_connected['date'], y=forecast_connected['AQI'],
        mode='lines+markers', name='AI Prediction',
        line=dict(color='#fab1a0', width=3, dash='dot'),
        marker=dict(size=8, color='#fab1a0')
    ))

    fig_combined.update_layout(
        template="plotly_dark", # Native Dark Mode Template
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="AQI Level")
    )
    st.plotly_chart(fig_combined, use_container_width=True)

with tab2:
    st.dataframe(features_df.sort_values("date", ascending=False), use_container_width=True)
with tab3:
    st.subheader("📊 Model Performance")

    metric_df = pd.DataFrame(pred_doc["evaluation_metrics"]).T
    st.dataframe(metric_df)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image("https://static.vecteezy.com/system/resources/previews/026/571/030/large_2x/weather-icon-with-sun-and-cloud-on-transparent-background-free-png.png", width=800)
st.sidebar.title("System Info")

st.sidebar.success(f"**Status:** Operational")

st.sidebar.markdown("### ⚙ Pipeline Architecture")
st.sidebar.info("""
1. **Ingest:** Hourly AQI Collection
2. **Process:** Daily Aggregation
3. **ML Core:** ElasticNet MultiOutput
4. **Store:** MongoDB Atlas
5. **View:** Streamlit UI
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.caption("""
This dashboard monitors Air Quality Index (AQI) in Karachi using Machine Learning.
""")

if st.sidebar.checkbox("Enable Auto Refresh", value=False):
    st.sidebar.caption("Refreshes every 60 seconds")

    st.rerun()
