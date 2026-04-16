"""
app.py — Rain Prediction in Australia
Streamlit web application with two modes:
  • Explore Historical Data — pick a location + date from the BoM dataset
  • Live Forecast          — manually enter today's weather readings
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

from pipeline import (
    load_artifacts,
    get_location_data,
    compute_temporal_features,
    compute_cyclic_features,
    assemble_feature_vector,
    predict,
    get_location_defaults,
    COMPASS_MAP,
)

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain Prediction — Australia",
    page_icon="🌧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #F5F7FA;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid #065A82;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1E293B;
    }
    .result-rain {
        background: linear-gradient(135deg, #065A82, #1C7293);
        color: white;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-norain {
        background: linear-gradient(135deg, #B45309, #D97706);
        color: white;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .result-sub {
        font-size: 1rem;
        opacity: 0.85;
    }
    .ground-truth-correct {
        background: #D1FAE5;
        border: 1.5px solid #10B981;
        border-radius: 10px;
        padding: 10px 16px;
        color: #065F46;
        font-weight: 600;
    }
    .ground-truth-wrong {
        background: #FEE2E2;
        border: 1.5px solid #EF4444;
        border-radius: 10px;
        padding: 10px 16px;
        color: #991B1B;
        font-weight: 600;
    }
    .disclaimer {
        font-size: 0.75rem;
        color: #94A3B8;
        font-style: italic;
        margin-top: 6px;
    }
    div[data-testid="metric-container"] {
        background: #F5F7FA;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data & artifacts (cached) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and pipeline…")
def load_model():
    return load_artifacts()

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv("data/weatherAUS_3.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

artifacts = load_model()
df = load_data()

LOCATIONS = sorted(df['Location'].unique().tolist())
DIRECTIONS = list(COMPASS_MAP.keys())
DATE_MIN = df['Date'].min().date()
DATE_MAX = df['Date'].max().date()

# ── Helper: gauge chart ──────────────────────────────────────────────────
def make_gauge(prob: float) -> go.Figure:
    pct = prob * 100
    if pct < 30:
        color = "#D97706"
    elif pct < 60:
        color = "#1C7293"
    else:
        color = "#065A82"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%', 'font': {'size': 36, 'color': '#1E293B'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94A3B8'},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30],  'color': '#FEF3C7'},
                {'range': [30, 60], 'color': '#DBEAFE'},
                {'range': [60, 100],'color': '#BFDBFE'},
            ],
            'threshold': {
                'line': {'color': '#EF4444', 'width': 3},
                'thickness': 0.75,
                'value': 50,
            },
        },
        title={'text': "Rain Probability", 'font': {'size': 14, 'color': '#64748B'}},
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=10, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def confidence_label(prob: float) -> str:
    dist = abs(prob - 0.5)
    if dist > 0.30: return "🟢 High Confidence"
    if dist > 0.15: return "🟡 Medium Confidence"
    return "🔴 Low Confidence"


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/120px-Above_Gotham.jpg",
             width=60)
    st.title("🌧 Rain Prediction")
    st.caption("Australian Bureau of Meteorology · XGBoost Model")
    st.divider()

    mode = st.radio(
        "**Mode**",
        [
            "🔍 Historical Data",
            "⚡ Quick Forecast",
            "🛠 Full Live Forecast",
        ],
        help=(
            "Historical: pick a real date and compare to ground truth.\n"
            "Quick: 4 inputs only — everything else auto-filled.\n"
            "Full: all 16 weather inputs for maximum control."
        ),
    )
    st.divider()

    location = st.selectbox("**Location**", LOCATIONS, index=LOCATIONS.index("Sydney"))

    if mode == "🔍 Historical Data":
        selected_date = st.date_input(
            "**Date**",
            value=date(2016, 6, 1),
            min_value=DATE_MIN,
            max_value=DATE_MAX,
            help="Restricted to BoM dataset range (2007–2017)",
        )
    else:
        selected_month = st.selectbox(
            "**Current Month**",
            options=list(range(1, 13)),
            format_func=lambda m: date(2000, m, 1).strftime("%B"),
            index=5,
        )

    st.divider()
    st.caption("AT82.03 Machine Learning · AIT 2/2025")


# ── HISTORICAL MODE ──────────────────────────────────────────────────────
if mode == "🔍 Historical Data":
    st.header(f"📅 {location} — {selected_date.strftime('%d %B %Y')}")

    loc_df = get_location_data(df, location)
    target_ts = pd.Timestamp(selected_date)

    # Find the exact row
    row_match = loc_df[loc_df['Date'] == target_ts]

    if row_match.empty:
        st.warning(f"No data found for **{location}** on **{selected_date}**. "
                   "Try a different date — some dates may be missing for this station.")
        st.stop()

    row = row_match.iloc[0]

    # ── Weather reading display ───────────────────────────────────────
    st.subheader("☁ Today's Conditions")
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, label, value, unit=""):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} <span style="font-size:0.85rem;color:#94A3B8">{unit}</span></div>
        </div>""", unsafe_allow_html=True)

    with c1:
        metric_card(c1, "Min Temperature", f"{row.get('MinTemp', 'N/A'):.1f}", "°C")
        metric_card(c1, "Max Temperature", f"{row.get('MaxTemp', 'N/A'):.1f}", "°C")
        metric_card(c1, "Temp 9am",  f"{row.get('Temp9am', 'N/A'):.1f}", "°C")
        metric_card(c1, "Temp 3pm",  f"{row.get('Temp3pm', 'N/A'):.1f}", "°C")
    with c2:
        metric_card(c2, "Humidity 9am", f"{row.get('Humidity9am', 'N/A'):.0f}", "%")
        metric_card(c2, "Humidity 3pm", f"{row.get('Humidity3pm', 'N/A'):.0f}", "%")
        metric_card(c2, "Rainfall",     f"{row.get('Rainfall', 'N/A'):.1f}", "mm")
        rain_today_val = "Yes ☔" if row.get('RainToday', 'No') == 'Yes' or row.get('RainToday', 0) == 1 else "No ☀"
        metric_card(c2, "Rain Today", rain_today_val, "")
    with c3:
        metric_card(c3, "Pressure 9am", f"{row.get('Pressure9am', 'N/A'):.1f}", "hPa")
        metric_card(c3, "Pressure 3pm", f"{row.get('Pressure3pm', 'N/A'):.1f}", "hPa")
        metric_card(c3, "Sunshine",     f"{row.get('Sunshine', 'N/A'):.1f}", "hrs")
        metric_card(c3, "Evaporation",  f"{row.get('Evaporation', 'N/A'):.1f}", "mm")
    with c4:
        metric_card(c4, "Wind Gust Speed", f"{row.get('WindGustSpeed', 'N/A'):.0f}", "km/h")
        metric_card(c4, "Wind Gust Dir",   str(row.get('WindGustDir', 'N/A')), "")
        metric_card(c4, "Cloud 9am",       f"{row.get('Cloud9am', 'N/A'):.0f}", "/8")
        metric_card(c4, "Cloud 3pm",       f"{row.get('Cloud3pm', 'N/A'):.0f}", "/8")

    # ── Lag context ───────────────────────────────────────────────────
    with st.expander("🕐 Previous days' context (lag features used for prediction)"):
        lag_data = []
        for days_back in [1, 2, 3]:
            past_date = target_ts - pd.Timedelta(days=days_back)
            past_row = loc_df[loc_df['Date'] == past_date]
            if not past_row.empty:
                pr = past_row.iloc[0]
                lag_data.append({
                    'Date': past_date.date(),
                    'Rainfall (mm)': pr.get('Rainfall', np.nan),
                    'Humidity3pm (%)': pr.get('Humidity3pm', np.nan),
                    'Pressure3pm (hPa)': pr.get('Pressure3pm', np.nan),
                })
        if lag_data:
            st.dataframe(pd.DataFrame(lag_data).set_index('Date'), use_container_width=True)
        else:
            st.info("No historical lag data available for this date.")

    # ── Build feature vector & predict ────────────────────────────────
    temporal = compute_temporal_features(loc_df, target_ts)

    # Reconstruct wind direction strings from the raw row (already string in data)
    def safe_dir(val):
        if isinstance(val, str) and val in COMPASS_MAP:
            return val
        return 'N'

    month = int(target_ts.month)
    day   = int(target_ts.day)
    cyclic = compute_cyclic_features(
        month, day,
        safe_dir(row.get('WindGustDir', 'N')),
        safe_dir(row.get('WindDir9am', 'N')),
        safe_dir(row.get('WindDir3pm', 'N')),
    )

    rain_today_bin = 1 if (row.get('RainToday', 'No') == 'Yes' or row.get('RainToday', 0) == 1) else 0
    raw_inputs = {
        'MinTemp':      float(row.get('MinTemp', 0)),
        'MaxTemp':      float(row.get('MaxTemp', 0)),
        'Rainfall':     float(row.get('Rainfall', 0)),
        'Evaporation':  float(row.get('Evaporation', 0)),
        'Sunshine':     float(row.get('Sunshine', 0)),
        'WindGustSpeed':float(row.get('WindGustSpeed', 0)),
        'WindSpeed9am': float(row.get('WindSpeed9am', 0)),
        'WindSpeed3pm': float(row.get('WindSpeed3pm', 0)),
        'Humidity9am':  float(row.get('Humidity9am', 0)),
        'Humidity3pm':  float(row.get('Humidity3pm', 0)),
        'Pressure9am':  float(row.get('Pressure9am', 0)),
        'Pressure3pm':  float(row.get('Pressure3pm', 0)),
        'Cloud9am':     float(row.get('Cloud9am', 0)),
        'Cloud3pm':     float(row.get('Cloud3pm', 0)),
        'Temp9am':      float(row.get('Temp9am', 0)),
        'Temp3pm':      float(row.get('Temp3pm', 0)),
        'RainToday':    float(rain_today_bin),
        'Year':         float(target_ts.year),
    }

    fvec = assemble_feature_vector(
        raw_inputs, temporal, cyclic,
        artifacts['feature_columns']
    )
    prob, binary = predict(fvec, artifacts['scaler'], artifacts['pca'], artifacts['model'])

    # ── Result panel ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Model Prediction")
    r1, r2, r3 = st.columns([1, 1, 1])

    with r1:
        st.plotly_chart(make_gauge(prob), use_container_width=True)

    with r2:
        css_cls = "result-rain" if binary == 1 else "result-norain"
        icon    = "🌧" if binary == 1 else "☀"
        label   = "Rain Tomorrow" if binary == 1 else "No Rain Tomorrow"
        st.markdown(f"""
        <div class="{css_cls}" style="margin-top:20px">
            <div class="result-label">{icon} {label}</div>
            <div class="result-sub">Probability: {prob*100:.1f}%</div>
            <div class="result-sub" style="margin-top:6px">{confidence_label(prob)}</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        # Ground truth
        actual = row.get('RainTomorrow', None)
        if actual is not None:
            actual_bin = 1 if actual in ['Yes', 1, 1.0] else 0
            correct = (binary == actual_bin)
            actual_str = "🌧 Rained" if actual_bin == 1 else "☀ No Rain"
            css_gt = "ground-truth-correct" if correct else "ground-truth-wrong"
            verdict = "✓ Correct prediction" if correct else "✗ Incorrect prediction"
            st.markdown(f"""
            <div style="margin-top:24px">
                <p style="font-size:0.85rem;color:#64748B;margin-bottom:6px">ACTUAL OUTCOME</p>
                <div class="{css_gt}">{actual_str}<br>{verdict}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Actual outcome not available for this date.")


# ── QUICK FORECAST MODE ──────────────────────────────────────────────────
elif mode == "⚡ Quick Forecast":
    st.header(f"📡 Live Forecast — {location}")
    st.caption(f"Month: {date(2000, selected_month, 1).strftime('%B')} · "
               f"Sliders pre-populated with {location} historical medians for this month")

    defaults = get_location_defaults(df, location, selected_month)
    month_name = date(2000, selected_month, 1).strftime('%B')

    with st.form("live_forecast_form"):
        col1, col2 = st.columns(2)

        with col1:
            hum_3pm = st.slider(
                "💧 Humidity this afternoon (%)",
                0, 100,
                int(defaults.get('Humidity3pm', 50)),
                help="How humid does it feel at 3pm? Higher = more likely to rain tomorrow.",
            )
            pres_3pm = st.slider(
                "📊 Pressure this afternoon (hPa)",
                970, 1040,
                int(defaults.get('Pressure3pm', 1012)),
                help="Check your weather app or barometer. Falling pressure = incoming weather.",
            )

        with col2:
            wind_dir = st.selectbox(
                "🌬 Wind direction",
                list(COMPASS_MAP.keys()),
                index=list(COMPASS_MAP.keys()).index(defaults.get('WindGustDir', 'N')),
                help="Which way is the wind mostly blowing? Onshore winds bring more rain.",
            )
            rain_today = st.toggle(
                "☔ Did it rain today?",
                value=False,
                help="Rain tends to persist for multiple days.",
            )

        st.caption("💡 Temperature, cloud cover, wind speed, and all other variables are auto-filled from historical averages for this location and month.")

        submitted = st.form_submit_button("🔮 Get Forecast", use_container_width=True, type="primary")

    if submitted:
        target_day = 15
        cyclic = compute_cyclic_features(
            selected_month, target_day,
            wind_dir, wind_dir, wind_dir,   # single direction for all wind columns
        )

        loc_df = get_location_data(df, location)
        month_rows = loc_df[loc_df['Date'].dt.month == selected_month]
        rain_median = float(month_rows['Rainfall'].median()) if 'Rainfall' in month_rows.columns else 0.0

        temporal = {}
        for feat, val in [
            ('Rainfall',    rain_median),
            ('Humidity3pm', float(hum_3pm)),
            ('Pressure3pm', float(pres_3pm)),
        ]:
            for lag in [1, 2, 3]:
                temporal[f'{feat}_lag{lag}'] = val
            for w in [3, 7, 14]:
                temporal[f'{feat}_SMA{w}'] = val
                temporal[f'{feat}_EMA{w}'] = val

        raw_inputs = {
            'MinTemp':       float(defaults.get('MinTemp', 15.0)),
            'MaxTemp':       float(defaults.get('MaxTemp', 25.0)),
            'Rainfall':      rain_median,
            'Evaporation':   float(defaults.get('Evaporation', 4.0)),
            'Sunshine':      float(defaults.get('Sunshine', 7.0)),
            'WindGustSpeed': float(defaults.get('WindGustSpeed', 35.0)),
            'WindSpeed9am':  float(defaults.get('WindSpeed9am', 15.0)),
            'WindSpeed3pm':  float(defaults.get('WindSpeed3pm', 20.0)),
            'Humidity9am':   float(defaults.get('Humidity9am', 65.0)),
            'Humidity3pm':   float(hum_3pm),
            'Pressure9am':   float(defaults.get('Pressure9am', 1015.0)),
            'Pressure3pm':   float(pres_3pm),
            'Cloud9am':      float(defaults.get('Cloud9am', 4.0)),
            'Cloud3pm':      float(defaults.get('Cloud3pm', 4.0)),
            'Temp9am':       float(defaults.get('Temp9am', 18.0)),
            'Temp3pm':       float(defaults.get('Temp3pm', 23.0)),
            'RainToday':     float(rain_today),
            'Year':          2024.0,
        }

        fvec = assemble_feature_vector(raw_inputs, temporal, cyclic, artifacts['feature_columns'])
        prob, binary = predict(fvec, artifacts['scaler'], artifacts['pca'], artifacts['model'])

        st.divider()
        fr1, fr2 = st.columns([1, 1])

        with fr1:
            st.plotly_chart(make_gauge(prob), use_container_width=True)

        with fr2:
            css_cls = "result-rain" if binary == 1 else "result-norain"
            icon    = "🌧" if binary == 1 else "☀"
            label   = "Rain Tomorrow" if binary == 1 else "No Rain Tomorrow"
            month_name = date(2000, selected_month, 1).strftime("%B")
            st.markdown(f"""
            <div class="{css_cls}" style="margin-top:20px">
                <div class="result-label">{icon} {label}</div>
                <div class="result-sub">Probability: {prob*100:.1f}%</div>
                <div class="result-sub" style="margin-top:6px">{confidence_label(prob)}</div>
                <div class="result-sub" style="margin-top:10px; font-size:0.8rem; opacity:0.7">
                    {location} · {month_name}
                </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 What was auto-filled from historical averages"):
            auto = {
                "Min / Max Temp":  f"{defaults.get('MinTemp', 0):.1f}°C / {defaults.get('MaxTemp', 0):.1f}°C",
                "Humidity 9am":    f"{defaults.get('Humidity9am', 0):.0f}%",
                "Pressure 9am":    f"{defaults.get('Pressure9am', 0):.1f} hPa",
                "Wind Speed":      f"Gust {defaults.get('WindGustSpeed', 0):.0f} km/h",
                "Cloud Cover":     f"9am {defaults.get('Cloud9am', 0):.0f}/8 · 3pm {defaults.get('Cloud3pm', 0):.0f}/8",
                "Sunshine":        f"{defaults.get('Sunshine', 0):.1f} hrs",
                "Rainfall (used for lags)": f"{rain_median:.1f} mm ({date(2000, selected_month, 1).strftime('%B')} median for {location})",
            }
            for k, v in auto.items():
                st.markdown(f"**{k}:** {v}")

# ── FULL LIVE FORECAST MODE ───────────────────────────────────────────────
else:
    st.header(f"🛠 Full Live Forecast — {location}")
    month_name_full = date(2000, selected_month, 1).strftime("%B")
    st.caption(f"{month_name_full} · All 16 inputs available · sliders pre-set to {location} historical medians")

    defaults_full = get_location_defaults(df, location, selected_month)

    with st.form("full_live_form"):
        st.subheader("🌡 Temperature")
        ft1, ft2, ft3, ft4 = st.columns(4)
        f_min_temp  = ft1.slider("Min Temp (°C)",  -10.0, 50.0, float(defaults_full.get("MinTemp",  15.0)), 0.5)
        f_max_temp  = ft2.slider("Max Temp (°C)",    0.0, 55.0, float(defaults_full.get("MaxTemp",  25.0)), 0.5)
        f_temp_9am  = ft3.slider("Temp 9am (°C)", -10.0, 50.0, float(defaults_full.get("Temp9am",  18.0)), 0.5)
        f_temp_3pm  = ft4.slider("Temp 3pm (°C)",   0.0, 55.0, float(defaults_full.get("Temp3pm",  23.0)), 0.5)

        st.subheader("💧 Moisture")
        fm1, fm2, fm3, fm4, fm5 = st.columns(5)
        f_hum_9am   = fm1.slider("Humidity 9am (%)",   0, 100, int(defaults_full.get("Humidity9am", 65)))
        f_hum_3pm   = fm2.slider("Humidity 3pm (%)",   0, 100, int(defaults_full.get("Humidity3pm", 50)))
        f_rainfall  = fm3.slider("Rainfall (mm)", 0.0, 100.0, float(defaults_full.get("Rainfall", 0.0)), 0.1)
        f_evap      = fm4.slider("Evaporation (mm)", 0.0, 30.0, float(defaults_full.get("Evaporation", 4.0)), 0.1)
        f_sunshine  = fm5.slider("Sunshine (hrs)", 0.0, 14.0, float(defaults_full.get("Sunshine", 7.0)), 0.1)

        st.subheader("📊 Pressure")
        fp1, fp2 = st.columns(2)
        f_pres_9am  = fp1.slider("Pressure 9am (hPa)", 970.0, 1040.0, float(defaults_full.get("Pressure9am", 1015.0)), 0.5)
        f_pres_3pm  = fp2.slider("Pressure 3pm (hPa)", 970.0, 1040.0, float(defaults_full.get("Pressure3pm", 1012.0)), 0.5)

        st.subheader("🌬 Wind")
        fw1, fw2, fw3 = st.columns(3)
        f_gust_dir  = fw1.selectbox("Gust Direction", list(COMPASS_MAP.keys()),
                        index=list(COMPASS_MAP.keys()).index(defaults_full.get("WindGustDir", "N")))
        f_dir_9am   = fw2.selectbox("Direction 9am",  list(COMPASS_MAP.keys()),
                        index=list(COMPASS_MAP.keys()).index(defaults_full.get("WindDir9am",  "N")))
        f_dir_3pm   = fw3.selectbox("Direction 3pm",  list(COMPASS_MAP.keys()),
                        index=list(COMPASS_MAP.keys()).index(defaults_full.get("WindDir3pm",  "N")))
        fw4, fw5, fw6 = st.columns(3)
        f_gust_spd  = fw4.slider("Gust Speed (km/h)", 0, 150, int(defaults_full.get("WindGustSpeed", 35)))
        f_spd_9am   = fw5.slider("Wind Speed 9am",    0, 100, int(defaults_full.get("WindSpeed9am",  15)))
        f_spd_3pm   = fw6.slider("Wind Speed 3pm",    0, 100, int(defaults_full.get("WindSpeed3pm",  20)))

        st.subheader("⛅ Sky")
        fc1, fc2 = st.columns(2)
        f_cloud_9am = fc1.slider("Cloud 9am (oktas)", 0, 8, int(defaults_full.get("Cloud9am", 4)))
        f_cloud_3pm = fc2.slider("Cloud 3pm (oktas)", 0, 8, int(defaults_full.get("Cloud3pm", 4)))

        f_rain_today = st.toggle("☔ Did it rain today?", value=False)

        f_submitted = st.form_submit_button("🔮 Get Forecast", use_container_width=True, type="primary")

    if f_submitted:
        f_cyclic = compute_cyclic_features(selected_month, 15, f_gust_dir, f_dir_9am, f_dir_3pm)

        loc_df_f = get_location_data(df, location)
        month_rows_f = loc_df_f[loc_df_f["Date"].dt.month == selected_month]

        f_temporal = {}
        for feat, val in [
            ("Rainfall",    float(f_rainfall)),
            ("Humidity3pm", float(f_hum_3pm)),
            ("Pressure3pm", float(f_pres_3pm)),
        ]:
            for lag in [1, 2, 3]:
                f_temporal[f"{feat}_lag{lag}"] = val
            for w in [3, 7, 14]:
                f_temporal[f"{feat}_SMA{w}"] = val
                f_temporal[f"{feat}_EMA{w}"] = val

        f_raw = {
            "MinTemp": f_min_temp, "MaxTemp": f_max_temp,
            "Rainfall": float(f_rainfall), "Evaporation": float(f_evap),
            "Sunshine": float(f_sunshine),
            "WindGustSpeed": float(f_gust_spd),
            "WindSpeed9am": float(f_spd_9am), "WindSpeed3pm": float(f_spd_3pm),
            "Humidity9am": float(f_hum_9am), "Humidity3pm": float(f_hum_3pm),
            "Pressure9am": float(f_pres_9am), "Pressure3pm": float(f_pres_3pm),
            "Cloud9am": float(f_cloud_9am), "Cloud3pm": float(f_cloud_3pm),
            "Temp9am": float(f_temp_9am), "Temp3pm": float(f_temp_3pm),
            "RainToday": float(f_rain_today), "Year": 2024.0,
        }

        f_fvec = assemble_feature_vector(f_raw, f_temporal, f_cyclic, artifacts["feature_columns"])
        f_prob, f_binary = predict(f_fvec, artifacts["scaler"], artifacts["pca"], artifacts["model"])

        st.divider()
        ff1, ff2 = st.columns([1, 1])
        with ff1:
            st.plotly_chart(make_gauge(f_prob), use_container_width=True)
        with ff2:
            css = "result-rain" if f_binary else "result-norain"
            icon = "🌧" if f_binary else "☀"
            label = "Rain Tomorrow" if f_binary else "No Rain Tomorrow"
            st.markdown(f"""
            <div class="{css}" style="margin-top:20px">
                <div class="result-label">{icon} {label}</div>
                <div class="result-sub">Probability: {f_prob*100:.1f}%</div>
                <div class="result-sub" style="margin-top:6px">{confidence_label(f_prob)}</div>
                <div class="result-sub" style="margin-top:10px;font-size:0.8rem;opacity:0.7">
                    {location} · {month_name_full} · all inputs user-specified
                </div>
            </div>""", unsafe_allow_html=True)
