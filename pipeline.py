"""
pipeline.py — Feature engineering and inference pipeline.
Reproduces the notebook training pipeline exactly.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# ── Compass map (must match notebook cell 2 exactly) ─────────────────────
COMPASS_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}
ALPHABETICAL_COMPASS = sorted(COMPASS_MAP.keys())
CODE_TO_COMPASS = {i: d for i, d in enumerate(ALPHABETICAL_COMPASS)}

TEMPORAL_FEATURES = ['Rainfall', 'Humidity3pm', 'Pressure3pm']
WINDOWS = [3, 7, 14]


def load_artifacts():
    """Load all deployment artifacts from models/ directory."""
    return {
        'model':           joblib.load(MODELS_DIR / 'model.pkl'),
        'scaler':          joblib.load(MODELS_DIR / 'scaler.pkl'),
        'pca':             joblib.load(MODELS_DIR / 'pca.pkl'),
        'feature_columns': joblib.load(MODELS_DIR / 'feature_columns.pkl'),
        'scale_pos':       joblib.load(MODELS_DIR / 'scale_pos.pkl'),
    }


def get_location_data(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Return the chronologically sorted rows for one location."""
    loc = df[df['Location'] == location].copy()
    loc['Date'] = pd.to_datetime(loc['Date'])
    return loc.sort_values('Date').reset_index(drop=True)


def compute_temporal_features(
    location_series: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict:
    """
    Given a location's full time-series and a target date, compute the
    27 temporal features (9 lag + 9 SMA + 9 EMA).

    Lag and rolling use shift(1) before the window — exactly as in the
    notebook — so we look at days *before* the target date only.
    """
    # Include only rows strictly before the target date
    history = location_series[location_series['Date'] < target_date].copy()
    history = history.sort_values('Date').reset_index(drop=True)

    temporal = {}

    for feature in TEMPORAL_FEATURES:
        if feature not in history.columns:
            # Fill with zeros if missing
            for lag in [1, 2, 3]:
                temporal[f'{feature}_lag{lag}'] = 0.0
            for w in WINDOWS:
                temporal[f'{feature}_SMA{w}'] = 0.0
                temporal[f'{feature}_EMA{w}'] = 0.0
            continue

        series = history[feature].astype(float)

        # Lag features: value i days before target
        for lag in [1, 2, 3]:
            temporal[f'{feature}_lag{lag}'] = (
                float(series.iloc[-lag]) if len(series) >= lag else 0.0
            )

        # SMA: rolling mean of (n days before target), shift(1) already handled
        # by only using history (no target row included)
        for w in WINDOWS:
            window = series.tail(w)
            temporal[f'{feature}_SMA{w}'] = float(window.mean()) if len(window) > 0 else 0.0

        # EMA: ewm with span=w, adjust=False (matches notebook)
        for w in WINDOWS:
            if len(series) > 0:
                ema_val = series.ewm(span=w, adjust=False).mean().iloc[-1]
                temporal[f'{feature}_EMA{w}'] = float(ema_val)
            else:
                temporal[f'{feature}_EMA{w}'] = 0.0

    return temporal


def compute_cyclic_features(
    month: int,
    day: int,
    wind_gust_dir: str,
    wind_dir_9am: str,
    wind_dir_3pm: str,
) -> dict:
    """
    Compute the 10 cyclic features.
    Month: /12, Day: /31 (matches notebook comment exactly).
    Wind directions: degrees / 360.
    """
    cyc = {}
    # Month
    cyc['Month_sin'] = np.sin(2 * np.pi * month / 12)
    cyc['Month_cos'] = np.cos(2 * np.pi * month / 12)
    # Day (notebook uses /31 — "approximation")
    cyc['Day_sin'] = np.sin(2 * np.pi * day / 31)
    cyc['Day_cos'] = np.cos(2 * np.pi * day / 31)
    # Wind directions
    for col, val in [
        ('WindGustDir', wind_gust_dir),
        ('WindDir9am',  wind_dir_9am),
        ('WindDir3pm',  wind_dir_3pm),
    ]:
        deg = COMPASS_MAP.get(val, 0.0)
        cyc[f'{col}_sin'] = np.sin(2 * np.pi * deg / 360)
        cyc[f'{col}_cos'] = np.cos(2 * np.pi * deg / 360)
    return cyc


def assemble_feature_vector(
    raw_inputs: dict,
    temporal_features: dict,
    cyclic_features: dict,
    feature_columns: list,
) -> np.ndarray:
    """
    Combine raw_inputs, temporal_features, and cyclic_features into a
    single 55-element array in the exact column order of feature_columns.
    NaN values (from raw CSV rows) are replaced with 0.0.
    """
    combined = {**raw_inputs, **temporal_features, **cyclic_features}
    vector = np.array(
        [combined.get(col, 0.0) for col in feature_columns],
        dtype=float
    )
    # Replace any NaN that leaked in from raw CSV data
    vector = np.nan_to_num(vector, nan=0.0)
    return vector.reshape(1, -1)


def predict(
    feature_vector: np.ndarray,
    scaler,
    pca,
    model,
) -> tuple[float, int]:
    """
    Apply scale → PCA → predict_proba.
    Returns (probability_of_rain, binary_prediction).
    """
    # Guard against NaN before scaling
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    scaled = scaler.transform(feature_vector)
    projected = pca.transform(scaled)
    prob = float(model.predict_proba(projected)[0, 1])
    binary = int(prob >= 0.5)
    return prob, binary


def get_location_defaults(
    df: pd.DataFrame,
    location: str,
    month: int,
) -> dict:
    """
    Compute median weather values for a location+month combination.
    Used to pre-populate sliders in Live Forecast mode.
    """
    loc_month = df[(df['Location'] == location) & (
        pd.to_datetime(df['Date']).dt.month == month
    )]
    if len(loc_month) == 0:
        loc_month = df[df['Location'] == location]

    numeric_cols = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
    ]
    defaults = {}
    for col in numeric_cols:
        if col in loc_month.columns:
            defaults[col] = float(loc_month[col].median())
        else:
            defaults[col] = 0.0

    # Most common wind direction for this location/month
    for wind_col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if wind_col in loc_month.columns:
            mode_val = loc_month[wind_col].mode()
            defaults[wind_col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else 'N'
        else:
            defaults[wind_col] = 'N'

    defaults['RainToday'] = int(
        loc_month['RainToday'].map({'Yes': 1, 'No': 0}).median() >= 0.5
    ) if 'RainToday' in loc_month.columns else 0

    return defaults
