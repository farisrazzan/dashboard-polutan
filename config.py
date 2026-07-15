"""
Konfigurasi Global Dashboard Streamlit
"""
import os
from pathlib import Path

# ========== PATH CONFIGURATION ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "data_udara_clean(update).csv"

# ========== MODEL CONFIGURATION ==========
LOOK_BACK = 48  # 48 step = 24 jam (30 menit per step)
FORECAST_STEPS = 336  # 7 hari x 48 step/hari

# XGBoost Parameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.08,
    'max_depth': 8,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42
}

# Train-Validation-Test Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
# Test ratio = 0.10 (calculated from remainder)

# ========== COLUMN NAMES ==========
TIME_COLUMN = 'waktu'
PM10_COLUMN = 'PM10'
PM25_COLUMN = 'PM2.5'
POLLUTANTS = [PM10_COLUMN, PM25_COLUMN]

# ========== FEATURE ENGINEER CONFIGURATION ==========
LAG_STEPS = [1, 2, 48]  # lag 30 min, 1 hour, 24 hours

# ========== DISPLAY CONFIGURATION ==========
# Untuk aggregation per hari (rata-rata)
FREQ_30MIN = '30T'
FREQ_DAILY = 'D'
RECORDS_PER_DAY = 48  # 24 jam / 30 menit

# Color scheme
COLOR_PM10 = '#FF6B6B'      # Red
COLOR_PM25 = '#4ECDC4'      # Teal
COLOR_PREDICTION = '#FFD93D' # Yellow

# ========== STREAMLIT CONFIG ==========
APP_TITLE = "Dashboard Polutan PM - Kota Samarinda"
MENU_DASHBOARD = "📊 Dashboard"
MENU_DATA = "💾 Data"

# ========== COLORS ==========
COLOR_PM10 = "#FF6B6B"  # Red
COLOR_PM25 = "#4ECDC4"  # Teal
COLOR_PREDICTION = "#FFD93D"  # Yellow
