"""
Model Utilities - XGBoost Training dan Forecasting
Ekstrak dari notebook
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

from config import (
    LOOK_BACK, FORECAST_STEPS, LAG_STEPS, XGBOOST_PARAMS,
    TRAIN_RATIO, VAL_RATIO
)


def create_cyclical_features(df, target_col):
    """
    Buat fitur siklus jam dan lag untuk XGBoost
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame dengan datetime index
    target_col : str
        Nama kolom target (PM10 atau PM2.5)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame dengan fitur yang sudah ditambahkan, NaN rows dihapus
    """
    data = df[[target_col]].copy()
    
    # Fitur Siklus Jam (Penting agar model mengerti pola daily)
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['dayofweek'] = data.index.dayofweek
    
    # Fitur Lag (Gunakan lag strategis: 30 min, 1 hour, 24 hours)
    for lag in LAG_STEPS:
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    
    # Drop rows dengan NaN (dari lag features)
    return data.dropna()


def train_xgboost_model(df, target_col):
    """
    Train XGBoost model pada data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame dengan datetime index
    target_col : str
        Nama kolom target (PM10 atau PM2.5)
    
    Returns:
    --------
    dict : {
        'model': trained XGBoost model,
        'metrics': {'rmse': float, 'mape': float},
        'y_test': actual test values,
        'y_pred': predicted test values,
        'df_model': feature engineered dataframe
    }
    """
    
    # 1. Feature Engineering
    df_model = create_cyclical_features(df, target_col)
    
    # 2. Split data 70:20:10 (train:val:test)
    n = len(df_model)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train = df_model.iloc[:train_end]
    val = df_model.iloc[train_end:val_end]
    test = df_model.iloc[val_end:]
    
    X_train, y_train = train.drop(columns=[target_col]), train[target_col]
    X_val, y_val = val.drop(columns=[target_col]), val[target_col]
    X_test, y_test = test.drop(columns=[target_col]), test[target_col]
    
    # 3. Train XGBoost
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 4. Predict on test set
    y_pred = model.predict(X_test)
    
    # 5. Hitung metrik
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return {
        'model': model,
        'metrics': {'rmse': rmse, 'mape': mape},
        'y_test': y_test,
        'y_pred': y_pred,
        'df_model': df_model,
        'feature_names': X_train.columns.tolist()
    }


def forecast_recursive(model, df_model, target_col, steps=FORECAST_STEPS):
    """
    Prediksi recursive untuk masa depan menggunakan trained XGBoost model
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained XGBoost model
    df_model : pd.DataFrame
        DataFrame dengan fitur yang sudah disiapkan
    target_col : str
        Nama kolom target
    steps : int
        Jumlah step untuk diprediksi (default 336 = 7 hari)
    
    Returns:
    --------
    tuple : (forecast_values, forecast_dates)
        forecast_values: np.array prediksi
        forecast_dates: pd.DatetimeIndex untuk plotting
    """
    
    future_preds = []
    current_series = df_model[target_col].tolist()
    last_time = df_model.index[-1]
    
    for i in range(steps):
        next_time = last_time + pd.Timedelta(minutes=30)
        
        # Construct features untuk prediksi
        feat_row = np.array([
            np.sin(2 * np.pi * next_time.hour / 24),     # hour_sin
            np.cos(2 * np.pi * next_time.hour / 24),     # hour_cos
            next_time.dayofweek,                         # dayofweek
            current_series[-1],                          # lag_1
            current_series[-2],                          # lag_2
            current_series[-48]                          # lag_48 (24 jam sebelumnya)
        ]).reshape(1, -1)
        
        # Predict
        pred = model.predict(feat_row)[0]
        future_preds.append(pred)
        current_series.append(pred)  # Add untuk lag berikutnya
        
        last_time = next_time
    
    # Generate datetime index untuk prediksi
    forecast_start = df_model.index[-1] + pd.Timedelta(minutes=30)
    forecast_dates = pd.date_range(
        start=forecast_start,
        periods=steps,
        freq='30T'
    )
    
    return np.array(future_preds), forecast_dates


def generate_forecast_dataframe(model, df, target_col, steps=FORECAST_STEPS):
    """
    Convenience function untuk generate forecast dan return sebagai DataFrame
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained model
    df : pd.DataFrame
        Original data dengan datetime index
    target_col : str
        Nama kolom target
    steps : int
        Jumlah step untuk diprediksi
    
    Returns:
    --------
    pd.DataFrame
        Forecast dengan index datetime
    """
    
    df_model = create_cyclical_features(df, target_col)
    forecast_values, forecast_dates = forecast_recursive(
        model, df_model, target_col, steps
    )
    
    forecast_df = pd.DataFrame({
        target_col: forecast_values
    }, index=forecast_dates)
    
    return forecast_df


def train_and_forecast(df, target_col):
    """
    High-level function: train model dan langsung generate forecast
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data dengan datetime index dan kolom target_col
    target_col : str
        Nama kolom target (PM10 atau PM2.5)
    
    Returns:
    --------
    dict : {
        'model': trained model,
        'metrics': evaluation metrics,
        'forecast_df': DataFrame prediksi 7 hari,
        'actual_test': actual test values,
        'pred_test': predicted test values
    }
    """
    
    # Train
    training_result = train_xgboost_model(df, target_col)
    model = training_result['model']
    df_model = training_result['df_model']
    
    # Forecast
    forecast_df = generate_forecast_dataframe(model, df, target_col)
    
    return {
        'model': model,
        'metrics': training_result['metrics'],
        'forecast_df': forecast_df,
        'actual_test': training_result['y_test'],
        'pred_test': training_result['y_pred'],
        'df_model': df_model
    }
