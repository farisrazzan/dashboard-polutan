"""
Data Manager untuk Load, Process, dan CRUD data CSV
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from config import (
    DATA_FILE, TIME_COLUMN, PM10_COLUMN, PM25_COLUMN, 
    FREQ_30MIN, FREQ_DAILY, RECORDS_PER_DAY
)


class DataManager:
    """Mengelola data CSV time series"""
    
    def __init__(self, filepath=DATA_FILE):
        self.filepath = filepath
        self._df = None
        self._df_aggregated = None  # Cache untuk data aggregated harian
    
    @property
    def df(self):
        """Lazy load dataframe"""
        if self._df is None:
            self.load_data()
        return self._df
    
    def load_data(self):
        """Load CSV dan set datetime index"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {self.filepath}")
        
        df = pd.read_csv(self.filepath)
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
        df = df.set_index(TIME_COLUMN)
        df = df.sort_index()
        
        self._df = df
        self._df_aggregated = None  # Reset cache
        return df
    
    def get_data_raw(self):
        """Return raw dataframe"""
        return self.df.copy()
    
    def get_data_aggregated(self, freq='D'):
        """
        Aggregate data dengan rata-rata per period
        freq: 'D' untuk daily, H untuk hourly, dll
        """
        if freq == FREQ_DAILY and self._df_aggregated is not None:
            return self._df_aggregated
        
        df_agg = self.df.resample(freq).mean()
        
        if freq == FREQ_DAILY:
            self._df_aggregated = df_agg
        
        return df_agg
    
    def get_last_n_days_aggregated(self, n_days=7):
        """Return data aggregated per hari untuk n hari terakhir"""
        df_agg = self.get_data_aggregated(FREQ_DAILY)
        return df_agg.tail(n_days)
    
    def get_last_n_rows(self, n=None):
        """Return n baris terakhir"""
        if n is None:
            n = RECORDS_PER_DAY * 7  # Default 7 hari
        return self.df.tail(n)
    
    def get_statistics(self):
        """Return statistik overview data"""
        df = self.df
        
        # Pengecekan jika DataFrame kosong
        if len(df) == 0:
            return {
                'total_records': 0,
                'latest_date': pd.NaT,
                'earliest_date': pd.NaT,
                'duration_days': 0,
                'pm10_mean': 0,
                'pm10_max': 0,
                'pm10_max_date': pd.NaT,
                'pm10_min': 0,
                'pm10_min_date': pd.NaT,
                'pm25_mean': 0,
                'pm25_max': 0,
                'pm25_max_date': pd.NaT,
                'pm25_min': 0,
                'pm25_min_date': pd.NaT,
            }
        
        stats = {
            'total_records': len(df),
            'latest_date': df.index[-1],
            'earliest_date': df.index[0],
            'duration_days': (df.index[-1] - df.index[0]).days,
            'pm10_mean': df[PM10_COLUMN].mean(),
            'pm10_max': df[PM10_COLUMN].max(),
            'pm10_max_date': df[PM10_COLUMN].idxmax(),
            'pm10_min': df[PM10_COLUMN].min(),
            'pm10_min_date': df[PM10_COLUMN].idxmin(),
            'pm25_mean': df[PM25_COLUMN].mean(),
            'pm25_max': df[PM25_COLUMN].max(),
            'pm25_max_date': df[PM25_COLUMN].idxmax(),
            'pm25_min': df[PM25_COLUMN].min(),
            'pm25_min_date': df[PM25_COLUMN].idxmin(),
        }
        
        return stats
    
    def save_data(self):
        """Simpan current dataframe ke file CSV"""
        if self._df is None or len(self._df) == 0:
            raise ValueError("Tidak ada data untuk disimpan")
        
        # Pastikan direktori ada
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df_to_save = self._df.reset_index()
        df_to_save[TIME_COLUMN] = df_to_save[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save.to_csv(self.filepath, index=False)
        return True
    
    def replace_data(self, new_df):
        """
        Replace semua data dengan data baru (untuk mengatasi data rusak)
        new_df: DataFrame dengan index datetime dan kolom PM10, PM2.5
        """
        if new_df is None or len(new_df) == 0:
            raise ValueError("Data baru kosong")
        
        # Pastikan direktori ada
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Update internal dataframe
        self._df = new_df.copy()
        self._df_aggregated = None
        
        # Save ke file (replace/overwrite)
        df_to_save = self._df.reset_index()
        df_to_save[TIME_COLUMN] = df_to_save[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save.to_csv(self.filepath, index=False)
        
        return True
    
    def add_forecast_data(self, forecast_df):
        """
        Tambah data forecast ke file CSV (untuk 7 hari ke depan)
        forecast_df: DataFrame dengan index datetime dan kolom PM10, PM2.5
        """
        # Append forecast ke existing data
        df_combined = pd.concat([self.df, forecast_df])
        
        # Sort by index
        df_combined = df_combined.sort_index()
        
        # Remove duplicates (jika ada)
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        # Save kembali
        df_to_save = df_combined.reset_index()
        df_to_save[TIME_COLUMN] = df_to_save[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save.to_csv(self.filepath, index=False)
        
        # Update internal dataframe
        self._df = df_combined
        self._df_aggregated = None
        
        return df_combined
    
    def remove_last_week(self):
        """Hapus data 1 minggu paling lama (untuk menjaga data tetap recent)"""
        n_to_remove = RECORDS_PER_DAY * 7
        df_trimmed = self.df.iloc[n_to_remove:]
        
        # Save
        df_to_save = df_trimmed.reset_index()
        df_to_save[TIME_COLUMN] = df_to_save[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save.to_csv(self.filepath, index=False)
        
        # Update
        self._df = df_trimmed
        self._df_aggregated = None
        
        return df_trimmed
    
    def reset_to_latest_backup(self):
        """Load ulang data dari file (discard perubahan di memory)"""
        self._df = None
        self._df_aggregated = None
        return self.load_data()
    
    def validate_datetime_format(self, df, time_col):
        """
        Validasi format datetime dari DataFrame
        Return: (is_valid, error_message, parsed_df)
        """
        try:
            df_temp = df.copy()
            df_temp[time_col] = pd.to_datetime(df_temp[time_col])
            return (True, None, df_temp)
        except Exception as e:
            return (False, f"Tidak bisa parse kolom waktu: {str(e)}", None)
    
    def check_missing_values(self, df):
        """
        Periksa missing values di PM10 dan PM2.5
        Return: {column: missing_count, ...}
        """
        return {col: df[col].isna().sum() for col in [PM10_COLUMN, PM25_COLUMN] if col in df.columns}
    
    def detect_gaps(self, df):
        """
        Deteksi gap/loncat dalam time series (data yang tidak kontinyu)
        Ekspektasi: 30 menit interval untuk setiap data
        Return: list of gaps {
            'start': timestamp,
            'end': timestamp,
            'duration_hours': float,
            'missing_records': int
        }
        """
        if len(df) < 2:
            return []
        
        gaps = []
        df_sorted = df.sort_index()
        time_index = df_sorted.index
        
        expected_delta = pd.Timedelta(minutes=30)
        
        for i in range(len(time_index) - 1):
            actual_delta = time_index[i + 1] - time_index[i]
            
            if actual_delta > expected_delta:
                gap_hours = actual_delta.total_seconds() / 3600
                missing_records = int((actual_delta.total_seconds() / 1800) - 1)  # 30 min = 1800 sec
                
                gaps.append({
                    'start': time_index[i],
                    'end': time_index[i + 1],
                    'duration_hours': round(gap_hours, 2),
                    'missing_records': missing_records
                })
        
        return gaps
    
    def validate_data(self, df, time_col):
        """
        Validasi komprehensif untuk data baru
        Return: {
            'is_valid': bool,
            'errors': [list of fatal errors],
            'warnings': [list of warnings],
            'missing_values': dict,
            'gaps': list,
            'validated_df': df with parsed datetime
        }
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_values': {},
            'gaps': [],
            'validated_df': None
        }
        
        # Check required columns
        required_cols = [time_col, PM10_COLUMN, PM25_COLUMN]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            result['errors'].append(f"Kolom tidak ditemukan: {', '.join(missing_cols)}")
            result['is_valid'] = False
            return result
        
        # Validate datetime format
        is_datetime_valid, datetime_error, df_parsed = self.validate_datetime_format(df, time_col)
        if not is_datetime_valid:
            result['errors'].append(datetime_error)
            result['is_valid'] = False
            return result
        
        df_parsed = df_parsed.set_index(time_col)
        df_parsed = df_parsed.sort_index()
        
        # Check missing values
        missing_values = self.check_missing_values(df_parsed)
        result['missing_values'] = missing_values
        
        for col, count in missing_values.items():
            if count > 0:
                result['warnings'].append(f"Missing values di {col}: {count} baris")
        
        # Detect gaps
        gaps = self.detect_gaps(df_parsed)
        result['gaps'] = gaps
        
        if gaps:
            total_missing = sum(gap['missing_records'] for gap in gaps)
            result['warnings'].append(f"Ditemukan {len(gaps)} gap dalam data (total {total_missing} record yang hilang)")
        
        result['validated_df'] = df_parsed
        
        return result
    
    def append_data(self, new_df):
        """
        Tambah data baru ke existing data
        new_df: DataFrame dengan index datetime dan kolom PM10, PM2.5
        Return: combined DataFrame yang sudah di-sort
        """
        if new_df is None or len(new_df) == 0:
            raise ValueError("Data baru kosong")
        
        # Combine
        df_combined = pd.concat([self.df, new_df])
        
        # Sort by index
        df_combined = df_combined.sort_index()
        
        # Remove duplicates (keep first)
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        # Update internal dataframe
        self._df = df_combined
        self._df_aggregated = None
        
        # Save ke file
        df_to_save = self._df.reset_index()
        df_to_save[TIME_COLUMN] = df_to_save[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save.to_csv(self.filepath, index=False)
        
        return df_combined
    
    def clean_data(self, df, method='interpolate'):
        """
        Bersihkan data dengan method tertentu
        - interpolate: Linear interpolation dengan method='time'
        - fill_gaps: Isi gap dengan baris baru, kemudian interpolate
        
        Return: cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Method: Fill gaps dulu sebelum interpolate
        if method == 'fill_gaps':
            # Reindex dengan frekuensi 30 menit untuk mengisi gaps
            df_clean = df_clean.asfreq('30T')
        
        # Interpolate missing values dengan method time-based linear
        for col in [PM10_COLUMN, PM25_COLUMN]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].interpolate(method='time', limit_direction='both')
        
        # Jika masih ada NaN (di awal/akhir), gunakan forward fill lalu backward fill
        df_clean = df_clean.ffill().bfill()
        
        return df_clean


# ========== HELPER FUNCTIONS ==========

def load_data(filepath=DATA_FILE):
    """Helper function untuk load data sederhana"""
    manager = DataManager(filepath)
    return manager.get_data_raw()


def get_data_7_days_aggregated(filepath=DATA_FILE):
    """Helper untuk mendapat data 7 hari terakhir (aggregated per hari)"""
    manager = DataManager(filepath)
    return manager.get_last_n_days_aggregated(7)


def get_statistics(filepath=DATA_FILE):
    """Helper untuk mendapat statistik data"""
    manager = DataManager(filepath)
    return manager.get_statistics()
