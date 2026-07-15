# 🌍 Dashboard Polutan PM - Samarinda

Web dashboard sederhana untuk monitoring kualitas udara (Polutan PM10 dan PM2.5) di Kota Samarinda berbasis Streamlit.

## 📁 Struktur Folder

```
d:\Kuliah\TA\code\
├── app.py                              # File utama Streamlit
├── config.py                            # Konfigurasi global (path, konstanta, parameter)
├── data_manager.py                      # Class untuk manage data CSV (load, CRUD, aggregation)
├── model_utils.py                       # Functions untuk XGBoost training & forecasting
├── requirements.txt                     # Daftar dependencies Python
├── data/
│   └── data_udara_clean(update).csv    # File data time series (waktu, PM10, PM2.5)
└── README.md                            # Dokumentasi (file ini)
```

## 🚀 Instalasi & Setup

### Prerequisites
- Python 3.8+ (sudah terinstall)
- pip (Python package manager)

### Step 1: Install Dependencies

Buka terminal/PowerShell di folder `d:\Kuliah\TA\code\` dan jalankan:

```bash
pip install -r requirements.txt
```

Packages yang akan diinstall:
- `streamlit` - Framework untuk web dashboard
- `pandas`, `numpy` - Data processing
- `xgboost` - Machine learning model untuk forecasting
- `scikit-learn` - Tools untuk machine learning
- `plotly` - Interactive charts
- `matplotlib` - Static visualization

### Step 2: Jalankan Application

Masih di folder yang sama, jalankan command:

```bash
streamlit run app.py
```

Setelah itu:
1. Browser otomatis akan membuka di **http://localhost:8501**
2. Anda akan melihat Dashboard Polutan PM

## 📊 Fitur Dashboard

### 🎯 Header & Navigation
- **Header Tetap** - Judul dashboard yang selalu ada di semua halaman
- **Navigation Buttons** - 2 tombol untuk berpindah menu:
  - 📊 **Dashboard** - View data aktual & prediksi
  - 💾 **Data** - View tabel lengkap & manage data

---

### 📊 Menu "Dashboard"

#### Bagian 1: Score Cards & Preview
- **Score Card Kiri** - Menampilkan jumlah total data yang dimiliki
- **Score Card Tengah** - Menampilkan tanggal dan waktu data terbaru
- **Score Card Kanan** - Tabel preview dengan 2 tabs:
  - Tab "Data Terakhir" - 5 baris data terakhir
  - Tab "Statistik" - Min, Max, Rata-rata untuk PM10 dan PM2.5

#### Bagian 2: Chart Aktual 7 Hari
- **Line Chart** menampilkan data PM10 7 hari terakhir (rata-rata per hari)
- **Dropdown Menu** untuk switch antara PM10 → PM2.5
- **Interactive Chart** - bisa di-pan, di-zoom, hover untuk detail

#### Bagian 3: Chart Prediksi 7 Hari
- **Line Chart** menampilkan prediksi PM10 7 hari ke depan (menggunakan XGBoost)
- **Dropdown Menu** untuk switch antara PM10 → PM2.5
- **Interactive Chart** - sama seperti chart aktual

---

### 💾 Menu "Data"

#### Bagian 1: Data Table
- **Table Lengkap** menampilkan semua data (waktu, PM10, PM2.5)
- **Dropdown "Tampilkan Data"** untuk filter:
  - Semua - Tampilkan semua kolom
  - PM10 - Hanya kolom PM10
  - PM2.5 - Hanya kolom PM2.5
  - Perbandingan - Tampilkan semua untuk perbandingan
- **Scrollable** - Bisa di-scroll horizontal & vertikal

#### Bagian 2: Data Management Buttons
- **🔄 Refresh Data** - Reload data dari file CSV (discard semua perubahan di memory)
- **➕ Tambah Data Minggu Depan** - Generate prediksi 7 hari ke depan dan append ke CSV
  - Setelah menambah data 7 hari, otomatis **hapus data 1 minggu paling lama**
  - Ini menjaga data tetap "recent" dan ukuran file manageable

---

## 🔧 Backend Processing

Semua processing data menggunakan kode dari notebook `Coba coba stress.ipynb` bagian **XGBoost**:

### 1. **Data Loading** (`data_manager.py`)
```python
# Load CSV dengan datetime index
df = DataManager().get_data_raw()
```

### 2. **Feature Engineering** (`model_utils.py`)
```python
# Create cyclical features (hourly pattern, day of week, lags)
df_features = create_cyclical_features(df, 'PM10')
```

### 3. **Model Training** (`model_utils.py`)
```python
# Train XGBoost dengan 70:20:10 split
result = train_and_forecast(df, 'PM10')
```

### 4. **Forecasting** (`model_utils.py`)
```python
# Recursive prediction untuk 7 hari ke depan (336 titik x 30 menit)
forecast_df = result['forecast_df']
```

---

## 📝 Penjelasan Parameter (di `config.py`)

```python
LOOK_BACK = 48              # Sliding window = 24 jam (48 x 30 menit)
FORECAST_STEPS = 336        # Prediksi 7 hari = 336 step (7 x 48)

LAG_STEPS = [1, 2, 48]      # Features lag: 30min, 1jam, 24jam sebelumnya

XGBOOST_PARAMS = {
    'n_estimators': 500,    # Jumlah trees
    'learning_rate': 0.08,  # Learning rate untuk tidak oversmooth
    'max_depth': 8,         # Kedalaman tree untuk capture variansi
}

TRAIN_RATIO = 0.70         # 70% data untuk training
VAL_RATIO = 0.20           # 20% data untuk validation
# Test = sisanya (10%)
```

---

## 📊 Data Format

File CSV harus memiliki 3 kolom:

| waktu | PM10 | PM2.5 |
|-------|------|-------|
| 2024-01-01 00:00:00 | 45.2 | 22.5 |
| 2024-01-01 00:30:00 | 46.1 | 23.1 |
| 2024-01-01 01:00:00 | 44.8 | 21.9 |

- **waktu** - Format ISO datetime (YYYY-MM-DD HH:MM:SS)
- **PM10** - Floating point value (µg/m³)
- **PM2.5** - Floating point value (µg/m³)
- **Interval** - 30 menit antar baris

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solusi:** Install dependencies dulu
```bash
pip install -r requirements.txt
```

### Error: "File tidak ditemukan: data/data_udara_clean(update).csv"
**Solusi:** Pastikan file CSV berada di folder `d:\Kuliah\TA\code\data\`

### Error: "LSTM model not found" atau XGBoost error saat forecasting
**Solusi:** Data CSV harus memiliki minimal 100+ baris agar model bisa train dengan baik. Jika data kurang, tambahkan data dummy atau gunakan data yang lebih panjang.

### Dashboard loading lambat
**Solusi:** Forecasting berjalan sekali per session. Gunakan Streamlit cache dengan `@st.cache_data` untuk optimasi lebih lanjut.

---

## 📚 File Reference

### [config.py](config.py)
- Global configuration, paths, constants
- XGBoost parameters
- Feature engineering settings

### [data_manager.py](data_manager.py)
- `DataManager` class untuk manage CSV
- Methods: load, aggregate, get_statistics, add_forecast_data, remove_last_week
- Helper functions untuk convenience

### [model_utils.py](model_utils.py)
- `create_cyclical_features()` - Feature engineering
- `train_xgboost_model()` - Train model
- `forecast_recursive()` - Generate predictions
- `train_and_forecast()` - High-level wrapper

### [app.py](app.py)
- Main Streamlit application
- UI Components: header, navigation, charts, tables
- Session state management

---

## 🎓 Educational Notes

Dashboard ini mendemonstrasikan:
1. **Time Series Forecasting** - XGBoost untuk data deret waktu
2. **Feature Engineering** - Cyclical features (hour_sin, hour_cos), lag features
3. **Web Dashboard** - Streamlit untuk quick UI tanpa frontend complexity
4. **Data Management** - CSV CRUD operations dengan Pandas
5. **Interactive Visualization** - Plotly untuk chart yang responsif

---

## 📞 Support

Untuk pertanyaan atau issue, silahkan review code atau dokumentasi di atas. 

**Selamat menggunakan Dashboard! 🎉**
