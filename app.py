"""
Dashboard Streamlit - Monitorig Polutan PM di Kota Samarinda
Bagian dari Tugas Akhir
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import (
    APP_TITLE, MENU_DASHBOARD, MENU_DATA,
    PM10_COLUMN, PM25_COLUMN, TIME_COLUMN,
    COLOR_PM10, COLOR_PM25, COLOR_PREDICTION,
    RECORDS_PER_DAY
)
from data_manager import DataManager
from model_utils import train_and_forecast

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== SESSION STATE INIT ==========
if 'selected_menu' not in st.session_state:
    st.session_state.selected_menu = MENU_DASHBOARD

if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None

if 'forecast_cache' not in st.session_state:
    st.session_state.forecast_cache = {}

if 'selected_date_detail' not in st.session_state:
    st.session_state.selected_date_detail = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'data_error' not in st.session_state:
    st.session_state.data_error = None

# Try to load default data if it exists
if not st.session_state.data_loaded and st.session_state.data_error is None:
    try:
        dm = DataManager()
        df_check = dm.get_data_raw()
        # Hanya tandai sebagai loaded jika data tidak kosong
        if len(df_check) > 0:
            st.session_state.data_manager = dm
            st.session_state.data_loaded = True
        else:
            st.session_state.data_loaded = False
    except Exception as e:
        # Jika error, simpan error message untuk ditampilkan nanti
        st.session_state.data_error = str(e)
        st.session_state.data_loaded = False

# ========== HELPER FUNCTIONS ==========

def get_forecast(target_col):
    """Get forecast dengan caching"""
    if target_col not in st.session_state.forecast_cache:
        try:
            df = st.session_state.data_manager.get_data_raw()
            result = train_and_forecast(df, target_col)
            st.session_state.forecast_cache[target_col] = result['forecast_df']
        except Exception as e:
            st.error(f"Error training model untuk {target_col}: {str(e)}")
            return None
    
    return st.session_state.forecast_cache[target_col]


def aggregate_daily(df, target_col):
    """Aggregate data per hari (rata-rata)"""
    return df[[target_col]].resample('D').mean()


def create_line_chart(df, title, color, y_label="Konsentrasi (µg/m³)"):
    """Create Plotly line chart yang interaktif"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.iloc[:, 0],
        mode='lines+markers',
        name=df.columns[0],
        line=dict(color=color, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f} µg/m³<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Waktu",
        yaxis_title=y_label,
        hovermode='x unified',
        height=350,
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    
    return fig


def create_scorecard(label, value, icon="📊"):
    """Create scorecard component"""
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<div style='font-size: 32px;'>{icon}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='font-size: 12px; color: gray;'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 24px; font-weight: bold;'>{value}</div>", unsafe_allow_html=True)


def get_daily_detail(df_raw, selected_date):
    """Get 48 data points untuk satu hari"""
    date_start = pd.Timestamp(selected_date)
    date_end = date_start + timedelta(days=1)
    
    mask = (df_raw.index >= date_start) & (df_raw.index < date_end)
    daily_data = df_raw[mask]
    
    return daily_data


def show_daily_detail(df_raw, selected_date, selected_pollutant):
    """Display detail data untuk satu hari (48 records)"""
    daily_data = get_daily_detail(df_raw, selected_date)
    
    if len(daily_data) == 0:
        st.warning(f"⚠️ Tidak ada data untuk tanggal {selected_date.strftime('%Y-%m-%d')}")
        return
    
    # Determine which column exists
    col_name = selected_pollutant
    if col_name not in daily_data.columns:
        st.error(f"❌ Kolom {col_name} tidak ditemukan dalam data")
        return
    
    # Stats untuk hari tersebut
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Jumlah Data", len(daily_data))
    
    with col_stat2:
        st.metric(f"{col_name} Rata-rata", f"{daily_data[col_name].mean():.2f} µg/m³")
    
    with col_stat3:
        st.metric(f"{col_name} Min", f"{daily_data[col_name].min():.2f} µg/m³")
    
    with col_stat4:
        st.metric(f"{col_name} Max", f"{daily_data[col_name].max():.2f} µg/m³")
    
    st.divider()
    
    # Chart hourly untuk hari tersebut
    col_chart_hourly = st.columns(1)[0]
    
    # Hourly chart
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=daily_data.index,
        y=daily_data[col_name],
        mode='lines+markers',
        name=col_name,
        line=dict(color=COLOR_PM10 if col_name == PM10_COLUMN else COLOR_PM25, width=2),
        marker=dict(size=6),
        hovertemplate='<b>%{x|%H:%M}</b><br>%{y:.2f} µg/m³<extra></extra>'
    ))
    
    fig_hourly.update_layout(
        title=f"{col_name} - {selected_date.strftime('%d %B %Y')}",
        xaxis_title="Waktu",
        yaxis_title="Konsentrasi (µg/m³)",
        hovermode='x unified',
        height=350,
        template="plotly_white"
    )
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    st.divider()
    
    # Detail table
    st.markdown("#### 📊 Detail Data Jam-an")
    
    df_display = daily_data.reset_index()
    
    # Check if TIME_COLUMN exists, otherwise use the index column name
    time_col = None
    for col in df_display.columns:
        if col == TIME_COLUMN or col == 'index':
            time_col = col
            break
    
    if time_col:
        df_display[time_col] = pd.to_datetime(df_display[time_col]).dt.strftime('%H:%M:%S')
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=800  # Show all 48 rows
    )
    
    # Download button
    csv = daily_data.reset_index().to_csv(index=False)
    st.download_button(
        label="📥 Download Detail Data CSV",
        data=csv,
        file_name=f"detail_{selected_date.strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def show_forecast_detail(forecast_df, selected_date, selected_pollutant):
    """Display detail forecast data untuk satu hari (48 forecast points)"""
    daily_forecast = get_daily_detail(forecast_df, selected_date)
    
    if len(daily_forecast) == 0:
        st.warning(f"⚠️ Tidak ada prediksi untuk tanggal {selected_date.strftime('%Y-%m-%d')}")
        return
    
    # Determine which column exists
    col_name = selected_pollutant
    if col_name not in daily_forecast.columns:
        st.error(f"❌ Kolom {col_name} tidak ditemukan dalam data prediksi")
        return
    
    # Stats untuk hari prediksi tersebut
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Jumlah Prediksi", len(daily_forecast))
    
    with col_stat2:
        st.metric(f"{col_name} Prediksi Rata-rata", f"{daily_forecast[col_name].mean():.2f} µg/m³")
    
    with col_stat3:
        st.metric(f"{col_name} Prediksi Min", f"{daily_forecast[col_name].min():.2f} µg/m³")
    
    with col_stat4:
        st.metric(f"{col_name} Prediksi Max", f"{daily_forecast[col_name].max():.2f} µg/m³")
    
    st.divider()
    
    # Chart hourly untuk hari prediksi tersebut
    col_chart_forecast = st.columns(1)[0]
    
    # Forecast hourly chart
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=daily_forecast.index,
        y=daily_forecast[col_name],
        mode='lines+markers',
        name=col_name,
        line=dict(color=COLOR_PREDICTION, width=2),
        marker=dict(size=6),
        hovertemplate='<b>%{x|%H:%M}</b><br>%{y:.2f} µg/m³<extra></extra>'
    ))
    
    fig_forecast.update_layout(
        title=f"Prediksi {col_name} - {selected_date.strftime('%d %B %Y')}",
        xaxis_title="Waktu",
        yaxis_title="Prediksi Konsentrasi (µg/m³)",
        hovermode='x unified',
        height=350,
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.divider()
    
    # Detail table
    st.markdown("#### 🔮 Detail Prediksi Jam-an")
    
    df_display = daily_forecast.reset_index()
    
    # Check if TIME_COLUMN exists, otherwise use the index column name
    time_col = None
    for col in df_display.columns:
        if col == TIME_COLUMN or col == 'index':
            time_col = col
            break
    
    if time_col:
        df_display[time_col] = pd.to_datetime(df_display[time_col]).dt.strftime('%H:%M:%S')
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=800  # Show all 48 rows
    )
    
    # Download button
    csv = daily_forecast.reset_index().to_csv(index=False)
    st.download_button(
        label="📥 Download Detail Prediksi CSV",
        data=csv,
        file_name=f"forecast_detail_{selected_date.strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ========== HEADER / KOP ==========
st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
        <h1>Dashboard Polutan PM - Kota Samarinda</h1>
        <p>Monitoring Kualitas Udara Real-time</p>
    </div>
""", unsafe_allow_html=True)

# ========== NAVIGATION MENU ==========
if st.session_state.data_loaded:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Dashboard", use_container_width=True, key="btn_dashboard"):
            st.session_state.selected_menu = MENU_DASHBOARD
    with col2:
        if st.button("💾 Data", use_container_width=True, key="btn_data"):
            st.session_state.selected_menu = MENU_DATA

st.divider()

# ========== PAGE: DATA ERROR (Jika ada error saat load) ==========
if st.session_state.data_error is not None:
    st.markdown("### ⚠️ Ada Masalah dengan Data")
    
    st.error(f"""
    ❌ **Error loading data:** {st.session_state.data_error}
    
    Kemungkinan penyebab:
    - File CSV rusak atau tidak lengkap
    - Format data tidak sesuai
    - Kolom yang hilang atau tidak valid
    """)
    
    st.markdown("#### 📁 Ganti dengan Data CSV Baru")
    st.info("Upload file CSV baru untuk menggantikan data yang rusak. Data lama akan dihapus dan diganti.")
    
    col_upload_error, col_info_error = st.columns([0.6, 0.4])
    
    with col_upload_error:
        uploaded_file = st.file_uploader(
            "Pilih file CSV untuk menggantikan data",
            type=["csv"],
            help="File harus memiliki kolom: waktu, PM10, PM2.5",
            key="file_uploader_error"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df_upload = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = [TIME_COLUMN, PM10_COLUMN, PM25_COLUMN]
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
                    st.info(f"File harus memiliki kolom: {', '.join(required_cols)}")
                else:
                    # Try to parse datetime
                    try:
                        df_upload[TIME_COLUMN] = pd.to_datetime(df_upload[TIME_COLUMN])
                        df_upload = df_upload.sort_values(TIME_COLUMN)
                        df_upload = df_upload.set_index(TIME_COLUMN)
                        
                        # Create data manager dan replace data lama
                        data_manager = DataManager()
                        
                        # Replace data (simpan langsung ke file CSV)
                        try:
                            data_manager.replace_data(df_upload)
                            st.success(f"✅ Data berhasil diganti! Total records: {len(df_upload)}")
                            st.info("Data lama telah dihapus dan diganti dengan data baru.")
                            
                            # Update session state
                            st.session_state.data_manager = data_manager
                            st.session_state.data_loaded = True
                            st.session_state.data_error = None
                            st.session_state.forecast_cache = {}
                            
                            # Rerun untuk ke dashboard
                            st.rerun()
                        except Exception as save_error:
                            st.error(f"❌ Gagal menyimpan data: {str(save_error)}")
                            
                    except Exception as e:
                        st.error(f"❌ Error parsing tanggal: {str(e)}")
                        st.info(f"Format tanggal yang didukung: YYYY-MM-DD atau YYYY-MM-DD HH:MM:SS")
                        
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    with col_info_error:
        st.markdown("**Format CSV yang diharapkan:**")
        st.code("""
waktu,PM10,PM2.5
2024-01-01 00:00:00,50.5,25.3
2024-01-01 00:30:00,48.2,24.1
...
        """, language="csv")
    
    st.stop()

# ========== PAGE: LOAD DATA (Jika belum ada data) ==========
if not st.session_state.data_loaded:
    st.markdown("### 📊 Selamat Datang di Dashboard Polutan PM")
    
    st.info("""
    Dashboard ini membantu Anda memantau kualitas udara (Polutan PM10 dan PM2.5) 
    di Kota Samarinda dengan visualisasi dan prediksi menggunakan Machine Learning.
    
    **Untuk memulai:** Silakan upload file CSV dengan data polutan Anda.
    """)
    
    st.markdown("#### 📁 Upload Dataset CSV")
    
    col_upload, col_info = st.columns([0.6, 0.4])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=["csv"],
            help="File harus memiliki kolom: waktu, PM10, PM2.5"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df_upload = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = [TIME_COLUMN, PM10_COLUMN, PM25_COLUMN]
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
                    st.info(f"File harus memiliki kolom: {', '.join(required_cols)}")
                else:
                    # Try to parse datetime
                    try:
                        df_upload[TIME_COLUMN] = pd.to_datetime(df_upload[TIME_COLUMN])
                        df_upload = df_upload.sort_values(TIME_COLUMN)
                        df_upload = df_upload.set_index(TIME_COLUMN)
                        
                        # Create data manager dan simpan langsung
                        data_manager = DataManager()
                        
                        # Replace/save data ke file CSV
                        try:
                            data_manager.replace_data(df_upload)
                            st.success(f"✅ Data berhasil dimuat dan disimpan! Total records: {len(df_upload)}")
                        except Exception as save_error:
                            st.error(f"❌ Gagal menyimpan data: {str(save_error)}")
                            st.stop()
                        
                        st.session_state.data_manager = data_manager
                        st.session_state.data_loaded = True
                        st.session_state.data_error = None
                        st.session_state.forecast_cache = {}
                        
                        # Rerun langsung ke dashboard
                        st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Error parsing tanggal: {str(e)}")
                        st.info(f"Format tanggal yang didukung: YYYY-MM-DD atau YYYY-MM-DD HH:MM:SS")
                        
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    with col_info:
        st.markdown("**Format CSV yang diharapkan:**")
        st.code("""
waktu,PM10,PM2.5
2024-01-01 00:00:00,50.5,25.3
2024-01-01 00:30:00,48.2,24.1
...
        """, language="csv")
    
    st.stop()

# ========== LOAD DATA (Jika sudah ada data) ==========
try:
    data_manager = st.session_state.data_manager
    df_raw = data_manager.get_data_raw()
    df_agg = data_manager.get_data_aggregated('D')
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# ========== PAGE: DASHBOARD ==========
if st.session_state.selected_menu == MENU_DASHBOARD:
    st.markdown("### Dashboard Overview")
    
    # ===== SCORE CARDS & PREVIEW TABLE =====
    stats = data_manager.get_statistics()
    
    score_col1, score_col2, score_col3 = st.columns(3)
    
    with score_col1:
        st.metric(
            label="📈 Total Data",
            value=f"{stats['total_records']:,}",
            delta=f"{stats['total_records'] // RECORDS_PER_DAY} hari"
        )
    
    with score_col2:
        # Handle NaT (Not a Time) jika data kosong
        if pd.isna(stats['latest_date']):
            st.metric(
                label="📅 Data Terbaru",
                value="N/A",
                delta="Tidak ada data"
            )
        else:
            st.metric(
                label="📅 Data Terbaru",
                value=stats['latest_date'].strftime('%Y-%m-%d'),
                delta=stats['latest_date'].strftime('%H:%M:%S')
            )
    
    with score_col3:
        # Preview table dengan stat selector
        st.markdown("**📊 Preview Data**")
        
        # Check if data is empty
        if len(df_raw) == 0:
            st.warning("⚠️ Belum ada data. Silakan upload file CSV terlebih dahulu.")
        else:
            # Tabs untuk switch view
            tab_preview, tab_stats = st.tabs(["Data Terakhir", "Statistik"])
            
            with tab_preview:
                # Show last 5 rows
                preview_data = df_raw.tail(5).reset_index()
                preview_data[TIME_COLUMN] = preview_data[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(preview_data, use_container_width=True, hide_index=True)
            
            with tab_stats:
                # Statistics table
                stats_display = pd.DataFrame({
                    'Metrik': ['Max', 'Min', 'Rata-rata'],
                    'PM10 (µg/m³)': [
                        f"{stats['pm10_max']:.2f}",
                        f"{stats['pm10_min']:.2f}",
                        f"{stats['pm10_mean']:.2f}"
                    ],
                    'PM2.5 (µg/m³)': [
                        f"{stats['pm25_max']:.2f}",
                        f"{stats['pm25_min']:.2f}",
                        f"{stats['pm25_mean']:.2f}"
                    ]
                })
                st.dataframe(stats_display, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ===== CHART 1: DATA AKTUAL 7 HARI =====
    st.markdown("### Data Aktual 7 Hari Terakhir (Rata-rata Harian)")
    
    col_chart1, col_dropdown1 = st.columns([0.85, 0.15])
    
    with col_dropdown1:
        selected_pollutant_1 = st.selectbox(
            "Pilih Polutan",
            [PM10_COLUMN, PM25_COLUMN],
            key="dropdown_actual"
        )
    
    with col_chart1:
        df_7days = df_agg.tail(7)[[selected_pollutant_1]]
        color = COLOR_PM10 if selected_pollutant_1 == PM10_COLUMN else COLOR_PM25
        
        fig_actual = create_line_chart(
            df_7days,
            f"Data Aktual - {selected_pollutant_1} (7 Hari Terakhir)",
            color
        )
        st.plotly_chart(fig_actual, use_container_width=True)
    
    # ===== DETAIL HARIAN VIEW =====
    with st.expander("📅 Lihat Detail Data Per Hari", expanded=False):
        st.markdown("Pilih tanggal untuk melihat data detail 48 jam dalam sehari")
        
        col_date_picker, col_spacer = st.columns([0.3, 0.7])
        
        with col_date_picker:
            # Date picker - default ke hari terakhir
            default_date = df_raw.index[-1].date() if len(df_raw) > 0 else datetime.now().date()
            selected_date = st.date_input(
                "Pilih Tanggal",
                value=default_date,
                min_value=df_raw.index[0].date() if len(df_raw) > 0 else datetime.now().date(),
                max_value=df_raw.index[-1].date() if len(df_raw) > 0 else datetime.now().date(),
                key="date_picker_detail"
            )
        
        if selected_date:
            show_daily_detail(df_raw, selected_date, PM10_COLUMN)
    
    st.divider()
    
    # ===== CHART 2: FORECAST 7 HARI =====
    st.markdown("### Prediksi 7 Hari ke Depan")
    
    col_chart2, col_dropdown2 = st.columns([0.85, 0.15])
    
    with col_dropdown2:
        selected_pollutant_2 = st.selectbox(
            "Pilih Polutan",
            [PM10_COLUMN, PM25_COLUMN],
            key="dropdown_forecast"
        )
    
    # Generate forecast (outside column to keep it accessible)
    with st.spinner(f"⏳ Generating forecast untuk {selected_pollutant_2}..."):
        forecast_df = get_forecast(selected_pollutant_2)
    
    with col_chart2:
        if forecast_df is not None:
            # Aggregate forecast per hari untuk 7 hari
            forecast_daily = forecast_df.resample('D').mean().head(7)
            color = COLOR_PM10 if selected_pollutant_2 == PM10_COLUMN else COLOR_PM25
            
            fig_forecast = create_line_chart(
                forecast_daily,
                f"Prediksi - {selected_pollutant_2} (7 Hari ke Depan)",
                COLOR_PREDICTION,
                "Prediksi Konsentrasi (µg/m³)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.error("❌ Gagal generate forecast. Periksa data atau coba refresh.")
    
    # ===== DETAIL FORECAST VIEW =====
    if forecast_df is not None:
        with st.expander("📅 Lihat Detail Prediksi Per Hari", expanded=False):
            st.markdown("Pilih tanggal untuk melihat prediksi detail 48 jam dalam sehari")
            
            col_date_picker_forecast, col_spacer_forecast = st.columns([0.3, 0.7])
            
            with col_date_picker_forecast:
                # Date picker - default ke hari pertama prediksi
                default_forecast_date = forecast_df.index[0].date() if len(forecast_df) > 0 else datetime.now().date()
                selected_forecast_date = st.date_input(
                    "Pilih Tanggal Prediksi",
                    value=default_forecast_date,
                    min_value=forecast_df.index[0].date() if len(forecast_df) > 0 else datetime.now().date(),
                    max_value=forecast_df.index[-1].date() if len(forecast_df) > 0 else datetime.now().date(),
                    key="date_picker_forecast_detail"
                )
            
            if selected_forecast_date:
                show_forecast_detail(forecast_df, selected_forecast_date, selected_pollutant_2)


# ========== PAGE: DATA ==========
elif st.session_state.selected_menu == MENU_DATA:
    st.markdown("### Manajemen Data Polutan PM")
    
    # Create tabs for different data views
    tab_view, tab_upload = st.tabs(["📊 Lihat Data", "📁 Upload Data"])
    
    # ===== TAB: VIEW DATA =====
    with tab_view:
        # Data selector
        col_select, col_info = st.columns([0.2, 0.8])
        
        with col_select:
            data_view = st.selectbox(
                "Tampilkan Data",
                ["Semua", PM10_COLUMN, PM25_COLUMN, "Perbandingan"]
            )
        
        # Display data based on selection
        if data_view == "Semua":
            df_display = df_raw.reset_index()
        elif data_view == "Perbandingan":
            df_display = df_raw.reset_index()
        else:
            df_display = df_raw[[data_view]].reset_index()
        
        # Format waktu
        df_display[TIME_COLUMN] = df_display[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Show table dengan pagination
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        st.divider()
        
        # ===== DATA MANAGEMENT BUTTONS =====
        col_btn1, col_btn2, col_spacer = st.columns([0.3, 0.3, 0.4])
        
        with col_btn1:
            if st.button("🔄 Refresh Data", use_container_width=True, key="btn_refresh"):
                try:
                    st.session_state.data_manager.reset_to_latest_backup()
                    st.session_state.forecast_cache = {}
                    st.success("✅ Data berhasil di-refresh")
                    st.rerun()
                except:
                    st.warning("⚠️ Tidak ada backup data untuk di-refresh")
        
        with col_btn2:
            if st.button("➕ Tambah Data Minggu Depan", use_container_width=True, key="btn_add_data"):
                st.info("⚙️ Generate forecast dan tambahkan ke database...")
                
                try:
                    # Generate forecast untuk kedua polutan
                    forecast_pm10 = get_forecast(PM10_COLUMN)
                    forecast_pm25 = get_forecast(PM25_COLUMN)
                    
                    if forecast_pm10 is not None and forecast_pm25 is not None:
                        # Combine forecasts
                        forecast_combined = pd.DataFrame({
                            PM10_COLUMN: forecast_pm10[PM10_COLUMN],
                            PM25_COLUMN: forecast_pm25[PM25_COLUMN]
                        })
                        
                        # Add to data
                        data_manager.add_forecast_data(forecast_combined)
                        
                        # Remove old week
                        data_manager.remove_last_week()
                        
                        # Reset cache
                        st.session_state.forecast_cache = {}
                        
                        st.success("✅ Data minggu depan berhasil ditambahkan (data 1 minggu terakhir dihapus)")
                        st.rerun()
                    else:
                        st.error("❌ Gagal generate forecast")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ===== TAB: UPLOAD DATA =====
    with tab_upload:
        st.markdown("#### 📁 Ganti atau Tambah Data Baru")
        
        col_upload, col_info = st.columns([0.6, 0.4])
        
        with col_upload:
            st.markdown("**Upload file CSV baru untuk mengganti atau menambah data:**")
            
            if st.session_state.get('upload_message'):
                st.success(st.session_state['upload_message'])
                st.session_state.upload_message = None
            
            uploaded_file = st.file_uploader(
                "Pilih file CSV",
                type=["csv"],
                key="data_upload",
                help="File harus memiliki kolom: waktu, PM10, PM2.5"
            )
            
            upload_context = st.session_state.get('upload_context') or {}
            if uploaded_file is not None:
                try:
                    df_raw_upload = pd.read_csv(uploaded_file)
                    required_cols = [TIME_COLUMN, PM10_COLUMN, PM25_COLUMN]
                    missing_cols = [col for col in required_cols if col not in df_raw_upload.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
                        st.info(f"File harus memiliki kolom: {', '.join(required_cols)}")
                    else:
                        try:
                            df_raw_upload[TIME_COLUMN] = pd.to_datetime(df_raw_upload[TIME_COLUMN])
                        except Exception as e:
                            st.error(f"❌ Error parsing tanggal: {str(e)}")
                            st.info("Format tanggal yang didukung: YYYY-MM-DD atau YYYY-MM-DD HH:MM:SS")
                            st.stop()
                        
                        df_raw_upload = df_raw_upload.sort_values(TIME_COLUMN).reset_index(drop=True)
                        validation_result = st.session_state.data_manager.validate_data(
                            df_raw_upload.copy(), TIME_COLUMN
                        )
                        df_validated = validation_result['validated_df']

                        upload_context = {
                            'file_name': uploaded_file.name,
                            'df_raw_upload': df_raw_upload,
                            'df_validated': df_validated,
                            'validation_result': validation_result,
                            'action': upload_context.get('action'),
                            'confirm': upload_context.get('confirm', False)
                        }
                        st.session_state.upload_context = upload_context
                        
                        preview = df_validated.reset_index()
                        preview[TIME_COLUMN] = preview[TIME_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        preview_col, review_col = st.columns([0.65, 0.35])
                        with preview_col:
                            st.markdown("**Preview Data:**")
                            st.dataframe(preview.head(10), use_container_width=True, hide_index=True)
                        
                        with review_col:
                            st.markdown("### 🧪 Review Kondisi Data")
                            if validation_result['errors']:
                                for error in validation_result['errors']:
                                    st.error(f"❌ {error}")
                                st.info("Perbaiki format file CSV lalu upload ulang.")
                            else:
                                if validation_result['warnings']:
                                    st.warning("⚠️ Ditemukan kondisi data yang perlu diperhatikan:")
                                    for warning in validation_result['warnings']:
                                        st.markdown(f"- {warning}")
                                else:
                                    st.success("✅ Data tampak baik dan siap diproses.")
                                
                                st.markdown("---")
                                st.markdown("**Ringkasan Data Upload**")
                                st.write(f"- Total baris: {len(df_validated)}")
                                st.write(f"- Baris pertama: {df_validated.index[0].strftime('%Y-%m-%d %H:%M')}")
                                st.write(f"- Baris terakhir: {df_validated.index[-1].strftime('%Y-%m-%d %H:%M')}")
                                
                                if len(st.session_state.data_manager.get_data_raw()) > 0:
                                    last_existing = st.session_state.data_manager.get_data_raw().index[-1]
                                    st.write(f"- Data terakhir saat ini: {last_existing.strftime('%Y-%m-%d %H:%M')}")
                        
                        if validation_result['errors']:
                            st.stop()
                        
                        col_action1, col_action2 = st.columns(2)
                        with col_action1:
                            if st.button("✅ Ganti Data", use_container_width=True, key="btn_replace_data"):
                                st.session_state.upload_context['action'] = 'replace'
                                st.session_state.upload_context['confirm'] = False
                                st.rerun()
                        with col_action2:
                            if st.button("➕ Tambah Data", use_container_width=True, key="btn_append_data"):
                                st.session_state.upload_context['action'] = 'append'
                                st.session_state.upload_context['confirm'] = False
                                st.rerun()
                        
                        action = st.session_state.upload_context.get('action')
                        if action in ['replace', 'append']:
                            st.markdown("---")
                            st.markdown("### 🔔 Konfirmasi Pembersihan Otomatis")
                            st.info("Proses hanya dilanjutkan jika Anda memilih bersihkan otomatis. Jika tidak, proses akan dibatalkan.")
                            
                            if action == 'append' and st.session_state.upload_context is None:
                                st.error("❌ Context upload tidak tersedia lagi. Silakan upload ulang file.")
                                st.stop()
                            
                            if action == 'replace':
                                st.write("Anda memilih: **Ganti Data** — seluruh dataset akan diganti.")
                            else:
                                st.write("Anda memilih: **Tambah Data** — data baru akan ditambahkan ke dataset yang ada.")
                                if len(st.session_state.data_manager.get_data_raw()) > 0:
                                    existing_last = st.session_state.data_manager.get_data_raw().index[-1]
                                    next_expected = existing_last + pd.Timedelta(minutes=30)
                                    first_new = df_validated.index[0]
                                    if first_new != next_expected:
                                        st.error(
                                            f"❌ File upload tidak dapat ditambahkan karena baris pertama harus mulai dari {next_expected.strftime('%Y-%m-%d %H:%M')}.")
                                        st.info("Periksa apakah data upload berurutan dari data terakhir yang ada.")
                                        st.stop()
                            
                            col_confirm1, col_confirm2 = st.columns(2)
                            with col_confirm1:
                                if st.button("✅ Ya, bersihkan otomatis", use_container_width=True, key="btn_confirm_clean"):
                                    st.session_state.upload_context['confirm'] = True
                                    st.rerun()
                            with col_confirm2:
                                if st.button("❌ Tidak, batalkan", use_container_width=True, key="btn_cancel_action"):
                                    st.session_state.upload_context = None
                                    st.info("❌ Proses dibatalkan")
                                    st.rerun()
                        
                        if st.session_state.upload_context.get('confirm') and action in ['replace', 'append']:
                            with st.spinner("🔄 Membersihkan data dan menyimpan..."):
                                df_final = st.session_state.data_manager.clean_data(df_validated, method='fill_gaps')
                                if action == 'replace':
                                    st.session_state.data_manager.replace_data(df_final)
                                    st.session_state.forecast_cache = {}
                                    st.session_state['upload_message'] = "✅ Data berhasil diganti dan disimpan!"
                                else:
                                    st.session_state.data_manager.append_data(df_final)
                                    st.session_state.forecast_cache = {}
                                    st.session_state['upload_message'] = "✅ Data berhasil ditambahkan dan disimpan!"
                                st.session_state.upload_context = None
                                st.rerun()
                except Exception as e:
                    st.error(f"❌ Error membaca file: {str(e)}")
        
        with col_info:
            st.markdown("**Format CSV yang diharapkan:**")
            st.code("""
waktu,PM10,PM2.5
2024-01-01 00:00:00,50.5,25.3
2024-01-01 00:30:00,48.2,24.1
2024-01-01 01:00:00,52.1,26.5
...
            """, language="csv")
            
            st.markdown("**Catatan:**")
            st.markdown("""
            - Kolom **waktu** harus dalam format tanggal/waktu
            - Kolom **PM10** dan **PM2.5** harus berisi nilai numerik
            - Data akan otomatis disort berdasarkan waktu
            - Format spasi di nama kolom harus persis (PM2.5 bukan PM2,5)
            """)


# ========== FOOTER ==========
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
        <p>Dashboard Monitoring Polutan PM - Samarinda | Data per 30 menit</p>
        <p>Powered by Streamlit + XGBoost Forecasting</p>
    </div>
""", unsafe_allow_html=True)
