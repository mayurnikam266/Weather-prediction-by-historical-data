import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import uuid
import os
from datetime import timedelta, datetime
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns # Added for heatmap

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Title and description
st.title("🌤️ Weather Prediction App")
st.write("Upload historical weather data (CSV) to train a model and predict the next day's weather (temperature, humidity, wind speed, and precipitation type) for a selected or manually entered date.")

# Initialize session state (idempotent)
default_session_state = {
    'model': None,
    'scaler': None,
    'label_encoders': {},
    'feature_columns': None,
    'historical_data': None,
    'session_id': str(uuid.uuid4()), # For file uploader key
    'precip_type_encoder': None,
    'plots_generated': False,
    'previous_uploaded_file_name': None,
    'plot_start_date_ss': None,
    'plot_end_date_ss': None
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Define target columns
TARGET_COLUMNS = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Precip Type']
NUMERIC_FEATURES_FOR_CORRELATION = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']


# Define folder for storing model files
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_model_paths():
    return {
        'model': os.path.join(MODEL_DIR, 'weather_model.pkl'),
        'scaler': os.path.join(MODEL_DIR, 'scaler.pkl'),
        'encoders': os.path.join(MODEL_DIR, 'label_encoders.pkl'),
        'features': os.path.join(MODEL_DIR, 'feature_columns.pkl')
    }

def load_saved_model():
    paths = get_model_paths()
    if all(os.path.exists(path) for path in [paths['model'], paths['scaler'], paths['encoders'], paths['features']]):
        try:
            st.session_state.model = joblib.load(paths['model'])
            st.session_state.scaler = joblib.load(paths['scaler'])
            st.session_state.label_encoders = joblib.load(paths['encoders'])
            st.session_state.feature_columns = joblib.load(paths['features'])
            # Assuming Precip Type encoder might be stored differently or within label_encoders based on training
            if 'Precip Type' in st.session_state.label_encoders:
                 # If Precip Type was a feature handled by general encoders
                pass # It's already loaded
            elif os.path.exists(os.path.join(MODEL_DIR, 'precip_type_encoder.pkl')): # If saved separately
                 st.session_state.precip_type_encoder = joblib.load(os.path.join(MODEL_DIR, 'precip_type_encoder.pkl'))
            
            # st.success("Loaded previously trained model components from disk!") # Keep it silent initially
            return True
        except Exception as e:
            st.error(f"Error loading saved model components: {e}")
            st.session_state.model = None # Clear partially loaded components
            st.session_state.scaler = None
            st.session_state.label_encoders = {}
            st.session_state.feature_columns = None
            st.session_state.precip_type_encoder = None
    return False

def load_and_clean_data(uploaded_file, date_format=None):
    with st.spinner("Loading and cleaning data..."):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            
            expected_columns = [
                'Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)', 
                'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
                'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover', 
                'Pressure (millibars)', 'Daily Summary'
            ]
            if all(col.isdigit() for col in df.columns): # Heuristic for missing header
                if len(df.columns) == len(expected_columns):
                    st.warning("CSV seems to lack a header row. Assigning expected column names.")
                    df.columns = expected_columns
                else:
                    raise ValueError(f"CSV has {len(df.columns)} numerical columns (suspected missing header), expected {len(expected_columns)}.")

            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV is missing columns: {', '.join(missing_cols)}")
            
            # Robust date parsing
            non_date_strings = ['N/A', 'NA', 'nan', '', 'unknown', 'null', 'None', 'undefined'] # Added undefined
            invalid_date_mask = df['Formatted Date'].astype(str).str.strip().isin(non_date_strings) | df['Formatted Date'].isna()
            if invalid_date_mask.any():
                st.warning(f"Found {invalid_date_mask.sum()} rows with non-date string values in 'Formatted Date'. These will be dropped before date parsing.")
                df = df[~invalid_date_mask]

            date_parsing_failed_flag = False # Flag to track if initial custom/default parsing failed
            if date_format:
                try:
                    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], format=date_format, errors='coerce', utc=True)
                    if df['Formatted Date'].isna().all(): # if all became NaT, format was wrong
                        raise ValueError("Custom date format resulted in all NaT values.")
                except Exception:
                    st.warning(f"Custom date format '{date_format}' failed or resulted in all NaT. Trying other common formats.")
                    date_parsing_failed_flag = True # Mark as failed to try alternatives
            
            if not date_format or date_parsing_failed_flag: # If no custom format or it failed
                try: # Try specific common format first
                    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z', errors='coerce', utc=True)
                    if df['Formatted Date'].isna().all(): # if all became NaT
                        raise ValueError("Default ISO format resulted in all NaT values.")
                except Exception: # Fallback to flexible parsing
                    st.warning("Default ISO format '%Y-%m-%d %H:%M:%S.%f %z' failed or resulted in all NaT. Trying flexible parsing (slower).")
                    df['Formatted Date'] = df['Formatted Date'].apply(lambda x: pd.NaT if pd.isna(x) else parser.parse(str(x), fuzzy=True, default=datetime(1900,1,1)) if pd.notna(x) else pd.NaT)
                    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], errors='coerce', utc=True)

            valid_dates_mask = df['Formatted Date'].notna()
            if not valid_dates_mask.all():
                st.warning(f"Dropped {len(df) - valid_dates_mask.sum()} additional rows due to unparseable 'Formatted Date' values after all attempts.")
                df = df[valid_dates_mask]

            if df.empty:
                raise ValueError("No valid data remains after 'Formatted Date' parsing. Check your date column and format.")
            
            df['Formatted Date'] = df['Formatted Date'].dt.tz_convert(None) # To naive for consistency
            df['day_of_year'] = df['Formatted Date'].dt.dayofyear
            df['month'] = df['Formatted Date'].dt.month
            
            # Impute missing values
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols: # Impute per column to avoid issues with all-NaN columns
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
            
            non_numeric_cols = df.select_dtypes(exclude=[np.number, 'datetime64[ns]']).columns
            for col in non_numeric_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
            
            st.session_state.historical_data = df.copy()
            return df
        except Exception as e:
            st.error(f"Error loading and cleaning data: {str(e)}")
            st.session_state.historical_data = None
            return None

def generate_plots(df, start_date, end_date):
    with st.spinner("Generating plots..."):
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
            
            mask = (df['Formatted Date'] >= start_dt) & (df['Formatted Date'] <= end_dt)
            filtered_df = df[mask].copy()
            
            if filtered_df.empty:
                st.warning("No data found in the selected date range for plotting. Using entire dataset if available.")
                filtered_df = df.copy() 
                if filtered_df.empty:
                    st.error("Cannot generate plots: The dataset is empty.")
                    st.session_state.plots_generated = False
                    return

            # Interpolate and fill for plotting robustness
            plot_numeric_cols = filtered_df.select_dtypes(include=np.number).columns
            for col in plot_numeric_cols: # Interpolate/fill per column
                 if filtered_df[col].isnull().any():
                    filtered_df[col] = filtered_df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

            if 'Precip Type' in filtered_df.columns and filtered_df['Precip Type'].isnull().any():
                precip_mode = filtered_df['Precip Type'].mode()
                filtered_df['Precip Type'] = filtered_df['Precip Type'].fillna(precip_mode.iloc[0] if not precip_mode.empty else 'Unknown')
            
            st.markdown( # CSS for plot container
                """
                <style>
                .plot-container {
                    max-height: 900px; 
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                </style>
                """, unsafe_allow_html=True)
            
            plot_container = st.container() # Create a container for plots
            with plot_container:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True) # Apply custom CSS
                st.subheader(f"Data Visualizations for Range: {pd.to_datetime(start_date).strftime('%Y-%m-%d')} to {pd.to_datetime(end_date).strftime('%Y-%m-%d')}")
                
                # Row 1: Time Series Plots
                st.write("---")
                st.write("**Time Series Data**")
                col_ts1, col_ts2 = st.columns(2)
                try:
                    with col_ts1:
                        if 'Temperature (C)' in filtered_df.columns and not filtered_df[['Formatted Date', 'Temperature (C)']].dropna().empty:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(filtered_df['Formatted Date'], filtered_df['Temperature (C)'], label='Temperature (C)', color='red')
                            ax.set_xlabel("Date"); ax.set_ylabel("Temperature (C)"); ax.set_title("Temperature Over Time"); ax.legend(); plt.xticks(rotation=45); st.pyplot(fig); plt.close(fig)
                        else: st.warning("Not enough data for Temperature time series plot.")
                    with col_ts2:
                        if 'Humidity' in filtered_df.columns and not filtered_df[['Formatted Date', 'Humidity']].dropna().empty:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(filtered_df['Formatted Date'], filtered_df['Humidity'], label='Humidity', color='blue')
                            ax.set_xlabel("Date"); ax.set_ylabel("Humidity"); ax.set_title("Humidity Over Time"); ax.legend(); plt.xticks(rotation=45); st.pyplot(fig); plt.close(fig)
                        else: st.warning("Not enough data for Humidity time series plot.")
                except Exception as e: st.error(f"Error rendering time series plots: {e}")

                # Row 2: Pie Chart and Bar Chart
                st.write("---"); st.write("**Distributions and Averages**"); col_dist1, col_dist2 = st.columns(2)
                with col_dist1:
                    try:
                        st.write("**Precipitation Type Distribution**")
                        precip_counts = filtered_df['Precip Type'].value_counts()
                        if not precip_counts.empty:
                            fig, ax = plt.subplots(figsize=(6, 5)); ax.pie(precip_counts, labels=precip_counts.index, autopct='%1.1f%%', startangle=90); ax.axis('equal'); st.pyplot(fig); plt.close(fig)
                        else: st.warning("No 'Precip Type' data for pie chart.")
                    except Exception as e: st.error(f"Error rendering pie chart: {str(e)}")
                with col_dist2:
                    try:
                        st.write("**Average Temperature by Month**")
                        if 'month' not in filtered_df.columns and 'Formatted Date' in filtered_df.columns: filtered_df['month'] = filtered_df['Formatted Date'].dt.month
                        if 'Temperature (C)' in filtered_df.columns and 'month' in filtered_df.columns:
                            monthly_avg_temp = filtered_df.groupby('month')['Temperature (C)'].mean()
                            if not monthly_avg_temp.empty:
                                fig, ax = plt.subplots(figsize=(6, 5)); monthly_avg_temp.plot(kind='bar', ax=ax); ax.set_xlabel("Month"); ax.set_ylabel("Avg Temperature (C)"); st.pyplot(fig); plt.close(fig)
                            else: st.warning("No data for monthly average temperature bar chart.")
                        else: st.warning("Month and Temperature (C) columns needed for bar chart.")
                    except Exception as e: st.error(f"Error rendering bar chart: {str(e)}")

                # Row 3: Histograms
                st.write("---"); st.write("**Histograms of Key Metrics**"); hist_cols_plot = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']; cols_hist = st.columns(len(hist_cols_plot))
                for i, column in enumerate(hist_cols_plot):
                    with cols_hist[i]:
                        if column in filtered_df.columns and not filtered_df[column].dropna().empty:
                            fig, ax = plt.subplots(figsize=(6, 4)); filtered_df[column].hist(bins=15, ax=ax, grid=False, alpha=0.75); ax.set_title(f"Distribution of {column}"); ax.set_xlabel(column); ax.set_ylabel("Frequency"); st.pyplot(fig); plt.close(fig)
                        else: st.warning(f"No valid data for {column} histogram.")
                
                # Row 4: Scatter Plot and Correlation Heatmap
                st.write("---"); st.write("**Relationships Between Features**"); col_rel1, col_rel2 = st.columns([2,3])
                with col_rel1:
                    try:
                        st.write("**Temperature vs. Humidity**")
                        if 'Temperature (C)' in filtered_df.columns and 'Humidity' in filtered_df.columns and not filtered_df[['Temperature (C)', 'Humidity']].dropna().empty:
                            fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(filtered_df['Temperature (C)'], filtered_df['Humidity'], alpha=0.5, edgecolor='k', s=50); ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Humidity"); ax.set_title("Temp vs. Humidity"); st.pyplot(fig); plt.close(fig)
                        else: st.warning("Insufficient data for Temperature vs. Humidity scatter plot.")
                    except Exception as e: st.error(f"Error rendering scatter plot: {e}")
                with col_rel2:
                    try:
                        st.write("**Correlation Heatmap (Numeric Features)**")
                        # Select only numeric columns that exist in filtered_df for correlation
                        available_numeric_cols_for_corr = [col for col in NUMERIC_FEATURES_FOR_CORRELATION if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col])]
                        if available_numeric_cols_for_corr and len(available_numeric_cols_for_corr) > 1:
                            numeric_df_for_corr = filtered_df[available_numeric_cols_for_corr].copy().dropna()
                            if not numeric_df_for_corr.empty and len(numeric_df_for_corr.columns) > 1: # Double check after dropna
                                corr_matrix = numeric_df_for_corr.corr()
                                fig, ax = plt.subplots(figsize=(10, 7)); sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, annot_kws={"size":8}); ax.set_title("Correlation Heatmap"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); st.pyplot(fig); plt.close(fig)
                            else: st.warning("Not enough valid numeric data after cleaning for correlation heatmap.")
                        else: st.warning("Not enough numeric columns or data available for correlation heatmap.")
                    except Exception as e: st.error(f"Error rendering heatmap: {e}")

                st.markdown('</div>', unsafe_allow_html=True) # Close plot-container div
            
            st.success("Plots generated successfully!")
            st.session_state.plots_generated = True
        
        except Exception as e:
            st.error(f"An error occurred during plot generation: {str(e)}")
            st.session_state.plots_generated = False

def train_model(df_train): # Model training function
    with st.spinner("Training model... This may take a few moments."):
        try:
            df_processed = df_train.copy()
            if 'Precip Type' not in df_processed.columns: raise ValueError("'Precip Type' column is missing.")
            
            precip_encoder_train = LabelEncoder()
            df_processed['Precip Type'] = precip_encoder_train.fit_transform(df_processed['Precip Type'].astype(str)) # Ensure string type before encoding
            st.session_state.precip_type_encoder = precip_encoder_train
            joblib.dump(st.session_state.precip_type_encoder, os.path.join(MODEL_DIR, 'precip_type_encoder.pkl'))

            feature_cols_train = ['Apparent Temperature (C)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover', 'Pressure (millibars)', 'day_of_year', 'month', 'Summary']
            missing_features = [col for col in feature_cols_train if col not in df_processed.columns]
            if missing_features: raise ValueError(f"Missing feature columns for training: {', '.join(missing_features)}")

            X = df_processed[feature_cols_train]
            missing_targets = [col for col in TARGET_COLUMNS if col not in df_processed.columns]
            if missing_targets: raise ValueError(f"Missing target columns: {', '.join(missing_targets)}")
            y = df_processed[TARGET_COLUMNS]
            
            st.session_state.feature_columns = X.columns.tolist()
            
            current_label_encoders = {}
            for col in X.select_dtypes(include=['object', 'category']).columns: # Include category type
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str)) # Ensure string type
                current_label_encoders[col] = le
            st.session_state.label_encoders = current_label_encoders
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            current_scaler = StandardScaler()
            X_train_scaled = current_scaler.fit_transform(X_train)
            X_test_scaled = current_scaler.transform(X_test)
            st.session_state.scaler = current_scaler
            
            base_rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            trained_model = MultiOutputRegressor(base_rf_model)
            trained_model.fit(X_train_scaled, y_train)
            st.session_state.model = trained_model
            
            paths = get_model_paths(); joblib.dump(st.session_state.model, paths['model']); joblib.dump(st.session_state.scaler, paths['scaler']); joblib.dump(st.session_state.label_encoders, paths['encoders']); joblib.dump(st.session_state.feature_columns, paths['features'])
            
            y_pred_test = st.session_state.model.predict(X_test_scaled)
            st.subheader("Model Training Results (Test Set Performance)"); cols_metrics_res = st.columns(len(TARGET_COLUMNS))
            for i, target in enumerate(TARGET_COLUMNS):
                mse = mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i]); rmse = np.sqrt(mse); mae = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
                with cols_metrics_res[i]: st.metric(label=f"{target} (RMSE)", value=f"{rmse:.2f}"); st.metric(label=f"{target} (MAE)", value=f"{mae:.2f}")
            return True
        except Exception as e:
            st.error(f"Error training model: {str(e)}"); st.session_state.model = None; return False

# --- Attempt to load an existing model at the start ---
if not st.session_state.model: load_saved_model()

# --- Step 1: Upload Data, Visualize, and Train Model ---
st.header("Step 1: Upload Data, Visualize, and Train Model"); st.markdown("---")
st.subheader("1.1 Upload CSV Data")
date_format_user_input = st.text_input("Optional: Specify 'Formatted Date' column format (e.g., %Y-%m-%d %H:%M:%S.%f %z). Leave blank for auto.", value="%Y-%m-%d %H:%M:%S.%f %z", key="date_format_input_widget_step1")
uploaded_file_widget = st.file_uploader("Upload historical weather data CSV", type="csv", key=st.session_state.session_id)

if uploaded_file_widget is not None:
    if uploaded_file_widget.name != st.session_state.get('previous_uploaded_file_name', None):
        st.info(f"New file detected: {uploaded_file_widget.name}. Resetting plots and model state for new data.")
        st.session_state.plots_generated = False; st.session_state.previous_uploaded_file_name = uploaded_file_widget.name
        for key_to_del in ['plot_start_date_ss', 'plot_end_date_ss', 'model', 'scaler', 'label_encoders', 'feature_columns', 'precip_type_encoder', 'historical_data']:
            st.session_state[key_to_del] = default_session_state.get(key_to_del, None) # Reset to default or None
    
    df_current_upload = load_and_clean_data(uploaded_file_widget, date_format_user_input if date_format_user_input else None)

    if df_current_upload is not None and not df_current_upload.empty:
        st.subheader("1.2 Preview of Your Data (First 5 Rows)"); st.dataframe(df_current_upload.head())
        st.subheader("1.3 Visualize Data")
        unique_dates_in_df = df_current_upload['Formatted Date'].dt.date.unique()
        
        if len(unique_dates_in_df) > 0:
            min_plot_date, max_plot_date = min(unique_dates_in_df), max(unique_dates_in_df)
            st.write(f"Data available for plots from: {min_plot_date.strftime('%Y-%m-%d')} to {max_plot_date.strftime('%Y-%m-%d')}")
            default_start = st.session_state.get('plot_start_date_ss', min_plot_date); default_end = st.session_state.get('plot_end_date_ss', max_plot_date)
            try: default_start = pd.to_datetime(default_start).date()
            except: default_start = min_plot_date
            try: default_end = pd.to_datetime(default_end).date()
            except: default_end = max_plot_date
            default_start = max(min_plot_date, min(max_plot_date, default_start)); default_end = max(min_plot_date, min(max_plot_date, default_end))
            if default_start > default_end: default_end = default_start

            with st.form(key="plot_display_form"):
                f_col1, f_col2 = st.columns(2)
                with f_col1: form_start_date = st.date_input("Select start date for plots", value=default_start, min_value=min_plot_date, max_value=max_plot_date, key="plot_form_start_date_w")
                with f_col2: form_end_date = st.date_input("Select end date for plots", value=default_end, min_value=min_plot_date, max_value=max_plot_date, key="plot_form_end_date_w")
                submit_plot_form_button = st.form_submit_button("📊 Generate / Update Plots")

            if submit_plot_form_button:
                st.session_state.plot_start_date_ss, st.session_state.plot_end_date_ss = form_start_date, form_end_date
                st.session_state.plots_generated = False; generate_plots(df_current_upload, form_start_date, form_end_date); st.experimental_rerun()

            if not submit_plot_form_button and st.session_state.get('plots_generated', False):
                if st.session_state.plot_start_date_ss and st.session_state.plot_end_date_ss:
                    s_start, s_end = pd.to_datetime(st.session_state.plot_start_date_ss).date(), pd.to_datetime(st.session_state.plot_end_date_ss).date()
                    if min_plot_date <= s_start <= max_plot_date and min_plot_date <= s_end <= max_plot_date and s_start <= s_end:
                        generate_plots(df_current_upload, s_start, s_end)
                    else: st.session_state.plots_generated = False
        else: st.warning("No valid dates in uploaded data for plotting range selection.")
        st.subheader("1.4 Model Training")
        if st.button("🧠 Train Model with Current Uploaded Data", key="train_model_button_step1"):
            if train_model(df_current_upload): st.success("Model trained and saved successfully!"); st.balloons(); st.experimental_rerun()
            else: st.error("Model training failed. Please check data and configurations.")
    elif uploaded_file_widget is not None: st.error("File uploaded, but data could not be loaded/cleaned. Check CSV content and format.")

st.markdown("---")
if st.button("🔄 Clear All Data & Reset Full Application State", key="reset_all_app_button"):
    paths_to_clear = get_model_paths()
    for path_val in paths_to_clear.values():
        if os.path.exists(path_val):
            try: os.remove(path_val)
            except OSError as e: st.warning(f"Could not remove {path_val}: {e}")
    if os.path.exists(os.path.join(MODEL_DIR, 'precip_type_encoder.pkl')):
        try: os.remove(os.path.join(MODEL_DIR, 'precip_type_encoder.pkl'))
        except OSError: pass
    for key_to_reset, default_val in default_session_state.items():
        st.session_state[key_to_reset] = default_val if key_to_reset != 'session_id' else str(uuid.uuid4()) # Keep session_id unique
    st.success("Application state and any saved models cleared. Please re-upload data."); st.experimental_rerun()

# --- Step 2: Predict Next Day's Weather ---
st.markdown("---")
if st.session_state.get('model') and st.session_state.get('historical_data') is not None and st.session_state.get('feature_columns') is not None:
    st.header("Step 2: Predict Next Day's Weather")
    df_for_prediction = st.session_state.historical_data.sort_values('Formatted Date', ascending=False)
    unique_dates_for_pred_input = sorted(df_for_prediction['Formatted Date'].dt.date.unique(), reverse=True)

    if not unique_dates_for_pred_input: st.warning("No unique dates in historical data for prediction input selection.")
    else:
        min_pred_input_date, max_pred_input_date = min(unique_dates_for_pred_input), max(unique_dates_for_pred_input)
        st.subheader("2.1 Select Base Date for Prediction Input Features")
        pred_date_mode = st.radio("Choose date input method:", ("Select from dropdown", "Enter manually"), key="pred_date_mode_radio", horizontal=True)
        
        base_date_for_pred = None
        if pred_date_mode == "Select from dropdown":
            base_date_for_pred = st.selectbox("Select a date (its data will be used as a base for next day's feature inputs):", options=unique_dates_for_pred_input, format_func=lambda x: x.strftime('%Y-%m-%d'), key="pred_date_selectbox")
        else:
            manual_date_str_pred = st.text_input("Enter base date (YYYY-MM-DD):", placeholder="YYYY-MM-DD", key="pred_manual_date_input")
            if manual_date_str_pred:
                try:
                    manual_dt_pred = datetime.strptime(manual_date_str_pred, '%Y-%m-%d').date()
                    if min_pred_input_date <= manual_dt_pred <= max_pred_input_date:
                        if manual_dt_pred in unique_dates_for_pred_input: base_date_for_pred = manual_dt_pred
                        else: st.error(f"Date {manual_dt_pred.strftime('%Y-%m-%d')} not found in dataset's unique dates.")
                    else: st.error(f"Date {manual_dt_pred.strftime('%Y-%m-%d')} is outside dataset range ({min_pred_input_date.strftime('%Y-%m-%d')} to {max_pred_input_date.strftime('%Y-%m-%d')}).")
                except ValueError: st.error("Invalid date format. Use YYYY-MM-DD.")
        
        if base_date_for_pred:
            # MODIFICATION: Display full data for the selected base date
            base_data_series = df_for_prediction[df_for_prediction['Formatted Date'].dt.date == base_date_for_pred].iloc[-1].copy()
            st.subheader(f"📋 Full Weather Data for Selected Base Date: {base_date_for_pred.strftime('%Y-%m-%d')}")
            st.dataframe(pd.DataFrame([base_data_series])) # Display all columns for that record
            st.markdown("---") # Add a separator

            target_prediction_date = pd.to_datetime(base_date_for_pred) + timedelta(days=1)
            st.subheader(f"2.2 Adjust Features for Prediction ({target_prediction_date.strftime('%Y-%m-%d')})")
            
            input_features_dict = {}
            pred_form_cols = st.columns(3)
            derived_day_of_year = target_prediction_date.timetuple().tm_yday; derived_month = target_prediction_date.month
            feature_val_limits = {'Apparent Temperature (C)': (-50.0, 50.0), 'Wind Bearing (degrees)': (0.0, 360.0), 'Visibility (km)': (0.0, 50.0), 'Loud Cover': (0.0, 1.0), 'Pressure (millibars)': (900.0, 1100.0)}

            for i, feat_col in enumerate(st.session_state.feature_columns):
                current_widget_col = pred_form_cols[i % 3]; default_feature_val = base_data_series.get(feat_col)
                if feat_col == 'day_of_year': input_features_dict[feat_col] = derived_day_of_year; current_widget_col.metric(label=feat_col, value=derived_day_of_year); continue
                if feat_col == 'month': input_features_dict[feat_col] = derived_month; current_widget_col.metric(label=feat_col, value=derived_month); continue
                if feat_col in st.session_state.label_encoders:
                    encoder = st.session_state.label_encoders[feat_col]; cats = encoder.classes_.tolist(); default_cat_val = default_feature_val
                    if isinstance(default_feature_val, (int, np.integer)): # Attempt to decode if it's an int
                        try: default_cat_val = encoder.inverse_transform([int(default_feature_val)])[0]
                        except: pass # If fails, use as is or fallback
                    idx = cats.index(str(default_cat_val)) if str(default_cat_val) in cats else (cats.index(default_cat_val) if default_cat_val in cats else 0) # Handle if default_cat_val is not string
                    input_features_dict[feat_col] = current_widget_col.selectbox(f"{feat_col}", options=cats, index=idx, key=f"pred_feat_widget_{feat_col}")
                else:
                    min_v, max_v = feature_val_limits.get(feat_col, (-200.0, 200.0))
                    try: default_num_val = float(default_feature_val if pd.notna(default_feature_val) else 0.0); default_num_val = max(min_v, min(max_v, default_num_val))
                    except (ValueError, TypeError): default_num_val = (min_v + max_v) / 2 if pd.notna(min_v) and pd.notna(max_v) else 0.0
                    input_features_dict[feat_col] = current_widget_col.number_input(f"{feat_col}", min_value=min_v, max_value=max_v, value=default_num_val, step=0.1, key=f"pred_feat_widget_{feat_col}")
            
            if st.button("✨ Predict Next Day's Weather", key="predict_final_button"):
                with st.spinner("Forecasting..."):
                    try:
                        pred_input_df = pd.DataFrame([input_features_dict], columns=st.session_state.feature_columns)
                        for col_to_enc, enc in st.session_state.label_encoders.items():
                            if col_to_enc in pred_input_df.columns:
                                current_val_enc = pred_input_df[col_to_enc].iloc[0]
                                if current_val_enc not in enc.classes_: st.warning(f"Value '{current_val_enc}' for '{col_to_enc}' wasn't seen in training. Using '{enc.classes_[0]}' as fallback."); pred_input_df[col_to_enc] = enc.classes_[0]
                                pred_input_df[col_to_enc] = enc.transform(pred_input_df[col_to_enc])
                        pred_input_scaled = st.session_state.scaler.transform(pred_input_df); final_prediction = st.session_state.model.predict(pred_input_scaled)[0]
                        st.subheader(f"🌤️ Predicted Weather for {target_prediction_date.strftime('%Y-%m-%d')}"); pred_res_cols = st.columns(len(TARGET_COLUMNS))
                        for i, target_name in enumerate(TARGET_COLUMNS):
                            with pred_res_cols[i]:
                                if target_name == 'Precip Type' and st.session_state.precip_type_encoder:
                                    enc_precip_val = int(round(final_prediction[i])); precip_classes = st.session_state.precip_type_encoder.classes_
                                    if 0 <= enc_precip_val < len(precip_classes): pred_precip_label = precip_classes[enc_precip_val]
                                    else: pred_precip_label = f"Unknown (Code: {enc_precip_val})"; st.warning(f"Predicted 'Precip Type' code {enc_precip_val} is out of range for known types ({len(precip_classes)}).")
                                    st.metric(target_name, pred_precip_label)
                                else: st.metric(target_name, f"{final_prediction[i]:.2f}")
                        st.balloons()
                    except Exception as e_pred: st.error(f"Error during prediction: {str(e_pred)}"); st.exception(e_pred)
else:
    st.info("➡️ Complete Step 1 (upload data and train a model) to enable predictions in Step 2.")
