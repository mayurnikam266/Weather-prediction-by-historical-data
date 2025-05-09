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

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Title and description
st.title("🌤️ Weather Prediction App")
st.write("Upload historical weather data (CSV) to train a model and predict the next day's weather (temperature, humidity, wind speed, and precipitation type) for a selected or manually entered date.")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.label_encoders = {}
    st.session_state.feature_columns = None
    st.session_state.historical_data = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.precip_type_encoder = None

# Define target columns
TARGET_COLUMNS = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Precip Type']

# Define folder for storing model files
MODEL_DIR = "models"

# Ensure the models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Paths for saved model components
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
            st.session_state.precip_type_encoder = st.session_state.label_encoders.get('Precip Type', None)
            st.success("Loaded previously trained model!")
            return True
        except Exception as e:
            st.error(f"Error loading saved model: {e}")
    return False

# Function to load and clean data
def load_and_clean_data(uploaded_file, date_format=None):
    with st.spinner("Loading and cleaning data..."):
        try:
            uploaded_file.seek(0)
            raw_content = uploaded_file.read().decode('utf-8').splitlines()[:10]
            st.write("Raw CSV content (first 10 lines):", raw_content)
            
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            
            df.columns = df.columns.str.strip()
            st.write("Columns found in CSV:", list(df.columns))
            
            expected_columns = [
                'Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)', 
                'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
                'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover', 
                'Pressure (millibars)', 'Daily Summary'
            ]
            if all(col.isdigit() for col in df.columns):
                if len(df.columns) == len(expected_columns):
                    st.warning("CSV lacks header row. Assigning expected column names.")
                    df.columns = expected_columns
                else:
                    raise ValueError(f"CSV has {len(df.columns)} columns, but expected {len(expected_columns)} columns: {expected_columns}")
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV is missing these columns: {missing_columns}")
            
            unique_dates = df['Formatted Date'].unique()
            st.write(f"Unique 'Formatted Date' values (showing up to 50 of {len(unique_dates)}):", unique_dates[:50].tolist())
            
            non_date_values = ['N/A', 'NA', 'nan', '', 'unknown', 'null', 'None']
            invalid_mask = df['Formatted Date'].isin(non_date_values) | df['Formatted Date'].isna()
            if invalid_mask.any():
                st.warning(f"Found {invalid_mask.sum()} rows with non-date values in 'Formatted Date' (e.g., {df['Formatted Date'][invalid_mask].unique()[:5].tolist()}). These will be dropped.")
                df = df[~invalid_mask]
            
            if date_format:
                try:
                    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], format=date_format, errors='coerce', utc=True)
                except ValueError as e:
                    st.warning(f"Custom date format '{date_format}' failed: {str(e)}. Falling back to default parsing.")
                    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z', errors='coerce', utc=True)
            else:
                try:
                    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z', errors='coerce', utc=True)
                except ValueError:
                    st.warning("Default format '%Y-%m-%d %H:%M:%S.%f %z' failed. Trying flexible parsing.")
                    df['Formatted Date'] = df['Formatted Date'].apply(lambda x: parser.parse(str(x), fuzzy=True) if pd.notna(x) else pd.NaT)
            
            invalid_dates = df[df['Formatted Date'].isna()]
            if not invalid_dates.empty:
                st.warning(f"Found {len(invalid_dates)} rows with unparseable 'Formatted Date' values. These rows will be dropped.")
                st.write("Sample invalid rows:", invalid_dates[['Formatted Date', 'Summary', 'Temperature (C)']].head())
                df = df.dropna(subset=['Formatted Date'])
            
            if df.empty:
                raise ValueError("No valid data remains after dropping rows with invalid 'Formatted Date'.")
            
            if not pd.api.types.is_datetime64_any_dtype(df['Formatted Date']):
                raise ValueError("Failed to convert 'Formatted Date' to datetime. Please check the date format and ensure all values are valid dates.")
            
            df['Formatted Date'] = df['Formatted Date'].dt.tz_convert(None)
            df['day_of_year'] = df['Formatted Date'].dt.dayofyear
            df['month'] = df['Formatted Date'].dt.month
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            df[non_numeric_columns] = df[non_numeric_columns].fillna(df[non_numeric_columns].mode().iloc[0])
            
            st.session_state.historical_data = df
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

# Function to train model
def train_model(df):
    with st.spinner("Training model..."):
        try:
            precip_encoder = LabelEncoder()
            df['Precip Type'] = precip_encoder.fit_transform(df['Precip Type'])
            st.session_state.precip_type_encoder = precip_encoder
            
            feature_columns = [
                'Apparent Temperature (C)', 'Wind Bearing (degrees)', 'Visibility (km)', 
                'Loud Cover', 'Pressure (millibars)', 'day_of_year', 'month', 'Summary'
            ]
            X = df[feature_columns]
            y = df[TARGET_COLUMNS]
            
            st.session_state.feature_columns = X.columns.tolist()
            
            categorical_columns = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            for column in categorical_columns:
                label_encoders[column] = LabelEncoder()
                X[column] = label_encoders[column].fit_transform(X[column])
            st.session_state.label_encoders = label_encoders
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
            
            base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model = MultiOutputRegressor(base_model)
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            
            paths = get_model_paths()
            joblib.dump(model, paths['model'])
            joblib.dump(scaler, paths['scaler'])
            joblib.dump(label_encoders, paths['encoders'])
            joblib.dump(st.session_state.feature_columns, paths['features'])
            
            y_pred = model.predict(X_test_scaled)
            st.subheader("Model Training Results")
            cols = st.columns(len(TARGET_COLUMNS))
            for i, target in enumerate(TARGET_COLUMNS):
                mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
                with cols[i]:
                    st.metric(f"{target}", f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}")
            
            return True
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False

# File uploader section
st.header("Step 1: Train Your Model or Load Existing")
if not st.session_state.model:
    st.subheader("Upload CSV")
    date_format = st.text_input(
        "Optional: Specify 'Formatted Date' format (e.g., %Y-%m-%d %H:%M:%S.%f %z for 2006-04-01 00:00:00.000 +0200). Leave blank for automatic parsing.",
        value="%Y-%m-%d %H:%M:%S.%f %z"
    )
    uploaded_file = st.file_uploader("Upload historical weather data CSV", type="csv", key=st.session_state.session_id)
    
    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file, date_format if date_format else None)
        
        if df is not None:
            st.subheader("Preview of Your Data")
            st.dataframe(df.head())
            
            if not load_saved_model():
                if st.button("Train Model"):
                    success = train_model(df)
                    if success:
                        st.success("Model trained and saved!")
            
            if st.button("Reset and Train New Model"):
                paths = get_model_paths()
                for path in paths.values():
                    if os.path.exists(path):
                        os.remove(path)
                st.session_state.model = None
                st.session_state.scaler = None
                st.session_state.label_encoders = {}
                st.session_state.feature_columns = None
                st.session_state.historical_data = None
                st.session_state.precip_type_encoder = None
                st.experimental_rerun()

# Prediction section
st.header("Step 2: Predict Next Day's Weather")

if st.session_state.model and st.session_state.feature_columns is not None and st.session_state.historical_data is not None:
    df = st.session_state.historical_data.sort_values('Formatted Date', ascending=False)
    
    # Get date range and unique dates
    unique_dates = df['Formatted Date'].dt.date.unique()
    min_date = min(unique_dates)
    max_date = max(unique_dates)
    st.write(f"Dataset date range: {min_date} to {max_date}")
    
    # Get recent and random dates
    recent_dates = unique_dates[:20]  # Up to 20 most recent dates
    # Sample up to 5 random dates, ensuring no overlap with recent_dates
    remaining_dates = [d for d in unique_dates if d not in recent_dates]
    random_dates = np.random.choice(remaining_dates, size=min(5, len(remaining_dates)), replace=False) if remaining_dates else []
    # Combine and sort dates
    combined_dates = sorted(set(list(recent_dates) + list(random_dates)))
    
    if len(combined_dates) < 15:
        st.warning(f"Only {len(combined_dates)} unique dates available. Need at least 15 for full functionality.")
    
    # Date selection
    st.subheader("Select or Enter Date")
    date_selection_mode = st.radio("Choose date input method:", ("Select from dropdown", "Enter manually"))
    
    if date_selection_mode == "Select from dropdown":
        selected_date = st.selectbox(
            "Choose a date to predict the next day's weather:",
            options=combined_dates,
            format_func=lambda x: x.strftime('%Y-%m-%d')
        )
    else:
        manual_date_str = st.text_input(
            "Enter date (YYYY-MM-DD, e.g., 2006-04-01):",
            placeholder="YYYY-MM-DD"
        )
        selected_date = None
        if manual_date_str:
            try:
                manual_date = datetime.strptime(manual_date_str, '%Y-%m-%d').date()
                if min_date <= manual_date <= max_date:
                    if manual_date in unique_dates:
                        selected_date = manual_date
                    else:
                        st.error(f"Date {manual_date} not found in dataset. Please choose a date present in the data.")
                else:
                    st.error(f"Date {manual_date} is outside the dataset range ({min_date} to {max_date}).")
            except ValueError:
                st.error("Invalid date format. Please use YYYY-MM-DD (e.g., 2006-04-01).")
    
    if selected_date:
        selected_data = df[df['Formatted Date'].dt.date == selected_date].iloc[-1].copy()
        st.subheader(f"Data for Selected Date ({selected_date})")
        st.dataframe(pd.DataFrame([selected_data]))
        
        next_day_data = selected_data.copy()
        next_day_date = pd.to_datetime(selected_date) + timedelta(days=1)
        next_day_data['day_of_year'] = next_day_date.timetuple().tm_yday
        next_day_data['month'] = next_day_date.month
        
        st.subheader(f"Edit Features for Next Day ({next_day_date.strftime('%Y-%m-%d')})")
        input_data = {}
        cols = st.columns(3)
        
        feature_limits = {
            'Apparent Temperature (C)': (-50.0, 50.0),
            'Wind Bearing (degrees)': (0.0, 360.0),
            'Visibility (km)': (0.0, 20.0),
            'Loud Cover': (0.0, 100.0),
            'Pressure (millibars)': (900.0, 1100.0),
            'day_of_year': (1, 366),
            'month': (1, 12)
        }
        
        for i, column in enumerate(st.session_state.feature_columns):
            col = cols[i % 3]
            if column in st.session_state.label_encoders:
                categories = st.session_state.label_encoders[column].classes_
                default_value = next_day_data.get(column, categories[0])
                try:
                    default_index = list(categories).index(default_value)
                except ValueError:
                    default_index = 0
                selected = col.selectbox(f"{column}", options=categories, index=default_index)
                input_data[column] = selected
            else:
                min_val, max_val = feature_limits.get(column, (-1000.0, 1000.0))
                default_val = float(next_day_data.get(column, 0.0))
                default_val = max(min_val, min(max_val, default_val))
                if column in ['day_of_year', 'month']:
                    col.text_input(f"{column}", value=str(int(default_val)), disabled=True)
                    input_data[column] = default_val
                else:
                    value = col.number_input(f"{column}", min_value=min_val, max_value=max_val, value=default_val)
                    input_data[column] = value
        
        if st.button("Predict Next Day's Weather"):
            with st.spinner("Making prediction..."):
                try:
                    input_df = pd.DataFrame([input_data])
                    
                    for column in st.session_state.label_encoders:
                        input_df[column] = st.session_state.label_encoders[column].transform(input_df[column])
                    
                    input_scaled = st.session_state.scaler.transform(input_df)
                    
                    prediction = st.session_state.model.predict(input_scaled)[0]
                    
                    precip_type_pred = int(round(prediction[3]))
                    if st.session_state.precip_type_encoder:
                        precip_type_label = st.session_state.precip_type_encoder.inverse_transform([precip_type_pred])[0]
                    else:
                        precip_type_label = str(precip_type_pred)
                    
                    st.subheader(f"Prediction for {next_day_date.strftime('%Y-%m-%d')}")
                    st.write("Predicted Weather for the Next Day:")
                    cols = st.columns(len(TARGET_COLUMNS))
                    for i, target in enumerate(TARGET_COLUMNS):
                        if target == 'Precip Type':
                            with cols[i]:
                                st.metric(target, precip_type_label)
                        else:
                            with cols[i]:
                                st.metric(target, f"{prediction[i]:.2f}")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.write("Ensure input values match the training data format.")
else:
    st.warning("Please train a model first by uploading historical data above or ensure all saved model files and historical data are available.")