import streamlit as st
import pandas as pd
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> fe2b6ed (Initial commit with all necessary files and scripts)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import joblib
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler

# Load config file
with open("config.json") as f:
    config = json.load(f)

data_file = config["LOGGING"]["anomaly_log_file"]
model_path = config["MODEL"]["model_path"]

# Load trained model and scaler using joblib
with open(model_path, 'rb') as f:
    model_data = joblib.load(f)

model = model_data["model"]
scaler = model_data["scaler"]

# Load data function
def load_data():
    df = pd.read_csv(data_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

# Streamlit layout enhancements
st.set_page_config(page_title="Real-Time Sensor Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>📊 Real-Time Sensor Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Styling and Controls
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Explore real-time sensor data and anomalies.")
time_range = st.sidebar.slider("Select time range", 1, 24, 2, 1)  # hours
anomaly_toggle = st.sidebar.checkbox("Show Anomalies", True)

# Plot function with improved styling
def plot_graph(df):
    # Convert Timestamp to a more readable format: HH:MM (hours:minutes)
    df['Time'] = df['Timestamp'].dt.strftime('%H:%M')  # This will show time in Hours:Minutes

    # Scale the data for prediction
    scaled_data = scaler.transform(df[['Temperature', 'Humidity', 'Motion']])

    # Predict anomalies using the Isolation Forest model
    df['is_anomaly'] = model.predict(scaled_data)

=======
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page configuration with custom theme and favicon
st.set_page_config(
    page_title="Advanced Sensor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #FF5733, #FFC300);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stProgress > div > div {
        background-color: #FF5733;
    }
    .stSidebar {
        background-color: #f5f5f5;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #888;
    }
    .anomaly-badge {
        background-color: #FF5733;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        white-space: nowrap;
        z-index: 1;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo and title
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 class='main-header'>📊 Advanced Sensor Dashboard</h1>", unsafe_allow_html=True)

# Function to generate sample data with improved variation
def generate_sample_data():
    """Generate sample sensor data with realistic variation"""
    now = datetime.now()
    # Create timestamps for last 24 hours with 5-minute intervals
    timestamps = [now - timedelta(minutes=i*5) for i in range(288)]
    timestamps.reverse()  # Make chronological
    
    # Generate sample data with MORE pronounced randomness and patterns
    temperature_base = 25
    humidity_base = 50
    
    data = {
        'Timestamp': timestamps,
        'Temperature': [
            temperature_base + 
            4*np.sin(i/24) +  # Larger sine wave amplitude (4 instead of 2)
            1.5*np.sin(i/8) +  # Add second higher frequency component
            np.random.normal(0, 1.2)  # More random noise (1.2 instead of 0.5)
            for i in range(288)
        ],
        'Humidity': [
            humidity_base + 
            10*np.sin(i/12) +  # Larger amplitude (10 instead of 5)
            5*np.sin(i/6) +    # Add second higher frequency component 
            np.random.normal(0, 3.5)  # More random noise (3.5 instead of 2)
            for i in range(288)
        ],
        'Motion': [
            # More varied pattern with clusters of motion
            1 if (i % 40 < 15) or np.random.random() < 0.2 else 0
            for i in range(288)
        ]
    }
    
    return pd.DataFrame(data)

# Cache config loading
@st.cache_data(ttl=60)
def load_config():
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Config file not found. Using default configuration.")
        return {
            "LOGGING": {"anomaly_log_file": "data.csv"},
            "MODEL": {"model_path": "model.joblib"},
            "THRESHOLDS": {
                "temperature": {"min": 15, "max": 35},
                "humidity": {"min": 30, "max": 70},
                "motion": {"min": 0, "max": 1}
            }
        }
    except json.JSONDecodeError:
        st.error("Error parsing config.json. Using default configuration.")
        return {
            "LOGGING": {"anomaly_log_file": "data.csv"},
            "MODEL": {"model_path": "model.joblib"},
            "THRESHOLDS": {
                "temperature": {"min": 15, "max": 35},
                "humidity": {"min": 30, "max": 70},
                "motion": {"min": 0, "max": 1}
            }
        }

# Load config
config = load_config()
data_file = config["LOGGING"]["anomaly_log_file"]
model_path = config["MODEL"]["model_path"]
thresholds = config.get("THRESHOLDS", {
    "temperature": {"min": 15, "max": 35},
    "humidity": {"min": 30, "max": 70},
    "motion": {"min": 0, "max": 1}
})

# Cache model loading with fallback to a mock model
@st.cache_resource
def load_model_and_scaler():
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = joblib.load(f)
            return model_data["model"], model_data["scaler"]
        else:
            # Create a simple mock model and scaler
            st.warning(f"Model file not found at {model_path}. Using a default model.")
            
            # Create a simple isolation forest model
            model = IsolationForest(contamination=0.05, random_state=42)
            
            # Create a simple scaler
            scaler = StandardScaler()
            
            # Generate some sample data to fit the model and scaler
            sample_data = np.random.normal(size=(100, 3))
            model.fit(sample_data)
            scaler.fit(sample_data)
            
            return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

# Load data with error handling and caching
@st.cache_data(ttl=10)  # Cache for 10 seconds
def load_data():
    try:
        if not os.path.exists(data_file):
            st.warning(f"Data file not found at {data_file}. Using sample data.")
            return generate_sample_data()
            
        df = pd.read_csv(data_file)
        if 'Timestamp' not in df.columns:
            st.error("Timestamp column not found in data file. Using sample data.")
            return generate_sample_data()
            
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}. Using sample data.")
        return generate_sample_data()

# Enhanced sidebar with clean design
with st.sidebar:
    st.title("Dashboard Controls")
    
    st.markdown("---")
    
    st.subheader("⏱️ Time Settings")
    time_range = st.slider("Select time range (hours)", 1, 24, 2, 1)
    time_granularity = st.selectbox("Time Granularity", ["Minute", "Hour", "Day"], index=0)
    
    st.markdown("---")
    
    st.subheader("🔍 Display Options")
    anomaly_toggle = st.checkbox("Show Anomalies", True)
    show_stats = st.checkbox("Show Statistics", True)
    
    # Advanced settings in an expandable section
    with st.expander("🔧 Advanced Settings"):
        # Graph types
        chart_type = st.radio("Chart Type", ["Line", "Scatter", "Bar"], index=0)
        
        # Theme selection
        theme = st.selectbox("Dashboard Theme", 
                            ["Default", "Dark", "Light", "Professional"], 
                            index=0)
        
        # Anomaly detection sensitivity
        sensitivity = st.slider("Anomaly Detection Sensitivity", 0.1, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    
    st.subheader("🔄 Refresh Settings")
    auto_refresh = st.checkbox("Auto Refresh Dashboard", True)
    refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30, 5)
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.info("This dashboard displays real-time sensor data and detects anomalies using an Isolation Forest model.")
    
    # Manual refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()

# Function to filter data based on time range and granularity - FIXED for 'H'/'h' usage
def filter_and_aggregate_data(df, time_range, granularity="Minute"):
    if df.empty:
        return df
        
    latest_time = df['Timestamp'].max()
    start_time = latest_time - pd.Timedelta(hours=time_range)
    filtered_df = df[df['Timestamp'] >= start_time].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Apply time granularity aggregation
    if granularity == "Minute":
        filtered_df.loc[:, 'Time'] = filtered_df['Timestamp'].dt.strftime('%H:%M')
    elif granularity == "Hour":
        # Fixed to use 'h' instead of 'H' to avoid FutureWarning
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        
        filtered_df = filtered_df.set_index('Timestamp')
        aggregated = filtered_df[numeric_cols].resample('h').mean()  # Use 'h' instead of 'H'
        
        aggregated = aggregated.reset_index()
        aggregated['Time'] = aggregated['Timestamp'].dt.strftime('%H:00')
        
        return aggregated
    elif granularity == "Day":
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        
        filtered_df = filtered_df.set_index('Timestamp')
        aggregated = filtered_df[numeric_cols].resample('D').mean()
        
        aggregated = aggregated.reset_index()
        aggregated['Time'] = aggregated['Timestamp'].dt.strftime('%Y-%m-%d')
        
        return aggregated
    
    return filtered_df

# Enhanced analytics functions
def calculate_statistics(df):
    """Calculate advanced statistics for sensor data"""
    if df.empty:
        return {}
    
    stats = {}
    
    # Basic statistics
    for column in ['Temperature', 'Humidity', 'Motion']:
        if column in df.columns:
            stats[column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'change': df[column].iloc[-1] - df[column].iloc[0] if len(df) > 1 else 0
            }
    
    # Time-based statistics
    if 'Timestamp' in df.columns and len(df) > 1:
        df_copy = df.copy()
        df_copy['hour'] = df_copy['Timestamp'].dt.hour
        
        # Hourly patterns
        stats['hourly_temp'] = df_copy.groupby('hour')['Temperature'].mean().to_dict()
        stats['hourly_humidity'] = df_copy.groupby('hour')['Humidity'].mean().to_dict()
        stats['hourly_motion'] = df_copy.groupby('hour')['Motion'].mean().to_dict()
        
        # Calculate rates of change
        df_copy['temp_change'] = df_copy['Temperature'].diff()
        df_copy['humidity_change'] = df_copy['Humidity'].diff()
        
        stats['temp_change_rate'] = df_copy['temp_change'].mean()
        stats['humidity_change_rate'] = df_copy['humidity_change'].mean()
    
    return stats

def detect_correlations(df):
    """Detect correlations between different sensors"""
    if df.empty or len(df) < 3:
        return {}
    
    correlations = {}
    if all(col in df.columns for col in ['Temperature', 'Humidity', 'Motion']):
        corr_matrix = df[['Temperature', 'Humidity', 'Motion']].corr()
        correlations = corr_matrix.to_dict()
    
    return correlations

# Improved function to create interactive plots with better visualization
def create_dashboard_plots(df, anomaly_toggle=True, chart_type="Line", sensitivity=0.5):
    if df.empty:
        st.warning("No data available for the selected time range.")
        return None, None, None
    
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Scale the data for prediction if model and scaler are available
    if model is not None and scaler is not None and all(col in df.columns for col in ['Temperature', 'Humidity', 'Motion']):
        try:
            # Make a copy to avoid SettingWithCopyWarning
            features = df[['Temperature', 'Humidity', 'Motion']].copy()
            
            # Ensure data is in the right range by clipping extreme values
            features['Temperature'] = features['Temperature'].clip(0, 50)  # Reasonable temperature range
            features['Humidity'] = features['Humidity'].clip(0, 100)  # Humidity percentage
            features['Motion'] = features['Motion'].clip(0, 1)  # Binary value
            
            try:
                scaled_data = scaler.transform(features)
                raw_scores = model.decision_function(scaled_data)
                
                # Adjust threshold using sensitivity slider from UI
                threshold = -sensitivity
                df.loc[:, 'is_anomaly'] = (raw_scores < threshold).astype(int)
                
                # Limit maximum anomalies to 20% of data points to avoid excessive flagging
                if df['is_anomaly'].mean() > 0.2:
                    # If too many anomalies, keep only the most extreme ones
                    anomaly_count = int(len(df) * 0.2)
                    anomaly_indices = np.argsort(raw_scores)[:anomaly_count]
                    df.loc[:, 'is_anomaly'] = 0
                    df.loc[df.index[anomaly_indices], 'is_anomaly'] = 1
                
            except Exception as e:
                st.sidebar.warning(f"Scaling error: {str(e)}")
                df.loc[:, 'is_anomaly'] = 0
        except Exception as e:
            st.sidebar.warning(f"Anomaly detection error: {str(e)}")
            df.loc[:, 'is_anomaly'] = 0
    else:
        df.loc[:, 'is_anomaly'] = 0
    
    # Create subplots
>>>>>>> fe2b6ed (Initial commit with all necessary files and scripts)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "🌡️ Temperature Over Time",
            "💧 Humidity Over Time",
            "🚶 Motion Over Time"
        ),
        vertical_spacing=0.12
    )
<<<<<<< HEAD

    # Temperature Plot
    fig.add_trace(go.Scatter(
        x=df['Time'], y=df['Temperature'],
        mode='lines+markers', name='Temperature',
        line=dict(color='orange', width=2, dash='solid'),
        marker=dict(size=8, color='orange', opacity=0.6)
    ), row=1, col=1)

    # Humidity Plot
    fig.add_trace(go.Scatter(
        x=df['Time'], y=df['Humidity'],
        mode='lines+markers', name='Humidity',
        line=dict(color='blue', width=2, dash='solid'),
        marker=dict(size=8, color='blue', opacity=0.6)
    ), row=2, col=1)

    # Motion Plot
    fig.add_trace(go.Scatter(
        x=df['Time'], y=df['Motion'],
        mode='lines+markers', name='Motion',
        line=dict(color='green', width=2, dash='solid'),
        marker=dict(size=8, color='green', opacity=0.6)
    ), row=3, col=1)

    # Highlight Anomalies if toggle is checked
    if anomaly_toggle:
        anomalies = df[df['is_anomaly'] == -1]  # Isolation Forest uses -1 for anomalies
        fig.add_trace(go.Scatter(
            x=anomalies['Time'], y=anomalies['Temperature'],
            mode='markers', name='Anomalies',
=======
    
    # Set trace mode based on chart type
    if chart_type == "Line":
        mode = 'lines+markers'
    elif chart_type == "Scatter":
        mode = 'markers'
    else:  # Bar chart
        mode = 'lines'  # We'll use Bar traces separately
    
    # FIXED: Ensure time is properly sorted for better display
    if 'Time' in df.columns:
        # Try to convert string time to datetime for proper sorting
        if isinstance(df['Time'].iloc[0], str):
            try:
                # Add a dummy date to the time strings for proper sorting
                if ':' in df['Time'].iloc[0]:  # Format like "HH:MM"
                    df['TimeSort'] = pd.to_datetime("2023-01-01 " + df['Time'])
                else:  # Format might be date only
                    df['TimeSort'] = pd.to_datetime(df['Time'])
                df = df.sort_values('TimeSort')
            except:
                # If conversion fails, try to sort as strings
                df = df.sort_values('Time')
    
    # Add Temperature trace
    if chart_type == "Bar":
        fig.add_trace(go.Bar(
            x=df['Time'], y=df['Temperature'],
            name='Temperature',
            marker=dict(color='rgba(255, 87, 51, 0.7)')
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['Temperature'],
            mode=mode, name='Temperature',
            line=dict(color='#FF5733', width=2, dash='solid'),
            marker=dict(size=8, color='#FF5733', opacity=0.6)
        ), row=1, col=1)
    
    # Add Humidity trace
    if chart_type == "Bar":
        fig.add_trace(go.Bar(
            x=df['Time'], y=df['Humidity'],
            name='Humidity',
            marker=dict(color='rgba(66, 133, 244, 0.7)')
        ), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['Humidity'],
            mode=mode, name='Humidity',
            line=dict(color='#4285F4', width=2, dash='solid'),
            marker=dict(size=8, color='#4285F4', opacity=0.6)
        ), row=2, col=1)
    
    # Add Motion trace
    if chart_type == "Bar":
        fig.add_trace(go.Bar(
            x=df['Time'], y=df['Motion'],
            name='Motion',
            marker=dict(color='rgba(52, 168, 83, 0.7)')
        ), row=3, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['Motion'],
            mode=mode, name='Motion',
            line=dict(color='#34A853', width=2, dash='solid'),
            marker=dict(size=8, color='#34A853', opacity=0.6)
        ), row=3, col=1)
    
    # Add threshold reference lines
    if 'temperature' in thresholds:
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="red"),
            y0=thresholds['temperature']['max'], y1=thresholds['temperature']['max'],
            x0=0, x1=1, xref="paper", row=1, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="blue"),
            y0=thresholds['temperature']['min'], y1=thresholds['temperature']['min'],
            x0=0, x1=1, xref="paper", row=1, col=1
        )
    
    if 'humidity' in thresholds:
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="red"),
            y0=thresholds['humidity']['max'], y1=thresholds['humidity']['max'],
            x0=0, x1=1, xref="paper", row=2, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="blue"),
            y0=thresholds['humidity']['min'], y1=thresholds['humidity']['min'],
            x0=0, x1=1, xref="paper", row=2, col=1
        )
    
    # Highlight Anomalies if toggle is checked
    if anomaly_toggle and 'is_anomaly' in df.columns and df['is_anomaly'].sum() > 0:
        anomalies = df[df['is_anomaly'] == 1]
        
        # Add anomaly traces with distinct names
        fig.add_trace(go.Scatter(
            x=anomalies['Time'], y=anomalies['Temperature'],
            mode='markers', name='Temperature Anomalies',
>>>>>>> fe2b6ed (Initial commit with all necessary files and scripts)
            marker=dict(size=12, color='red', symbol='x', opacity=0.9)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=anomalies['Time'], y=anomalies['Humidity'],
<<<<<<< HEAD
            mode='markers', name='Anomalies',
=======
            mode='markers', name='Humidity Anomalies',
>>>>>>> fe2b6ed (Initial commit with all necessary files and scripts)
            marker=dict(size=12, color='red', symbol='x', opacity=0.9)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=anomalies['Time'], y=anomalies['Motion'],
<<<<<<< HEAD
            mode='markers', name='Anomalies',
            marker=dict(size=12, color='red', symbol='x', opacity=0.9)
        ), row=3, col=1)

    # Update Layout
    fig.update_layout(
        height=800,
        width=1000,
        title="📊 Live Sensor Monitoring (Temperature, Humidity, Motion)",
        xaxis_title="Time (HH:MM)",  # X-axis title updated
        template="plotly_dark",  # Dark background for contrast
        showlegend=True
    )

    # Pass a unique key to each chart to avoid duplicate element ID error
    st.plotly_chart(fig, use_container_width=True, key=str(time.time()))  # Unique key using timestamp

# Streamlit real-time loop
st.markdown("### 📈 Live Data and Anomalies")
while True:
    df = load_data()
    if not df.empty:
        plot_graph(df)
    time.sleep(config["LOGGING"]["interval_sec"])
=======
            mode='markers', name='Motion Anomalies',
            marker=dict(size=12, color='red', symbol='x', opacity=0.9)
        ), row=3, col=1)
    
    # Update Layout based on selected theme
    template = "plotly"
    if theme == "Dark":
        template = "plotly_dark"
    elif theme == "Light":
        template = "plotly_white"
    elif theme == "Professional":
        template = "plotly_white"
        fig.update_layout(
            font=dict(family="Arial", size=12),
            plot_bgcolor='rgba(240, 240, 240, 0.7)',
            paper_bgcolor='rgba(255, 255, 255, 0.8)',
        )
    
    # FIXED: Ensure the y-axis has appropriate range to show data variations
    y_margin = 0.1  # 10% margin above and below data
    
    if 'Temperature' in df.columns and len(df) > 0:
        temp_min = df['Temperature'].min()
        temp_max = df['Temperature'].max()
        if temp_min == temp_max:  # If all values are the same
            temp_range = abs(temp_min) * 0.1 if temp_min != 0 else 1.0
            fig.update_yaxes(range=[temp_min - temp_range, temp_max + temp_range], row=1, col=1)
        else:
            range_size = temp_max - temp_min
            fig.update_yaxes(range=[temp_min - range_size * y_margin, temp_max + range_size * y_margin], row=1, col=1)
    
    if 'Humidity' in df.columns and len(df) > 0:
        hum_min = df['Humidity'].min()
        hum_max = df['Humidity'].max()
        if hum_min == hum_max:  # If all values are the same
            hum_range = abs(hum_min) * 0.1 if hum_min != 0 else 1.0
            fig.update_yaxes(range=[hum_min - hum_range, hum_max + hum_range], row=2, col=1)
        else:
            range_size = hum_max - hum_min
            fig.update_yaxes(range=[hum_min - range_size * y_margin, hum_max + range_size * y_margin], row=2, col=1)
    
    if 'Motion' in df.columns and len(df) > 0:
        # For motion, typically binary 0-1, set a fixed range
        fig.update_yaxes(range=[-0.1, 1.1], row=3, col=1)
    
    fig.update_layout(
        height=800,
        title={
            'text': "📊 Live Sensor Monitoring",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template=template,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=2, col=1)
    fig.update_yaxes(title_text="Motion", row=3, col=1)
    
    # Add range selector to bottom x-axis
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        row=3, col=1
    )
    
    # Create histogram subplots for data distribution
    hist_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Temperature Distribution", "Humidity Distribution", "Motion Distribution")
    )
    
    hist_fig.add_trace(go.Histogram(
        x=df['Temperature'],
        nbinsx=20,
        marker_color='#FF5733',
        name='Temperature'
    ), row=1, col=1)
    
    hist_fig.add_trace(go.Histogram(
        x=df['Humidity'],
        nbinsx=20,
        marker_color='#4285F4',
        name='Humidity'
    ), row=1, col=2)
    
    hist_fig.add_trace(go.Histogram(
        x=df['Motion'],
        nbinsx=2,
        marker_color='#34A853',
        name='Motion'
    ), row=1, col=3)
    
    hist_fig.update_layout(
        height=300,
        template=template,
        showlegend=False,
        bargap=0.1
    )
    
    # Create correlation heatmap
    if len(df) > 5:  # Need minimum data points for correlation
        corr_matrix = df[['Temperature', 'Humidity', 'Motion']].corr()
        
        corr_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text:.2f}",
            textfont={"size":14},
        ))
        
        corr_fig.update_layout(
            title="Sensor Correlation Matrix",
            height=300,
            width=500,
            template=template,
        )
    else:
        corr_fig = None
    
    return fig, hist_fig, corr_fig

# Main dashboard area
# Load and filter data
df = load_data()
filtered_df = filter_and_aggregate_data(df, time_range, time_granularity)

# Main container with data visualization
if not filtered_df.empty:
    # Display summary metrics
    st.markdown("### 📈 Live Sensor Metrics")
    
    # Metrics with improved styling - Remove problematic div wrapping
    col1, col2, col3 = st.columns(3)
    
    # Calculate trend indicators
    if len(filtered_df) > 1:
        temp_trend = filtered_df['Temperature'].iloc[-1] - filtered_df['Temperature'].iloc[-2]
        humidity_trend = filtered_df['Humidity'].iloc[-1] - filtered_df['Humidity'].iloc[-2]
    else:
        temp_trend = 0
        humidity_trend = 0
    
    with col1:
        st.metric(
            "Average Temperature", 
            f"{filtered_df['Temperature'].mean():.1f}°C", 
            f"{temp_trend:.1f}°C"
        )
        
    with col2:
        st.metric(
            "Average Humidity", 
            f"{filtered_df['Humidity'].mean():.1f}%", 
            f"{humidity_trend:.1f}%"
        )
        
    with col3:
        motion_percent = filtered_df['Motion'].mean() * 100
        st.metric(
            "Motion Detected", 
            f"{motion_percent:.1f}% of time"
        )
    
    # Create and display main plots
    main_fig, hist_fig, corr_fig = create_dashboard_plots(filtered_df, anomaly_toggle, chart_type, sensitivity)
    
    if main_fig:
        st.plotly_chart(main_fig, use_container_width=True)
    
    # Display anomaly statistics if anomalies are toggled on
    if anomaly_toggle and 'is_anomaly' in filtered_df.columns:
        anomaly_count = filtered_df['is_anomaly'].sum()
        
        if anomaly_count > 0:
            st.markdown(f"<div class='anomaly-badge'>Detected {anomaly_count} anomalies</div> in the selected time range.", unsafe_allow_html=True)
            
            # Display recent anomalies table
            st.markdown("### 🚨 Recent Anomalies")
            
            anomaly_df = filtered_df[filtered_df['is_anomaly'] == 1].sort_values('Timestamp', ascending=False)
            
            # Enhanced table with highlighting
            def highlight_anomalies(val):
                if isinstance(val, (int, float)):
                    # Check if value exceeds thresholds
                    if val > 30 and val < 100:  # Likely temperature or humidity
                        return 'background-color: rgba(255, 87, 51, 0.2)'
                return ''
            
            st.dataframe(
                anomaly_df[['Timestamp', 'Temperature', 'Humidity', 'Motion']].head(10)
                .style.applymap(highlight_anomalies),
                use_container_width=True
            )
        else:
            st.success("No anomalies detected in the selected time range.")
    
    # Display additional statistical visualizations if enabled
    if show_stats and not filtered_df.empty:
        st.markdown("### 📊 Data Distribution and Statistics")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if hist_fig:
                st.plotly_chart(hist_fig, use_container_width=True)
        
        with col2:
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
        
        # Calculate and display advanced statistics
        stats = calculate_statistics(filtered_df)
        
        if stats:
            st.markdown("### 📝 Detailed Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Temperature Stats")
                st.markdown(f"**Min:** {stats['Temperature']['min']:.1f}°C")
                st.markdown(f"**Max:** {stats['Temperature']['max']:.1f}°C")
                st.markdown(f"**Mean:** {stats['Temperature']['mean']:.1f}°C")
                st.markdown(f"**Median:** {stats['Temperature']['median']:.1f}°C")
                st.markdown(f"**Std Dev:** {stats['Temperature']['std']:.2f}")
            
            with col2:
                st.markdown("#### Humidity Stats")
                st.markdown(f"**Min:** {stats['Humidity']['min']:.1f}%")
                st.markdown(f"**Max:** {stats['Humidity']['max']:.1f}%")
                st.markdown(f"**Mean:** {stats['Humidity']['mean']:.1f}%")
                st.markdown(f"**Median:** {stats['Humidity']['median']:.1f}%")
                st.markdown(f"**Std Dev:** {stats['Humidity']['std']:.2f}")
            
            with col3:
                st.markdown("#### Motion Stats")
                st.markdown(f"**Active Time:** {stats['Motion']['mean']*100:.1f}%")
                st.markdown(f"**Total Motion Events:** {int(filtered_df['Motion'].sum())}")
                
                # Calculate motion patterns if available
                if 'hourly_motion' in stats:
                    peak_hour = max(stats['hourly_motion'].items(), key=lambda x: x[1])[0]
                    st.markdown(f"**Peak Motion Hour:** {peak_hour}:00")
    
    # Data export options
    with st.expander("Export Data", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name=f'sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )
        
        with col2:
            # Fixed Excel export with proper buffer
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Sensor Data')
                # No need to call writer.save() explicitly - it's handled by the context manager
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer,
                file_name=f'sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
else:
    st.warning("No data available. Check your data file path or try increasing the time range.")

# Display information about the graph - outside the update loop
with st.expander("Graph Information", expanded=False):
    st.markdown("""
    ### Dashboard Information
    
    This advanced dashboard displays real-time sensor data with the following features:
    
    - **Time Series Visualization**: Track temperature, humidity, and motion over time
    - **Anomaly Detection**: Machine learning-based anomaly detection using Isolation Forest
    - **Statistical Analysis**: View data distributions and correlations between sensors
    - **Customization Options**: Change time ranges, chart types, and display settings
    - **Auto-Refresh**: Keep data current with automatic dashboard updates
    
    #### Sensor Details:
    - **Temperature**: Measured in degrees Celsius (°C)
    - **Humidity**: Measured as percentage (%)
    - **Motion**: Binary detection (1 = motion detected, 0 = no motion)
    
    #### Anomaly Detection:
    Anomalies (shown as red X markers) are detected when the sensor readings deviate significantly from normal patterns as determined by the machine learning model.
    """)

# Footer with auto-refresh indicator
st.markdown("---")
if auto_refresh:
    st.markdown(f"*Auto-refreshing every {refresh_interval} seconds. Last updated: {datetime.now().strftime('%H:%M:%S')}*")
    st.markdown("<div class='footer'>© 2025 Advanced Sensor Dashboard | Version 2.0</div>", unsafe_allow_html=True)

# If auto refresh is enabled, add automatic page refreshing using JavaScript
if auto_refresh:
    refresh_js = f"""
    <script>
        var timer = setTimeout(function() {{
            window.location.reload();
        }}, {refresh_interval * 1000});
        window.onbeforeunload = function() {{
            clearTimeout(timer);
        }}
    </script>
    """
    st.components.v1.html(refresh_js, height=0)
>>>>>>> fe2b6ed (Initial commit with all necessary files and scripts)
