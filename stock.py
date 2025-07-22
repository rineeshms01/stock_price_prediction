# Enhanced Stock Price Prediction Dashboard
# Comprehensive improvements with multiple models, sentiment analysis, and advanced features

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

# Feature Importance & Explainability
import shap

# Utilities
from datetime import datetime, timedelta
import time
import io
import base64
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Advanced Stock Prediction",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown('<h1 class="main-header">🚀 Advanced Stock Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🎛️ Configuration Panel")

    # Popular stocks with logos
    popular_tickers = {
        "🍎 Apple": "AAPL",
        "🖥️ Microsoft": "MSFT",
        "📦 Amazon": "AMZN",
        "🔍 Google": "GOOGL",
        "⚡ Tesla": "TSLA",
        "🎮 Nvidia": "NVDA",
        "👥 Meta": "META",
        "🎬 Netflix": "NFLX",
        "💻 Intel": "INTC",
        "🔵 IBM": "IBM",
        "🎨 Adobe": "ADBE",
        "☁️ Salesforce": "CRM",
        "💰 Berkshire Hathaway": "BRK-B",
        "🏦 JP Morgan": "JPM",
        "💳 Visa": "V",
        "🥤 Coca-Cola": "KO",
        "🛒 Walmart": "WMT",
        "🍟 McDonald's": "MCD",
        "🧴 Procter & Gamble": "PG"
    }

    selected_ticker_name = st.selectbox("📈 Select Popular Stock:", list(popular_tickers.keys()), index=0)
    ticker = st.text_input("✏️ Or Enter Custom Ticker:", value=popular_tickers[selected_ticker_name])

    st.divider()

    # Date range selection
    st.subheader("📅 Date Range")
    # Changed default selection to 'Max' for "beginning to current date"
    date_presets = st.radio(
        "Quick Presets:",
        ["6M", "1Y", "2Y", "3Y", "5Y", "Max", "Custom"],
        horizontal=True,
        index=5 # Set index to 5 for 'Max'
    )

    if date_presets != "Custom":
        if date_presets == "6M":
            start_date = datetime.now() - timedelta(days=180)
        elif date_presets == "1Y":
            start_date = datetime.now() - timedelta(days=365)
        elif date_presets == "2Y":
            start_date = datetime.now() - timedelta(days=2*365)
        elif date_presets == "3Y":
            start_date = datetime.now() - timedelta(days=3*365)
        elif date_presets == "5Y":
            start_date = datetime.now() - timedelta(days=5*365)
        else: # 'Max' selected
            start_date = datetime(2000, 1, 1) # yfinance typically goes back to ~2000 or IPO date
        end_date = datetime.now()
    else:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", value=datetime.now())
        if start_date > end_date:
            st.error("❌ Start date must be before end date.")
            st.stop()

    st.divider()

    # Model selection
    st.subheader("🤖 Model Configuration")
    model_choice = st.selectbox(
        "Choose Prediction Model:",
        ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "LSTM", "Prophet"]
    )

    # Model hyperparameters
    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
        max_depth = st.slider("Max Depth", 3, 20, 10)
    elif model_choice == "XGBoost":
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        n_estimators = st.slider("Number of Estimators", 50, 1000, 100, 50)
    elif model_choice == "LightGBM":
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        num_leaves = st.slider("Number of Leaves", 10, 300, 31, 10)
    elif model_choice == "LSTM":
        sequence_length = st.slider("Sequence Length", 10, 100, 60, 10)
        lstm_units = st.slider("LSTM Units", 32, 256, 64, 32)

    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100

    # Forecasting options
    st.subheader("🔮 Forecasting")
    # Changed to fix forecast_days to 1
    forecast_days = 1
    st.write(f"Days to Forecast: **{forecast_days} (Next Day)**")


    st.divider()

    # Feature engineering options
    st.subheader("⚙️ Feature Engineering")
    use_technical_indicators = st.checkbox("Technical Indicators", True)
    use_sentiment = st.checkbox("Sentiment Analysis", False)
    use_lag_features = st.checkbox("Lag Features", True)

    if use_sentiment:
        st.info("💡 Sentiment analysis uses recent news headlines. For this demo, sentiment is simulated based on the selected stock. Real-time web scraping is not performed and financial advice is not provided.")

# Tab structure for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔮 Predictions", "😊 Sentiment", "📈 Analysis", "⚙️ Settings"])

# Cache functions for better performance
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.error(f"❌ No data found for ticker: {ticker}")
            return None, None

        # Get company info
        try:
            info = stock.info
            company_name = info.get('longName', ticker)
            sector = info.get('sector', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
        except:
            company_name = ticker
            sector = 'N/A'
            market_cap = 'N/A'

        company_info = {
            'name': company_name,
            'sector': sector,
            'market_cap': market_cap
        }

        return data, company_info
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None, None

@st.cache_data
def get_news_sentiment(ticker, days=7):
    """Get news sentiment for the stock (simulated for demo purposes)"""
    try:
        # This is a placeholder - in production, you'd use a News API (e.g., NewsAPI, Alpha Vantage news)
        # For demo, we'll simulate sentiment scores consistent for the given ticker
        np.random.seed(hash(ticker) % (2**32 - 1)) # Consistent random for same ticker, avoid 0 seed if hash can be 0

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        sentiments = np.random.normal(0, 0.3, days)  # Random sentiment around neutral

        sentiment_data = pd.DataFrame({
            'date': dates,
            'sentiment': sentiments,
            'sentiment_label': ['Positive' if s > 0.1 else 'Negative' if s < -0.1 else 'Neutral' for s in sentiments]
        })

        return sentiment_data
    except Exception as e:
        st.warning(f"Could not generate simulated sentiment data: {e}")
        return None

def compute_technical_indicators(df):
    """Compute comprehensive technical indicators"""
    df = df.copy()

    # Moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()

    # RSI
    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = compute_rsi(df['Close'])

    # MACD
    def compute_macd(series, slow=26, fast=12, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = compute_macd(df['Close'])

    # Bollinger Bands
    def compute_bollinger_bands(series, window=20, num_std=2):
        ma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower

    df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(df['Close'])
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']

    # Average True Range (ATR)
    def compute_atr(df, window=14):
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift())
        low_close_prev = np.abs(df['Low'] - df['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return tr.rolling(window).mean()

    df['ATR'] = compute_atr(df)

    # On Balance Volume (OBV)
    def compute_obv(df):
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return obv

    df['OBV'] = compute_obv(df)

    # Stochastic Oscillator
    def compute_stochastic(df, k_window=14, d_window=3):
        low_min = df['Low'].rolling(window=k_window).min()
        high_max = df['High'].rolling(window=k_window).max()
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

    df['Stoch_K'], df['Stoch_D'] = compute_stochastic(df)

    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()

    # Price features
    df['Daily_Return'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

    return df

def add_lag_features(df, lags=[1, 2, 3, 5, 10]):
    """Add lag features for time series"""
    for lag in lags:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Daily_Return_lag_{lag}'] = df['Daily_Return'].shift(lag)
    return df

def prepare_features(df, use_technical=True, use_lags=True, use_sentiment_data=None):
    """Prepare features for machine learning"""
    df = df.copy()

    # Add target variable
    df['Target'] = df['Close'].shift(-1)

    # Technical indicators
    if use_technical:
        df = compute_technical_indicators(df)

    # Lag features
    if use_lags:
        df = add_lag_features(df)

    # Sentiment features
    if use_sentiment_data is not None:
        sentiment_df = use_sentiment_data.set_index('date')
        df = df.join(sentiment_df[['sentiment']], how='left')
        df['sentiment'] = df['sentiment'].fillna(0) # Fill NaN from join with 0 (neutral)

    # Drop NaN values introduced by shifting or rolling window calculations
    # Prioritize 'Target' and other essential columns for dropping NaNs
    initial_rows = len(df)
    df.dropna(inplace=True)
    if len(df) < initial_rows:
        st.warning(f"Removed {initial_rows - len(df)} rows with NaN values after feature engineering.")
    if df.empty:
        st.error("DataFrame is empty after dropping NaN values. Not enough data for analysis.")

    return df

def train_model(X_train, y_train, model_type, **kwargs):
    """Train the selected model"""
    model = None # Initialize model to None
    scaler = None
    sequence_length = None

    if X_train.empty or y_train.empty:
        st.error(f"❌ Training data is empty for {model_type}. Cannot train model.")
        return None, None, None # Consistent return for all branches

    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(
            learning_rate=kwargs.get('learning_rate', 0.1),
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=42
        )
        model.fit(X_train, y_train)

    elif model_type == "LightGBM":
        model = lgb.LGBMRegressor(
            learning_rate=kwargs.get('learning_rate', 0.1),
            num_leaves=kwargs.get('num_leaves', 31),
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

    elif model_type == "LSTM":
        # Reshape data for LSTM
        sequence_length = kwargs.get('sequence_length', 60)
        lstm_units = kwargs.get('lstm_units', 64)

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Check if X_train_scaled has enough data for sequences
        if len(X_train_scaled) < sequence_length:
            st.error(f"❌ Not enough data for LSTM sequence length {sequence_length}. Requires at least {sequence_length} data points. Current: {len(X_train_scaled)}")
            return None, None, None

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, sequence_length)

        # Check if sequences were created successfully
        if len(X_train_seq) == 0:
            st.error("❌ Failed to create LSTM sequences. Check data and sequence length.")
            return None, None, None

        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(32),
            Dense(1)
        ])

        model.compile(optimizer=Adam(), loss='mse')
        # verbose=0 to suppress training output in Streamlit
        try:
            model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
        except Exception as e:
            st.error(f"Error during LSTM training: {e}")
            return None, None, None

        return model, scaler, sequence_length # Return all three for LSTM

    elif model_type == "Prophet":
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': X_train.index,
            'y': y_train
        })

        if prophet_data.empty:
            st.error("Prophet data is empty. Cannot train model.")
            return None, None, None # Consistent return

        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_data)

    return model, None, None # Return None for scaler and seq_len for non-LSTM models

def create_forecast(model, X_test, model_type, forecast_days=7, **kwargs):
    """Create multi-day forecast"""
    y_pred = None
    forecasts = []

    if model is None:
        st.error("Prediction model is not trained.")
        return None, []

    if model_type in ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"]:
        # Check if X_test is not empty
        if not X_test.empty:
            y_pred = model.predict(X_test)

            # Multi-day forecast (simplified approach)
            last_features_df = X_test.iloc[-1:].copy()
            forecasts = []

            for _ in range(forecast_days):
                pred = model.predict(last_features_df)[0]
                forecasts.append(pred)
                # Update features for next prediction.
                # This is a simplification: only updating the 'Close' related features.
                # A more robust approach would simulate future values for all features.
                if 'Close' in last_features_df.columns:
                    last_features_df['Close'] = pred
                # Attempt to update lag features assuming 'Close_lag_1' is a common one.
                # This logic is highly simplified and might need refinement based on exact features.
                for col in last_features_df.columns:
                    if 'Close_lag_' in col:
                        lag_val = int(col.split('_')[-1])
                        # This assumes we update lags based on the new 'pred'
                        # For true multi-step forecasting, you'd roll all relevant features.
                        if lag_val == 1: # Update lag 1 with current pred
                            last_features_df[col] = pred
                        # More complex logic needed for other lags if they depend on each other.

        else:
            st.warning("Test data is empty for prediction.")
            return None, []


    elif model_type == "Prophet":
        # Prophet handles forecasting natively
        # Need a future dataframe that includes historical dates for y_pred matching X_test
        # and then extends into the future for forecasts.
        # It's better to predict the entire range and then split.
        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast_full = model.predict(future)

        # Align historical predictions with X_test index
        y_pred_prophet_series = forecast_full.set_index('ds')['yhat'].reindex(X_test.index)
        y_pred = y_pred_prophet_series.values

        # Future forecasts are the last 'forecast_days'
        forecasts = forecast_full['yhat'].iloc[-forecast_days:].values


    else:  # LSTM
        scaler = kwargs.get('scaler')
        sequence_length = kwargs.get('sequence_length', 60)

        if scaler is None:
            st.error("LSTM scaler not found.")
            return None, []

        if X_test.empty:
            st.warning("Test data is empty for LSTM prediction.")
            return None, []

        X_test_scaled = scaler.transform(X_test)

        def create_sequences_for_prediction(data_scaled, seq_length):
            X = []
            if len(data_scaled) < seq_length:
                return np.array(X) # Return empty if not enough data
            # Generate sequences for prediction of existing test set
            for i in range(seq_length, len(data_scaled) + 1):
                X.append(data_scaled[i-seq_length:i])
            return np.array(X)

        X_test_seq = create_sequences_for_prediction(X_test_scaled, sequence_length)

        if X_test_seq.shape[0] > 0:
            y_pred_scaled = model.predict(X_test_seq)

            # Inverse transform y_pred_scaled
            # Need to create a temporary array to inverse transform correctly as scaler expects multiple features
            temp_array = np.zeros((len(y_pred_scaled), X_test.shape[1]))
            temp_array[:, 0] = y_pred_scaled.flatten() # Assuming Close is the first feature
            y_pred = scaler.inverse_transform(temp_array)[:, 0]
        else:
            st.warning("Not enough test data to create LSTM sequences for prediction.")
            y_pred = np.array([]) # Ensure y_pred is an array, even if empty

        # Multi-day forecast for LSTM
        forecasts = []
        if len(X_test_scaled) >= sequence_length:
            # Use the very last sequence from the test set as the starting point for forecasting
            last_seq_for_forecast = X_test_scaled[-sequence_length:].copy()

            for _ in range(forecast_days):
                # Predict next value (scaled)
                pred_scaled = model.predict(last_seq_for_forecast.reshape(1, sequence_length, X_test.shape[1]))[0, 0]

                # Inverse transform the predicted scaled value to get actual price
                temp_array = np.zeros((1, X_test.shape[1]))
                temp_array[0, 0] = pred_scaled # Place prediction in the 'Close' position (index 0)
                pred_actual = scaler.inverse_transform(temp_array)[0, 0]
                forecasts.append(pred_actual)

                # Update the sequence for the next prediction
                # Shift all existing values one position to the left (removing the oldest)
                last_seq_for_forecast = np.roll(last_seq_for_forecast, -1, axis=0)
                # Add the new scaled prediction to the end of the sequence at the 'Close' feature position
                last_seq_for_forecast[-1, 0] = pred_scaled
                # NOTE: For a real application, you'd need to intelligently predict/update *all* features
                # for the next time step, not just the close price. This is a common simplification
                # in many single-feature LSTM forecasting examples.

        else:
            st.warning("Not enough data history to generate multi-day LSTM forecast.")

    return y_pred, forecasts

# Global variables to store results after button click
data_global = None
company_info_global = None
y_test_global = None
y_pred_global = None
forecasts_global = None
processed_data_global = None
feature_columns_global = None
model_global = None
scaler_global = None # For LSTM
seq_len_global = None # For LSTM

if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
    # Reset global variables at the start of analysis
    data_global = None
    company_info_global = None
    y_test_global = None
    y_pred_global = None
    forecasts_global = None
    processed_data_global = None
    feature_columns_global = None
    model_global = None
    scaler_global = None
    seq_len_global = None

    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Load data
        status_text.text("📊 Loading stock data...")
        progress_bar.progress(10)

        data, company_info = load_stock_data(ticker, start_date, end_date)
        if data is None or data.empty:
            st.error("❌ Failed to load stock data or data is empty.")
            st.stop()
        data_global = data # Store globally
        company_info_global = company_info # Store globally

        progress_bar.progress(30)
        status_text.text("✅ Data loaded successfully!")

        # Step 2: Load sentiment data if requested
        sentiment_data = None
        if use_sentiment:
            status_text.text("😊 Analyzing market sentiment...")
            sentiment_data = get_news_sentiment(ticker) # Now ticker is explicitly passed
            progress_bar.progress(45)

        # Step 3: Feature engineering
        status_text.text("⚙️ Engineering features...")
        processed_data = prepare_features(
            data,
            use_technical=use_technical_indicators,
            use_lags=use_lag_features,
            use_sentiment_data=sentiment_data
        )
        if processed_data.empty:
            st.error("❌ Not enough data after feature engineering to proceed. Try a longer date range.")
            st.stop()
        processed_data_global = processed_data # Store globally
        progress_bar.progress(60)

        # Step 4: Prepare training data
        status_text.text("🎯 Preparing training data...")
        feature_columns = [col for col in processed_data.columns if col != 'Target']
        if 'Target' not in processed_data.columns or len(feature_columns) == 0:
            st.error("❌ Target column or features not found after processing data.")
            st.stop()

        X = processed_data[feature_columns]
        y = processed_data['Target']

        if len(X) < 2: # Need at least 2 data points for train/test split
            st.error("❌ Not enough data points for training and testing after feature engineering. Please select a longer date range.")
            st.stop()

        # Ensure that test_size is not too large for the available data
        min_train_size = 1 # At least one data point for training
        min_test_size = 1  # At least one data point for testing
        if len(X) * (1 - test_size) < min_train_size or len(X) * test_size < min_test_size:
            st.error(f"❌ Data split (test size {int(test_size*100)}%) is too aggressive for {len(X)} data points. Adjust test size or data range.")
            st.stop()


        split_idx = int(len(X) * (1 - test_size))

        # Ensure split_idx leaves at least one sample for test set
        if split_idx >= len(X): # If split_idx is at or beyond the last index
            split_idx = len(X) - 1 # Set to the second to last element to ensure at least one test sample
            if split_idx <= 0: # If even after adjustment, there's no data for training/testing
                 st.error("❌ Data set is too small to perform a train/test split. Please choose a larger data range.")
                 st.stop()

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Store globally
        feature_columns_global = feature_columns
        y_test_global = y_test

        progress_bar.progress(75)

        # Step 5: Train model
        status_text.text(f"🤖 Training {model_choice} model...")

        model_kwargs = {}
        if model_choice == "Random Forest":
            model_kwargs = {'n_estimators': n_estimators, 'max_depth': max_depth}
        elif model_choice == "XGBoost":
            model_kwargs = {'learning_rate': learning_rate, 'n_estimators': n_estimators}
        elif model_choice == "LightGBM":
            model_kwargs = {'learning_rate': learning_rate, 'num_leaves': num_leaves}
        elif model_choice == "LSTM":
            model_kwargs = {'sequence_length': sequence_length, 'lstm_units': lstm_units}

        # Train model, handling multiple returns for LSTM
        model_result = train_model(X_train, y_train, model_choice, **model_kwargs)
        if model_result[0] is None: # Check if training failed (model is None)
            st.error(f"❌ {model_choice} model training failed. Please check your data or model parameters.")
            st.stop()

        model_global = model_result[0] # Store model globally
        scaler_global = model_result[1] # Store scaler globally (will be None for non-LSTM)
        seq_len_global = model_result[2] # Store seq_len globally (will be None for non-LSTM)

        progress_bar.progress(90)

        # Step 6: Make predictions
        status_text.text("🔮 Generating predictions...")
        # Pass all relevant model_kwargs, including scaler and seq_len for LSTM
        y_pred, forecasts = create_forecast(model_global, X_test, model_choice, forecast_days,
                                            scaler=scaler_global, sequence_length=seq_len_global)
        if y_pred is None and not forecasts: # Check if prediction failed
            st.error("❌ Prediction generation failed. Please check the model and data.")
            st.stop()

        y_pred_global = y_pred # Store globally
        forecasts_global = forecasts # Store globally

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)

        # Clear progress indicators
        progress_container.empty()

    # Display results in tabs - ALL TAB CONTENT SHOULD BE INSIDE THIS BUTTON BLOCK
    with tab1:  # Overview
        st.header(f"📊 {company_info_global['name']} Overview")

        # Company info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Company", company_info_global['name'])
        with col2:
            st.metric("Sector", company_info_global['sector'])
        with col3:
            if company_info_global['market_cap'] != 'N/A':
                market_cap_b = company_info_global['market_cap'] / 1e9
                st.metric("Market Cap", f"${market_cap_b:.1f}B")
            else:
                st.metric("Market Cap", "N/A")

        # Price overview
        st.subheader("💰 Price Summary")
        col1, col2, col3, col4 = st.columns(4)

        if data_global is not None and not data_global.empty:
            latest_price = float(data_global['Close'].iloc[-1])
            if len(data_global) > 1:
                price_change = float(data_global['Close'].iloc[-1] - data_global['Close'].iloc[-2])
                price_change_pct = (price_change / data_global['Close'].iloc[-2]) * 100
            else:
                price_change = 0.0
                price_change_pct = 0.0
            volume = int(data_global['Volume'].iloc[-1])

            col1.metric("Current Price", f"${latest_price:.2f}", f"{price_change:.2f} ({price_change_pct:.1f}%)")
            col2.metric("Volume", f"{volume:,}")
            col3.metric("52W High", f"${data_global['High'].max():.2f}")
            col4.metric("52W Low", f"${data_global['Low'].min():.2f}")
        else:
            st.warning("No historical data to display price summary.")

        # Interactive price chart
        st.subheader("📈 Interactive Price Chart")

        if data_global is not None and not data_global.empty:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Price & Volume'], # Removed 'Technical Indicators' subtitle here as it's not plotted in this section
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data_global.index,
                    open=data_global['Open'],
                    high=data_global['High'],
                    low=data_global['Low'],
                    close=data_global['Close'],
                    name="Price"
                ),
                row=1, col=1
            )

            # Volume
            fig.add_trace(
                go.Bar(x=data_global.index, y=data_global['Volume'], name="Volume", yaxis="y2"),
                row=2, col=1
            )

            fig.update_layout(
                title=f"{ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                showlegend=True,
                 xaxis_rangeslider_visible=False # Hide range slider for cleaner look
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available for price chart.")

        # Forecast summary
        if forecasts_global and data_global is not None and not data_global.empty:
            st.subheader("🎯 Forecast Summary")

            current_price = data_global['Close'].iloc[-1]
            forecast_change = forecasts_global[-1] - current_price
            forecast_change_pct = (forecast_change / current_price) * 100

            # Prediction box with styling
            prediction_html = f"""
            <div class="prediction-box">
                <h3>📊 {forecast_days}-Day Forecast</h3>
                <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                <p><strong>Predicted Price:</strong> ${forecasts_global[-1]:.2f}</p>
                <p><strong>Expected Change:</strong> ${forecast_change:.2f} ({forecast_change_pct:+.1f}%)</p>
            </div>
            """
            st.markdown(prediction_html, unsafe_allow_html=True)

            # Forecast table
            forecast_df = pd.DataFrame({
                'Day': range(1, len(forecasts_global) + 1),
                'Date': pd.date_range(start=data_global.index[-1] + timedelta(days=1), periods=len(forecasts_global), freq='D'),
                'Predicted Price': [f"${p:.2f}" for p in forecasts_global],
                'Change from Today': [f"${p - current_price:.2f} ({((p - current_price) / current_price) * 100:+.1f}%)" for p in forecasts_global]
            })

            st.dataframe(forecast_df, use_container_width=True)
        else:
            st.info("No forecast available. Run the prediction model first.")

    with tab2:  # Predictions
        st.header("🔮 Predictions & Forecasting")

        # Model performance metrics
        # Ensure y_test_global and y_pred_global are valid and have comparable lengths
        if y_test_global is not None and y_pred_global is not None and len(y_test_global) > 0 and len(y_pred_global) > 0 and len(y_test_global) == len(y_pred_global):
            mse = mean_squared_error(y_test_global, y_pred_global)
            mae = mean_absolute_error(y_test_global, y_pred_global)
            r2 = r2_score(y_test_global, y_pred_global)

            # Relative accuracy
            # Avoid division by zero if y_test contains 0 values, or handle very small values
            y_test_non_zero = y_test_global[y_test_global != 0]
            if not y_test_non_zero.empty:
                mape = np.mean(np.abs((y_test_global[y_test_global != 0] - y_pred_global[y_test_global != 0]) / y_test_non_zero)) * 100
                accuracy = 100 - mape
            else:
                mape = np.nan
                accuracy = np.nan


            st.subheader("📊 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² Score", f"{r2:.4f}")
            col2.metric("MAE", f"${mae:.2f}")
            col3.metric("MSE", f"{mse:.2f}")
            if not np.isnan(accuracy):
                col4.metric("Accuracy", f"{accuracy:.1f}%")
            else:
                col4.metric("Accuracy", "N/A")
        else:
            st.info("No prediction performance metrics available. Ensure model was trained and predictions generated successfully.")


        # Prediction vs Actual chart
        st.subheader("📈 Prediction vs Actual")

        fig = go.Figure()

        # Actual prices
        if y_test_global is not None and not y_test_global.empty:
            fig.add_trace(
                go.Scatter(
                    x=y_test_global.index,
                    y=y_test_global,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                )
            )
        else:
            st.warning("No actual test data to display.")

        # Predicted prices
        # Ensure y_pred_global is a numpy array for consistent behavior
        if y_pred_global is not None and isinstance(y_pred_global, np.ndarray) and len(y_test_global) == len(y_pred_global) and not y_test_global.empty:
            fig.add_trace(
                go.Scatter(
                    x=y_test_global.index,
                    y=y_pred_global,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
        else:
            st.warning("No predicted data to display or mismatch in lengths.")

        # Future forecasts
        if forecasts_global and data_global is not None and not data_global.empty:
            future_dates = pd.date_range(
                start=data_global.index[-1] + timedelta(days=1),
                periods=len(forecasts_global),
                freq='D'
            )

            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=forecasts_global,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='green', width=3),
                    marker=dict(size=6)
                )
            )
        else:
            st.warning("No future forecast data available.")

        fig.update_layout(
            title=f"{ticker} - Actual vs Predicted vs Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:  # Sentiment Analysis
        st.header("😊 Market Sentiment Analysis")

        if use_sentiment and sentiment_data is not None and not sentiment_data.empty:
            # Sentiment overview
            avg_sentiment = sentiment_data['sentiment'].mean()
            sentiment_trend = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Sentiment", f"{avg_sentiment:.3f}")
            col2.metric("Sentiment Trend", sentiment_trend)
            col3.metric("Data Points", len(sentiment_data))

            # Sentiment chart
            fig = go.Figure()

            colors = ['green' if s > 0.1 else 'red' if s < -0.1 else 'gray' for s in sentiment_data['sentiment']]

            fig.add_trace(
                go.Bar(
                    x=sentiment_data['date'],
                    y=sentiment_data['sentiment'],
                    marker_color=colors,
                    name='Daily Sentiment'
                )
            )

            fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Neutral")
            fig.add_hline(y=0.1, line_dash="dot", line_color="green", annotation_text="Positive Threshold")
            fig.add_hline(y=-0.1, line_dash="dot", line_color="red", annotation_text="Negative Threshold")

            fig.update_layout(
                title="Market Sentiment Over Time",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Sentiment distribution
            st.subheader("📊 Sentiment Distribution")

            sentiment_counts = sentiment_data['sentiment_label'].value_counts()

            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    marker_colors=['green', 'gray', 'red'] # Ensure order matches counts if possible
                )
            ])

            fig_pie.update_layout(title="Sentiment Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        else:
            st.info("🔄 Sentiment analysis not enabled or no sentiment data available. Enable it in the sidebar to see market emotion insights.")
            st.markdown("""
            **What is Sentiment Analysis?**

            Sentiment analysis examines news headlines, social media posts, and other text data to gauge market emotion:

            - **Positive sentiment** (>0.1): Optimistic market outlook
            - **Neutral sentiment** (-0.1 to 0.1): Balanced market sentiment
            - **Negative sentiment** (<-0.1): Pessimistic market outlook

            This data can be used as an additional feature in prediction models to improve accuracy.
            """)

    with tab4:  # Technical Analysis
        st.header("📈 Technical Analysis & Feature Importance")

        if use_technical_indicators and processed_data_global is not None and not processed_data_global.empty:
            # Technical indicators overview
            st.subheader("📊 Key Technical Indicators")

            latest_data = processed_data_global.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            # Check if columns exist before trying to access them
            if 'RSI' in processed_data_global.columns:
                rsi_val = latest_data['RSI']
                rsi_signal = "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Normal"
                col1.metric("RSI", f"{rsi_val:.1f}", rsi_signal)

            if 'MACD' in processed_data_global.columns and 'MACD_Signal' in processed_data_global.columns:
                macd_val = latest_data['MACD']
                macd_signal = "Bullish" if macd_val > latest_data['MACD_Signal'] else "Bearish"
                col2.metric("MACD", f"{macd_val:.3f}", macd_signal)

            if 'BB_Position' in processed_data_global.columns:
                bb_pos = latest_data['BB_Position']
                bb_signal = "Upper Band" if bb_pos > 0.8 else "Lower Band" if bb_pos < 0.2 else "Middle"
                col3.metric("Bollinger Position", f"{bb_pos:.2f}", bb_signal)

            if 'Stoch_K' in processed_data_global.columns:
                stoch_k = latest_data['Stoch_K']
                stoch_signal = "Oversold" if stoch_k < 20 else "Overbought" if stoch_k > 80 else "Normal"
                col4.metric("Stochastic %K", f"{stoch_k:.1f}", stoch_signal)

            # Technical indicators chart
            st.subheader("📈 Technical Indicators Chart")

            # Ensure data_global is available for the Close Price trace
            if data_global is not None and not data_global.empty:
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=['Price & Bollinger Bands', 'RSI', 'MACD', 'Stochastic Oscillator'],
                    vertical_spacing=0.05,
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )

                # Price and Bollinger Bands
                fig.add_trace(
                    go.Scatter(x=data_global.index, y=data_global['Close'], name='Close Price', line=dict(color='blue')),
                    row=1, col=1
                )

                if 'BB_Upper' in processed_data_global.columns and 'BB_Lower' in processed_data_global.columns:
                    fig.add_trace(
                        go.Scatter(x=processed_data_global.index, y=processed_data_global['BB_Upper'],
                                 name='BB Upper', line=dict(color='red', dash='dash')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=processed_data_global.index, y=processed_data_global['BB_Lower'],
                                 name='BB Lower', line=dict(color='red', dash='dash')),
                        row=1, col=1
                    )

                # RSI
                if 'RSI' in processed_data_global.columns:
                    fig.add_trace(
                        go.Scatter(x=processed_data_global.index, y=processed_data_global['RSI'], name='RSI'),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # MACD
                if 'MACD' in processed_data_global.columns and 'MACD_Signal' in processed_data_global.columns:
                    fig.add_trace(
                        go.Scatter(x=processed_data_global.index, y=processed_data_global['MACD'], name='MACD'),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=processed_data_global.index, y=processed_data_global['MACD_Signal'], name='Signal'),
                        row=3, col=1
                    )

                # Stochastic
                if 'Stoch_K' in processed_data_global.columns and 'Stoch_D' in processed_data_global.columns:
                    fig.add_trace(
                        go.Scatter(x=processed_data_global.index, y=processed_data_global['Stoch_K'], name='%K'),
                        row=4, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=processed_data_global.index, y=processed_data_global['Stoch_D'], name='%D'),
                        row=4, col=1
                    )
                    fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)

                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to compute technical indicators.")
        else:
            st.info("🔄 Technical indicators not enabled or no processed data available. Enable them in the sidebar to see detailed analysis.")


        # Feature importance (for tree-based models)
        if model_choice in ["Random Forest", "XGBoost", "LightGBM"] and model_global is not None and feature_columns_global is not None:
            st.subheader("🎯 Feature Importance Analysis")

            try:
                if hasattr(model_global, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_columns_global,
                        'Importance': model_global.feature_importances_
                    }).sort_values('Importance', ascending=True)

                    # Top 15 most important features
                    top_features = importance_df.tail(15)

                    fig = go.Figure(go.Bar(
                        x=top_features['Importance'],
                        y=top_features['Feature'],
                        orientation='h',
                        marker_color='skyblue'
                    ))

                    fig.update_layout(
                        title="Top 15 Most Important Features",
                        xaxis_title="Importance Score",
                        yaxis_title="Features",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Feature importance table
                    st.dataframe(importance_df.sort_values('Importance', ascending=False), height=300)

            except Exception as e:
                st.error(f"Could not generate feature importance: {str(e)}")
        elif model_global is None:
            st.info(f"Feature importance is not available for {model_choice} model, or model was not trained successfully.")
        else:
            st.info(f"Feature importance is only available for tree-based models (Random Forest, XGBoost, LightGBM).")


        # Model comparison section
        st.subheader("⚖️ Model Comparison")

        comparison_data = {
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'LSTM', 'Prophet'],
            'Training Time': ['Fast', 'Medium', 'Medium', 'Fast', 'Slow', 'Medium'],
            'Interpretability': ['High', 'Medium', 'Medium', 'Medium', 'Low', 'Medium'],
            'Complexity': ['Low', 'Medium', 'High', 'Medium', 'High', 'Medium'],
            'Best For': ['Linear trends', 'Non-linear patterns', 'Competition', 'Speed & accuracy', 'Sequential data', 'Seasonal patterns']
        }

        comparison_df = pd.DataFrame(comparison_data)

        # Highlight the selected model
        def highlight_selected_model(row):
            if row['Model'] == model_choice:
                return ['background-color: lightblue'] * len(row)
            return [''] * len(row)

        st.dataframe(
            comparison_df.style.apply(highlight_selected_model, axis=1),
            use_container_width=True
        )

    with tab5:  # Settings & Export
        st.header("⚙️ Settings & Export Tools")

        # Export section
        st.subheader("📤 Export Data & Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Download Options:**")

            # Prepare data for download
            if forecasts_global and data_global is not None:
                forecast_df = pd.DataFrame({
                    'Date': pd.date_range(start=data_global.index[-1] + timedelta(days=1),
                                        periods=len(forecasts_global), freq='D'),
                    'Predicted_Price': forecasts_global
                })

                # Download forecast data
                csv_forecast = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Forecast CSV",
                    data=csv_forecast,
                    file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No forecast data available for download.")

            # Download historical data
            if data_global is not None and not data_global.empty:
                csv_data = data_global.to_csv()
                st.download_button(
                    label="📈 Download Historical Data",
                    data=csv_data,
                    file_name=f"{ticker}_historical_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No historical data available for download.")

            # Model performance report
            if (y_test_global is not None and y_pred_global is not None and
                len(y_test_global) > 0 and len(y_pred_global) > 0 and
                len(y_test_global) == len(y_pred_global) and not y_test_global.empty and
                data_global is not None and not data_global.empty and forecasts_global):

                mse = mean_squared_error(y_test_global, y_pred_global)
                mae = mean_absolute_error(y_test_global, y_pred_global)
                r2 = r2_score(y_test_global, y_pred_global)
                y_test_non_zero = y_test_global[y_test_global != 0]
                if not y_test_non_zero.empty:
                    mape = np.mean(np.abs((y_test_global[y_test_global != 0] - y_pred_global[y_test_global != 0]) / y_test_non_zero)) * 100
                    accuracy = 100 - mape
                else:
                    accuracy = np.nan

                report_text = f"""
Stock Prediction Report
=======================

Ticker: {ticker}
Company: {company_info_global.get('name', 'N/A')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Used: {model_choice}

Performance Metrics:
- R² Score: {r2:.4f}
- Mean Absolute Error: ${mae:.2f}
- Mean Squared Error: {mse:.2f}
- Accuracy: {accuracy:.1f}%

Forecast Summary:
- Current Price: ${data_global['Close'].iloc[-1]:.2f}
- {forecast_days}-Day Forecast: ${forecasts_global[-1]:.2f}
- Expected Change: ${forecasts_global[-1] - data_global['Close'].iloc[-1]:.2f} ({((forecasts_global[-1] - data_global['Close'].iloc[-1]) / data_global['Close'].iloc[-1]) * 100:+.1f}%)

Data Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Total Data Points: {len(data_global)}
Training Data: {len(X_train)} points
Testing Data: {len(X_test)} points
                """

                st.download_button(
                    label="📋 Download Report",
                    data=report_text,
                    file_name=f"{ticker}_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            else:
                st.info("No performance report available. Ensure prediction was successful and data is complete.")

        with col2:
            st.write("**Visualization Export:**")

            st.info("💡 Tip: Right-click on any chart and select 'Download plot as a PNG' to save visualizations.")

            # Model settings summary
            st.write("**Current Model Settings:**")
            settings_info = {
                'Model': model_choice,
                'Test Size': f"{int(test_size * 100)}%",
                'Forecast Days': forecast_days,
                'Technical Indicators': "✅" if use_technical_indicators else "❌",
                'Lag Features': "✅" if use_lag_features else "❌",
                'Sentiment Analysis': "✅" if use_sentiment else "❌"
            }

            for key, value in settings_info.items():
                st.write(f"- **{key}:** {value}")

        # Advanced settings
        st.subheader("🔧 Advanced Configuration")

        with st.expander("⚠️ Advanced Model Parameters"):
            st.warning("⚠️ These settings affect model performance. Change only if you understand the implications.")

            st.write("**Feature Engineering Options:**")
            col1, col2 = st.columns(2)

            with col1:
                st.checkbox("Include Volume Features", True, disabled=True)
                st.checkbox("Include Price Ratios", True, disabled=True)
                st.checkbox("Include Volatility Measures", True, disabled=True)

            with col2:
                st.checkbox("Include Moving Averages", True, disabled=True)
                st.checkbox("Include Momentum Indicators", True, disabled=True)
                st.checkbox("Include Market Breadth", False, disabled=True)

            st.write("**Data Processing:**")
            st.selectbox("Data Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler"], disabled=True)
            st.selectbox("Missing Value Strategy", ["Forward Fill", "Backward Fill", "Interpolation"], disabled=True)

        # API and data sources info
        st.subheader("📊 Data Sources & API Information")

        st.markdown("""
        **Data Sources:**
        - **Stock Data:** Yahoo Finance API (yfinance)
        - **Company Information:** Yahoo Finance
        - **Sentiment Data:** Simulated (in production, would use NewsAPI, Reddit API, or Twitter API)

        **API Limitations:**
        - Yahoo Finance: Free tier with rate limits
        - Real-time data may have 15-20 minute delays
        - Sentiment analysis is currently simulated for demo purposes

        **For Production Use:**
        - Consider premium data providers (Alpha Vantage, Quandl, Bloomberg API)
        - Implement proper sentiment analysis with NewsAPI or social media APIs
        - Add real-time data streaming capabilities
        - Include additional fundamental analysis data
        """)

        # Performance tips
        st.subheader("💡 Performance Tips")

        st.markdown("""
        **Improving Prediction Accuracy:**

        1. **Data Quality:** Use longer historical periods (3+ years) for better pattern recognition
        2. **Feature Engineering:** Enable all technical indicators and lag features
        3. **Model Selection:** Try different models - Random Forest and XGBoost often perform well
        4. **Sentiment Integration:** Enable sentiment analysis for additional market context
        5. **Regular Updates:** Retrain models with fresh data weekly or monthly

        **Model-Specific Tips:**
        - **Linear Regression:** Works best for trending stocks with clear patterns
        - **Random Forest:** Excellent for non-linear relationships, robust to outliers
        - **XGBoost/LightGBM:** Best overall performance, handles complex patterns
        - **LSTM:** Great for capturing sequential patterns in volatile stocks
        - **Prophet:** Excellent for stocks with strong seasonal patterns
        """)

    # Data preview, now correctly placed inside the button's execution block
    with st.expander("📋 View Raw Data (Last 100 Rows)"):
        if data_global is not None and not data_global.empty:
            st.dataframe(data_global.tail(100), height=300)
        else:
            st.info("No raw data to display. Please run the analysis first.")

# Footer - this should be outside the if st.button block so it's always displayed.
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>📊 Advanced Stock Prediction Dashboard | Built with Streamlit & Python</p>
        <p>⚠️ Disclaimer: This tool is for educational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)