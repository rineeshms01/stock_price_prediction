# 1. Imports
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide", page_icon="📈")

# 2. Streamlit UI
st.title("📈 Stock Price Prediction Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    popular_tickers = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Tesla": "TSLA",
        "Nvidia": "NVDA",
        "Meta": "META",
        "Netflix": "NFLX",
        "Intel": "INTC",
        "IBM": "IBM",
        "Adobe": "ADBE",
        "Salesforce": "CRM",
        "Berkshire Hathaway": "BRK-B",
        "JP Morgan": "JPM",
        "Visa": "V",
        "Coca-Cola": "KO",
        "Walmart": "WMT",
        "McDonald's": "MCD",
        "Procter & Gamble": "PG"
    }
    selected_ticker_name = st.selectbox("Select popular stock:", list(popular_tickers.keys()), index=0)
    ticker = st.text_input("Or enter custom ticker:", value=popular_tickers[selected_ticker_name])

    st.subheader("Date Range")
    date_presets = st.radio("Presets:", ["1 Year", "3 Years", "5 Years", "Max", "Custom"], horizontal=True)

    if date_presets != "Custom":
        if date_presets == "1 Year":
            start_date = datetime.now() - timedelta(days=365)
        elif date_presets == "3 Years":
            start_date = datetime.now() - timedelta(days=3*365)
        elif date_presets == "5 Years":
            start_date = datetime.now() - timedelta(days=5*365)
        else:
            start_date = datetime(2000, 1, 1)
        end_date = datetime.now()
    else:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365), min_value=datetime(2000, 1, 1))
        end_date = st.date_input("End Date", value=datetime.now(), min_value=datetime(2000, 1, 1))
        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()

    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100

with st.expander("ℹ️ How to use this dashboard"):
    st.markdown("""**1. Select a stock** - Choose from popular stocks or enter any valid ticker symbol  
    **2. Set date range** - Use presets or select custom dates  
    **3. Analyze results** - View predictions, performance metrics, and insights  
    *Tip: Hover over charts for interactive exploration*""")

@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# 3. Streamlit button to start the prediction process
start_button = st.button("Start Prediction Process")

if start_button:
    with st.spinner("Fetching stock data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(101):
            progress_bar.progress(i)
            status_text.text(f"Downloading data: {i}%")
            time.sleep(0.01)

        data = load_data(ticker, start_date, end_date)

        if data is None:
            st.stop()

        progress_bar.empty()
        status_text.text("Data loaded successfully!")

    st.write(f"### 📊 {ticker} Stock Data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

    # 3b. Show summary stats
    if not data.empty and 'Close' in data.columns:
        col1, col2, col3 = st.columns(3)
        try:
            close_series = data['Close'].dropna()
            first_close = float(close_series.iloc[0]) if not close_series.empty else None
            last_close = float(close_series.iloc[-1]) if not close_series.empty else None
            total_days = len(close_series)

            if first_close is not None:
                col1.metric("First Close", f"${first_close:.2f}")
            else:
                col1.metric("First Close", "N/A")

            if last_close is not None:
                col2.metric("Last Close", f"${last_close:.2f}")
            else:
                col2.metric("Last Close", "N/A")

            col3.metric("Total Days", total_days)
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
    else:
        st.error("No valid closing price data available.")
        st.stop()

    # Preview raw data
    with st.expander("View Raw Data"):
        if not data.empty:
            st.dataframe(data.sort_index(ascending=False).style.format("{:.2f}"), height=300)
        else:
            st.warning("No data available to display")

    # 4. Feature Engineering
    def create_features(df):
        df = df.copy()
        df['Target'] = df['Close'].shift(-1)
        for window in [5, 10, 20, 50]:
            df[f'{window}-day MA'] = df['Close'].rolling(window=window).mean()
        df['Daily Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily Return'].rolling(20).std()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'] = compute_macd(df['Close'])
        df.dropna(inplace=True)
        return df

    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def compute_macd(series, slow=26, fast=12, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    with st.spinner("Creating features..."):
        processed_data = create_features(data)

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                '5-day MA', '10-day MA', '20-day MA', '50-day MA',
                'Daily Return', 'Volatility', 'RSI', 'MACD', 'Signal']
    X = processed_data[features]
    y = processed_data['Target']

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 6. Train model
    with st.spinner("Training model..."):
        progress_bar = st.progress(0)

        # Only using Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        progress_bar.empty()

    # 7. Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Relative accuracy metric
    accuracy_percentage = 100 - np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100

    # Plot actual vs predicted
    fig = plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual", color='blue')
    plt.plot(y_test.index, y_pred, label="Predicted", color='red')
    plt.title(f"{ticker} Stock Price - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("R²", f"{r2:.2f}")

    st.metric("Relative Accuracy (%)", f"{accuracy_percentage:.2f}%")

    # Predict next day's price
    next_day_pred = model.predict([X.iloc[-1].values])[0]
    st.write(f"### Predicted next day's closing price: ${next_day_pred:.2f}")
# stock_price_prediction
