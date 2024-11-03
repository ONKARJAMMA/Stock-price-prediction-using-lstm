import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor App")

# Input for the stock ticker
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set date range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start=start, end=end)

# Load the pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(google_data)

# Set splitting length for test data
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data['Adj Close'][splitting_len:])

# Define plotting function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label="Moving Average")
    plt.plot(full_data['Adj Close'], 'b', label="Close Price")
    if extra_data:
        plt.plot(extra_dataset, 'g', label="Extra Moving Average")
    plt.legend()
    return fig

# Plot moving averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Adj Close'].rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Adj Close'].rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Adj Close'].rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price with MAs for 100 and 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scale and reshape data for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict using the model
predictions = model.predict(x_data)

# Inverse transform to get original scale
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create a DataFrame for plotting
plotting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_predictions.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

# Display original vs predicted values
st.subheader("Original Values vs Predicted Values")
st.write(plotting_data)

# Plot original close price vs predicted close price
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data['Adj Close'][:splitting_len + 100], plotting_data], axis=0))
plt.legend(["Data (Not Used)", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)
