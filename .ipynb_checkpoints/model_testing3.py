# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime

# Load the saved LSTM model
model = load_model("Latest_stock_price_model.keras")

# Set up date range for new test data
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)  # Last year for more recent data

# Download recent data for testing
stock = "GOOG"
google_data = yf.download(stock, start=start, end=end)

# Prepare data for testing
Adj_close_price = google_data[['Adj Close']]

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(Adj_close_price)

# Create sequences for testing
x_test = []
for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i-100:i])
x_test = np.array(x_test)

# Make predictions using the model
predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)  # Inverse scaling for predictions

# Prepare actual test data for RMSE calculation
actual_prices = Adj_close_price['Adj Close'][100:].values.reshape(-1, 1)  # Actual prices after the first 100
inv_actual_prices = scaler.inverse_transform(actual_prices)

# Calculate RMSE
rmse = np.sqrt(np.mean((inv_predictions.flatten() - inv_actual_prices.flatten()) ** 2))
print("RMSE:", rmse)

# Prepare data for plotting
plotting_data = pd.DataFrame({
    'actual': inv_actual_prices.flatten(),  # Flatten to 1D array
    'predictions': inv_predictions.flatten()  # Flatten to 1D array
}, index=google_data.index[100:])

# Plotting function
def plot_graph(figsize, data, title):
    plt.figure(figsize=figsize)
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(data.columns)
    plt.show()

# Plot original vs. predicted test data
plot_graph((15, 6), plotting_data, "Actual vs Predicted Test Data")
