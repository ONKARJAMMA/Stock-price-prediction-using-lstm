import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
import seaborn as sns

# Function to download stock data
def download_data(stock, start, end):
    google_data = yf.download(stock, start, end)
    return google_data

# Function to prepare the data
def prepare_data(google_data):
    # Prepare Adj Close Price
    Adj_close_price = google_data[['Adj Close']]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(Adj_close_price)
    
    # Prepare x_data and y_data
    x_data, y_data = [], []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])
    
    return np.array(x_data), np.array(y_data), scaler

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to evaluate model performance
def evaluate_model(model, x_data, y_data, scaler, title):
    # Splitting data
    splitting_len = int(len(x_data) * 0.7)
    x_test = x_data[splitting_len:]
    y_test = y_data[splitting_len:]

    # Make predictions
    predictions = model.predict(x_test)

    # Inverse transform predictions and actual values
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test)

    # Convert to binary classes
    y_test_binary = np.where(np.diff(inv_y_test.flatten()) > 0, 1, 0)
    predictions_binary = np.where(np.diff(inv_predictions.flatten()) > 0, 1, 0)

    # Calculate confusion matrix and metrics
    cm = confusion_matrix(y_test_binary, predictions_binary)
    accuracy = accuracy_score(y_test_binary, predictions_binary)
    precision = precision_score(y_test_binary, predictions_binary)
    recall = recall_score(y_test_binary, predictions_binary)
    f1 = f1_score(y_test_binary, predictions_binary)

    print(f"{title} Model Evaluation")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, title)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_binary, predictions_binary)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{title} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.show()

# Main function to run everything
def main(stock):
    # Define time period
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)

    # Download data
    google_data = download_data(stock, start, end)

    # Prepare the data
    x_data, y_data, scaler = prepare_data(google_data)

    # Load the model
    model = load_model("Latest_stock_price_model.keras")

    # Evaluate the model
    evaluate_model(model, x_data, y_data, scaler, title="Stock Price Prediction")

if __name__ == "__main__":
    # Specify the stock you want to analyze
    stock = "GOOG"  # Change this to your desired stock
    main(stock)
