# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import LabelEncoder
#
# # Load your dataset
# df = pd.read_excel('Coffee_Shop_Sales.xlsx')
#
# # Convert transaction_date to datetime
# df['transaction_date'] = pd.to_datetime(df['transaction_date'])
#
# # Aggregate the data by date and product category
# df_grouped = df.groupby(['transaction_date', 'product_category']).agg({'transaction_qty': 'sum'}).reset_index()
#
# # Pivot the data to get time series for each product category
# df_pivot = df_grouped.pivot(index='transaction_date', columns='product_category', values='transaction_qty').fillna(0)
#
# # Function to prepare the data for LSTM
# def prepare_lstm_data(series, n_steps):
#     X, y = [], []
#     for i in range(len(series)):
#         end_idx = i + n_steps
#         if end_idx > len(series) - 1:
#             break
#         X.append(series[i:end_idx])
#         y.append(series[end_idx])
#     return np.array(X), np.array(y)
#
# # Function to fit LSTM model and predict next day's sales for a category
# def predict_next_day_sales_lstm(series, n_steps=5, epochs=50, batch_size=32):
#     # Normalize the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))
#
#     # Prepare the data for LSTM
#     X, y = prepare_lstm_data(series_scaled, n_steps)
#     X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for LSTM input
#
#     # Build LSTM model
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
#     model.add(Dense(1))
#     model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
#
#     # Train the model
#     model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
#
#     # Prepare the input for prediction (last n_steps values)
#     input_data = series_scaled[-n_steps:].reshape((1, n_steps, 1))
#
#     # Make prediction and inverse the normalization
#     forecast_scaled = model.predict(input_data, verbose=0)
#     forecast = scaler.inverse_transform(forecast_scaled)[0, 0]
#
#     return forecast
#
# # Store predictions for each category
# predictions = {}
#
# for category in df_pivot.columns:
#     series = df_pivot[category]
#     predictions[category] = predict_next_day_sales_lstm(series)
#
# # Sort categories by predicted sales
# sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
#
# # Output the ranked categories
# for category, pred_sales in sorted_predictions:
#     print(f"Category: {category}, Predicted Sales: {pred_sales}")


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score

# Load your dataset
df = pd.read_excel('Coffee_Shop_Sales.xlsx')

# Convert transaction_date to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Aggregate the data by date and product category
df_grouped = df.groupby(['transaction_date', 'product_category']).agg({'transaction_qty': 'sum'}).reset_index()

# Pivot the data to get time series for each product category
df_pivot = df_grouped.pivot(index='transaction_date', columns='product_category', values='transaction_qty').fillna(0)


# Function to prepare the data for LSTM
def prepare_lstm_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)):
        end_idx = i + n_steps
        if end_idx > len(series) - 1:
            break
        X.append(series[i:end_idx])
        y.append(series[end_idx])
    return np.array(X), np.array(y)


# Function to fit LSTM model and predict next day's sales for a category
def predict_next_day_sales_lstm(series, n_steps=5, epochs=50, batch_size=32):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    # Split the data into training (80%) and testing (20%) sets
    train_size = int(len(series_scaled) * 0.8)
    train_series, test_series = series_scaled[:train_size], series_scaled[train_size:]

    # Prepare the data for LSTM
    X_train, y_train = prepare_lstm_data(train_series, n_steps)
    X_test, y_test = prepare_lstm_data(test_series, n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshaping for LSTM input
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict the next day's sales for the test set
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    return y_pred.flatten(), y_test.flatten()


# Function to classify the prediction as "accurate" or "inaccurate"
def classify_prediction(actual, predicted, tolerance=0.1):
    lower_bound = actual * (1 - tolerance)
    upper_bound = actual * (1 + tolerance)
    return 1 if lower_bound <= predicted <= upper_bound else 0


# Store predictions and actuals for each category
predictions = {}
actuals = {}

for category in df_pivot.columns:
    series = df_pivot[category]
    predicted_sales, actual_sales = predict_next_day_sales_lstm(series)

    # For simplicity, use the first value from the test set
    predictions[category] = predicted_sales[0]
    actuals[category] = actual_sales[0]

# Convert to lists for easy comparison
predicted_values = list(predictions.values())
actual_values = list(actuals.values())

# Classify the predictions (1 for accurate, 0 for inaccurate)
predicted_classes = [classify_prediction(actual, predicted) for actual, predicted in
                     zip(actual_values, predicted_values)]
actual_classes = [1] * len(predicted_classes)  # All actual values are assumed "accurate"

# Compute accuracy and confusion matrix
accuracy = accuracy_score(actual_classes, predicted_classes)
conf_matrix = confusion_matrix(actual_classes, predicted_classes)

# Output accuracy and confusion matrix
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Output the predicted and actual sales for each category
for category, (pred_sales, actual_sales) in zip(predictions.keys(), zip(predicted_values, actual_values)):
    print(f"Category: {category}, Predicted Sales: {pred_sales}, Actual Sales: {actual_sales}")
