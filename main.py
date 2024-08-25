import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_excel('Coffee_Shop_Sales.xlsx')

# Convert transaction_date to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Aggregate the data by date and product category
df_grouped = df.groupby(['transaction_date', 'product_category']).agg({'transaction_qty': 'sum'}).reset_index()

# Pivot the data to get time series for each product category
df_pivot = df_grouped.pivot(index='transaction_date', columns='product_category', values='transaction_qty').fillna(0)


# Function to fit ARIMA model and predict next day's sales for a category
def predict_next_day_sales(series):
    # Differencing to make the series stationary
    series_diff = series.diff().dropna()

    # Determine p, d, q parameters (can be automated using AIC/BIC)
    model = ARIMA(series_diff, order=(1, 1, 1))  # Example: ARIMA(1, 1, 1)
    model_fit = model.fit()

    # Forecast the next value (revert differencing)
    forecast = model_fit.forecast()[0]
    predicted_sales = series.iloc[-1] + forecast
    return predicted_sales


# Store predictions for each category
predictions = {}

for category in df_pivot.columns:
    series = df_pivot[category]
    predictions[category] = predict_next_day_sales(series)

# Sort categories by predicted sales
sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

# Output the ranked categories
for category, pred_sales in sorted_predictions:
    print(f"Category: {category}, Predicted Sales: {pred_sales}")



















# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score
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
# # Function to fit ARIMA model and predict next day's sales for a category
# def predict_next_day_sales(series):
#     # Differencing to make the series stationary
#     series_diff = series.diff().dropna()
#
#     # Determine p, d, q parameters (can be automated using AIC/BIC)
#     model = ARIMA(series_diff, order=(1, 1, 1))  # Example: ARIMA(1, 1, 1)
#     model_fit = model.fit()
#
#     # Forecast the next value (revert differencing)
#     forecast = model_fit.forecast()[0]
#     predicted_sales = series.iloc[-1] + forecast
#     return predicted_sales
#
# # Function to classify the prediction as "accurate" or "inaccurate"
# def classify_prediction(actual, predicted, tolerance=0.1):
#     lower_bound = actual * (1 - tolerance)
#     upper_bound = actual * (1 + tolerance)
#     return 1 if lower_bound <= predicted <= upper_bound else 0
#
# # Store predictions for each category and track actual values
# predictions = {}
# actuals = {}
#
# for category in df_pivot.columns:
#     series = df_pivot[category]
#
#     # Split the data into training (80%) and testing (20%) sets
#     train_size = int(len(series) * 0.8)
#     train_series, test_series = series[:train_size], series[train_size:]
#
#     # Predict on the last value of the training series
#     predicted_sales = predict_next_day_sales(train_series)
#     actual_sales = test_series.iloc[0]
#
#     predictions[category] = predicted_sales
#     actuals[category] = actual_sales
#
# # Convert to lists for easy comparison
# predicted_values = list(predictions.values())
# actual_values = list(actuals.values())
#
# # Classify the predictions (1 for accurate, 0 for inaccurate)
# predicted_classes = [classify_prediction(actual, predicted) for actual, predicted in zip(actual_values, predicted_values)]
# actual_classes = [1] * len(predicted_classes)  # All actual values are assumed "accurate"
#
# # Compute accuracy and confusion matrix
# accuracy = accuracy_score(actual_classes, predicted_classes)
# conf_matrix = confusion_matrix(actual_classes, predicted_classes)
#
# # Output accuracy and confusion matrix
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print("Confusion Matrix:")
# print(conf_matrix)
#
# # Output the predicted and actual sales for each category
# for category, (pred_sales, actual_sales) in zip(predictions.keys(), zip(predicted_values, actual_values)):
#     print(f"Category: {category}, Predicted Sales: {pred_sales}, Actual Sales: {actual_sales}")


# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
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
#
# # Function to split the data into training and testing sets
# def train_test_split_series(series, train_size=0.8):
#     train_size = int(len(series) * train_size)
#     train, test = series[:train_size], series[train_size:]
#     return train, test
#
#
# # Function to fit ARIMA model and predict sales
# def predict_sales(series_train, series_test):
#     # Differencing to make the series stationary
#     series_diff = series_train.diff().dropna()
#
#     # Fit the ARIMA model (p, d, q parameters can be tuned)
#     model = ARIMA(series_diff, order=(1, 1, 1))
#     model_fit = model.fit()
#
#     # Forecast the next values for the testing period
#     forecast_diff = model_fit.forecast(steps=len(series_test))
#
#     # Convert differenced forecast back to original scale
#     forecast = series_train.iloc[-1] + forecast_diff.cumsum()
#
#     return forecast
#
#
# # Store predictions and calculate accuracy for each category
# for category in df_pivot.columns:
#     series = df_pivot[category]
#
#     # Split the data into training and testing sets
#     train, test = train_test_split_series(series)
#
#     # Predict sales for the testing period
#     predicted_sales = predict_sales(train, test)
#
#     # Plot the actual vs. predicted sales
#     plt.figure(figsize=(12, 6))
#     plt.plot(train.index, train, label='Training Data')
#     plt.plot(test.index, test, label='Actual Sales', color='green')
#     plt.plot(test.index, predicted_sales, label='Predicted Sales', color='red')
#     plt.title(f'Sales Prediction for Category: {category}')
#     plt.xlabel('Date')
#     plt.ylabel('Sales Quantity')
#     plt.legend()
#     plt.show()
#
#     # Calculate accuracy (Mean Squared Error)
#     mse = mean_squared_error(test, predicted_sales)
#     accuracy = 100 - np.sqrt(mse) / np.mean(test) * 100
#     print(f"Category: {category}, Prediction Accuracy: {accuracy:.2f}%\n")


