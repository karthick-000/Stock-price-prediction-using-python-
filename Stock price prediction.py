# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Step 1: Data Collection
ticker = 'AAPL'  # Example stock
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

# Step 2: Data Preprocessing
data['Close'] = data['Adj Close']  # Using adjusted close price for prediction
data.drop(['Adj Close'], axis=1, inplace=True)

# Feature Engineering: Create Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data.dropna(inplace=True)

# Step 3: Prepare data for modeling
X = data[['MA50', 'MA200']]  # You can add more features like Volume
y = data['Close']

# Scaling data (Optional)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 4: Model Selection & Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Step 7: Visualization
plt.figure(figsize=(10,6))
plt.plot(data.index[-len(y_test):], y_test, label="Actual Prices")
plt.plot(data.index[-len(y_test):], y_pred, label="Predicted Prices", linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.legend()
plt.show()
