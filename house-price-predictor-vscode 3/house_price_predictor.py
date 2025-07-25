# house_price_predictor.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os

# Confirm dataset exists
if not os.path.exists("house_data.csv"):
    raise FileNotFoundError("⚠️ 'house_data.csv' not found. Make sure you're running this script from the correct folder.")

# Load the dataset
df = pd.read_csv("house_data.csv")

# Encode 'location'
df = pd.get_dummies(df, columns=["location"], drop_first=True)

# Features and target
X = df.drop(columns=["price"])
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Evaluation:")
print(f"   - Mean Absolute Error: ${mae:,.2f} USD")
print(f"   - R² Score: {r2:.3f}")

# Plot predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.title("Predicted vs Actual House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
plt.grid(True)
plt.tight_layout()
plt.savefig("price_plot.png")
plt.show()

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
