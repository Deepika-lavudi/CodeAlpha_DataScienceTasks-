# ==========================================
# CODEALPHA TASK 3
# Car Price Prediction using Machine Learning
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------
# Load Dataset
# ------------------------------------------
df = pd.read_csv("car data.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ------------------------------------------
# Feature Engineering
# ------------------------------------------
current_year = 2026
df["Car_Age"] = current_year - df["Year"]

# Drop unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# ------------------------------------------
# Convert Categorical Data into Numbers
# ------------------------------------------
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

print("\nProcessed Dataset")
print(df.head())

# ------------------------------------------
# Correlation Heatmap
# ------------------------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------
# Selling Price Distribution
# ------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Selling_Price"], bins=30, kde=True)
plt.title("Selling Price Distribution")
plt.show()

# ------------------------------------------
# Present Price vs Selling Price
# ------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=df["Present_Price"],
    y=df["Selling_Price"]
)
plt.title("Present Price vs Selling Price")
plt.show()

# ------------------------------------------
# Car Age vs Selling Price
# ------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=df["Car_Age"],
    y=df["Selling_Price"]
)
plt.title("Car Age vs Selling Price")
plt.show()

# ------------------------------------------
# Prepare Features and Target
# ------------------------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ------------------------------------------
# Split Dataset
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------
# Train Random Forest Model
# ------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------
# Predictions
# ------------------------------------------
predictions = model.predict(X_test)

# ------------------------------------------
# Evaluation
# ------------------------------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n========== Model Performance ==========")
print("Mean Absolute Error :", round(mae,3))
print("Mean Squared Error  :", round(mse,3))
print("Root Mean Squared Error :", round(rmse,3))
print("R2 Score :", round(r2,3))

# ------------------------------------------
# Actual vs Predicted Plot
# ------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

# ------------------------------------------
# Feature Importance
# ------------------------------------------
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(10,6)
)

plt.title("Feature Importance")
plt.show()

# ------------------------------------------
# Predict Price of a Sample Car
# ------------------------------------------
sample = X.iloc[[0]]

pred