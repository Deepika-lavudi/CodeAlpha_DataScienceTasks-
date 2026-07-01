# ==========================================
# CODEALPHA TASK 4 - SALES PREDICTION
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Advertising.csv")

# Display first 5 rows
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary column if present
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Statistical Summary
print("\nSummary:")
print(df.describe())

# ==========================================
# Exploratory Data Analysis
# ==========================================

sns.pairplot(df)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# Feature Selection
# ==========================================

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# Model Training
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==========================================
# Model Evaluation
# ==========================================

print("\nModel Performance")
print("---------------------------")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ==========================================
# Actual vs Predicted Plot
# ==========================================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ==========================================
# Predict New Data
# ==========================================

new_data = pd.DataFrame({
    'TV':[230],
    'Radio':[37],
    'Newspaper':[69]
})

prediction = model.predict(new_data)

print("\nPredicted Sales:", prediction[0])ss