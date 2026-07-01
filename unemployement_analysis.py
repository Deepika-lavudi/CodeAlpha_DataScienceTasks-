# ==========================================
# CODEALPHA TASK 2
# UNEMPLOYMENT ANALYSIS WITH PYTHON
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment in India.csv")

# Display First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# ------------------------------------------
# DATA CLEANING
# ------------------------------------------

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Check Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Missing Values
df = df.dropna()

# Convert Date Column into Datetime Format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# ------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ------------------------------------------

print("\nStatistical Summary:")
print(df.describe())

# Average Unemployment Rate
avg_unemployment = df['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Rate:")
print(round(avg_unemployment, 2), "%")

# ------------------------------------------
# STATE-WISE UNEMPLOYMENT ANALYSIS
# ------------------------------------------

state_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Rate by State:")
print(state_unemployment.sort_values(ascending=False))

# ------------------------------------------
# VISUALIZATION 1
# UNEMPLOYMENT RATE OVER TIME
# ------------------------------------------

plt.figure(figsize=(12,6))

sns.lineplot(
    x='Date',
    y='Estimated Unemployment Rate (%)',
    data=df
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------
# VISUALIZATION 2
# STATE-WISE UNEMPLOYMENT RATE
# ------------------------------------------

plt.figure(figsize=(12,8))

state_unemployment.sort_values().plot(
    kind='barh'
)

plt.title("Average Unemployment Rate by State")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# ------------------------------------------
# VISUALIZATION 3
# HEATMAP
# ------------------------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------------
# COVID-19 IMPACT ANALYSIS
# ------------------------------------------

covid_data = df[df['Date'] >= '2020-03-01']

plt.figure(figsize=(12,6))

sns.lineplot(
    x='Date',
    y='Estimated Unemployment Rate (%)',
    data=covid_data
)

plt.title("Impact of Covid-19 on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------
# REGIONAL ANALYSIS
# ------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(
    x='Region',
    y='Estimated Unemployment Rate (%)',
    data=df
)

plt.xticks(rotation=90)
plt.title("Region-wise Unemployment Distribution")
plt.tight_layout()
plt.show()

# ------------------------------------------
# HIGHEST AND LOWEST UNEMPLOYMENT STATES
# ------------------------------------------

highest_state = state_unemployment.idxmax()
highest_rate = state_unemployment.max()

lowest_state = state_unemployment.idxmin()
lowest_rate = state_unemployment.min()

print("\nHighest Unemployment State:")
print(highest_state, "-", round(highest_rate, 2), "%")

print("\nLowest Unemployment State:")
print(lowest_state, "-", round(lowest_rate, 2), "%")

# ------------------------------------------
# INSIGHTS
# ------------------------------------------

print("\n========== PROJECT INSIGHTS ==========")
print("1. Covid-19 caused a significant rise in unemployment.")
print("2. Unemployment trends vary across different states.")
print("3. Some states consistently recorded higher unemployment rates.")
print("4. Economic restrictions during lockdown impacted employment.")
print("5. Unemployment gradually decreased after lockdown relaxation.")
print("======================================")