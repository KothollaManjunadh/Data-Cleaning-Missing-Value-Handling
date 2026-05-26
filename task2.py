# Data Cleaning & Missing Value Handling

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv("titanic.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize Missing Values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Handle Missing Values

# Fill Age column using Mean
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Fill Embarked column using Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Remove Remaining Null Values
df.dropna(inplace=True)

# Verify Missing Values Removed
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Detect Outliers using Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()

# Remove Outliers using IQR Method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_limit) & (df['Fare'] <= upper_limit)]

# Save Cleaned Dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nData Cleaning Completed Successfully!")
