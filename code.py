# 1️⃣ Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2️⃣ Load the dataset
df = sns.load_dataset('diamonds')

# 3️⃣ View first few rows
print(df.head())

# 4️⃣ Check dataset information
df.info()

# 5️⃣ Basic statistical summary
print(df.describe())

# 6️⃣ Check for missing values
print(df.isnull().sum())

# 7️⃣ Univariate Analysis

# Price distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
plt.show()

# Carat distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['carat'], kde=True, color='orange')
plt.title("Carat Distribution")
plt.show()

# 8️⃣ Bivariate Analysis

# Scatter plot for Carat vs Price
plt.figure(figsize=(8, 4))
sns.scatterplot(x='carat', y='price', data=df)
plt.title("Carat vs Price")
plt.show()

# Boxplot for Price vs Cut
plt.figure(figsize=(8, 4))
sns.boxplot(x='cut', y='price', data=df)
plt.title("Price vs Cut")
plt.show()

# Boxplot for Price vs Color
plt.figure(figsize=(8, 4))
sns.boxplot(x='color', y='price', data=df)
plt.title("Price vs Color")
plt.show()

# Boxplot for Price vs Clarity
plt.figure(figsize=(8, 4))
sns.boxplot(x='clarity', y='price', data=df)
plt.title("Price vs Clarity")
plt.show()

# 9️⃣ Correlation Analysis

# Select only numeric columns for correlation to avoid error
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 🔟 Outlier Detection using boxplot for 'price'
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['price'])
plt.title("Outliers in Price")
plt.show()

# 1️⃣1️⃣ Outlier Detection using boxplot for 'carat'
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['carat'])
plt.title("Outliers in Carat")
plt.show()
