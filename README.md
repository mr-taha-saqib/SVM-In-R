# Exploratory Data Analysis (EDA) on Iris Dataset

```python
# Importing Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# Loading the Iris Dataset
data = load_iris(as_frame=True)
df = data.frame
df['species'] = df['target']  # Add target as species for clarity
df.drop(columns=['target'], inplace=True)

# 1. Classifying Dependent and Independent Variables
dependent_var = "species"
independent_vars = df.columns.drop(dependent_var).tolist()
print(f"Dependent Variable: {dependent_var}")
print(f"Independent Variables: {independent_vars}")

# 2. Reporting Examples of Each Feature
print("Dataset Shape:", df.shape)
print("Data Types:\n", df.dtypes)
print("Value Counts:\n", df.nunique())

# 3. Checking for Missing Values
print("Missing Values:\n", df.isnull().sum())

# 4. Descriptive Statistics
print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Mode:\n", df.mode().iloc[0])
print("Range:\n", df.max(numeric_only=True) - df.min(numeric_only=True))
print("Standard Deviation:\n", df.std(numeric_only=True))

# 5. Data Visualizations
# Pair Plot
sns.pairplot(df, hue='species', palette='Set2')
plt.show()

# Scatter Plot
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='Set1')
plt.title("Scatter Plot of Sepal Dimensions")
plt.show()

# Histograms
df.hist(bins=15, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Iris Features")
plt.show()

# Heatmap Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap of Feature Correlations")
plt.show()
