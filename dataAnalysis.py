# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris(as_frame=True)
data = iris.frame

# Display the first few rows
print("First few rows of the Iris dataset:")
print(data.head())

# Check data types and missing values
print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

# Clean the dataset (in this case, no missing values, so no cleaning is needed)
cleaned_data = data

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(data.describe())

print("Columns in the DataFrame:", data.columns)
# Group by a categorical column and compute mean
grouped_data = data.groupby("target").mean()
print("\nMean of numerical columns grouped by species:")
print(grouped_data)

# Identify patterns or findings
print("\nFindings:")
print("1. The average sepal and petal dimensions differ significantly between species.")
print("2. Setosa has the smallest petal length and width, while Virginica has the largest.")


# Line chart (not relevant for Iris, but demonstrating with the first 20 rows)
plt.figure(figsize=(10, 6))
plt.plot(cleaned_data.index[:20], cleaned_data['sepal length (cm)'][:20], marker='o', label='Sepal Length')
plt.title('Line Chart: Sepal Length for First 20 Samples')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# Bar chart: Average sepal width by class
plt.figure(figsize=(8, 5))
sns.barplot(x='target', y='sepal width (cm)', data=cleaned_data, ci=None)
plt.title('Bar Chart: Average Sepal Width by Target Class')
plt.xlabel('Target Class')
plt.ylabel('Average Sepal Width (cm)')
plt.show()

# Histogram: Distribution of petal length
plt.figure(figsize=(8, 5))
plt.hist(cleaned_data['petal length (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram: Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=cleaned_data, palette='viridis')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
