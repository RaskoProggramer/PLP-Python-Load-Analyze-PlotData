import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Loading
file_path = input("Enter the path to your dataset file (e.g., 'data.csv'): ")

# Try loading the dataset
try:
    # Load CSV or Excel files based on the file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")
    
    print(f"\nDataset loaded successfully! Here's a preview:\n")
    print(df.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Data Exploration
# Display summary statistics for numerical columns
print("\nSummary statistics:")
print(df.describe())

# Check for missing values in the dataset
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# 3. Basic Data Analysis
# Calculate mean of each numerical column
mean_values = df.mean(numeric_only=True)
print("\nMean values of numerical columns:")
print(mean_values)

# Calculate correlation (if numerical columns exist)
# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Check if there are enough numeric columns
if len(numeric_df.columns) > 1:
    correlation_matrix = numeric_df.corr()
    print("\nCorrelation matrix:")
    print(correlation_matrix)
else:
    print("\nNot enough numeric columns for a correlation matrix.")

# 4. Visualizations
# Set up a seaborn style
sns.set(style="whitegrid")

# Try to plot visualizations only if the dataset has the required columns
try:
    # Pairplot (only if at least two numerical columns are present)
    if len(df.select_dtypes(include='number').columns) >= 2:
        print("\nGenerating pairplot...")
        sns.pairplot(df)
        plt.show()

    # Boxplot example (requires categorical and numerical columns)
    categorical_columns = df.select_dtypes(include='object').columns
    numerical_columns = df.select_dtypes(include='number').columns
    if len(categorical_columns) > 0 and len(numerical_columns) > 0:
        print(f"\nGenerating boxplot for '{numerical_columns[0]}' by '{categorical_columns[0]}'...")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=categorical_columns[0], y=numerical_columns[0], data=df)
        plt.title(f"{numerical_columns[0]} Distribution by {categorical_columns[0]}")
        plt.show()
except Exception as e:
    print(f"Visualization error: {e}")

# 5. Findings or Observations
print("\nFindings/Observations:")
print("- Dataset successfully analyzed. See the output and visualizations for details.")
