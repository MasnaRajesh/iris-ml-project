import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the downloaded CSV into a Pandas DataFrame
df = pd.read_csv('C:/Users/sushm/OneDrive/Desktop/iris_project/Iris.csv')
print("Dataset loaded successfully!\n")

print("2. Data Cleaning...")
# Display the first few rows to see what we are working with
print("Initial Data Snapshot:")
print(df.head(), "\n")

# The Kaggle dataset usually comes with an 'Id' column which is useless for ML. Let's drop it.
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)
    print("Dropped 'Id' column.")

# Check for missing (null) values
missing_values = df.isnull().sum().sum()
print(f"Total missing values found: {missing_values}")
if missing_values > 0:
    df = df.dropna() # Drop rows with missing values
    print("Dropped rows with missing values.")

# Check for and remove duplicate rows
duplicates = df.duplicated().sum()
print(f"Total duplicate rows found: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Dropped duplicate rows.")

print("\nData cleaning complete. Here is the cleaned data snapshot:")
print(df.head(), "\n")

# Save the cleaned dataset so train.py can use it later
df.to_csv('Cleaned_Iris.csv', index=False)
print("Cleaned data saved as 'Cleaned_Iris.csv'.\n")

print("3. Data Visualization...")
# Set the visual style for our plots
sns.set_theme(style="ticks")

# Visualization 1: Pairplot
# This shows the relationship between all combinations of features, colored by species.
# It helps us see which features best separate the different iris flowers.
print("Generating Pairplot... (Close the image window to continue)")
sns.pairplot(df, hue="Species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()


# Visualization 2: Boxplots
# This helps us spot outliers and understand the distribution of each feature.
print("Generating Boxplot... (Close the image window to finish)")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop('Species', axis=1), orient="h", palette="Set2")
plt.title("Distribution of Iris Features")
plt.show()

print("Data Exploration Finished!")