# ==========================================================
# Applied AI & ML Essentials - Capstone Project
# Part 1A - Data Acquisition & Cleaning
# Dataset: Indian Overseas Migration Dataset
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Create plots directory
os.makedirs("plots", exist_ok=True)

# ==========================================================
# 1. LOAD DATASET
# ==========================================================
import os
import kagglehub
import pandas as pd

# Download dataset
path = kagglehub.dataset_download("aryanmdev/india-global-migration-dataset")

print("Dataset downloaded to:")
print(path)

# List files in the downloaded folder
files = os.listdir(path)

print("\nFiles in dataset:")
for file in files:
    print(file)

# Find the CSV automatically
csv_files = [f for f in files if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset.")

csv_path = os.path.join(path, csv_files[0])

print("\nLoading:", csv_path)

# Load dataset
df = pd.read_csv(csv_path)

print("\nDataset loaded successfully!")
print(df.head())
# ==========================================================
# 2. NULL VALUE ANALYSIS
# ==========================================================

print("\n")
print("="*80)
print("NULL VALUE ANALYSIS")
print("="*80)

null_count = df.isnull().sum()

null_percent = (df.isnull().sum() / len(df)) * 100

null_table = pd.DataFrame({
    "Missing Count": null_count,
    "Missing Percentage": null_percent.round(2)
})

print(null_table)

print("\n")

print("Columns with more than 20% missing values")

high_null_columns = null_table[
    null_table["Missing Percentage"] > 20
]

if len(high_null_columns) == 0:
    print("No columns exceed 20% missing values.")
else:
    print(high_null_columns)

# ==========================================================
# 3. DUPLICATE ANALYSIS
# ==========================================================

print("\n")
print("="*80)
print("DUPLICATE ANALYSIS")
print("="*80)

duplicates = df.duplicated().sum()

print("Duplicate Rows Found :", duplicates)

rows_before = len(df)

df = df.drop_duplicates()

rows_after = len(df)

print("Rows Before :", rows_before)
print("Rows After  :", rows_after)
print("Rows Removed:", rows_before - rows_after)

print("\n")

print("Null Percentage After Removing Duplicates")

new_null_percent = (
    df.isnull().sum()/len(df)
)*100

print(new_null_percent.round(2))

# ==========================================================
# 4. MEMORY USAGE BEFORE CONVERSION
# ==========================================================

print("\n")
print("="*80)
print("MEMORY USAGE")
print("="*80)

memory_before = df.memory_usage(deep=True).sum()

print(f"Memory Before Conversion : {memory_before:,} bytes")

# ==========================================================
# 5. DATA TYPE CORRECTION
# ==========================================================

print("\n")
print("="*80)
print("DATA TYPE CORRECTION")
print("="*80)

# Keep original value column
df["Value_Original"] = df["Value"]

# Remove unwanted symbols

df["Value"] = (
    df["Value"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.replace("B", "e9", regex=False)
    .str.replace("M", "e6", regex=False)
    .str.replace("%", "", regex=False)
    .str.replace("NOT AVAILABLE", "", regex=False)
    .str.strip()
)

# Convert to numeric

df["Value"] = pd.to_numeric(
    df["Value"],
    errors="coerce"
)

print("\n")

print("Data Type After Conversion")

print(df["Value"].dtype)

# ==========================================================
# Convert repetitive text columns into category
# ==========================================================

categorical_columns = [
    "Destination_Country",
    "Region",
    "Column_Name",
    "Data_Type"
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype("category")

print("\n")

print("Updated Data Types")

print(df.dtypes)

# ==========================================================
# MEMORY AFTER CONVERSION
# ==========================================================

memory_after = df.memory_usage(deep=True).sum()

print("\n")

print(f"Memory After Conversion : {memory_after:,} bytes")

print(f"Memory Saved            : {memory_before-memory_after:,} bytes")

# ==========================================================
# 6. IMPUTE NUMERIC MISSING VALUES (<20%)
# ==========================================================

print("\n")
print("="*80)
print("MISSING VALUE IMPUTATION")
print("="*80)

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    percentage = (df[col].isnull().sum()/len(df))*100

    if percentage < 20:

        median_value = df[col].median()

        df[col] = df[col].fillna(median_value)

        print(f"{col:<20} Median Used : {median_value}")

print("\n")

print("Remaining Missing Values")

print(df[numeric_columns].isnull().sum())

# ==========================================================
# 7. FINAL DATA INFORMATION
# ==========================================================

print("\n")
print("="*80)
print("FINAL DATAFRAME INFORMATION")
print("="*80)

print(df.info())

print("\n")

print(df.head())

# ==========================================================
# 8. SAVE INTERMEDIATE CLEANED DATASET
# ==========================================================

df.to_csv("cleaned_data_part1A.csv", index=False)

print("\n")
print("Intermediate cleaned dataset saved successfully.")
print("Filename : cleaned_data_part1A.csv")

# ==========================================================
# PART 1B - EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================
df["Value_Log"] = np.log1p(df["Value"])
df["Year_Index"] = df["Year"] - df["Year"].min()
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

print(df.describe(include="all"))

# ==========================================================
# NUMERIC COLUMNS
# ==========================================================

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

print("\nNumeric Columns:")
print(numeric_columns)

# ==========================================================
# SKEWNESS
# ==========================================================

print("\n" + "="*80)
print("SKEWNESS ANALYSIS")
print("="*80)

skewness = {}

for col in numeric_columns:

    skew = df[col].skew()

    skewness[col] = skew

    print(f"{col:<20} {skew:.4f}")

# Sort by absolute skewness

sorted_skew = sorted(
    skewness.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("\n")

print("Columns Sorted by Absolute Skewness")

for item in sorted_skew:

    print(item)

most_skewed = sorted_skew[0][0]

print("\n")

print("Most Skewed Column :", most_skewed)

# ==========================================================
# DESCRIPTIVE STATISTICS OF MOST SKEWED COLUMN
# ==========================================================

print("\n")

print(df[most_skewed].describe())

# ==========================================================
# IQR OUTLIER ANALYSIS
# ==========================================================

print("\n" + "="*80)
print("IQR OUTLIER ANALYSIS")
print("="*80)

iqr_columns = ["Year", "Value"]

for col in iqr_columns:

    print("\n")

    print(f"Column : {col}")

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print("Q1 :", Q1)
    print("Q3 :", Q3)
    print("IQR :", IQR)
    print("Lower Bound :", lower)
    print("Upper Bound :", upper)
    print("Number of Outliers :", len(outliers))

# ==========================================================
# LINE PLOT
# ==========================================================

print("\nGenerating Line Plot...")

line_data = df.sort_values("Year")

plt.figure(figsize=(12,6))

plt.plot(
    line_data["Year"],
    line_data["Value"],
    color="blue",
    linewidth=2
)

plt.title("Migration Value Across Years")

plt.xlabel("Year")

plt.ylabel("Value")

plt.grid(True)

plt.tight_layout()

plt.savefig("plots/line_plot.png")

plt.show()

# ==========================================================
# BAR CHART
# ==========================================================

print("Generating Bar Chart...")

region_mean = (
    df.groupby("Region")["Value"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,6))

region_mean.plot(kind="bar")

plt.title("Average Migration Value by Region")

plt.xlabel("Region")

plt.ylabel("Average Value")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("plots/bar_chart.png")

plt.show()

# ==========================================================
# HISTOGRAM
# ==========================================================

print("Generating Histogram...")

plt.figure(figsize=(10,6))

sns.histplot(
    df[most_skewed],
    bins=20,
    kde=True
)

plt.title(f"Distribution of {most_skewed}")

plt.xlabel(most_skewed)

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("plots/histogram.png")

plt.show()

# ==========================================================
# SCATTER PLOT
# ==========================================================

print("Generating Scatter Plot...")

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="Year",
    y="Value",
    hue="Region"
)

plt.title("Year vs Migration Value")

plt.xlabel("Year")

plt.ylabel("Value")

plt.tight_layout()

plt.savefig("plots/scatter_plot.png")

plt.show()

# ==========================================================
# BOX PLOT
# ==========================================================

print("Generating Box Plot...")

plt.figure(figsize=(14,6))

sns.boxplot(
    data=df,
    x="Region",
    y="Value"
)

plt.xticks(rotation=45)

plt.title("Migration Value Distribution Across Regions")

plt.xlabel("Region")

plt.ylabel("Value")

plt.tight_layout()

plt.savefig("plots/box_plot.png")

plt.show()

# ==========================================================
# PEARSON CORRELATION
# ==========================================================

print("\n" + "="*80)
print("PEARSON CORRELATION MATRIX")
print("="*80)

pearson_corr = df[numeric_columns].corr()

print(pearson_corr)

# ==========================================================
# HEATMAP
# ==========================================================

plt.figure(figsize=(10,8))

sns.heatmap(
    pearson_corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Pearson Correlation Heatmap")

plt.tight_layout()

plt.savefig("plots/correlation_heatmap.png")

plt.show()

# ==========================================================
# HIGHEST CORRELATED PAIR
# ==========================================================

corr_matrix = pearson_corr.abs().copy()

# Mask the diagonal instead of using fill_diagonal
for col in corr_matrix.columns:
    corr_matrix.loc[col, col] = np.nan

highest_pair = corr_matrix.stack().idxmax()

highest_corr = pearson_corr.loc[
    highest_pair[0],
    highest_pair[1]
]

print(f"\nHighest Correlated Pair: {highest_pair}")
print(f"Correlation: {highest_corr:.4f}")

print("\n")

print(pearson_corr)

print(pearson_corr.shape)

print(pearson_corr.dtypes)

print("Highest Correlated Pair")

print("-----------------------")

print(f"{highest_pair[0]}  <-->  {highest_pair[1]}")

print(f"Correlation : {highest_corr:.4f}")

# ==========================================================
# SAVE UPDATED DATA
# ==========================================================

df.to_csv("cleaned_data_part1B.csv", index=False)

print("\nEDA Completed Successfully.")

print("Plots saved inside 'plots' folder.")

print("Updated cleaned dataset saved as cleaned_data_part1B.csv")

# ==========================================================
# PART 1C - ADVANCED ANALYSIS
# ==========================================================

print("\n" + "="*80)
print("MEAN vs MEDIAN COMPARISON")
print("="*80)

# Top two skewed numeric columns
top2_skewed = [item[0] for item in sorted_skew[:2]]

comparison = []

for col in top2_skewed:

    mean_value = df[col].mean()
    median_value = df[col].median()

    comparison.append([col, mean_value, median_value])

comparison_df = pd.DataFrame(
    comparison,
    columns=["Column","Mean","Median"]
)

print(comparison_df)

# ==========================================================
# IMPUTE REMAINING NULLS USING MEDIAN
# ==========================================================

print("\n")

print("Imputing Remaining Null Values...")

for col in top2_skewed:

    df[col] = df[col].fillna(df[col].median())

print("\nRemaining Null Values")

print(df[top2_skewed].isnull().sum())

# ==========================================================
# SPEARMAN CORRELATION
# ==========================================================

print("\n" + "="*80)
print("SPEARMAN CORRELATION MATRIX")
print("="*80)

spearman_corr = df[numeric_columns].corr(method="spearman")

print(spearman_corr)

# ==========================================================
# PEARSON vs SPEARMAN DIFFERENCE
# ==========================================================

difference_rows = []

for i in range(len(numeric_columns)):

    for j in range(i+1, len(numeric_columns)):

        c1 = numeric_columns[i]
        c2 = numeric_columns[j]

        pearson_value = pearson_corr.loc[c1,c2]

        spearman_value = spearman_corr.loc[c1,c2]

        difference = abs(spearman_value - pearson_value)

        difference_rows.append([
            c1,
            c2,
            pearson_value,
            spearman_value,
            difference
        ])

difference_df = pd.DataFrame(
    difference_rows,
    columns=[
        "Column 1",
        "Column 2",
        "Pearson",
        "Spearman",
        "Absolute Difference"
    ]
)

difference_df = difference_df.sort_values(
    "Absolute Difference",
    ascending=False
)

print("\n")

print("Top 3 Column Pairs")

print(difference_df.head(3))

# ==========================================================
# GROUPED AGGREGATION
# ==========================================================

print("\n" + "="*80)
print("GROUPED AGGREGATION")
print("="*80)

grouped = df.groupby("Region")["Value"].agg(
    ["mean","std","count"]
)

print(grouped)

# ==========================================================
# HIGHEST MEAN
# ==========================================================

highest_mean_group = grouped["mean"].idxmax()

highest_mean_value = grouped["mean"].max()

print("\n")

print("Highest Mean Group")

print(highest_mean_group)

print(highest_mean_value)

# ==========================================================
# HIGHEST STANDARD DEVIATION
# ==========================================================

highest_std_group = grouped["std"].idxmax()

highest_std_value = grouped["std"].max()

print("\n")

print("Highest Standard Deviation Group")

print(highest_std_group)

print(highest_std_value)

# ==========================================================
# MEAN RATIO
# ==========================================================

highest_mean = grouped["mean"].max()

lowest_mean = grouped["mean"].min()

ratio = highest_mean / lowest_mean

print("\n")

print("Highest Mean :", highest_mean)

print("Lowest Mean :", lowest_mean)

print("Ratio :", ratio)

# ==========================================================
# SAVE DIFFERENCE TABLE
# ==========================================================

difference_df.to_csv(
    "pearson_vs_spearman_difference.csv",
    index=False
)

# ==========================================================
# FINAL CLEAN DATASET
# ==========================================================

df.to_csv(
    "cleaned_data.csv",
    index=False
)

print("\n")

print("="*80)

print("PART 1 COMPLETED SUCCESSFULLY")

print("="*80)

print("Generated Files")

print("----------------")

print("✔ cleaned_data.csv")

print("✔ pearson_vs_spearman_difference.csv")

print("✔ plots/line_plot.png")

print("✔ plots/bar_chart.png")

print("✔ plots/histogram.png")

print("✔ plots/scatter_plot.png")

print("✔ plots/box_plot.png")

print("✔ plots/correlation_heatmap.png")