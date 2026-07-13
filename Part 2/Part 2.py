# ==========================================================
# Applied AI & ML Essentials – Capstone Project
# Part 2A – Data Loading & Preprocessing
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Create plots folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "cleaned_data.csv")

print(csv_path)

df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully")
print(df.head())

print("\nShape:", df.shape)

print("\nData Types")
print(df.dtypes)

# ==========================================================
# DEFINE TARGETS
# ==========================================================

# Continuous target
y_reg = df["Value"]

# Binary target
y_clf = (y_reg > y_reg.median()).astype(int)

# ==========================================================
# FEATURE MATRIX
# ==========================================================

X = df.drop(columns=["Value", "Value_Log"])

print("\nFeature Matrix Shape:", X.shape)

# ==========================================================
# IDENTIFY COLUMN TYPES
# ==========================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

print("\nCategorical Columns")
print(categorical_columns)

print("\nNumeric Columns")
print(numeric_columns)

# ==========================================================
# ONE-HOT ENCODING
# ==========================================================

print("\nApplying One-Hot Encoding...")

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)
feature_names = X.columns.tolist()
print("Encoded Shape:", X.shape)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaling Completed")

print("Training Shape:", X_train_scaled.shape)
print("Testing Shape :", X_test_scaled.shape)

print("\nPart 2A Completed Successfully")


# ==========================================================
# PART 2B - REGRESSION MODELS
# ==========================================================

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

print("\n" + "="*80)
print("LINEAR REGRESSION")
print("="*80)

# ----------------------------------------------------------
# Train Linear Regression
# ----------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_reg_train
)

# Predictions
y_pred_reg = linear_model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(
    y_reg_test,
    y_pred_reg
)

r2 = r2_score(
    y_reg_test,
    y_pred_reg
)

print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# ----------------------------------------------------------
# Feature Coefficients
# ----------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": linear_model.coef_
})

coefficients["Absolute"] = coefficients["Coefficient"].abs()

coefficients = coefficients.sort_values(
    "Absolute",
    ascending=False
)

print("\nTop 10 Features by Absolute Coefficient")

print(coefficients.head(10))

print("\nTop 3 Most Influential Features")

print(coefficients.head(3))

# ----------------------------------------------------------
# Ridge Regression
# ----------------------------------------------------------

print("\n" + "="*80)
print("RIDGE REGRESSION")
print("="*80)

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(
    X_train_scaled,
    y_reg_train
)

ridge_pred = ridge_model.predict(
    X_test_scaled
)

ridge_mse = mean_squared_error(
    y_reg_test,
    ridge_pred
)

ridge_r2 = r2_score(
    y_reg_test,
    ridge_pred
)

print(f"Ridge MSE : {ridge_mse:.4f}")
print(f"Ridge R²  : {ridge_r2:.4f}")

# ----------------------------------------------------------
# Comparison Table
# ----------------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression"
    ],
    "MSE": [
        mse,
        ridge_mse
    ],
    "R²": [
        r2,
        ridge_r2
    ]
})

print("\n")
print("="*80)
print("MODEL COMPARISON")
print("="*80)

print(comparison)

# Save comparison table
comparison.to_csv(
    "regression_comparison.csv",
    index=False
)

print("\nRegression comparison saved as regression_comparison.csv")

# ==========================================================
# PART 2C - LOGISTIC REGRESSION
# ==========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

print("\n" + "="*80)
print("CLASSIFICATION MODEL")
print("="*80)

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

print("\nClass Distribution (Training Set)")
print(y_clf_train.value_counts())

class_ratio = y_clf_train.value_counts(normalize=True)

print("\nClass Percentage")
print(class_ratio)

# ==========================================================
# HANDLE CLASS IMBALANCE
# ==========================================================

if class_ratio.min() < 0.35:
    print("\nClass imbalance detected.")
    class_weight = "balanced"
else:
    print("\nClasses are reasonably balanced.")
    class_weight = None

# ==========================================================
# BASELINE LOGISTIC REGRESSION (C = 1.0)
# ==========================================================

log_model = LogisticRegression(
    max_iter=1000,
    class_weight=class_weight,
    random_state=42
)

log_model.fit(
    X_train_scaled,
    y_clf_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = log_model.predict(X_test_scaled)

y_prob = log_model.predict_proba(X_test_scaled)[:,1]

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_clf_test,
    y_pred
)

print("\nConfusion Matrix")
print(cm)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report")

print(
    classification_report(
        y_clf_test,
        y_pred
    )
)

# ==========================================================
# AUC
# ==========================================================

auc = roc_auc_score(
    y_clf_test,
    y_prob
)

print(f"\nAUC : {auc:.4f}")

# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, thresholds = roc_curve(
    y_clf_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "plots/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ==========================================================
# THRESHOLD ANALYSIS
# ==========================================================

print("\n" + "="*80)
print("THRESHOLD SENSITIVITY")
print("="*80)

threshold_results = []

for threshold in np.arange(0.30,0.71,0.10):

    prediction = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_clf_test,
        prediction
    )

    recall = recall_score(
        y_clf_test,
        prediction
    )

    f1 = f1_score(
        y_clf_test,
        prediction
    )

    threshold_results.append([
        threshold,
        precision,
        recall,
        f1
    ])

threshold_df = pd.DataFrame(
    threshold_results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1"
    ]
)

print(threshold_df)

threshold_df.to_csv(
    "threshold_analysis.csv",
    index=False
)

# ==========================================================
# REGULARIZATION EXPERIMENT
# ==========================================================

print("\n" + "="*80)
print("LOGISTIC REGULARIZATION")
print("="*80)

log_model_smallC = LogisticRegression(
    C=0.01,
    max_iter=1000,
    class_weight=class_weight,
    random_state=42
)

log_model_smallC.fit(
    X_train_scaled,
    y_clf_train
)

prob_small = log_model_smallC.predict_proba(
    X_test_scaled
)[:,1]

pred_small = log_model_smallC.predict(
    X_test_scaled
)

auc_small = roc_auc_score(
    y_clf_test,
    prob_small
)

comparison = pd.DataFrame({
    "Model":[
        "C=1.0",
        "C=0.01"
    ],
    "Precision":[
        precision_score(y_clf_test,y_pred),
        precision_score(y_clf_test,pred_small)
    ],
    "Recall":[
        recall_score(y_clf_test,y_pred),
        recall_score(y_clf_test,pred_small)
    ],
    "AUC":[
        auc,
        auc_small
    ]
})

print(comparison)

comparison.to_csv(
    "logistic_comparison.csv",
    index=False
)

# ==========================================================
# BOOTSTRAP AUC
# ==========================================================

print("\n" + "="*80)
print("BOOTSTRAP CONFIDENCE INTERVAL")
print("="*80)

auc_difference = []

np.random.seed(42)

for i in range(500):

    index = np.random.choice(
        len(y_clf_test),
        size=len(y_clf_test),
        replace=True
    )

    sample_true = y_clf_test.iloc[index]

    # Skip samples with only one class
    if len(np.unique(sample_true)) < 2:
        continue

    sample_prob1 = y_prob[index]
    sample_prob2 = prob_small[index]

    auc1 = roc_auc_score(sample_true, sample_prob1)
    auc2 = roc_auc_score(sample_true, sample_prob2)

    auc_difference.append(auc1 - auc2)

# ----------------------------------------------------------
# Results
# ----------------------------------------------------------

if len(auc_difference) == 0:
    print("No valid bootstrap samples were generated.")
else:
    mean_difference = np.mean(auc_difference)

    lower = np.percentile(auc_difference, 2.5)
    upper = np.percentile(auc_difference, 97.5)

    print(f"Mean Difference : {mean_difference:.4f}")
    print(f"95% CI Lower    : {lower:.4f}")
    print(f"95% CI Upper    : {upper:.4f}")

print("\nPART 2 COMPLETED SUCCESSFULLY")