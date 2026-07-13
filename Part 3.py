# ==========================================================
# Applied AI & ML Essentials – Capstone Project
# Part 3 – Advanced Modeling
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

print("=" * 80)
print("PART 3 - ADVANCED MODELING")
print("=" * 80)

# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "cleaned_data.csv")

df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully")
print(df.shape)

# ==========================================================
# TARGET VARIABLES
# ==========================================================

y_reg = df["Value"]

y_clf = (y_reg > y_reg.median()).astype(int)

# ==========================================================
# FEATURE MATRIX
# ==========================================================

X = df.drop(columns=["Value", "Value_Log"])

# ==========================================================
# ONE-HOT ENCODING
# ==========================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

feature_names = X.columns.tolist()

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

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nPreprocessing Completed Successfully")

# ==========================================================
# BASELINE LOGISTIC REGRESSION (Required for Task 5 & Summary)
# ==========================================================

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(
    X_train_scaled,
    y_clf_train
)

log_pred = log_model.predict(
    X_test_scaled
)

log_prob = log_model.predict_proba(
    X_test_scaled
)[:, 1]

log_accuracy = accuracy_score(
    y_clf_test,
    log_pred
)

log_auc = roc_auc_score(
    y_clf_test,
    log_prob
)

print("\nBaseline Logistic Regression")
print(f"Test Accuracy : {log_accuracy:.4f}")
print(f"Test ROC-AUC  : {log_auc:.4f}")

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import joblib

print("\n" + "="*80)
print("TASK 1 - DECISION TREE (DEFAULT)")
print("="*80)


dt_default = DecisionTreeClassifier(
    random_state=42
)

dt_default.fit(
    X_train_scaled,
    y_clf_train
)

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

dt_train_pred = dt_default.predict(
    X_train_scaled
)

dt_test_pred = dt_default.predict(
    X_test_scaled
)

# ----------------------------------------------------------
# Accuracy
# ----------------------------------------------------------

dt_train_acc = accuracy_score(
    y_clf_train,
    dt_train_pred
)

dt_test_acc = accuracy_score(
    y_clf_test,
    dt_test_pred
)

print(f"Training Accuracy : {dt_train_acc:.4f}")
print(f"Testing Accuracy  : {dt_test_acc:.4f}")

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_clf_test,
        dt_test_pred
    )
)

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    y_clf_test,
    dt_test_pred
)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Overfitting Check
# ----------------------------------------------------------

gap = dt_train_acc - dt_test_acc

print(f"\nTrain-Test Gap : {gap:.4f}")

if gap > 0.10:
    print("Observation : Model shows signs of overfitting.")
else:
    print("Observation : No significant overfitting observed.")

print("\n" + "="*80)
print("TASK 2 - CONTROLLED DECISION TREE")
print("="*80)

controlled_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

controlled_tree.fit(
    X_train_scaled,
    y_clf_train
)

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

controlled_train_pred = controlled_tree.predict(
    X_train_scaled
)

controlled_test_pred = controlled_tree.predict(
    X_test_scaled
)

controlled_test_prob = controlled_tree.predict_proba(
    X_test_scaled
)[:, 1]

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

controlled_train_acc = accuracy_score(
    y_clf_train,
    controlled_train_pred
)

controlled_test_acc = accuracy_score(
    y_clf_test,
    controlled_test_pred
)

controlled_auc = roc_auc_score(
    y_clf_test,
    controlled_test_prob
)

print(f"Training Accuracy : {controlled_train_acc:.4f}")
print(f"Testing Accuracy  : {controlled_test_acc:.4f}")
print(f"Test ROC-AUC      : {controlled_auc:.4f}")

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report\n")

print(
    classification_report(
        y_clf_test,
        controlled_test_pred
    )
)

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    y_clf_test,
    controlled_test_pred
)

print("\nConfusion Matrix")

print(cm)
# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_clf_test,
        controlled_test_pred
    )
)

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    y_clf_test,
    controlled_test_pred
)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Comparison
# ----------------------------------------------------------

comparison_tree = pd.DataFrame({
    "Model": [
        "Default Decision Tree",
        "Controlled Decision Tree"
    ],
    "Training Accuracy": [
        dt_train_acc,
        controlled_train_acc
    ],
    "Testing Accuracy": [
        dt_test_acc,
        controlled_test_acc
    ]
})

comparison_tree["Train-Test Gap"] = (
    comparison_tree["Training Accuracy"]
    - comparison_tree["Testing Accuracy"]
)

print("\nDecision Tree Comparison")

print(comparison_tree)

print("\n" + "="*80)
print("TASK 3 - GINI vs ENTROPY")
print("="*80)

# ----------------------------------------------------------
# Gini Tree
# ----------------------------------------------------------

gini_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

gini_tree.fit(
    X_train_scaled,
    y_clf_train
)

gini_pred = gini_tree.predict(
    X_test_scaled
)

gini_accuracy = accuracy_score(
    y_clf_test,
    gini_pred
)

# ----------------------------------------------------------
# Entropy Tree
# ----------------------------------------------------------

entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

entropy_tree.fit(
    X_train_scaled,
    y_clf_train
)

entropy_pred = entropy_tree.predict(
    X_test_scaled
)

entropy_accuracy = accuracy_score(
    y_clf_test,
    entropy_pred
)

# ----------------------------------------------------------
# Comparison
# ----------------------------------------------------------

gini_entropy = pd.DataFrame({
    "Criterion": [
        "Gini",
        "Entropy"
    ],
    "Test Accuracy": [
        gini_accuracy,
        entropy_accuracy
    ]
})

print(gini_entropy)

print("\n" + "=" * 80)
print("TASK 4 - RANDOM FOREST")
print("=" * 80)

# ----------------------------------------------------------
# Train Random Forest
# ----------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_clf_train
)

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

rf_train_pred = rf_model.predict(
    X_train_scaled
)

rf_test_pred = rf_model.predict(
    X_test_scaled
)

rf_test_prob = rf_model.predict_proba(
    X_test_scaled
)[:, 1]

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

rf_train_accuracy = accuracy_score(
    y_clf_train,
    rf_train_pred
)

rf_test_accuracy = accuracy_score(
    y_clf_test,
    rf_test_pred
)

rf_auc = roc_auc_score(
    y_clf_test,
    rf_test_prob
)

print(f"\nTraining Accuracy : {rf_train_accuracy:.4f}")
print(f"Testing Accuracy  : {rf_test_accuracy:.4f}")
print(f"Test ROC-AUC      : {rf_auc:.4f}")

print("\n" + "=" * 80)
print("TOP 5 FEATURE IMPORTANCE")
print("=" * 80)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top5_features = importance_df.head(5)

print(top5_features)

# Save for README

top5_features.to_csv(
    "top5_feature_importance.csv",
    index=False
)

print("\n" + "=" * 80)
print("TASK 4A - GRADIENT BOOSTING")
print("=" * 80)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(
    X_train_scaled,
    y_clf_train
)

# ----------------------------------------------------------
# Predictions
# ----------------------------------------------------------

gb_train_pred = gb_model.predict(
    X_train_scaled
)

gb_test_pred = gb_model.predict(
    X_test_scaled
)

gb_test_prob = gb_model.predict_proba(
    X_test_scaled
)[:, 1]

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

gb_train_accuracy = accuracy_score(
    y_clf_train,
    gb_train_pred
)

gb_test_accuracy = accuracy_score(
    y_clf_test,
    gb_test_pred
)

gb_auc = roc_auc_score(
    y_clf_test,
    gb_test_prob
)

print(f"\nTraining Accuracy : {gb_train_accuracy:.4f}")
print(f"Testing Accuracy  : {gb_test_accuracy:.4f}")
print(f"Test ROC-AUC      : {gb_auc:.4f}")

print("\n" + "=" * 80)
print("TASK 4B - FEATURE ABLATION")
print("=" * 80)

least5_features = importance_df.tail(5)

print("\nLowest 5 Important Features\n")

print(least5_features)

lowest_features = least5_features["Feature"].tolist()

X_train_reduced = X_train.drop(
    columns=lowest_features
)

X_test_reduced = X_test.drop(
    columns=lowest_features
)

scaler_reduced = StandardScaler()

X_train_reduced_scaled = scaler_reduced.fit_transform(
    X_train_reduced
)

X_test_reduced_scaled = scaler_reduced.transform(
    X_test_reduced
)

rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_reduced.fit(
    X_train_reduced_scaled,
    y_clf_train
)

reduced_prob = rf_reduced.predict_proba(
    X_test_reduced_scaled
)[:,1]

reduced_auc = roc_auc_score(
    y_clf_test,
    reduced_prob
)

print(f"\nOriginal Model AUC : {rf_auc:.4f}")
print(f"Reduced Model AUC  : {reduced_auc:.4f}")

ablation_results = pd.DataFrame({
    "Model": [
        "Full Feature Set",
        "Reduced Feature Set"
    ],
    "Test ROC-AUC": [
        rf_auc,
        reduced_auc
    ]
})

print("\nFeature Ablation Comparison\n")

print(ablation_results)

ablation_results.to_csv(
    "feature_ablation_results.csv",
    index=False
)

print("\nInterpretation")

difference = rf_auc - reduced_auc

if abs(difference) < 0.01:
    print(
        "Removing the least important features had minimal impact "
        "on model performance, suggesting they contribute little "
        "predictive information."
    )

elif difference > 0:
    print(
        "Model performance decreased after removing the features, "
        "indicating that they still contributed useful information."
    )

else:
    print(
        "The reduced model slightly outperformed the full model, "
        "suggesting the removed features mainly added noise."
    )

print("\n" + "=" * 80)
print("TASK 5 - CROSS VALIDATION")
print("=" * 80)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Controlled Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

cv_results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_scaled,
        y_clf_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_results.append([
        name,
        scores.mean(),
        scores.std()
    ])

cv_results = pd.DataFrame(
    cv_results,
    columns=[
        "Model",
        "Mean AUC",
        "Std AUC"
    ]
)

print(cv_results)

print("\n" + "=" * 80)
print("TASK 6 - GRID SEARCH")
print("=" * 80)

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(
    X_train,
    y_clf_train
)

best_pipeline = grid.best_estimator_

print("\nBest Parameters")

print(grid.best_params_)

print(f"\nBest CV Score : {grid.best_score_:.4f}")

total_configurations = (
    len(param_grid["randomforestclassifier__n_estimators"])
    * len(param_grid["randomforestclassifier__max_depth"])
    * len(param_grid["randomforestclassifier__min_samples_leaf"])
)

print(f"\nConfigurations Evaluated : {total_configurations}")

print(f"Total Model Fits : {total_configurations * 5}")

print("\n" + "=" * 80)
print("TASK 7 - MANUAL LEARNING CURVE")
print("=" * 80)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

learning_curve = []

for frac in fractions:

    rows = int(frac * len(X_train))

    X_subset = X_train.iloc[:rows]

    y_subset = y_clf_train.iloc[:rows]

    best_pipeline.fit(
        X_subset,
        y_subset
    )

    train_prob = best_pipeline.predict_proba(
        X_subset
    )[:, 1]

    test_prob = best_pipeline.predict_proba(
        X_test
    )[:, 1]

    train_auc = roc_auc_score(
        y_subset,
        train_prob
    )

    test_auc = roc_auc_score(
        y_clf_test,
        test_prob
    )

    learning_curve.append([
        f"{int(frac*100)}%",
        train_auc,
        test_auc
    ])

learning_curve = pd.DataFrame(
    learning_curve,
    columns=[
        "Training Fraction",
        "Training AUC",
        "Test AUC"
    ]
)

print(learning_curve)

print("\n" + "=" * 80)
print("TASK 8 - MODEL SERIALIZATION")
print("=" * 80)

joblib.dump(
    best_pipeline,
    "best_model.pkl"
)

print("best_model.pkl saved successfully.")

print("\n" + "=" * 80)
print("TASK 9 - RELOAD MODEL")
print("=" * 80)

loaded_model = joblib.load(
    "best_model.pkl"
)

sample_rows = X_test.iloc[:2]

predictions = loaded_model.predict(
    sample_rows
)

print("\nPredictions")

print(predictions)

print("\n" + "=" * 80)
print("TASK 10 - FINAL COMPARISON")
print("=" * 80)

test_results = {
    "Logistic Regression": log_auc,
    "Controlled Decision Tree": controlled_auc,
    "Random Forest": rf_auc,
    "Gradient Boosting": gb_auc
}

summary = cv_results.copy()

summary["Test AUC"] = summary["Model"].map(test_results)

summary = summary.sort_values(
    by="Test AUC",
    ascending=False
)

print(summary)

summary.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved.")

print("\n" + "=" * 80)
print("PART 3 COMPLETED SUCCESSFULLY")
print("=" * 80)