# Applied AI & ML Essentials Capstone Project – Part 3

## Project Overview

This project demonstrates advanced machine learning techniques using ensemble models, hyperparameter tuning, feature selection, cross-validation, learning curves, and model serialization. The objective is to identify the most robust classification model while following a reproducible machine learning workflow.

---

# Dataset

- Dataset: cleaned_data.csv
- Target Variable: Value (converted into a binary classification target using the median value)

---

# Machine Learning Pipeline

The project follows the following workflow:

1. Load dataset
2. One-Hot Encode categorical features
3. Train-Test Split (80:20)
4. Feature Scaling using StandardScaler
5. Train ensemble models
6. Hyperparameter tuning using GridSearchCV
7. Cross-validation
8. Feature importance analysis
9. Feature ablation study
10. Manual learning curve
11. Model serialization using Joblib

---

# Models Implemented

- Logistic Regression
- Decision Tree (Default)
- Decision Tree (Controlled)
- Random Forest
- Gradient Boosting

---

# Decision Tree Comparison

## Default Decision Tree

Training Accuracy: **<value>**

Testing Accuracy: **<value>**

The default decision tree achieves very high training accuracy but lower testing accuracy, indicating signs of overfitting. Decision trees are considered high-variance models because they greedily choose the best split at each node without revisiting previous decisions, making them sensitive to variations in the training data.

---

## Controlled Decision Tree

Parameters used:

- max_depth = 5
- min_samples_split = 20

Training Accuracy: **<value>**

Testing Accuracy: **<value>**

The controlled decision tree reduces overfitting by limiting tree growth and preventing splits on very small sample groups. This decreases variance while introducing a small increase in bias, resulting in better generalization.

---

# Gini vs Entropy

## Gini Impurity

\[
Gini = 1 - \sum p_i^2
\]

## Entropy

\[
Entropy = -\sum p_i\log_2(p_i)
\]

A Gini impurity of **0** indicates that all samples within the node belong to a single class.

| Criterion | Test Accuracy |
|-----------|--------------:|
| Gini | <value> |
| Entropy | <value> |

---

# Random Forest

Parameters:

- n_estimators = 100
- max_depth = 10
- random_state = 42

Training Accuracy: **<value>**

Testing Accuracy: **<value>**

Test ROC-AUC: **<value>**

## Top 5 Important Features

| Feature | Importance |
|---------|-----------:|
| <Feature1> | <value> |
| <Feature2> | <value> |
| <Feature3> | <value> |
| <Feature4> | <value> |
| <Feature5> | <value> |

### Feature Importance

Random Forest computes feature importance using the average reduction in Gini impurity produced by each feature across all trees in the ensemble. Unlike linear regression coefficients, these values represent the contribution of a feature to reducing classification uncertainty rather than the direction or magnitude of a linear relationship.

---

# Bagging Concept

Random Forest uses bootstrap aggregation (bagging), where each tree is trained on a random sample of the training data selected with replacement. At every split, only a random subset of features is considered. Averaging predictions from many diverse trees significantly reduces model variance compared with a single deep decision tree, resulting in better generalization.

---

# Gradient Boosting

Parameters:

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3

Training Accuracy: **<value>**

Testing Accuracy: **<value>**

Test ROC-AUC: **<value>**

Gradient Boosting builds trees sequentially, where each tree attempts to correct the errors made by the previous ensemble, gradually improving predictive performance.

---

# Feature Ablation Study

Five least important features identified from the Random Forest model were removed.

| Model | Test ROC-AUC |
|------|-------------:|
| Full Feature Set | <value> |
| Reduced Feature Set | <value> |

### Interpretation

If the reduced model achieves similar ROC-AUC, the removed features contribute little predictive value and may safely be removed in production to reduce inference cost and maintenance complexity.

If ROC-AUC decreases noticeably, the removed features still provide useful predictive information and should be retained.

---

# Cross-Validation Results

5-Fold Stratified Cross Validation (ROC-AUC)

| Model | Mean AUC | Std AUC |
|------|---------:|--------:|
| Logistic Regression | <value> | <value> |
| Controlled Decision Tree | <value> | <value> |
| Random Forest | <value> | <value> |
| Gradient Boosting | <value> | <value> |

Cross-validation provides a more reliable estimate of model generalization because each sample is used for both training and validation across different folds, reducing dependence on a single train-test split.

---

# Hyperparameter Tuning

Pipeline:

```
SimpleImputer
→ StandardScaler
→ RandomForestClassifier
```

Parameter Grid

```python
{
    'randomforestclassifier__n_estimators':[50,100,200],
    'randomforestclassifier__max_depth':[5,10,None],
    'randomforestclassifier__min_samples_leaf':[1,5]
}
```

Total Configurations:

**18**

Total Model Fits:

**90**

Best Parameters

```text
<best_params>
```

Best Cross Validation Score

**<best_score>**

Grid Search exhaustively evaluates every parameter combination, whereas Randomized Search evaluates only a subset, reducing computation time while potentially finding near-optimal solutions.

---

# Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
|------------------|-------------:|---------:|
|20%|<value>|<value>|
|40%|<value>|<value>|
|60%|<value>|<value>|
|80%|<value>|<value>|
|100%|<value>|<value>|

### Bias-Variance Interpretation

As the training set grows, training AUC is expected to decrease slightly while test AUC generally improves. If the test AUC continues to increase at the largest training size, the model is likely data-limited. If the test AUC plateaus, model capacity rather than data quantity is likely the limiting factor.

---

# Model Serialization

The best-performing pipeline was saved as:

```
best_model.pkl
```

The model was successfully reloaded using `joblib.load()` and used to make predictions on unseen samples.

---

# Final Model Comparison

| Model | 5-Fold Mean AUC | 5-Fold Std AUC | Test AUC |
|------|----------------:|---------------:|---------:|
| Logistic Regression | <value> | <value> | <value> |
| Controlled Decision Tree | <value> | <value> | <value> |
| Random Forest | <value> | <value> | <value> |
| Gradient Boosting | <value> | <value> | <value> |

---

# Recommendation

The **<best model>** is recommended for deployment because it achieved the strongest balance between cross-validation performance and test-set ROC-AUC while maintaining good generalization. Compared with the other models, it demonstrates greater robustness across folds and is therefore expected to perform more reliably on unseen data. The serialized pipeline (`best_model.pkl`) also simplifies deployment by encapsulating preprocessing and prediction within a single reusable model.

---

# Repository Contents

```
Capstone Project/
│
├── Part 1.py
├── Part 2.py
├── Part 3.py
├── cleaned_data.csv
├── best_model.pkl
├── README.md
└── plots/
```