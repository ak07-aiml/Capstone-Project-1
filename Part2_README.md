# Applied AI & ML Essentials – Capstone Project

# Part 2 – Supervised Machine Learning Model

---

## Student Information

**Course:** Applied AI & ML Essentials

**Part:** Part 2 – Supervised Machine Learning

---

# Project Overview

This project demonstrates the implementation of supervised machine learning techniques using the cleaned dataset generated in Part 1. Two predictive models were developed:

1. Regression Model (Linear Regression and Ridge Regression)
2. Binary Classification Model (Logistic Regression)

The project covers the complete machine learning pipeline from preprocessing to model evaluation and regularization.

---

# Dataset

The cleaned dataset (`cleaned_data.csv`) generated in Part 1 was used.

The dataset had already undergone:

- Missing value treatment
- Duplicate removal
- Data type correction
- Category conversion
- Feature engineering
- Outlier analysis

No additional cleaning was required before modelling.

---

# Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Regression Target

The regression target is:

```
Value
```

The objective of the regression model is to predict the migration value.

---

# Classification Target

A binary target variable was created by comparing each migration value with the median value.

```python
y_clf = (Value > Value.median()).astype(int)
```

Where:

- 1 = Migration value greater than the median
- 0 = Migration value less than or equal to the median

This converts the regression problem into a binary classification problem.

---

# Feature Selection

The feature matrix contains all predictor variables except:

- Value
- Value_Log

`Value` is the prediction target.

`Value_Log` is a logarithmic transformation of the target variable and was excluded to prevent **target leakage**, ensuring the models do not indirectly access information about the prediction target.

---

# Categorical Encoding

Categorical variables were encoded using One-Hot Encoding with:

```python
pd.get_dummies(drop_first=True)
```

## Why One-Hot Encoding?

The categorical variables (such as Region and Destination Country) have no natural ordering.

Using Label Encoding would incorrectly assign numerical order to categories, introducing artificial relationships.

One-Hot Encoding creates independent binary columns for each category and prevents false ordinal relationships.

The first dummy column was dropped to reduce multicollinearity.

---

# Train-Test Split

The dataset was divided into:

- Training Set: 80%
- Testing Set: 20%

using

```python
train_test_split(random_state=42)
```

Using a fixed random state ensures reproducible results.

---

# Feature Scaling

Feature scaling was performed using StandardScaler.

The scaler was fitted only on the training dataset:

```python
scaler.fit(X_train)
```

The fitted scaler was then applied to both training and testing datasets.

## Why only fit on training data?

Fitting the scaler on the complete dataset would introduce **data leakage**, because the test data statistics (mean and standard deviation) would influence the training process.

Using only the training dataset preserves an unbiased evaluation of model performance.

---

# Linear Regression

A Linear Regression model was trained using the scaled training data.

Evaluation Metrics:

- Mean Squared Error (MSE)
- R² Score

## Results

Mean Squared Error (MSE)

> **<Replace with your output>**

R² Score

> **<Replace with your output>**

---

# Feature Importance

The model coefficients were extracted and sorted according to their absolute values.

The three most influential features were:

| Feature | Coefficient |
|----------|------------:|
| <Feature 1> | <Value> |
| <Feature 2> | <Value> |
| <Feature 3> | <Value> |

### Interpretation

A positive coefficient indicates that increasing the standardized feature value increases the predicted migration value.

A negative coefficient indicates that increasing the standardized feature value decreases the predicted migration value.

Features with larger absolute coefficient values have greater influence on the regression prediction.

---

# Ridge Regression

A Ridge Regression model was trained using:

```
alpha = 1.0
```

## Model Comparison

| Model | MSE | R² |
|------|------:|------:|
| Linear Regression | <Value> | <Value> |
| Ridge Regression | <Value> | <Value> |

### Interpretation

Ridge Regression introduces L2 regularization which penalizes excessively large coefficients.

Unlike ordinary least squares regression, Ridge Regression reduces model complexity and improves stability when predictor variables are correlated.

The parameter **alpha** controls the amount of regularization.

Higher alpha values increase coefficient shrinkage.

---

# Logistic Regression

A Logistic Regression classifier was developed for binary classification.

The model predicts whether migration value is above or below the dataset median.

---

# Class Distribution

Training class counts before modelling:

```
<Paste value_counts() output>
```

If the minority class represented fewer than 35% of the training data, `class_weight="balanced"` would be applied automatically.

Otherwise, the original class distribution is retained.

This approach avoids unnecessary oversampling while addressing class imbalance when required.

---

# Classification Performance

The following evaluation metrics were computed:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

Classification Report

```
<Paste classification report here>
```

---

# Precision and Recall

Precision

```
Precision = TP / (TP + FP)
```

Recall

```
Recall = TP / (TP + FN)
```

### Metric Selection

For this migration classification task, **Recall** is considered more important because failing to identify genuinely high migration values (False Negatives) may have a greater impact than incorrectly predicting a high migration event (False Positives).

---

# ROC Curve

The Receiver Operating Characteristic (ROC) curve was generated to evaluate the classifier across different thresholds.

Area Under Curve (AUC)

> **<Replace with output>**

### Interpretation

An AUC value close to 1 indicates excellent discrimination between the two classes.

An AUC close to 0.5 represents performance similar to random guessing.

---

# Decision Threshold Sensitivity

Predicted probabilities were converted into class labels using five different thresholds:

- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

For each threshold, the following metrics were computed:

- Precision
- Recall
- F1 Score

| Threshold | Precision | Recall | F1 Score |
|-----------:|----------:|-------:|---------:|
| <Copy threshold table generated by the script> |

### Interpretation

The threshold producing the highest F1 Score was:

> **<Replace with threshold>**

Lower thresholds generally improve Recall but reduce Precision.

Higher thresholds generally improve Precision but reduce Recall.

The preferred threshold depends on the business objective and the relative cost of false positives and false negatives.

---

# Logistic Regression Regularization

A second Logistic Regression model was trained using

```
C = 0.01
```

to study the effect of stronger regularization.

## Model Comparison

| Model | Precision | Recall | AUC |
|------|----------:|-------:|----:|
| C = 1.0 | <Value> | <Value> | <Value> |
| C = 0.01 | <Value> | <Value> | <Value> |

### Interpretation

The parameter **C** controls the inverse strength of regularization.

Smaller values of C produce stronger regularization by shrinking coefficient values.

Depending on the dataset, stronger regularization may improve generalization or reduce predictive performance if the model becomes overly constrained.

---

# Bootstrap Confidence Interval

A bootstrap experiment with **500 resamples** was performed.

For each bootstrap sample:

- Test observations were sampled with replacement.
- AUC was calculated for both Logistic Regression models.
- The AUC difference (C=1.0 − C=0.01) was recorded.

## Results

Mean AUC Difference

> **<Replace with output>**

95% Confidence Interval

Lower Limit

> **<Replace with output>**

Upper Limit

> **<Replace with output>**

### Interpretation

If the confidence interval excludes zero, the baseline model consistently outperforms the regularized model.

If the confidence interval includes zero, the observed difference is likely due to sampling variability rather than a genuine performance difference.

---

# Generated Files

The following files are generated automatically:

- regression_comparison.csv
- logistic_comparison.csv
- threshold_analysis.csv

Generated Plot

- plots/roc_curve.png

---

# Conclusion

This project successfully implemented a complete supervised machine learning pipeline using the cleaned migration dataset.

The workflow included:

- Feature selection
- One-Hot Encoding
- Leak-free train-test split
- Standardization
- Linear Regression
- Ridge Regression
- Logistic Regression
- ROC Analysis
- Threshold Optimization
- Logistic Regularization
- Bootstrap Confidence Interval Estimation

The implementation demonstrates proper machine learning practices while ensuring reproducibility, prevention of data leakage, and rigorous model evaluation.