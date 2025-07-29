

# Model Metrics for Classification and Regression Problems

## Classification Metrics

### **Binary Classification Metrics**

#### **Core Metrics**

- **Accuracy**: Overall correctness of predictions

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

    - **When to use**: Balanced datasets, equal class importance
    - **Range**: 0 to 1 (higher is better)

- **Precision**: Quality of positive predictions

```
Precision = TP / (TP + FP)
```

    - **When to use**: When false positives are costly (e.g., spam detection)
    - **Interpretation**: "When model predicts positive, how often is it correct?"

- **Recall (Sensitivity)**: Coverage of actual positives

```
Recall = TP / (TP + FN)
```

    - **When to use**: When false negatives are costly (e.g., medical diagnosis)
    - **Interpretation**: "Of all actual positives, how many did we catch?"

- **F1-Score**: Harmonic mean of precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

    - **When to use**: Imbalanced datasets, need balance between precision/recall
    - **Best for**: General-purpose classification evaluation

#### **Advanced Classification Metrics**

- **Specificity**: True negative rate

```
Specificity = TN / (TN + FP)
```

- **AUC-ROC**: Area under ROC curve
  - **Range**: 0 to 1 (0.5 = random, 1.0 = perfect)
  - **Use**: Model ranking, threshold-independent evaluation
- **AUC-PR**: Area under Precision-Recall curve
  - **Better than ROC**: For highly imbalanced datasets
- **Log Loss**: Probabilistic classification loss
  - **Lower is better**: Penalizes confident wrong predictions

### **Multi-class Classification Metrics**

- **Macro-averaged**: Average metric across all classes (treats all classes equally)
- **Micro-averaged**: Global average (accounts for class frequency)
- **Weighted-average**: Weighted by class support

```python
# Example code for classification metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Comprehensive report
print(classification_report(y_true, y_pred))
```

## Regression Metrics

### **Core Regression Metrics**

#### **Error-Based Metrics**

- **Mean Absolute Error (MAE)**:

```
MAE = (1/n) × Σ|y_true - y_pred|
```

    - **Interpretation**: Average absolute difference between actual and predicted
    - **Units**: Same as target variable
    - **Robust**: Less sensitive to outliers

- **Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ(y_true - y_pred)²
```

    - **Interpretation**: Average squared difference
    - **Sensitive**: Heavily penalizes large errors
    - **Units**: Squared units of target variable

- **Root Mean Squared Error (RMSE)**:

```
RMSE = √MSE
```

    - **Interpretation**: Square root of MSE
    - **Units**: Same as target variable
    - **Popular**: Widely used standard metric

#### **Percentage-Based Metrics**

- **Mean Absolute Percentage Error (MAPE)**:

```
MAPE = (100/n) × Σ|(y_true - y_pred)/y_true|
```

    - **Interpretation**: Average percentage error
    - **Scale-independent**: Good for comparing different datasets
    - **Problem**: Infinite when actual value is zero

#### **Coefficient-Based Metrics**

- **R² Score (Coefficient of Determination)**:

```
R² = 1 - (SS_res / SS_tot)
```

    - **Range**: -∞ to 1 (1 = perfect, 0 = as good as mean)
    - **Interpretation**: Proportion of variance explained by model
    - **Most common**: Standard regression evaluation metric

- **Adjusted R²**: Adjusts R² for number of features
  - **Better for**: Model comparison with different feature counts

```python
# Example code for regression metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np

# Basic regression metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"MAPE: {mape:.3f}%")
```

## Metric Selection Guide

### **For Classification Problems**

| Problem Type                   | Primary Metric | Secondary Metrics | Use Case                           |
|:------------------------------ |:-------------- |:----------------- |:---------------------------------- |
| **Balanced Binary**            | Accuracy       | Precision, Recall | General classification             |
| **Imbalanced Binary**          | F1-Score       | AUC-ROC, AUC-PR   | Fraud detection, medical diagnosis |
| **Cost-sensitive (FP costly)** | Precision      | Specificity       | Spam detection, security           |
| **Cost-sensitive (FN costly)** | Recall         | Sensitivity       | Medical screening, safety          |
| **Multi-class Balanced**       | Accuracy       | Macro F1          | Image classification               |
| **Multi-class Imbalanced**     | Weighted F1    | Macro/Micro F1    | Text classification                |

### **For Regression Problems**

| Problem Type             | Primary Metric | Secondary Metrics | Use Case                 |
|:------------------------ |:-------------- |:----------------- |:------------------------ |
| **General Regression**   | RMSE           | R², MAE           | House price prediction   |
| **Outlier-sensitive**    | MAE            | RMSE              | Robust predictions       |
| **Scale-independent**    | MAPE           | R²                | Cross-dataset comparison |
| **Variance explanation** | R²             | Adjusted R²       | Model interpretation     |
| **Business metrics**     | Custom         | MAE, MAPE         | Domain-specific KPIs     |

## MLflow Integration

### **Tracking Classification Metrics**

```python
with mlflow.start_run():
    # Log classification metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc_score)

    # Log confusion matrix as artifact
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
```

### **Tracking Regression Metrics**

```python
with mlflow.start_run():
    # Log regression metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mape", mape)

    # Log prediction vs actual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.savefig("predictions_plot.png")
    mlflow.log_artifact("predictions_plot.png")
```

## Common Mistakes to Avoid

### **Classification Mistakes**

- **Using accuracy for imbalanced data** → Use F1-score instead
- **Ignoring business context** → Consider cost of false positives vs false negatives
- **Not checking class distribution** → Always examine your dataset balance

### **Regression Mistakes**

- **Only using R²** → Combine with RMSE and MAE for full picture
- **Ignoring outliers** → Check MAE vs RMSE difference
- **Wrong scale interpretation** → Understand units of your metrics
