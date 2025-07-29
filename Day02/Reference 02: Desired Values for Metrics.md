

# Desired Values for Classification and Regression Metrics

## Classification Metrics - Desired Values

### **Binary Classification**

| Metric          | Range     | Desired Value     | Interpretation                           | Business Context                         |
|:--------------- |:--------- |:----------------- |:---------------------------------------- |:---------------------------------------- |
| **Accuracy**    | 0.0 - 1.0 | **≥ 0.80** (80%+) | 80%+ correct predictions                 | General baseline for balanced datasets   |
| **Precision**   | 0.0 - 1.0 | **≥ 0.85** (85%+) | 85%+ of positive predictions are correct | Critical when false positives are costly |
| **Recall**      | 0.0 - 1.0 | **≥ 0.80** (80%+) | Catch 80%+ of actual positive cases      | Critical when false negatives are costly |
| **F1-Score**    | 0.0 - 1.0 | **≥ 0.75** (75%+) | Balanced precision-recall performance    | Good general-purpose target              |
| **Specificity** | 0.0 - 1.0 | **≥ 0.85** (85%+) | 85%+ of negatives correctly identified   | Important for ruling out conditions      |

### **Advanced Classification Metrics**

| Metric       | Range     | Desired Value     | Interpretation                         | When to Aim Higher               |
|:------------ |:--------- |:----------------- |:-------------------------------------- |:-------------------------------- |
| **AUC-ROC**  | 0.0 - 1.0 | **≥ 0.80** (80%+) | Model discrimination ability           | 0.90+ for high-stakes decisions  |
| **AUC-PR**   | 0.0 - 1.0 | **≥ 0.70** (70%+) | Performance on imbalanced data         | 0.85+ for rare event detection   |
| **Log Loss** | 0.0 - ∞   | **≤ 0.50**        | Lower = better probability calibration | <0.30 for well-calibrated models |

### **Domain-Specific Targets**

| Application           | Accuracy | Precision | Recall   | F1-Score | Priority                                        |
|:--------------------- |:-------- |:--------- |:-------- |:-------- |:----------------------------------------------- |
| **Medical Diagnosis** | ≥85%     | ≥90%      | **≥95%** | ≥90%     | **Recall** (catch all cases)                    |
| **Spam Detection**    | ≥95%     | **≥98%**  | ≥85%     | ≥90%     | **Precision** (avoid blocking important emails) |
| **Fraud Detection**   | ≥90%     | ≥80%      | **≥90%** | ≥85%     | **Recall** (catch all fraud)                    |
| **Marketing**         | ≥75%     | ≥70%      | ≥70%     | **≥70%** | **Balance** (cost-effective targeting)          |
| **Quality Control**   | ≥95%     | **≥95%**  | ≥90%     | ≥92%     | **Precision** (minimize false alarms)           |

## Regression Metrics - Desired Values

### **Core Regression Metrics**

| Metric       | Range     | Desired Value             | Interpretation                | Context                     |
|:------------ |:--------- |:------------------------- |:----------------------------- |:--------------------------- |
| **R² Score** | -∞ to 1.0 | **≥ 0.70** (70%+)         | Explains 70%+ of variance     | 0.80+ for predictive models |
| **MAE**      | 0 to ∞    | **≤ 10% of target range** | Average error within 10%      | Domain-dependent            |
| **RMSE**     | 0 to ∞    | **≤ 15% of target range** | RMS error within 15%          | Should be close to MAE      |
| **MAPE**     | 0% to ∞   | **≤ 10%**                 | Average percentage error <10% | 5% for high accuracy needs  |

### **Domain-Specific Regression Targets**

| Application                | R² Score | MAE        | RMSE       | MAPE | Business Impact                |
|:-------------------------- |:-------- |:---------- |:---------- |:---- |:------------------------------ |
| **House Price Prediction** | ≥0.75    | ≤\$20K     | ≤\$30K     | ≤8%  | \$20K average error acceptable |
| **Stock Price Prediction** | ≥0.60    | ≤\$5       | ≤\$8       | ≤5%  | High accuracy needed           |
| **Sales Forecasting**      | ≥0.80    | ≤500 units | ≤750 units | ≤10% | Inventory planning accuracy    |
| **Energy Consumption**     | ≥0.85    | ≤100 kWh   | ≤150 kWh   | ≤7%  | Grid management precision      |
| **Temperature Prediction** | ≥0.90    | ≤2°C       | ≤3°C       | ≤5%  | Weather forecasting accuracy   |

## Performance Categories

### **Classification Performance Levels**

| Performance Level | Accuracy | F1-Score | Description                       | Use Case                 |
|:----------------- |:-------- |:-------- |:--------------------------------- |:------------------------ |
| **Excellent**     | ≥90%     | ≥90%     | Production-ready, high confidence | Critical applications    |
| **Good**          | 80-90%   | 75-90%   | Acceptable for most use cases     | Business applications    |
| **Fair**          | 70-80%   | 65-75%   | Needs improvement, baseline       | Development/testing      |
| **Poor**          | <70%     | <65%     | Requires major changes            | Research/experimentation |

### **Regression Performance Levels**

| Performance Level | R² Score | MAPE   | Description             | Business Decision            |
|:----------------- |:-------- |:------ |:----------------------- |:---------------------------- |
| **Excellent**     | ≥85%     | ≤5%    | Deploy to production    | High confidence predictions  |
| **Good**          | 70-85%   | 5-10%  | Acceptable performance  | Business use with monitoring |
| **Fair**          | 50-70%   | 10-20% | Improvement needed      | Limited business value       |
| **Poor**          | <50%     | >20%   | Major revision required | Back to development          |

## Context-Dependent Considerations

### **When to Aim Higher**

- **Safety-critical applications**: Medical, automotive, aviation (≥95% accuracy)
- **Financial applications**: Fraud, trading, credit (≥90% precision)
- **Legal/compliance**: Document classification (≥95% recall)
- **High-volume operations**: Manufacturing QC (≥98% precision)

### **When Lower Values May Be Acceptable**

- **Exploratory analysis**: Research, hypothesis testing (≥60% accuracy)
- **Human-in-the-loop systems**: Recommendations, suggestions (≥70% accuracy)
- **Early-stage prototypes**: Proof of concept (≥65% accuracy)
- **Highly imbalanced rare events**: Natural disasters, equipment failure (≥50% recall acceptable if precision is high)

## MLflow Target Tracking

### **Setting Performance Thresholds in Code**

```python
# Define performance thresholds
PERFORMANCE_THRESHOLDS = {
    'classification': {
        'accuracy_min': 0.80,
        'precision_min': 0.85,
        'recall_min': 0.80,
        'f1_min': 0.75
    },
    'regression': {
        'r2_min': 0.70,
        'mape_max': 0.10,
        'mae_max': 0.15  # As fraction of target range
    }
}

# Log thresholds and check performance
with mlflow.start_run():
    # Log actual metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log thresholds for reference
    mlflow.log_param("accuracy_threshold", PERFORMANCE_THRESHOLDS['classification']['accuracy_min'])
    mlflow.log_param("f1_threshold", PERFORMANCE_THRESHOLDS['classification']['f1_min'])

    # Performance check
    meets_accuracy = accuracy >= PERFORMANCE_THRESHOLDS['classification']['accuracy_min']
    meets_f1 = f1 >= PERFORMANCE_THRESHOLDS['classification']['f1_min']

    mlflow.log_metric("meets_accuracy_threshold", int(meets_accuracy))
    mlflow.log_metric("meets_f1_threshold", int(meets_f1))

    # Overall performance status
    production_ready = meets_accuracy and meets_f1
    mlflow.log_param("production_ready", production_ready)
```

## Red Flags - When Values Indicate Problems

### **Classification Warning Signs**

- **Accuracy >> Precision/Recall**: Model may be predicting majority class only
- **Perfect scores (1.0)**: Likely overfitting or data leakage
- **Precision very high, Recall very low**: Model too conservative
- **Recall very high, Precision very low**: Model too aggressive
- **F1 << Accuracy**: Severe class imbalance issues

### **Regression Warning Signs**

- **R² < 0**: Model performs worse than predicting the mean
- **RMSE >> MAE**: Presence of significant outliers
- **MAPE > 50%**: Model essentially random for percentage-based decisions
- **R² high but RMSE high**: Good relative performance but high absolute errors

## Quick Reference Card

### **Minimum Acceptable Values (Production)**

```
Classification:
✅ Accuracy ≥ 80%
✅ F1-Score ≥ 75%
✅ AUC-ROC ≥ 80%

Regression:
✅ R² ≥ 70%
✅ MAPE ≤ 10%
✅ RMSE ≤ 15% of target range
```

### **Excellence Targets**

```
Classification:
🎯 Accuracy ≥ 90%
🎯 F1-Score ≥ 85%
🎯 AUC-ROC ≥ 90%

Regression:
🎯 R² ≥ 85%
🎯 MAPE ≤ 5%
🎯 RMSE ≤ 10% of target range
```

**Remember**: These are general guidelines. Always consider your specific domain, business requirements, cost of errors, and available data quality when setting performance targets. The "desired value" should align with business value, not just mathematical optimization.
