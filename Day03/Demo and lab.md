

# Day 3 Demo & Lab: Model Validation, Metrics, and Deployment (Flask/Streamlit)



## 1. Model Validation Demo

### **A. Model Training & Evaluation**

**Objective:** Show how to assess if a model generalizes beyond its training data.

```python
# 1. Prepare Data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = load_breast_cancer()
X, y = data.data[:, :10], data.target  # Use first 10 features for consistency

# 2. Split Data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate with Multiple Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

**Show how cross-validation provides a more robust validation:**

```python
# 5. Cross-validation
cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
print("5-fold CV F1-Scores:", cv_f1)
print("Mean F1:", cv_f1.mean())
```

## 2. Metrics Monitoring Demo

**Goal:** Track training progress and metrics for ongoing monitoring.

```python
# Simple loss/accuracy monitoring is manual in scikit-learn.
# However, in your training loop, always print/log metrics after each epoch or during cross-validation runs.

# For extended monitoring:
import mlflow
mlflow.set_experiment("day3_model_validation")
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    # Save the model as well for deployment
    import joblib
    joblib.dump(model, "rf_breast_cancer.pkl")
    mlflow.log_artifact("rf_breast_cancer.pkl")
```

## 3. Simple Deployment Demo: Flask

**Objective:** Deploy the trained model as a REST API.

**Flask App (`app.py`):**

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("rf_breast_cancer.pkl")  # Load your trained model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expect {'features': [...]}
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
```

**Test with:**

```bash
curl -X POST http://localhost:5001/predict -H 'Content-Type: application/json' \
  -d '{"features": [15.2, 21.3, 98.2, 700.0, 0.1, 0.2, 0.07, 0.07, 0.18, 0.05]}'
```

## 🌐 4. Simple Deployment Demo: Streamlit

**Objective:** Create a graphical web app for human-friendly demo.

**Streamlit App (`app_streamlit.py`):**

```python
import streamlit as st
import numpy as np
import joblib

model = joblib.load("rf_breast_cancer.pkl")
st.title("Breast Cancer Prediction Demo")

st.write("Input the features for prediction:")
inputs = []
for i in range(10):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    inp_array = np.array(inputs).reshape(1, -1)
    pred = model.predict(inp_array)[0]
    st.write("Prediction:", "Malignant" if pred == 0 else "Benign")
```

**Run with:**

```bash
streamlit run app_streamlit.py
```
