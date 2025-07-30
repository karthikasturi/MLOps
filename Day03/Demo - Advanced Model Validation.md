# Advanced Demo & Lab: Model Validation with Training/Inference Monitoring

Since yesterday's content covered basic model validation, let's focus on **advanced monitoring during training and production inference** using the Breast Cancer dataset from Kaggle.

## 1. Setup: Load Kaggle Breast Cancer Dataset

### **Demo: Dataset Preparation**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(data.feature_names)}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train a simple model for demo
model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=200)
model.fit(X_train_scaled, y_train)

# Create models directory and save files
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'monitored_model.pkl')  # Save in current directory
joblib.dump(scaler, 'scaler.pkl')          # Save in current directory


print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples") 
print(f"Test set: {X_test.shape[0]} samples")

print("✅ Model and scaler saved successfully")
#print("Files created:", os.listdir('.'))
```

## 2. Training Phase Monitoring Demo

### **A. Custom Training Loop with Monitoring**

```python
import mlflow
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
import time

# Set up MLflow
mlflow.set_experiment("day3_training_monitoring")

class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epoch_times = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.epoch_times.append(epoch_time)

        # MLflow logging
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("epoch_time", epoch_time, step=epoch)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")

    def check_early_stopping(self, val_loss, patience=5):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                self.early_stopped = True
                return True
        return False

    def plot_training_curves(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss Curves - Convergence/Divergence Detection')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy Over Epochs - Overfitting Detection')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        # Training time per epoch
        ax3.plot(epochs, self.epoch_times, 'g-', marker='o')
        ax3.set_title('Training Time Per Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True)

        # Overfitting detection (gap between train and val)
        gap = np.array(self.train_accuracies) - np.array(self.val_accuracies)
        ax4.plot(epochs, gap, 'purple', label='Train-Val Gap')
        ax4.axhline(y=0.05, color='r', linestyle='--', label='Overfitting Threshold')
        ax4.set_title('Overfitting Detection (Train-Val Gap)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Gap')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('training_monitoring.png', dpi=150, bbox_inches='tight')
        plt.show()
        return fig

def train_with_monitoring(X_train, X_val, y_train, y_val, max_epochs=100, patience=10):
    monitor = TrainingMonitor()

    with mlflow.start_run(run_name="monitored_training"):
        # Log hyperparameters
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("hidden_layers", "(100, 50)")
        mlflow.log_param("learning_rate", 0.001)

        # Use MLPClassifier with warm_start for epoch-by-epoch training
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            learning_rate_init=0.001,
            max_iter=1,  # Train one iteration at a time
            warm_start=True,
            random_state=42,
            early_stopping=False  # We'll implement our own
        )

        print("Starting training with monitoring...")
        print("="*80)

        for epoch in range(1, max_epochs + 1):
            start_time = time.time()

            # Train for one epoch
            model.fit(X_train, y_train)

            # Calculate metrics
            train_pred_proba = model.predict_proba(X_train)
            val_pred_proba = model.predict_proba(X_val)

            train_loss = log_loss(y_train, train_pred_proba)
            val_loss = log_loss(y_val, val_pred_proba)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)

            epoch_time = time.time() - start_time

            # Log metrics
            monitor.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, epoch_time)

            # Early stopping check
            if monitor.check_early_stopping(val_loss, patience):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                mlflow.log_param("early_stopped", True)
                mlflow.log_param("stopped_at_epoch", epoch)
                break

        # Final logging
        mlflow.log_param("total_epochs", len(monitor.train_losses))
        mlflow.log_metric("best_val_loss", monitor.best_val_loss)
        mlflow.log_metric("final_train_acc", monitor.train_accuracies[-1])
        mlflow.log_metric("final_val_acc", monitor.val_accuracies[-1])

        # Save monitoring plots
        fig = monitor.plot_training_curves()
        mlflow.log_artifact('training_monitoring.png')

        # Save final model
        import joblib
        joblib.dump(model, 'monitored_model.pkl')
        mlflow.log_artifact('monitored_model.pkl')

        return model, monitor

# Run the training with monitoring
model, monitor = train_with_monitoring(X_train_scaled, X_val_scaled, y_train, y_val)
```

### **B. Learning Curve Analysis**

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y):
    """Plot learning curves to detect overfitting/underfitting"""

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves - Overfitting/Underfitting Detection')
    plt.legend()
    plt.grid(True)

    # Interpretation
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.05:
        plt.text(0.5, 0.02, f'⚠️ Possible Overfitting (Gap: {final_gap:.3f})', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='yellow'))
    elif val_mean[-1] < 0.85:
        plt.text(0.5, 0.02, f'⚠️ Possible Underfitting (Val Acc: {val_mean[-1]:.3f})', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='orange'))
    else:
        plt.text(0.5, 0.02, '✅ Good Generalization', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))

    plt.show()
    return train_sizes, train_scores, val_scores

# Analyze learning curves
train_sizes, train_scores, val_scores = plot_learning_curves(
    RandomForestClassifier(n_estimators=100, random_state=42), 
    X_train_scaled, y_train
)
```

## 3. Production Inference Monitoring Demo

In the python virtual environment install the following packages.

```bash
pip install streamlit plotly
```

### **A. Enhanced Flask API with Monitoring**

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
import time
import json
import os
from datetime import datetime
import logging
from collections import deque
import threading

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceMonitor:
    def __init__(self):
        self.prediction_times = deque(maxlen=1000)  # Last 1000 predictions
        self.predictions_log = deque(maxlen=1000)
        self.error_count = 0
        self.total_predictions = 0
        self.input_stats = {'mean': [], 'std': []}
        self.lock = threading.Lock()

    def log_prediction(self, features, prediction, latency, error=None):
        with self.lock:
            timestamp = datetime.now().isoformat()

            if error:
                self.error_count += 1
                log_entry = {
                    'timestamp': timestamp,
                    'error': str(error),
                    'latency': latency
                }
            else:
                self.total_predictions += 1
                self.prediction_times.append(latency)

                # Track input distribution for drift detection
                features_array = np.array(features)
                self.input_stats['mean'].append(np.mean(features_array))
                self.input_stats['std'].append(np.std(features_array))

                log_entry = {
                    'timestamp': timestamp,
                    'features_mean': np.mean(features_array),
                    'features_std': np.std(features_array),
                    'prediction': int(prediction),
                    'latency': latency
                }

            self.predictions_log.append(log_entry)

    def get_metrics(self):
        with self.lock:
            if not self.prediction_times:
                return {'error': 'No predictions yet'}

            return {
                'total_predictions': self.total_predictions,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(1, self.total_predictions + self.error_count),
                'avg_latency_ms': np.mean(self.prediction_times) * 1000,
                'p95_latency_ms': np.percentile(self.prediction_times, 95) * 1000,
                'p99_latency_ms': np.percentile(self.prediction_times, 99) * 1000,
                'input_drift_score': self._calculate_drift_score()
            }

    def _calculate_drift_score(self):
        """Simple drift detection based on input statistics"""
        if len(self.input_stats['mean']) < 50:
            return 0.0

        # Compare recent vs historical means
        recent_mean = np.mean(self.input_stats['mean'][-50:])
        historical_mean = np.mean(self.input_stats['mean'][:-50]) if len(self.input_stats['mean']) > 50 else recent_mean

        drift_score = abs(recent_mean - historical_mean) / (historical_mean + 1e-8)
        return min(drift_score, 1.0)  # Cap at 1.0

def load_model_and_scaler():
    """Load model and scaler with multiple path fallbacks"""
    model = None
    scaler = None

    # Define possible paths for model files
    model_paths = [
        'monitored_model.pkl',           # Current directory
        'models/monitored_model.pkl',    # Models directory
        'models/model.pkl',              # Alternative name
        'model.pkl'                      # Alternative current directory
    ]

    scaler_paths = [
        'scaler.pkl',                    # Current directory
        'models/scaler.pkl',             # Models directory
    ]

    # Try to load model
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info(f"✅ Model loaded from: {model_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                continue

    # Try to load scaler
    for scaler_path in scaler_paths:
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler loaded from: {scaler_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load scaler from {scaler_path}: {e}")
                continue

    return model, scaler

def create_fallback_model():
    """Create a simple fallback model if no model is found"""
    try:
        logger.info("🔄 Creating fallback model...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        # Load data and create simple model
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Save the fallback model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/monitored_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')

        logger.info("✅ Fallback model created and saved")
        return model, scaler

    except Exception as e:
        logger.error(f"❌ Failed to create fallback model: {e}")
        return None, None

# Load model and scaler with fallback
model, scaler = load_model_and_scaler()

# If no model found, create a fallback
if model is None:
    logger.warning("⚠️ No model found, creating fallback model...")
    model, scaler = create_fallback_model()

# Initialize monitor
monitor = InferenceMonitor()

# Log final status
if model is not None and scaler is not None:
    logger.info("🎉 Model and scaler ready for predictions")
else:
    logger.error("❌ Failed to load or create model and scaler")

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    try:
        if model is None:
            raise ValueError("Model not loaded")

        if scaler is None:
            raise ValueError("Scaler not loaded")

        data = request.json

        # Validate input
        if not data or 'features' not in data:
            raise ValueError("Missing 'features' in request")

        features = np.array(data['features'])

        # Handle different input shapes
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Validate feature count (breast cancer dataset has 30 features)
        if features.shape[1] != 30:
            raise ValueError(f"Expected 30 features, got {features.shape[1]}")

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].max()

        latency = time.time() - start_time

        # Log prediction
        monitor.log_prediction(data['features'], prediction, latency)

        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'latency_ms': latency * 1000,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Prediction made: {prediction}, Latency: {latency:.3f}s")
        return jsonify(response)

    except Exception as e:
        latency = time.time() - start_time
        monitor.log_prediction([], 0, latency, error=e)
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get inference monitoring metrics"""
    return jsonify(monitor.get_metrics())

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    metrics = monitor.get_metrics()

    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat(),
        'uptime_predictions': monitor.total_predictions,
        'error_rate': metrics.get('error_rate', 0),
        'avg_latency_ms': metrics.get('avg_latency_ms', 0)
    }

    # Set status based on metrics and model availability
    if model is None or scaler is None:
        health_status['status'] = 'unhealthy'
        health_status['warning'] = 'Model or scaler not loaded'
    elif metrics.get('error_rate', 0) > 0.05:  # 5% error rate threshold
        health_status['status'] = 'degraded'
        health_status['warning'] = 'High error rate detected'
    elif metrics.get('avg_latency_ms', 0) > 1000:  # 1 second threshold
        health_status['status'] = 'slow'
        health_status['warning'] = 'High latency detected'

    return jsonify(health_status)

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check file system and model status"""
    return jsonify({
        'current_directory': os.getcwd(),
        'files_in_current_dir': os.listdir('.'),
        'models_dir_exists': os.path.exists('models'),
        'models_dir_contents': os.listdir('models') if os.path.exists('models') else [],
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'model_type': str(type(model)) if model else None,
        'scaler_type': str(type(scaler)) if scaler else None
    })

@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload model and scaler"""
    global model, scaler

    try:
        logger.info("🔄 Reloading model and scaler...")
        model, scaler = load_model_and_scaler()

        if model is None:
            model, scaler = create_fallback_model()

        if model is not None and scaler is not None:
            logger.info("✅ Model and scaler reloaded successfully")
            return jsonify({
                'status': 'success',
                'message': 'Model and scaler reloaded successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.error("❌ Failed to reload model and scaler")
            return jsonify({
                'status': 'error',
                'message': 'Failed to reload model and scaler'
            }), 500

    except Exception as e:
        logger.error(f"❌ Error reloading model: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    if model is None or scaler is None:
        logger.warning("⚠️ Starting API without properly loaded model/scaler")
    else:
        logger.info("🚀 Starting API with loaded model and scaler")

    app.run(debug=True, host='0.0.0.0', port=5001)
```

### **B. Production Monitoring Dashboard (Streamlit)**

```python
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np

st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")

st.title("Production Model Monitoring Dashboard")

# API endpoint
API_URL = "http://localhost:5001"

# Sidebar controls
st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)
if st.sidebar.button("Manual Refresh") or auto_refresh:
    if auto_refresh:
        time.sleep(10)
        st.rerun()

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

try:
    # Get health status
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_data = health_response.json()

    # Get metrics
    metrics_response = requests.get(f"{API_URL}/metrics", timeout=5)
    metrics_data = metrics_response.json()

    # Display key metrics
    with col1:
        st.metric("Status", health_data.get('status', 'unknown').upper())

    with col2:
        st.metric("Total Predictions", health_data.get('uptime_predictions', 0))

    with col3:
        error_rate = metrics_data.get('error_rate', 0) * 100
        st.metric("Error Rate", f"{error_rate:.2f}%")

    with col4:
        avg_latency = metrics_data.get('avg_latency_ms', 0)
        st.metric("Avg Latency", f"{avg_latency:.1f}ms")

    # Latency distribution
    st.subheader("Performance Metrics")

    col5, col6 = st.columns(2)

    with col5:
        # Latency metrics
        latency_data = {
            'Metric': ['Average', 'P95', 'P99'],
            'Latency (ms)': [
                metrics_data.get('avg_latency_ms', 0),
                metrics_data.get('p95_latency_ms', 0),
                metrics_data.get('p99_latency_ms', 0)
            ]
        }

        fig_latency = px.bar(pd.DataFrame(latency_data), x='Metric', y='Latency (ms)',
                            title="Latency Distribution")
        st.plotly_chart(fig_latency, use_container_width=True)

    with col6:
        # Data drift score
        drift_score = metrics_data.get('input_drift_score', 0)

        fig_drift = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = drift_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Data Drift Score"},
            delta = {'reference': 0.1},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.1], 'color': "lightgray"},
                    {'range': [0.1, 0.3], 'color': "yellow"},
                    {'range': [0.3, 1], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.3}}))

        st.plotly_chart(fig_drift, use_container_width=True)

    # Test prediction section
    st.subheader(" Test Prediction")

    col7, col8 = st.columns([2, 1])

    with col7:
        st.write("Enter feature values for testing:")
        # Create input fields for all 30 features
        feature_values = []
        cols = st.columns(5)  # 5 columns for layout

        # Sample default values (you can adjust these)
        default_values = [13.0, 20.0, 85.0, 500.0, 0.1, 0.09, 0.03, 0.02, 0.18, 0.06,
                         0.4, 1.2, 3.0, 40.0, 0.007, 0.02, 0.01, 0.006, 0.02, 0.003,
                         15.0, 25.0, 100.0, 700.0, 0.15, 0.25, 0.08, 0.08, 0.3, 0.08]

        for i in range(30):
            with cols[i % 5]:
                val = st.number_input(f"F{i+1}", value=float(default_values[i]), 
                                    key=f"feature_{i}", step=0.01)
                feature_values.append(val)

    with col8:
        st.write("Prediction Results:")
        if st.button(" Make Prediction", type="primary"):
            try:
                pred_response = requests.post(
                    f"{API_URL}/predict",
                    json={'features': feature_values},
                    timeout=5
                )

                if pred_response.status_code == 200:
                    pred_data = pred_response.json()

                    prediction = pred_data.get('prediction', 0)
                    probability = pred_data.get('probability', 0)
                    latency = pred_data.get('latency_ms', 0)

                    result_color = "green" if prediction == 1 else "red"
                    result_text = "Benign" if prediction == 1 else "Malignant"

                    st.markdown(f"**Result:** :{result_color}[{result_text}]")
                    st.write(f"**Confidence:** {probability:.3f}")
                    st.write(f"**Latency:** {latency:.1f}ms")
                else:
                    st.error(f"Prediction failed: {pred_response.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")

except Exception as e:
    st.error(f"Dashboard error: {e}")
    st.write("Make sure the Flask API is running on localhost:5001")

# Warning thresholds
st.subheader("Alert Thresholds")
col9, col10, col11 = st.columns(3)

with col9:
    if error_rate > 5:
        st.error(f"High Error Rate: {error_rate:.2f}%")
    else:
        st.success("✅ Error Rate Normal")

with col10:
    if avg_latency > 1000:
        st.error(f" High Latency: {avg_latency:.1f}ms")
    else:
        st.success("✅ Latency Normal")

with col11:
    if drift_score > 0.3:
        st.error(f"Data Drift Detected: {drift_score:.3f}")
    elif drift_score > 0.1:
        st.warning(f" Potential Drift: {drift_score:.3f}")
    else:
        st.success("✅ No Drift Detected")
```

## 🧪 4. Complete Lab Exercise

### **Lab Steps:**

#### **Step 1: Training Phase Monitoring (30 minutes)**

1. **Run the monitored training script**
2. **Analyze the training curves** - identify overfitting/underfitting
3. **Experiment with different patience values** for early stopping
4. **Track all metrics in MLflow**

#### **Step 2: Production Monitoring Setup (25 minutes)**

1. **Deploy the enhanced Flask API** with monitoring
2. **Create the Streamlit monitoring dashboard**
3. **Make 20+ test predictions** to generate monitoring data
4. **Observe latency and error rate metrics**

#### **Step 3: Data Drift Simulation (15 minutes)**

1. **Make predictions with "normal" data**
2. **Make predictions with modified data** (multiply features by 2.0)
3. **Observe drift score changes** in the dashboard
4. **Document when alerts trigger**

#### **Step 4: Performance Analysis (20 minutes)**

1. **Load test the API** with concurrent requests
2. **Monitor latency under load**
3. **Identify performance bottlenecks**
4. **Propose optimization strategies**

### **Expected Lab Outcomes:**

✅ **Training monitoring plots** showing loss/accuracy curves
✅ **Early stopping demonstration** with patience mechanism
✅ **Production API** with comprehensive monitoring
✅ **Live dashboard** showing real-time metrics
✅ **Data drift detection** in action
✅ **Performance benchmarks** and optimization insights

### **Success Criteria:**

- Training curves clearly show convergence/divergence patterns
- Early stopping triggers appropriately based on validation loss
- Production API responds with <500ms latency for 95% of requests
- Dashboard updates monitoring metrics in real-time
- Data drift alerts trigger when input distribution changes significantly
- Complete documentation of monitoring insights and recommendations

This comprehensive lab provides hands-on experience with both training and production monitoring, using realistic scenarios that students will encounter in production MLOps environments.
