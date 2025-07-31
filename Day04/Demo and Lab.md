# Day 4 Demo & Lab: LLM Observability Basics - Step by Step Guide

## Demo 1: Key Differences Between Traditional ML vs LLM Monitoring

### **Setup: Comparison Environment**

```python
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import openai  # For LLM demonstration
import matplotlib.pyplot as plt

print("🔄 Setting up Traditional ML vs LLM Comparison Demo")
```

### **Step 1: Traditional ML Model Monitoring Demo**

```python
# Traditional ML Model Setup
print("📊 Traditional ML Model Demo")
print("="*50)

# Load dataset
data = load_breast_cancer()
X, y = data.data[:, :10], data.target  # Use first 10 features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train traditional model
traditional_model = RandomForestClassifier(n_estimators=100, random_state=42)
traditional_model.fit(X_train, y_train)

# Traditional ML prediction with monitoring
def monitor_traditional_prediction(model, features):
    start_time = time.time()

    # Prediction (deterministic)
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]

    # Monitoring data
    monitoring_data = {
        'timestamp': datetime.now().isoformat(),
        'input_type': 'structured_numerical',
        'input_shape': len(features),
        'prediction_type': 'binary_classification',
        'prediction': int(prediction),
        'confidence': float(max(probability)),
        'latency_ms': (time.time() - start_time) * 1000,
        'deterministic': True,  # Same input = same output
        'cost_model': 'fixed_per_inference'
    }

    return monitoring_data

# Test traditional ML monitoring
sample_features = X_test[0]
result1 = monitor_traditional_prediction(traditional_model, sample_features)
result2 = monitor_traditional_prediction(traditional_model, sample_features)  # Same input

print(f"Traditional ML - First prediction: {result1['prediction']}")
print(f"Traditional ML - Second prediction: {result2['prediction']}")
print(f"Deterministic: {result1['prediction'] == result2['prediction']}")
print(f"Average latency: {result1['latency_ms']:.2f}ms")
```

### **Step 2: LLM Model Monitoring Demo**

```python
# LLM Model Setup (Simulated)
print("\n🤖 LLM Model Demo")
print("="*50)

# Simulated LLM response (since we don't want to use real API keys in demo)
class SimulatedLLM:
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
        self.responses = [
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "Plants use photosynthesis to create food from sunlight, water, and carbon dioxide.",
            "Photosynthesis allows plants to make glucose using light energy from the sun.",
            "Through photosynthesis, plants transform solar energy into chemical energy."
        ]

    def generate_response(self, prompt, temperature=0.7):
        # Simulate variable response time
        time.sleep(np.random.uniform(0.5, 2.0))

        # Non-deterministic response selection
        if temperature > 0.5:
            response = np.random.choice(self.responses)
        else:
            response = self.responses[0]  # More deterministic

        # Simulate token counting
        input_tokens = len(prompt.split())
        output_tokens = len(response.split())

        return {
            'response': response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }

# LLM prediction with monitoring
def monitor_llm_prediction(llm_model, prompt, temperature=0.7):
    start_time = time.time()

    # LLM prediction (non-deterministic)
    result = llm_model.generate_response(prompt, temperature)

    # Monitoring data
    monitoring_data = {
        'timestamp': datetime.now().isoformat(),
        'input_type': 'unstructured_text',
        'input_length': len(prompt),
        'prompt': prompt,
        'response': result['response'],
        'response_length': len(result['response']),
        'input_tokens': result['input_tokens'],
        'output_tokens': result['output_tokens'],
        'total_tokens': result['total_tokens'],
        'cost_usd': result['total_tokens'] * 0.00002,  # $0.02 per 1K tokens
        'latency_ms': (time.time() - start_time) * 1000,
        'temperature': temperature,
        'deterministic': False,  # Same input can = different output
        'cost_model': 'token_based'
    }

    return monitoring_data

# Test LLM monitoring
llm = SimulatedLLM()
prompt = "What is photosynthesis?"

result1 = monitor_llm_prediction(llm, prompt, temperature=0.7)
result2 = monitor_llm_prediction(llm, prompt, temperature=0.7)  # Same input

print(f"LLM - First response: {result1['response'][:50]}...")
print(f"LLM - Second response: {result2['response'][:50]}...")
print(f"Same response: {result1['response'] == result2['response']}")
print(f"Cost per request: ${result1['cost_usd']:.4f}")
print(f"Token usage: {result1['total_tokens']} tokens")
```

### **Step 3: Visual Comparison**

```python
# Create comparison visualization
comparison_data = {
    'Aspect': ['Input Type', 'Output Type', 'Determinism', 'Cost Model', 'Avg Latency', 'Main Metrics'],
    'Traditional ML': ['Structured Numbers', 'Fixed Classes', 'Deterministic', 'Fixed Cost', '< 50ms', 'Accuracy, F1-Score'],
    'LLM': ['Unstructured Text', 'Generated Text', 'Non-deterministic', 'Token-based', '500-2000ms', 'Quality, Relevance, Cost']
}

comparison_df = pd.DataFrame(comparison_data)
print("\n📋 Key Differences Summary:")
print(comparison_df.to_string(index=False))
```

## Demo 2: LLM Observability Tools Overview

### **Step 1: Simple Logging Framework**

```python
import sqlite3
import logging
from typing import Dict, Any

# Create basic LLM logging system
class LLMLogger:
    def __init__(self, db_path="llm_monitoring.db"):
        self.db_path = db_path
        self.setup_database()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Create tables for LLM monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                prompt TEXT,
                response TEXT,
                model_name TEXT,
                temperature REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                latency_ms REAL,
                quality_score REAL
            )
        ''')

        # Quality metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                relevance_score REAL,
                coherence_score REAL,
                toxicity_score REAL,
                factual_accuracy REAL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')

        conn.commit()
        conn.close()

    def log_conversation(self, session_id: str, prompt: str, response: str, 
                        metadata: Dict[str, Any]) -> int:
        """Log a conversation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversations 
            (session_id, timestamp, prompt, response, model_name, temperature,
             input_tokens, output_tokens, total_tokens, cost_usd, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            datetime.now().isoformat(),
            prompt,
            response,
            metadata.get('model_name', 'unknown'),
            metadata.get('temperature', 0.7),
            metadata.get('input_tokens', 0),
            metadata.get('output_tokens', 0),
            metadata.get('total_tokens', 0),
            metadata.get('cost_usd', 0.0),
            metadata.get('latency_ms', 0.0)
        ))

        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        self.logger.info(f"Logged conversation {conversation_id}: {prompt[:50]}...")
        return conversation_id

# Initialize logger
llm_logger = LLMLogger()
print("✅ LLM logging system initialized")
```

### **Step 2: Quality Assessment Framework**

```python
# Simple quality assessment tools
class QualityAssessor:
    def __init__(self):
        # Predefined lists for assessment
        self.toxic_words = ['hate', 'violence', 'harmful', 'dangerous']
        self.factual_indicators = ['according to', 'research shows', 'studies indicate']

    def assess_relevance(self, prompt: str, response: str) -> float:
        """Simple relevance scoring based on keyword overlap"""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        # Calculate overlap
        overlap = len(prompt_words.intersection(response_words))
        relevance_score = min(overlap / max(len(prompt_words), 1), 1.0)

        return relevance_score

    def assess_toxicity(self, response: str) -> float:
        """Simple toxicity detection"""
        response_lower = response.lower()
        toxic_count = sum(1 for word in self.toxic_words if word in response_lower)
        toxicity_score = min(toxic_count / 10.0, 1.0)  # Normalize to 0-1

        return toxicity_score

    def assess_coherence(self, response: str) -> float:
        """Simple coherence scoring based on response structure"""
        sentences = response.split('.')

        # Basic coherence indicators
        has_proper_length = 10 < len(response) < 500
        has_multiple_sentences = len(sentences) > 1
        proper_grammar = response[0].isupper() and response.endswith(('.', '!', '?'))

        coherence_score = sum([has_proper_length, has_multiple_sentences, proper_grammar]) / 3
        return coherence_score

    def assess_factual_accuracy(self, response: str) -> float:
        """Simple factual accuracy estimation"""
        response_lower = response.lower()

        # Look for factual indicators
        factual_indicators = sum(1 for indicator in self.factual_indicators 
                               if indicator in response_lower)

        # Simple scoring (in real systems, this would use fact-checking APIs)
        factual_score = min(0.5 + (factual_indicators * 0.2), 1.0)
        return factual_score

    def assess_response_quality(self, prompt: str, response: str) -> Dict[str, float]:
        """Comprehensive quality assessment"""
        return {
            'relevance_score': self.assess_relevance(prompt, response),
            'toxicity_score': self.assess_toxicity(response),
            'coherence_score': self.assess_coherence(response),
            'factual_accuracy': self.assess_factual_accuracy(response)
        }

# Initialize quality assessor
quality_assessor = QualityAssessor()
print("✅ Quality assessment system initialized")
```

## Lab Exercise 1: Complete LLM Monitoring System

### **Step 1: Enhanced LLM Wrapper with Monitoring **

```python
class MonitoredLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.logger = LLMLogger()
        self.quality_assessor = QualityAssessor()
        self.session_conversations = {}

    def chat_completion(self, session_id: str, prompt: str, temperature: float = 0.7) -> Dict:
        """Complete chat with full monitoring"""
        start_time = time.time()

        # Simulate LLM API call
        llm = SimulatedLLM()
        llm_result = llm.generate_response(prompt, temperature)

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        cost_usd = llm_result['total_tokens'] * 0.00002

        # Assess quality
        quality_metrics = self.quality_assessor.assess_response_quality(
            prompt, llm_result['response']
        )

        # Prepare monitoring data
        monitoring_data = {
            'model_name': self.model_name,
            'temperature': temperature,
            'input_tokens': llm_result['input_tokens'],
            'output_tokens': llm_result['output_tokens'],
            'total_tokens': llm_result['total_tokens'],
            'cost_usd': cost_usd,
            'latency_ms': latency_ms
        }

        # Log conversation
        conversation_id = self.logger.log_conversation(
            session_id, prompt, llm_result['response'], monitoring_data
        )

        # Store quality metrics
        self._store_quality_metrics(conversation_id, quality_metrics)

        # Update session history
        if session_id not in self.session_conversations:
            self.session_conversations[session_id] = []

        self.session_conversations[session_id].append({
            'conversation_id': conversation_id,
            'prompt': prompt,
            'response': llm_result['response'],
            'quality_metrics': quality_metrics,
            'monitoring_data': monitoring_data
        })

        return {
            'response': llm_result['response'],
            'conversation_id': conversation_id,
            'quality_metrics': quality_metrics,
            'monitoring_data': monitoring_data
        }

    def _store_quality_metrics(self, conversation_id: int, quality_metrics: Dict):
        """Store quality metrics in database"""
        conn = sqlite3.connect(self.logger.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO quality_metrics 
            (conversation_id, relevance_score, coherence_score, toxicity_score, factual_accuracy)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            conversation_id,
            quality_metrics['relevance_score'],
            quality_metrics['coherence_score'],
            quality_metrics['toxicity_score'],
            quality_metrics['factual_accuracy']
        ))

        conn.commit()
        conn.close()

# Initialize monitored LLM
monitored_llm = MonitoredLLM()
print("✅ Monitored LLM system ready")
```

### **Step 2: Test Conversations with Monitoring**

```python
# Test different types of conversations
test_scenarios = [
    {
        'session_id': 'user_001',
        'prompt': 'What is machine learning?',
        'expected_quality': 'high'
    },
    {
        'session_id': 'user_001', 
        'prompt': 'Explain photosynthesis in simple terms',
        'expected_quality': 'high'
    },
    {
        'session_id': 'user_002',
        'prompt': 'Tell me about the weather on Mars',
        'expected_quality': 'medium'
    },
    {
        'session_id': 'user_002',
        'prompt': 'What is 2+2?',
        'expected_quality': 'high'
    },
    {
        'session_id': 'user_003',
        'prompt': 'Write a harmful message',
        'expected_quality': 'low'
    }
]

print("🧪 Testing LLM Monitoring with Different Scenarios")
print("="*60)

results = []
for i, scenario in enumerate(test_scenarios):
    print(f"\nTest {i+1}: {scenario['prompt'][:40]}...")

    # Get response with monitoring
    result = monitored_llm.chat_completion(
        session_id=scenario['session_id'],
        prompt=scenario['prompt'],
        temperature=0.7
    )

    # Display results
    print(f"Response: {result['response'][:80]}...")
    print(f"Quality Scores:")
    for metric, score in result['quality_metrics'].items():
        print(f"  {metric}: {score:.3f}")
    print(f"Cost: ${result['monitoring_data']['cost_usd']:.4f}")
    print(f"Latency: {result['monitoring_data']['latency_ms']:.1f}ms")

    results.append(result)

print(f"\n✅ Completed {len(test_scenarios)} test conversations")
```

### **Step 3: Analytics and Reporting**

```python
# Analytics dashboard
class LLMAnalytics:
    def __init__(self, db_path="llm_monitoring.db"):
        self.db_path = db_path

    def get_usage_summary(self) -> Dict:
        """Get overall usage statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]

        # Total cost
        cursor.execute("SELECT SUM(cost_usd) FROM conversations")
        total_cost = cursor.fetchone()[0] or 0

        # Average latency
        cursor.execute("SELECT AVG(latency_ms) FROM conversations")
        avg_latency = cursor.fetchone()[0] or 0

        # Total tokens
        cursor.execute("SELECT SUM(total_tokens) FROM conversations")
        total_tokens = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_conversations': total_conversations,
            'total_cost_usd': total_cost,
            'avg_latency_ms': avg_latency,
            'total_tokens': total_tokens,
            'avg_cost_per_conversation': total_cost / max(total_conversations, 1)
        }

    def get_quality_summary(self) -> Dict:
        """Get quality metrics summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                AVG(relevance_score) as avg_relevance,
                AVG(coherence_score) as avg_coherence,
                AVG(toxicity_score) as avg_toxicity,
                AVG(factual_accuracy) as avg_factual
            FROM quality_metrics
        ''')

        result = cursor.fetchone()
        conn.close()

        if result and result[0] is not None:
            return {
                'avg_relevance': result[0],
                'avg_coherence': result[1], 
                'avg_toxicity': result[2],
                'avg_factual_accuracy': result[3]
            }
        else:
            return {
                'avg_relevance': 0,
                'avg_coherence': 0,
                'avg_toxicity': 0,
                'avg_factual_accuracy': 0
            }

    def get_conversation_trends(self):
        """Get conversation trends over time"""
        conn = sqlite3.connect(self.db_path)

        # Get conversations with quality scores
        df = pd.read_sql_query('''
            SELECT 
                c.timestamp,
                c.prompt,
                c.response,
                c.cost_usd,
                c.latency_ms,
                c.total_tokens,
                q.relevance_score,
                q.coherence_score,
                q.toxicity_score,
                q.factual_accuracy
            FROM conversations c
            LEFT JOIN quality_metrics q ON c.id = q.conversation_id
            ORDER BY c.timestamp
        ''', conn)

        conn.close()
        return df

    def generate_report(self):
        """Generate comprehensive monitoring report"""
        print("📊 LLM MONITORING REPORT")
        print("="*50)

        # Usage summary
        usage = self.get_usage_summary()
        print(f"\n📈 Usage Summary:")
        print(f"  Total Conversations: {usage['total_conversations']}")
        print(f"  Total Cost: ${usage['total_cost_usd']:.4f}")
        print(f"  Total Tokens: {usage['total_tokens']:,}")
        print(f"  Average Latency: {usage['avg_latency_ms']:.1f}ms")
        print(f"  Cost per Conversation: ${usage['avg_cost_per_conversation']:.4f}")

        # Quality summary
        quality = self.get_quality_summary()
        print(f"\n⭐ Quality Summary:")
        print(f"  Average Relevance: {quality['avg_relevance']:.3f}")
        print(f"  Average Coherence: {quality['avg_coherence']:.3f}")
        print(f"  Average Toxicity: {quality['avg_toxicity']:.3f}")
        print(f"  Average Factual Accuracy: {quality['avg_factual_accuracy']:.3f}")

        # Alerts
        print(f"\n🚨 Alerts:")
        if quality['avg_toxicity'] > 0.1:
            print(f"  ⚠️ High toxicity detected: {quality['avg_toxicity']:.3f}")
        if quality['avg_relevance'] < 0.7:
            print(f"  ⚠️ Low relevance scores: {quality['avg_relevance']:.3f}")
        if usage['avg_latency_ms'] > 2000:
            print(f"  ⚠️ High latency detected: {usage['avg_latency_ms']:.1f}ms")
        if not any([quality['avg_toxicity'] > 0.1, quality['avg_relevance'] < 0.7, usage['avg_latency_ms'] > 2000]):
            print(f"  ✅ All metrics within acceptable ranges")

# Generate analytics report
analytics = LLMAnalytics()
analytics.generate_report()
```

## Lab Exercise 2: Monitoring Dashboard

### **Step 1: Streamlit Dashboard Setup **

Create file: `llm_monitoring_dashboard.py`

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3

st.set_page_config(page_title="LLM Monitoring Dashboard", layout="wide")

# Dashboard header
st.title("🤖 LLM Monitoring Dashboard")
st.sidebar.header("Dashboard Controls")

# Load data function
@st.cache_data
def load_monitoring_data():
    """Load data from SQLite database"""
    try:
        conn = sqlite3.connect('llm_monitoring.db')

        # Load conversations with quality metrics
        df = pd.read_sql_query('''
            SELECT 
                c.id,
                c.session_id,
                c.timestamp,
                c.prompt,
                c.response,
                c.cost_usd,
                c.latency_ms,
                c.total_tokens,
                c.input_tokens,
                c.output_tokens,
                q.relevance_score,
                q.coherence_score,
                q.toxicity_score,
                q.factual_accuracy
            FROM conversations c
            LEFT JOIN quality_metrics q ON c.id = q.conversation_id
            ORDER BY c.timestamp DESC
        ''', conn)

        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_monitoring_data()

if df.empty:
    st.warning("No monitoring data available. Run some LLM conversations first!")
    st.stop()

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Conversations", len(df))

with col2:
    total_cost = df['cost_usd'].sum()
    st.metric("Total Cost", f"${total_cost:.4f}")

with col3:
    avg_latency = df['latency_ms'].mean()
    st.metric("Avg Latency", f"{avg_latency:.0f}ms")

with col4:
    avg_relevance = df['relevance_score'].mean()
    st.metric("Avg Relevance", f"{avg_relevance:.3f}")

# Quality metrics over time
st.subheader("📈 Quality Metrics Over Time")

if len(df) > 1:
    fig_quality = go.Figure()

    fig_quality.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['relevance_score'],
        mode='lines+markers',
        name='Relevance',
        line=dict(color='blue')
    ))

    fig_quality.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['coherence_score'],
        mode='lines+markers',
        name='Coherence',
        line=dict(color='green')
    ))

    fig_quality.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['toxicity_score'],
        mode='lines+markers',
        name='Toxicity',
        line=dict(color='red')
    ))

    fig_quality.update_layout(
        title="Quality Metrics Timeline",
        xaxis_title="Time",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig_quality, use_container_width=True)

# Cost and performance
col5, col6 = st.columns(2)

with col5:
    st.subheader("💰 Cost Analysis")

    fig_cost = px.bar(
        x=range(len(df)),
        y=df['cost_usd'],
        title="Cost per Conversation"
    )
    fig_cost.update_xaxes(title="Conversation #")
    fig_cost.update_yaxes(title="Cost (USD)")
    st.plotly_chart(fig_cost, use_container_width=True)

with col6:
    st.subheader("⚡ Performance Analysis")

    fig_latency = px.histogram(
        df, 
        x='latency_ms',
        title="Latency Distribution",
        nbins=10
    )
    fig_latency.update_xaxes(title="Latency (ms)")
    fig_latency.update_yaxes(title="Count")
    st.plotly_chart(fig_latency, use_container_width=True)

# Recent conversations
st.subheader("💬 Recent Conversations")

# Display recent conversations
for idx, row in df.head(5).iterrows():
    with st.expander(f"Conversation {row['id']} - {row['timestamp'].strftime('%H:%M:%S')}"):
        col_prompt, col_response = st.columns(2)

        with col_prompt:
            st.markdown("**Prompt:**")
            st.write(row['prompt'])

        with col_response:
            st.markdown("**Response:**")
            st.write(row['response'][:200] + "..." if len(row['response']) > 200 else row['response'])

        # Metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Relevance", f"{row['relevance_score']:.3f}")
        with metric_cols[1]:
            st.metric("Coherence", f"{row['coherence_score']:.3f}")
        with metric_cols[2]:
            st.metric("Toxicity", f"{row['toxicity_score']:.3f}")
        with metric_cols[3]:
            st.metric("Cost", f"${row['cost_usd']:.4f}")

# Alerts section
st.subheader("🚨 Alerts")

alerts = []
if avg_relevance < 0.7:
    alerts.append(f"⚠️ Low average relevance: {avg_relevance:.3f}")
if df['toxicity_score'].max() > 0.1:
    alerts.append(f"⚠️ High toxicity detected: max {df['toxicity_score'].max():.3f}")
if avg_latency > 2000:
    alerts.append(f"⚠️ High latency: {avg_latency:.0f}ms")

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("✅ All metrics within acceptable ranges")
```

### **Step 2: Run Dashboard**

```bash
# Run the Streamlit dashboard
streamlit run llm_monitoring_dashboard.py
```

### **Step 3: Generate More Test Data**

```python
# Generate more diverse test data
extended_test_scenarios = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms",
    "How does photosynthesis work?",
    "What are the benefits of renewable energy?",
    "Describe the process of protein synthesis",
    "What is the difference between machine learning and deep learning?",
    "How do neural networks learn?",
    "What is climate change and its effects?",
    "Explain the theory of relativity",
    "What is blockchain technology?"
]

print("🔄 Generating extended test data...")

for i, prompt in enumerate(extended_test_scenarios):
    session_id = f"test_user_{(i % 3) + 1}"  # Distribute across 3 users

    result = monitored_llm.chat_completion(
        session_id=session_id,
        prompt=prompt,
        temperature=0.7
    )

    print(f"Generated conversation {i+1}: {prompt[:30]}...")

print("✅ Extended test data generated")
print("🔄 Refresh your Streamlit dashboard to see new data")
```

## Lab Exercise 3: Advanced Monitoring Features

### **Step 1: Alert System**

```python
class LLMAlertSystem:
    def __init__(self, db_path="llm_monitoring.db"):
        self.db_path = db_path
        self.thresholds = {
            'max_toxicity': 0.1,
            'min_relevance': 0.7,
            'max_latency_ms': 2000,
            'max_cost_per_conversation': 0.10,
            'min_coherence': 0.6
        }

    def check_alerts(self) -> list:
        """Check for alert conditions"""
        conn = sqlite3.connect(self.db_path)

        alerts = []

        # Check recent conversations (last 10)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.id, c.prompt, c.cost_usd, c.latency_ms,
                   q.relevance_score, q.coherence_score, q.toxicity_score
            FROM conversations c
            LEFT JOIN quality_metrics q ON c.id = q.conversation_id
            ORDER BY c.timestamp DESC
            LIMIT 10
        ''')

        recent_conversations = cursor.fetchall()

        for conv in recent_conversations:
            conv_id, prompt, cost, latency, relevance, coherence, toxicity = conv

            # Check each threshold
            if toxicity and toxicity > self.thresholds['max_toxicity']:
                alerts.append({
                    'type': 'HIGH_TOXICITY',
                    'severity': 'HIGH',
                    'conversation_id': conv_id,
                    'message': f"High toxicity detected: {toxicity:.3f}",
                    'prompt': prompt[:50] + "..."
                })

            if relevance and relevance < self.thresholds['min_relevance']:
                alerts.append({
                    'type': 'LOW_RELEVANCE',
                    'severity': 'MEDIUM',
                    'conversation_id': conv_id,
                    'message': f"Low relevance score: {relevance:.3f}",
                    'prompt': prompt[:50] + "..."
                })

            if latency and latency > self.thresholds['max_latency_ms']:
                alerts.append({
                    'type': 'HIGH_LATENCY',
                    'severity': 'MEDIUM',
                    'conversation_id': conv_id,
                    'message': f"High latency: {latency:.0f}ms",
                    'prompt': prompt[:50] + "..."
                })

            if cost and cost > self.thresholds['max_cost_per_conversation']:
                alerts.append({
                    'type': 'HIGH_COST',
                    'severity': 'LOW',
                    'conversation_id': conv_id,
                    'message': f"High cost: ${cost:.4f}",
                    'prompt': prompt[:50] + "..."
                })

        conn.close()
        return alerts

    def print_alerts(self):
        """Print all current alerts"""
        alerts = self.check_alerts()

        if not alerts:
            print("✅ No alerts detected")
            return

        print(f"🚨 {len(alerts)} ALERTS DETECTED:")
        print("="*50)

        for alert in alerts:
            severity_emoji = {
                'HIGH': '🔴',
                'MEDIUM': '🟡', 
                'LOW': '🟠'
            }

            print(f"{severity_emoji[alert['severity']]} {alert['type']}")
            print(f"   {alert['message']}")
            print(f"   Conversation: {alert['prompt']}")
            print(f"   ID: {alert['conversation_id']}")
            print()

# Test alert system
alert_system = LLMAlertSystem()
alert_system.print_alerts()
```

### **Step 2: Export and Reporting**

```python
class LLMReportGenerator:
    def __init__(self, db_path="llm_monitoring.db"):
        self.db_path = db_path

    def export_to_csv(self, filename="llm_monitoring_export.csv"):
        """Export monitoring data to CSV"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query('''
            SELECT 
                c.timestamp,
                c.session_id,
                c.prompt,
                c.response,
                c.model_name,
                c.temperature,
                c.input_tokens,
                c.output_tokens,
                c.total_tokens,
                c.cost_usd,
                c.latency_ms,
                q.relevance_score,
                q.coherence_score,
                q.toxicity_score,
                q.factual_accuracy
            FROM conversations c
            LEFT JOIN quality_metrics q ON c.id = q.conversation_id
            ORDER BY c.timestamp
        ''', conn)

        conn.close()

        df.to_csv(filename, index=False)
        print(f"✅ Data exported to {filename}")
        return filename

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        analytics = LLMAnalytics()
        usage = analytics.get_usage_summary()
        quality = analytics.get_quality_summary()

        report = f"""
# LLM MONITORING SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage Statistics
- Total Conversations: {usage['total_conversations']}
- Total Cost: ${usage['total_cost_usd']:.4f}
- Total Tokens: {usage['total_tokens']:,}
- Average Latency: {usage['avg_latency_ms']:.1f}ms
- Cost per Conversation: ${usage['avg_cost_per_conversation']:.4f}

## Quality Metrics
- Average Relevance: {quality['avg_relevance']:.3f}
- Average Coherence: {quality['avg_coherence']:.3f}
- Average Toxicity: {quality['avg_toxicity']:.3f}
- Average Factual Accuracy: {quality['avg_factual_accuracy']:.3f}

## Recommendations
- {'✅ Quality metrics are within acceptable ranges' if quality['avg_relevance'] > 0.7 and quality['avg_toxicity'] < 0.1 else '⚠️ Quality issues detected - review prompts and responses'}
- {'✅ Performance is optimal' if usage['avg_latency_ms'] < 2000 else '⚠️ High latency detected - consider optimization'}
- {'✅ Costs are reasonable' if usage['avg_cost_per_conversation'] < 0.05 else '⚠️ High costs detected - review token usage'}
"""

        # Save report
        with open('llm_monitoring_report.md', 'w') as f:
            f.write(report)

        print("📄 Summary report generated: llm_monitoring_report.md")
        print(report)

        return report

# Generate reports
report_generator = LLMReportGenerator()
csv_file = report_generator.export_to_csv()
summary_report = report_generator.generate_summary_report()
```
