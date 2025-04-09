```python
# ======================
# 1. Data Preparation
# ======================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

# Load data with temporal ordering
df = pd.read_csv('cyber_physical_smart_home.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Temporal split (3 weeks train, 3 days test)
train_end = df['timestamp'].quantile(0.85)  # 85% for training
train = df[df['timestamp'] <= train_end].copy()
test = df[df['timestamp'] > train_end].copy()

# Label engineering
train['label'] = np.where(train['actor'] == 2, 1, 0)
test['label'] = np.where(test['actor'] == 2, 1, 0)

# ======================
# 2. Feature Engineering
# ======================

# Numerical features (sensor readings)
numerical_features = [
    'temperature', 'humidity', 'packet_size', 
    'response_time', 'power_consumption'
]

# Categorical features with high cardinality
categorical_features = {
    'device_id': 100,      # 100 unique devices
    'src_ip': 1000,        # Use hashing for IPs
    'protocol': 10,        # TCP/UDP/ICMP etc.
    'sensor_type': 20      # Motion/temp/door etc.
}

# Preprocessing pipeline
def preprocess(df):
    # Numerical normalization
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Categorical hashing/encoding
    df['src_ip'] = df['src_ip'].apply(lambda x: hash(x) % 1000)
    df['device_id'] = df['device_id'].astype('category').cat.codes
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    
    return df

train = preprocess(train)
test = preprocess(test)

# ======================
# 3. DLRM Architecture
# ======================

def build_dlrm(numerical_dim, categorical_config):
    # Numerical input
    numerical_input = layers.Input(shape=(numerical_dim,), name='numerical')
    
    # Categorical embeddings
    categorical_embeddings = []
    for feature, cardinality in categorical_config.items():
        inp = layers.Input(shape=(1,), name=feature)
        emb = layers.Embedding(
            input_dim=cardinality, 
            output_dim=int(np.log2(cardinality)) + 1  # Dynamic embedding size
        )(inp)
        categorical_embeddings.append(layers.Flatten()(emb))
    
    # Feature interaction
    interaction = layers.concatenate([numerical_input] + categorical_embeddings)
    
    # Deep network
    x = layers.Dense(256, activation='relu')(interaction)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output with automatic class balancing
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return Model(
        inputs=[numerical_input] + [inp for inp, _ in categorical_config.items()],
        outputs=output
    )

# Initialize model
model = build_dlrm(len(numerical_features), categorical_features)

# ======================
# 4. Model Training
# ======================

# Class weights for imbalance
class_weight = {
    0: 1.0,
    1: len(train[train['label']==0])/len(train[train['label']==1])  # Auto-balance
}

# Custom metrics for anomaly detection
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Temporal validation
history = model.fit(
    x=[train[numerical_features]] + [train[col].values for col in categorical_features],
    y=train['label'],
    validation_data=(
        [test[numerical_features]] + [test[col].values for col in categorical_features],
        test['label']
    ),
    epochs=100,
    batch_size=1024,
    class_weight=class_weight,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# ======================
# 5. Deployment
# ======================

# Save full pipeline
import joblib
pipeline = {
    'scaler': scaler,
    'model': model,
    'features': {
        'numerical': numerical_features,
        'categorical': categorical_features
    }
}

joblib.dump(pipeline, 'anomaly_detection_pipeline.pkl')

# Inference function
def detect_anomalies(new_data):
    # Preprocess
    processed = preprocess(new_data.copy())
    
    # Predict
    inputs = [processed[numerical_features]] + \
             [processed[col].values for col in categorical_features]
    
    return model.predict(inputs)

# ======================
# 6. Interpretation
# ======================

def explain_anomalies(sample):
    import shap
    explainer = shap.DeepExplainer(model, [train[numerical_features]] + 
                                  [train[col].values for col in categorical_features])
    shap_values = explainer.shap_values(sample)
    
    # Visualize feature importance
    shap.summary_plot(shap_values, sample, feature_names=numerical_features + list(categorical_features.keys()))
```


**Usage:**

```python
# Load pipeline
pipeline = joblib.load('anomaly_detection_pipeline.pkl')

# Real-time inference
new_data = pd.read_csv('live_smart_home_data.csv')
predictions = detect_anomalies(new_data)

# Investigate anomalies
explain_anomalies(test.sample(100))  # Sample 100 suspicious events
```

