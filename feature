

#### 1. **Load the Dataset**
First, you need to load the dataset into a Pandas DataFrame.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('path_to_cyber_physical_smart_home_dataset.csv', parse_dates=['timestamp'])
```

#### 2. **Data Preprocessing**
This step involves cleaning the data, handling missing values, and normalizing numerical features.

```python
# Check for missing values
print(df.isnull().sum())

# Fill missing values (forward fill for sensor readings)
df.fillna(method='ffill', inplace=True)

# Drop irrelevant columns if necessary
# df.drop(columns=['irrelevant_column'], inplace=True)
```

#### 3. **Feature Engineering**
Create the necessary features based on the dataset's characteristics.

##### 3.1 **Numerical Features**
Normalize numerical features to ensure they are on a similar scale.

```python
from sklearn.preprocessing import MinMaxScaler

# Define numerical features
numerical_features = ['temperature', 'humidity', 'light_intensity', 'power_consumption', 'response_time']

# Normalize numerical features
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

##### 3.2 **Categorical Features**
Convert categorical features into numerical representations using encoding or embeddings.

```python
from sklearn.preprocessing import LabelEncoder

# Define categorical features
categorical_features = ['device_type', 'sensor_location', 'protocol', 'src_ip']

# Encode categorical features
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
```

##### 3.3 **Time-Based Features**
Extract time-based features from the timestamp.

```python
# Extract time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
```

##### 3.4 **Interaction Features**
Create interaction features if necessary. For example, you can create a feature that combines device type and power consumption.

```python
# Example of creating an interaction feature
df['device_power_interaction'] = df['device_type'] * df['power_consumption']
```

##### 3.5 **Anomaly Labels**
Label the data based on the actor's behavior.

```python
# Label anomalies (1 for anomalous, 0 for normal)
df['label'] = df['actor'].apply(lambda x: 1 if x == 2 else 0)
```

#### 4. **Split the Data**
Split the dataset into training and testing sets based on time to avoid data leakage.

```python
# Temporal split (e.g., first 3 weeks for training, last 3 days for testing)
train_end = df['timestamp'].quantile(0.85)  # Adjust as needed
train_data = df[df['timestamp'] <= train_end]
test_data = df[df['timestamp'] > train_end]
```

#### 5. **Prepare Data for DLRM**
Prepare the data for input into the DLRM model.

```python
# Prepare input features and labels
X_train = train_data[numerical_features + categorical_features + ['hour', 'day_of_week', 'month']]
y_train = train_data['label']

X_test = test_data[numerical_features + categorical_features + ['hour', 'day_of_week', 'month']]
y_test = test_data['label']

# Convert to appropriate format for DLRM
X_train_numerical = X_train[numerical_features].values
X_train_categorical = [X_train[feature].values for feature in categorical_features]

X_test_numerical = X_test[numerical_features].values
X_test_categorical = [X_test[feature].values for feature in categorical_features]
```

#### 6. **Build the DLRM Model**
Define the DLRM architecture using TensorFlow/Keras.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_dlrm(numerical_dim, categorical_dims):
    # Numerical input
    numerical_input = layers.Input(shape=(numerical_dim,), name='numerical_input')
    
    # Categorical embeddings
    categorical_embeddings = []
    for dim in categorical_dims:
        inp = layers.Input(shape=(1,), name='categorical_input')
        emb = layers.Embedding(input_dim=dim, output_dim=int(np.log2(dim)) + 1)(inp)
        emb = layers.Flatten()(emb)
        categorical_embeddings.append(emb)
    
    # Concatenate all features
    combined = layers.concatenate([numerical_input] + categorical_embeddings)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=[numerical_input] + [layers.Input(shape=(1,)) for _ in categorical_dims], outputs=output)

# Define categorical dimensions
categorical_dims = [len(df[feature].unique()) for feature in categorical_features]

# Build the model
model = build_dlrm(len(numerical_features), categorical_dims)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

#### 7. **Train the Model**
Fit the model to the training data.

```python
# Train the model
history = model.fit(
    [X_train_numerical] + X_train_categorical,
    y_train,
    validation_data=([X_test_numerical] + X_test_categorical, y_test),
    epochs=50,
   
