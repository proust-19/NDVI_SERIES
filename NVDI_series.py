
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from scipy.signal import savgol_filter

# Load data
train_df = pd.read_csv("C:\PROUST\hacktrain.csv")
test_df = pd.read_csv("C:\PROUST\hacktest.csv")

# test IDs
test_ids = test_df['ID']


# DATA PREPROCESSING

def preprocess_data(df):
    """Handle missing values, noise, and feature engineering"""

  # copy
    processed = df.copy()

    if 'ID' in processed.columns:
        processed.drop(columns=['ID'], inplace=True)

    # NDVI columns
    ndvi_cols = [col for col in processed.columns if col not in ['class'] and col != 'ID']

    # Handle outliers
    processed[ndvi_cols] = processed[ndvi_cols].clip(lower=-1, upper=1)

    # Smooth time series
    for col in ndvi_cols:
        processed[col] = savgol_filter(processed[col], window_length=5, polyorder=2)

    # Impute missing values
    if 'class' in processed.columns:
        for land_class in processed['class'].unique():
            class_mask = processed['class'] == land_class
            processed.loc[class_mask, ndvi_cols] = processed.loc[class_mask, ndvi_cols].fillna(
                processed.loc[class_mask, ndvi_cols].median()
            )
    else:
        imputer = SimpleImputer(strategy='median')
        processed[ndvi_cols] = imputer.fit_transform(processed[ndvi_cols])

    # Feature engineering
    processed['ndvi_mean'] = processed[ndvi_cols].mean(axis=1)
    processed['ndvi_slope'] = processed[ndvi_cols].apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], axis=1)
    processed['ndvi_amplitude'] = processed[ndvi_cols].max(axis=1) - processed[ndvi_cols].min(axis=1)

    return processed

# Preprocess both train and test data
train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df)

# TRAINING

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_processed['class'])
X = train_processed.drop(columns=['class'])

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Logistic regression
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,  # Increased from 10 to ensure convergence
    penalty='l2',
    C=0.1,  # Regularization strength
    random_state=42
)
model.fit(X_train, y_train)

# Validation performance
y_pred = model.predict(X_val)
print("Validation Report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=label_encoder.classes_
))

# PREDICTIONS

# Preprocess and scale test data
X_test = test_processed
X_test_scaled = scaler.transform(X_test)

# Make predictions
test_preds = model.predict(X_test_scaled)
test_preds_decoded = label_encoder.inverse_transform(test_preds)

# Create submission file
submission = pd.DataFrame({
    'ID': test_ids,
    'class': test_preds_decoded
})

submission.to_csv('C:\PROUST\submission.csv', index=False)
print("\nSubmission file created successfully!")