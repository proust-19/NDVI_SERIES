# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from scipy.signal import savgol_filter

# Load data
train_df = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktrain.csv")
test_df = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktest.csv")

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

# train and test data
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
    max_iter=1000,  
    penalty='l2',
    C=0.1,  # Regularization strength
    random_state=42
)
model.fit(X_train, y_train)

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

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")
