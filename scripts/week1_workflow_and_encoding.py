"""
Week 1: Fundamentals of Scikit-Learn & ML Workflow
Covering Chapters 1, 2, and 3 from Master Machine Learning with scikit-learn.

Topics:
- Basic classification workflow (fit, predict, predict_proba)
- Regression & Multiclass variations
- Cross-validation with cross_val_score
- Categorical feature encoding (OneHotEncoder, OrdinalEncoder, LabelEncoder)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

def demo_ml_workflow():
    print("--- 1. Basic ML Workflow (Classification) ---")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test[:3])

    print(f"Sample Test Predictions: {y_pred[:5]}")
    print(f"Sample Probabilities (first 3 samples):\n{y_proba}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")

def demo_categorical_encoding():
    print("--- 2. Categorical Feature Encoding ---")
    # Sample DataFrame with nominal and ordinal features
    df = pd.DataFrame({
        'city': ['NYC', 'Paris', 'London', 'NYC', 'Paris'],
        'size': ['small', 'medium', 'large', 'medium', 'small'],
        'purchased': ['no', 'yes', 'no', 'yes', 'yes']
    })
    print("Original DataFrame:\n", df)

    # One-Hot Encoding for nominal feature ('city')
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    city_encoded = ohe.fit_transform(df[['city']])
    print("\nOne-Hot Encoded 'city':\n", pd.DataFrame(city_encoded, columns=ohe.get_feature_names_out(['city'])))

    # Ordinal Encoding for ordered feature ('size')
    size_order = [['small', 'medium', 'large']]
    oe = OrdinalEncoder(categories=size_order)
    df['size_encoded'] = oe.fit_transform(df[['size']])
    print("\nOrdinal Encoded 'size':\n", df[['size', 'size_encoded']])

    # LabelEncoder for binary target variable
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['purchased'])
    print("\nLabel Encoded Target:\n", df[['purchased', 'target_encoded']])

if __name__ == '__main__':
    demo_ml_workflow()
    demo_categorical_encoding()
