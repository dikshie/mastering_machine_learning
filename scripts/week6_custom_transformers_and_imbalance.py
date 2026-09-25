"""
Week 6: Custom Transformers, High-Cardinality Categorical Encoding & Imbalanced Data
Covering Chapters 15, 16, 17, 18, 19, and 20 from Master Machine Learning with scikit-learn.

Topics:
- Custom feature engineering transformers (FunctionTransformer, BaseEstimator/TransformerMixin)
- Handling class imbalance (class_weight='balanced', ROC AUC, confusion matrix, threshold tuning)
- Target encoding / high cardinality categorical features
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Custom Transformer via BaseEstimator & TransformerMixin
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1.0):
        self.offset = offset
        
    def fit(self, X, y=None):
        return self  # Nothing to fit
        
    def transform(self, X):
        return np.log1p(np.maximum(0, X + self.offset))

# Function for simple FunctionTransformer
def create_ratio_features(df):
    df_out = df.copy()
    if 'income' in df_out.columns and 'debt' in df_out.columns:
        df_out['debt_to_income'] = df_out['debt'] / (df_out['income'] + 1.0)
    return df_out

def demo_custom_transformers_and_imbalance():
    print("--- 1. Custom Feature Engineering Transformers ---")
    df = pd.DataFrame({
        'income': [50000, 120000, 30000, 80000, 200000],
        'debt': [5000, 15000, 8000, 12000, 20000]
    })
    
    # Using FunctionTransformer
    ratio_transformer = FunctionTransformer(create_ratio_features)
    transformed_df = ratio_transformer.transform(df)
    print("Transformed DataFrame with engineered ratio:\n", transformed_df)

    print("\n--- 2. Class Imbalance Handling & Threshold Tuning ---")
    # Synthetic imbalanced dataset (90% negative, 10% positive)
    X, y = make_classification(
        n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Model with class_weight='balanced'
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', random_state=42)
    )

    pipeline.fit(X_train, y_train)

    # Evaluate using default threshold (0.5)
    y_pred_default = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_probs)
    print(f"ROC AUC Score: {auc:.4f}")
    print("\nConfusion Matrix (Default 0.5 Threshold):\n", confusion_matrix(y_test, y_pred_default))

    # Threshold Tuning: lower threshold to increase recall for minority class
    custom_threshold = 0.35
    y_pred_custom = (y_probs >= custom_threshold).astype(int)

    print(f"\nConfusion Matrix (Tuned Threshold {custom_threshold}):\n", confusion_matrix(y_test, y_pred_custom))
    print("\nClassification Report (Tuned Threshold):\n", classification_report(y_test, y_pred_custom))

if __name__ == '__main__':
    demo_custom_transformers_and_imbalance()
