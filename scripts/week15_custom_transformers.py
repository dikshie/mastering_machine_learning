"""
Week 15: Custom Transformers & Feature Engineering
Book Chapter: 15 - Feature engineering with custom transformers

Topics:
- FunctionTransformer for simple stateless feature transformations
- Subclassing BaseEstimator & TransformerMixin for stateful custom transformers
- Integrating custom transformers into Scikit-Learn Pipelines seamlessly
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 1. Stateful Custom Transformer: Outlier Capper
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = np.clip(X_df[col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df.values

def main():
    print("=" * 60)
    print(" Week 15: Custom Transformers & Feature Engineering ")
    print("=" * 60)

    # Sample dataset
    df = pd.DataFrame({
        'price': [10, 15, 20, 25, 1000, 30, 35, 40], # Contains outlier 1000
        'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000]
    })
    y = np.array([100, 150, 200, 250, 300, 350, 400, 450])

    print("\nOriginal 'price' values (contains outlier 1000):")
    print(df['price'].tolist())

    # Stateless transformation using FunctionTransformer (Log transform)
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    # Pipeline with custom OutlierCapper and Log Transformer
    pipeline = Pipeline([
        ('capper', OutlierCapper(lower_quantile=0.1, upper_quantile=0.9)),
        ('log_transform', log_transformer),
        ('model', Ridge())
    ])

    pipeline.fit(df, y)
    print("\nPipeline with Custom OutlierCapper & Log Transformation fitted successfully!")

    # Check transformed values inside pipeline
    capped_values = pipeline.named_steps['capper'].transform(df)
    print("Capped values after OutlierCapper:")
    print(capped_values[:, 0].tolist())

if __name__ == "__main__":
    main()
