"""
Week 16: Workflow Review #3
Book Chapter: 16 - Workflow review #3

Topics:
- Advanced modular production pipeline architecture
- Incorporating custom transformers, multi-type preprocessing, and cross-validation evaluation
- Production validation of end-to-end data transformation pipeline
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Custom Feature Engineering: Ratio Transformer
class RatioAdder(BaseEstimator, TransformerMixin):
    def __init__(self, num_col, denom_col, new_col_name):
        self.num_col = num_col
        self.denom_col = denom_col
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df[self.new_col_name] = X_df[self.num_col] / (X_df[self.denom_col] + 1e-5)
        return X_df

def main():
    print("=" * 60)
    print(" Week 16: Workflow Review #3 (Modular Production Pipelines) ")
    print("=" * 60)

    # Synthetic financial dataset
    df = pd.DataFrame({
        'debt': [5000, 12000, 3000, 45000, 8000, 15000, 2000],
        'income': [50000, 60000, 40000, 90000, 55000, 70000, 35000],
        'credit_score': [720, 680, 750, 590, 710, 640, 780],
        'employment': ['Employed', 'Self-Employed', 'Employed', 'Unemployed', 'Employed', 'Self-Employed', 'Employed'],
        'default': [0, 0, 0, 1, 0, 1, 0]
    })

    X = df.drop(columns=['default'])
    y = df['default']

    # Step 1: Custom Feature Engineering
    ratio_adder = RatioAdder(num_col='debt', denom_col='income', new_col_name='debt_to_income')

    # Step 2: Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['debt', 'income', 'credit_score', 'debt_to_income']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['employment'])
    ])

    # Step 3: Combined Modular Pipeline
    full_pipeline = Pipeline([
        ('ratio_feature', ratio_adder),
        ('preprocessing', preprocessor),
        ('model', HistGradientBoostingClassifier(random_state=42))
    ])

    full_pipeline.fit(X, y)
    print("\nModular Pipeline fitted successfully!")

    # Verify transformation
    X_transformed = ratio_adder.transform(X)
    print("\nDataFrame with Engineered 'debt_to_income' feature:")
    print(X_transformed[['debt', 'income', 'debt_to_income']].head(3))

if __name__ == "__main__":
    main()
