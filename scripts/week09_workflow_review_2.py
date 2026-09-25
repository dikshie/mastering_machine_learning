"""
Week 9: Workflow Review #2
Book Chapter: 9 - Workflow review #2

Topics:
- Comprehensive multi-step preprocessing architecture
- Nested Pipelines inside ColumnTransformers (SimpleImputer -> OneHotEncoder & SimpleImputer -> StandardScaler)
- Model evaluation with realistic dirty data containing missing values and mixed feature types
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

def main():
    print("=" * 60)
    print(" Week 9: Workflow Review #2 (Complex Pipeline Architectures) ")
    print("=" * 60)

    # Create synthetic dirty tabular dataset
    raw_data = {
        'age': [25, np.nan, 47, 51, np.nan, 23, 39, 29, 55, 31],
        'salary': [50000, 64000, np.nan, 85000, 95000, 42000, np.nan, 61000, 105000, 72000],
        'education': ['Bachelor', 'Master', np.nan, 'High School', 'PhD', 'Bachelor', 'Master', 'High School', 'PhD', 'Bachelor'],
        'city': ['NYC', 'London', 'Paris', np.nan, 'Paris', 'NYC', 'London', 'Paris', 'NYC', 'London'],
        'churn': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(raw_data)
    X = df.drop(columns=['churn'])
    y = df['churn']

    # 1. Numeric pipeline: Impute missing numbers with median, then scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical pipeline: Impute missing categories with most frequent, then One-Hot Encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine pipelines in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, ['age', 'salary']),
            ('cat', cat_pipeline, ['education', 'city'])
        ]
    )

    # 4. Final end-to-end model pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    full_pipeline.fit(X, y)
    print("\nEnd-to-end multi-step pipeline fitted successfully!")

    # Test prediction on new incoming dirty sample
    new_sample = pd.DataFrame({
        'age': [np.nan],
        'salary': [70000],
        'education': ['Master'],
        'city': [np.nan]
    })
    pred = full_pipeline.predict(new_sample)
    proba = full_pipeline.predict_proba(new_sample)
    print(f"Prediction for dirty new sample: Churn={pred[0]} (Probability: {proba[0][1]:.4f})")

if __name__ == "__main__":
    main()
