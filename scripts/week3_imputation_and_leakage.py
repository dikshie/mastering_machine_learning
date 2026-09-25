"""
Week 3: Handling Missing Values & Robust Preprocessing Workflows
Covering Chapters 7, 8, and 9 from Master Machine Learning with scikit-learn.

Topics:
- Imputation strategies (SimpleImputer, KNNImputer)
- Handling missing values in numerical vs categorical features
- Creating nested pipelines inside ColumnTransformer
- Preventing data leakage during preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def demo_imputation_and_leakage():
    print("--- 1. Handling Missing Data & Preventing Data Leakage ---")

    # Creating dataset with missing values (NaN) in both numeric and categorical features
    np.random.seed(42)
    df = pd.DataFrame({
        'age': [25, np.nan, 35, 50, np.nan, 40, 28, 62],
        'salary': [50000, 85000, np.nan, 110000, 48000, 95000, np.nan, 120000],
        'gender': ['M', 'F', np.nan, 'F', 'M', 'F', 'M', np.nan],
        'subscribed': [0, 1, 0, 1, 0, 1, 0, 1]
    })

    print("Dataset with Missing Values:\n", df)

    X = df.drop(columns=['subscribed'])
    y = df['subscribed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 1. Numerical Pipeline: Mean Imputation + Scaling
    num_cols = ['age', 'salary']
    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # 2. Categorical Pipeline: Constant Imputation + One-Hot Encoding
    cat_cols = ['gender']
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    )

    # Combined ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    # Complete Workflow Model Pipeline
    model_pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier(n_estimators=50, random_state=42)
    )

    # Fit pipeline ONLY on training data (prevents data leakage)
    model_pipeline.fit(X_train, y_train)

    print("\nPipeline successfully trained without data leakage!")
    print(f"Test Accuracy: {model_pipeline.score(X_test, y_test):.4f}")

    # Transform test sample safely
    sample_test = pd.DataFrame({
        'age': [np.nan],
        'salary': [75000],
        'gender': ['Unknown_Category']  # Unseen category
    })
    pred = model_pipeline.predict(sample_test)
    print(f"Prediction for unseen missing/new category sample: {pred[0]}")

if __name__ == '__main__':
    demo_imputation_and_leakage()
