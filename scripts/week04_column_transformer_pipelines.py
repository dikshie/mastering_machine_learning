"""
Week 4: Preprocessing Pipelines & ColumnTransformers
Book Chapter: 4 - Improving your workflow with ColumnTransformer and Pipeline

Topics:
- ColumnTransformer and make_column_transformer
- make_column_selector for dynamic data type targeting
- Pipeline and make_pipeline for clean modular preprocessing & estimation
"""

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    print("=" * 60)
    print(" Week 4: ColumnTransformer & Pipeline ")
    print("=" * 60)

    # Synthetic dataset with mixed data types
    df = pd.DataFrame({
        'age': [25, 32, 47, 51, 62, 23, 39, 29],
        'income': [50000, 64000, 120000, 85000, 95000, 42000, 78000, 61000],
        'department': ['Sales', 'Tech', 'HR', 'Tech', 'Sales', 'HR', 'Tech', 'Sales'],
        'city': ['NYC', 'London', 'NYC', 'Paris', 'Paris', 'NYC', 'London', 'Paris'],
        'promoted': [0, 1, 1, 0, 1, 0, 1, 0]
    })

    X = df.drop(columns=['promoted'])
    y = df['promoted']

    # 1. Define ColumnTransformer with automatic type selectors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), make_column_selector(dtype_include=['object']))
        ]
    )

    # 2. Build Pipeline combining preprocessing + estimator
    clf = make_pipeline(preprocessor, LogisticRegression(random_state=42))

    # 3. Fit pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"\nPipeline successfully fitted!")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    # Inspect features processed
    encoded_cols = clf.named_steps['columntransformer'].get_feature_names_out()
    print("\nTransformed Feature Names:")
    for col in encoded_cols:
        print(f" - {col}")

if __name__ == "__main__":
    main()
