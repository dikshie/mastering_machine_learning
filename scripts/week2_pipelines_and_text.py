"""
Week 2: Advanced Preprocessing Pipelines & Text Encoding
Covering Chapters 4, 5, and 6 from Master Machine Learning with scikit-learn.

Topics:
- ColumnTransformer (make_column_transformer, make_column_selector)
- Pipeline (make_pipeline)
- Encoding text features with CountVectorizer inside ColumnTransformer
- Full pipeline integration and predictions
"""

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def demo_pipeline_and_text():
    print("--- 1. Preprocessing Pipelines & Text Vectorization ---")
    
    # Dataset with numerical, categorical, and text features
    data = pd.DataFrame({
        'age': [25, 45, 35, 50, 23, 40],
        'income': [50000, 85000, 62000, 110000, 48000, 95000],
        'city': ['NYC', 'London', 'Paris', 'NYC', 'Paris', 'London'],
        'review': [
            'Great service and product!',
            'Poor customer support bad experience',
            'Neutral service average quality',
            'Excellent product love it',
            'Terrible experience bad customer service',
            'Good service overall happy'
        ],
        'target': [1, 0, 0, 1, 0, 1]
    })

    X = data.drop(columns=['target'])
    y = data['target']

    # Column selectors by data type
    numeric_selector = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_selector = make_column_selector(dtype_include=['object'], pattern='city')
    
    # Preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_selector),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['city']),
            ('text', CountVectorizer(stop_words='english'), 'review')
        ]
    )

    # Full Pipeline
    full_pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(random_state=42)
    )

    # Fit and Predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    full_pipeline.fit(X_train, y_train)

    preds = full_pipeline.predict(X_test)
    probas = full_pipeline.predict_proba(X_test)

    print("Pipeline fitted successfully.")
    print(f"Test Predictions: {preds}")
    print(f"Test Probabilities:\n{probas}")

    # Inspecting pipeline steps
    print("\nPipeline Steps:")
    for name, step in full_pipeline.named_steps.items():
        print(f" - {name}: {step}")

if __name__ == '__main__':
    demo_pipeline_and_text()
