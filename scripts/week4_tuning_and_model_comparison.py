"""
Week 4: Model Evaluation, Hyperparameter Tuning & Selection
Covering Chapters 10 and 11 from Master Machine Learning with scikit-learn.

Topics:
- GridSearchCV and RandomizedSearchCV on entire Pipelines
- Jointly tuning transformers and estimators
- Comparing linear (LogisticRegression) and non-linear (RandomForest) models
- Saving/loading trained pipelines with joblib
"""

import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def demo_pipeline_tuning():
    print("--- 1. Pipeline Hyperparameter Tuning & Model Comparison ---")
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    # Search space comparing LogisticRegression and RandomForestClassifier
    param_grid = [
        {
            'classifier': [LogisticRegression(solver='liblinear', random_state=42)],
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2']
        },
        {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 5, 10]
        }
    ]

    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    print("Running GridSearchCV across multiple models and hyperparameters...")
    grid_search.fit(X_train, y_train)

    print(f"Best Estimator: {grid_search.best_params_['classifier'].__class__.__name__}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best 5-Fold CV Score: {grid_search.best_score_:.4f}")
    print(f"Test Set Accuracy: {grid_search.score(X_test, y_test):.4f}")

    # Saving and Loading Pipeline
    model_filename = 'best_pipeline.joblib'
    joblib.dump(grid_search.best_estimator_, model_filename)
    print(f"\nSaved best pipeline to '{model_filename}'")

    # Reload model and verify
    loaded_pipe = joblib.load(model_filename)
    reloaded_score = loaded_pipe.score(X_test, y_test)
    print(f"Reloaded Pipeline Test Accuracy: {reloaded_score:.4f}")

    # Cleanup temporary model file
    if os.path.exists(model_filename):
        os.remove(model_filename)

if __name__ == '__main__':
    demo_pipeline_tuning()
