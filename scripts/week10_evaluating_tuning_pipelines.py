"""
Week 10: Evaluating & Tuning Pipelines
Book Chapter: 10 - Evaluating and tuning a Pipeline

Topics:
- GridSearchCV and RandomizedSearchCV across complete pipeline steps
- Joint hyperparameter search over preprocessing parameters (e.g., imputer strategy) and model hyperparameters
- Analyzing cv_results_ and extracting best estimators
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def main():
    print("=" * 60)
    print(" Week 10: Evaluating & Tuning Pipelines ")
    print("=" * 60)

    cancer = load_breast_cancer(as_frame=True)
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define base pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Grid search parameter space (targeting step__parameter)
    param_grid = {
        'imputer__strategy': ['mean', 'median'],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }

    print("\nRunning GridSearchCV on Pipeline...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters Found:\n{grid_search.best_params_}")
    print(f"Best 5-Fold CV Score: {grid_search.best_score_:.4f}")

    test_score = grid_search.score(X_test, y_test)
    print(f"Test Set Accuracy with Best Estimator: {test_score:.4f}")

if __name__ == "__main__":
    main()
