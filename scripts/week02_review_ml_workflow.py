"""
Week 2: Review of the Machine Learning Workflow & Data Splitting Strategies
Book Chapter: 2 - Review of the Machine Learning workflow

Topics:
- Stratified K-Fold Cross Validation
- Regression vs Classification workflow
- Evaluating variance across CV folds
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import Ridge, LogisticRegression

def demo_cross_validation():
    print("=" * 60)
    print(" Week 2: Review of ML Workflow & Cross-Validation ")
    print("=" * 60)

    # 1. Stratified K-Fold for Classification
    cancer = load_breast_cancer(as_frame=True)
    X_clf, y_clf = cancer.data, cancer.target
    clf_model = LogisticRegression(max_iter=5000, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_scores = cross_val_score(clf_model, X_clf, y_clf, cv=skf, scoring='accuracy')

    print("\n--- 1. Classification CV (Breast Cancer) ---")
    print(f"5-Fold CV Accuracy Scores: {clf_scores}")
    print(f"Mean Accuracy: {clf_scores.mean():.4f} +/- {clf_scores.std():.4f}")

    # 2. Standard K-Fold for Regression
    housing = fetch_california_housing(as_frame=True)
    X_reg, y_reg = housing.data.iloc[:2000], housing.target.iloc[:2000] # Subsample for speed
    reg_model = Ridge(alpha=1.0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_scores = cross_val_score(reg_model, X_reg, y_reg, cv=kf, scoring='r2')

    print("\n--- 2. Regression CV (California Housing Subsample) ---")
    print(f"5-Fold CV R^2 Scores: {reg_scores}")
    print(f"Mean R^2 Score: {reg_scores.mean():.4f} +/- {reg_scores.std():.4f}")

if __name__ == "__main__":
    demo_cross_validation()
