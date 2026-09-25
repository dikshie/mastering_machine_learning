"""
Week 1: Fundamentals of Scikit-Learn & Machine Learning Workflow
Book Chapter: 1 - Introduction to scikit-learn

Topics:
- Scikit-learn estimator API (fit, predict, predict_proba)
- Iris classification dataset example
- Model evaluation with standard train/test splitting
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 60)
    print(" Week 1: Fundamentals of Scikit-Learn & ML Workflow")
    print("=" * 60)

    # 1. Load dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    print(f"\nDataset shape: {X.shape}")
    print(f"Feature names: {iris.feature_names}")
    print(f"Target classes: {iris.target_names}")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 3. Model instantiation & training
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")

    # 4. Prediction & Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test[:3])

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nSample Probabilities (First 3 Test Samples):")
    print(pd.DataFrame(y_proba, columns=iris.target_names))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    main()
