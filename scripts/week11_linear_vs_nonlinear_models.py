"""
Week 11: Comparing Linear & Non-Linear Models
Book Chapter: 11 - Comparing linear and non-linear models

Topics:
- Comparing LogisticRegression, Support Vector Machines (SVC), and RandomForest/GradientBoosting
- Evaluating model complexity, interpretability, and decision boundaries
- Tuning regularized linear models vs ensemble trees
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def main():
    print("=" * 60)
    print(" Week 11: Comparing Linear & Non-Linear Models ")
    print("=" * 60)

    # Generate non-linearly separable dataset (Moons)
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

    models = {
        'Linear Model (Logistic Regression)': make_pipeline(StandardScaler(), LogisticRegression()),
        'Non-Linear SVM (RBF Kernel)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0)),
        'Tree Ensemble (Random Forest)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Boosting Ensemble (Gradient Boosting)': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    print("\nBenchmark on Non-Linear Moons Dataset (5-Fold CV Accuracy):")
    print("-" * 55)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"{name:<40}: {scores.mean():.4f} +/- {scores.std():.4f}")

if __name__ == "__main__":
    main()
