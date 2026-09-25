"""
Week 14: Feature Standardization & Scaling
Book Chapter: 14 - Feature standardization

Topics:
- StandardScaler (z-score scaling) vs MinMaxScaler (range scaling) vs RobustScaler (outlier resistant)
- Evaluating when feature scaling is critical (KNN, SVM, Linear models) vs invariant (Tree models)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

def main():
    print("=" * 60)
    print(" Week 14: Feature Standardization & Scaling ")
    print("=" * 60)

    # Create dataset with vastly different feature scales and outliers
    np.random.seed(42)
    feature_1 = np.random.normal(loc=1000, scale=500, size=200) # Large scale
    feature_2 = np.random.normal(loc=0.01, scale=0.005, size=200) # Small scale
    # Add outliers
    feature_1[0] = 50000.0
    feature_2[0] = 10.0

    y = (feature_1 * 0.001 + feature_2 * 100 > 1.5).astype(int)
    X = np.column_stack([feature_1, feature_2])

    scalers = {
        'No Scaling': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler (Outlier-Resistant)': RobustScaler()
    }

    print("\nImpact of Feature Scaling on KNN vs Decision Tree Performance (5-Fold CV Accuracy):")
    print("-" * 75)
    for scaler_name, scaler in scalers.items():
        if scaler is None:
            knn_model = KNeighborsClassifier(n_neighbors=5)
            dt_model = DecisionTreeClassifier(random_state=42)
        else:
            knn_model = make_pipeline(scaler, KNeighborsClassifier(n_neighbors=5))
            dt_model = make_pipeline(scaler, DecisionTreeClassifier(random_state=42))

        knn_score = cross_val_score(knn_model, X, y, cv=5).mean()
        dt_score = cross_val_score(dt_model, X, y, cv=5).mean()

        print(f"{scaler_name:<35} | KNN Score: {knn_score:.4f} | Decision Tree Score: {dt_score:.4f}")

if __name__ == "__main__":
    main()
