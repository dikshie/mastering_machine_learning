"""
Week 7: Handling Missing Values
Book Chapter: 7 - Handling missing values

Topics:
- SimpleImputer strategies (mean, median, most_frequent, constant)
- KNNImputer for nearest-neighbor based imputation
- IterativeImputer (MICE) for multivariate regression imputation
- MissingIndicator to capture missingness patterns as explicit features
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def main():
    print("=" * 60)
    print(" Week 7: Handling Missing Values ")
    print("=" * 60)

    # Synthetic dataset with missing values (NaN)
    X = np.array([
        [25.0, 50000.0, 3.0],
        [np.nan, 64000.0, 2.0],
        [47.0, np.nan, 5.0],
        [51.0, 85000.0, np.nan],
        [62.0, 95000.0, 4.0],
        [23.0, 42000.0, 1.0]
    ])

    print("\nOriginal Matrix with NaNs:")
    print(X)

    # 1. SimpleImputer (Mean strategy)
    simple_imp = SimpleImputer(strategy='mean')
    X_simple = simple_imp.fit_transform(X)
    print("\n1. SimpleImputer (Mean):")
    print(np.round(X_simple, 2))

    # 2. KNNImputer
    knn_imp = KNNImputer(n_neighbors=2)
    X_knn = knn_imp.fit_transform(X)
    print("\n2. KNNImputer (k=2):")
    print(np.round(X_knn, 2))

    # 3. IterativeImputer (Multivariate Imputation by Chained Equations - MICE)
    iter_imp = IterativeImputer(max_iter=10, random_state=42)
    X_iter = iter_imp.fit_transform(X)
    print("\n3. IterativeImputer (MICE):")
    print(np.round(X_iter, 2))

    # 4. MissingIndicator
    indicator = MissingIndicator()
    mask = indicator.fit_transform(X)
    print("\n4. MissingIndicator Mask (True where value was missing):")
    print(mask)

if __name__ == "__main__":
    main()
