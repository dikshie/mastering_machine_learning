"""
Week 17: High-Cardinality Categorical Features
Book Chapter: 17 - High-cardinality categorical features

Topics:
- The curse of dimensionality with One-Hot Encoding on high-cardinality variables
- TargetEncoder (Supervised encoding using out-of-fold target mean)
- HashingEncoder (Feature hashing for high-dimensional sparse representations)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def main():
    print("=" * 60)
    print(" Week 17: High-Cardinality Categorical Features ")
    print("=" * 60)

    # Generate synthetic high-cardinality categorical data (e.g. 100 unique Zip Codes)
    np.random.seed(42)
    zip_codes = [f"ZIP_{i:03d}" for i in range(100)]
    categories = np.random.choice(zip_codes, size=1000)
    y = np.random.normal(loc=50, scale=10, size=1000)

    df = pd.DataFrame({'zip_code': categories})
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # 1. OneHotEncoder (Explodes into 100 columns)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_ohe_train = ohe.fit_transform(X_train)
    print(f"\n1. OneHotEncoder matrix shape: {X_ohe_train.shape} (High memory & dimensionality!)")

    # 2. TargetEncoder (Compresses to 1 column containing smoothed target mean)
    te = TargetEncoder(smooth="auto", cv=5)
    X_te_train = te.fit_transform(X_train, y_train)
    X_te_test = te.transform(X_test)
    print(f"2. TargetEncoder matrix shape: {X_te_train.shape} (Compact 1D representation!)")

    # Train Ridge Regression model on Target Encoded feature
    model = Ridge()
    model.fit(X_te_train, y_train)
    score = model.score(X_te_test, y_test)
    print(f"TargetEncoder + Ridge Test R^2 Score: {score:.4f}")

if __name__ == "__main__":
    main()
