"""
Week 8: Fixing Common Workflow Problems & Data Leakage Prevention
Book Chapter: 8 - Fixing common workflow problems

Topics:
- Data Leakage: fitting transformers before train-test split vs inside pipeline
- Demonstrating artificially inflated cross-validation performance caused by leakage
- Ensuring strict separation between train and test distributions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def main():
    print("=" * 60)
    print(" Week 8: Fixing Workflow Problems & Data Leakage ")
    print("=" * 60)

    # Generate synthetic dataset with high dimensionality
    np.random.seed(42)
    X = np.random.normal(size=(100, 1000)) # Random noise
    y = np.random.choice([0, 1], size=100) # Random targets (true score should be ~50%)

    # WRONG WAY: Preprocessing (e.g. scaling or feature selection) on entire X before CV
    print("\n[INCORRECT] Fitting scaler on full dataset prior to cross-validation...")
    scaler_leaky = StandardScaler()
    X_scaled_leaky = scaler_leaky.fit_transform(X)
    model = LogisticRegression()
    leaky_cv_score = cross_val_score(model, X_scaled_leaky, y, cv=5).mean()
    print(f"Leaky CV Score: {leaky_cv_score:.4f} (Can mask distribution mismatches!)")

    # CORRECT WAY: Preprocessing strictly enclosed inside Pipeline during CV
    print("\n[CORRECT] Encapsulating scaler inside Scikit-Learn Pipeline during CV...")
    correct_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    correct_cv_score = cross_val_score(correct_pipeline, X, y, cv=5).mean()
    print(f"Correct Pipeline CV Score: {correct_cv_score:.4f} (Reflects true generalization!)")

if __name__ == "__main__":
    main()
