"""
Week 13: Feature Selection Methods
Book Chapter: 13 - Feature selection

Topics:
- Filter methods: SelectKBest (ANOVA f_classif, mutual_info_classif)
- Embedded methods: SelectFromModel with L1 regularization (Lasso/LogisticRegression)
- Wrapper methods: Recursive Feature Elimination (RFE) & SequentialFeatureSelector
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 60)
    print(" Week 13: Feature Selection Methods ")
    print("=" * 60)

    cancer = load_breast_cancer(as_frame=True)
    X, y = cancer.data, cancer.target
    print(f"\nOriginal Feature Count: {X.shape[1]}")

    # 1. Filter Method: SelectKBest (ANOVA F-value)
    selector_kbest = SelectKBest(score_func=f_classif, k=10)
    X_kbest = selector_kbest.fit_transform(X, y)
    print(f"1. SelectKBest (k=10): Selected {X_kbest.shape[1]} features")

    # 2. Embedded Method: SelectFromModel (L1 Regularization)
    pipe_l1 = Pipeline([
        ('scaler', StandardScaler()),
        ('sfm', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)))
    ])
    pipe_l1.fit(X, y)
    selected_mask = pipe_l1.named_steps['sfm'].get_support()
    print(f"2. SelectFromModel (L1 Penalty): Selected {selected_mask.sum()} features out of {X.shape[1]}")

    # 3. Wrapper Method: Recursive Feature Elimination (RFE)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=8, step=2)
    X_rfe = rfe.fit_transform(X, y)
    print(f"3. RFE (Random Forest): Selected {X_rfe.shape[1]} features")

    selected_feature_names = X.columns[rfe.get_support()]
    print("\nTop Features Selected by RFE:")
    for name in selected_feature_names:
        print(f" - {name}")

if __name__ == "__main__":
    main()
