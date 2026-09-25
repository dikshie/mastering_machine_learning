"""
Week 18: Class Imbalance Fundamentals & Evaluation Metrics
Book Chapter: 18 - Class imbalance

Topics:
- Evaluating imbalanced datasets: why accuracy is misleading
- Class weighting (`class_weight='balanced'`) in cost-sensitive learning
- Key metrics: Precision, Recall, F1-score, ROC AUC, Precision-Recall AUC
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def main():
    print("=" * 60)
    print(" Week 18: Class Imbalance Fundamentals & Evaluation Metrics ")
    print("=" * 60)

    # Generate severely imbalanced dataset (95% majority, 5% minority)
    X, y = make_classification(
        n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nTarget Class Distribution in Test Set: 0s = {(y_test==0).sum()}, 1s = {(y_test==1).sum()}")

    # 1. Unweighted Logistic Regression
    model_unweighted = LogisticRegression(random_state=42)
    model_unweighted.fit(X_train, y_train)
    y_pred_un = model_unweighted.predict(X_test)
    y_proba_un = model_unweighted.predict_proba(X_test)[:, 1]

    print("\n--- 1. Unweighted Logistic Regression ---")
    print(classification_report(y_test, y_pred_un, target_names=['Majority', 'Minority']))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba_un):.4f}")

    # 2. Cost-Sensitive Logistic Regression (class_weight='balanced')
    model_balanced = LogisticRegression(class_weight='balanced', random_state=42)
    model_balanced.fit(X_train, y_train)
    y_pred_bal = model_balanced.predict(X_test)
    y_proba_bal = model_balanced.predict_proba(X_test)[:, 1]

    print("\n--- 2. Cost-Sensitive (class_weight='balanced') Logistic Regression ---")
    print(classification_report(y_test, y_pred_bal, target_names=['Majority', 'Minority']))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba_bal):.4f}")
    print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_proba_bal):.4f}")

if __name__ == "__main__":
    main()
