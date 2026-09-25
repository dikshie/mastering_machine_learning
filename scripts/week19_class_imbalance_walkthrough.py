"""
Week 19: Class Imbalance Walkthrough & Decision Threshold Tuning
Book Chapter: 19 - Class imbalance walkthrough

Topics:
- Decision Threshold Tuning: Shifting default 0.5 classification threshold to optimize Recall/F1
- Precision-Recall Curve analysis
- Practical strategy for business-driven cost function optimization
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, classification_report, f1_score

def main():
    print("=" * 60)
    print(" Week 19: Decision Threshold Tuning Walkthrough ")
    print("=" * 60)

    # Generate imbalanced dataset (90:10)
    X, y = make_classification(n_samples=1000, n_features=12, weights=[0.90, 0.10], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_proba = rf.predict_proba(X_test)[:, 1]

    # Evaluate default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    print("\nDefault Threshold (0.50) Performance:")
    print(classification_report(y_test, y_pred_default, target_names=['Negative', 'Positive']))

    # Find optimal threshold maximizing F1-Score via Precision-Recall Curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nOptimal Threshold found for max F1-Score: {best_threshold:.4f}")

    y_pred_tuned = (y_proba >= best_threshold).astype(int)
    print(f"\nTuned Threshold ({best_threshold:.2f}) Performance:")
    print(classification_report(y_test, y_pred_tuned, target_names=['Negative', 'Positive']))

if __name__ == "__main__":
    main()
