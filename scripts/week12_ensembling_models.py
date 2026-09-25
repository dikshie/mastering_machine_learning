"""
Week 12: Ensembling Multiple Models
Book Chapter: 12 - Ensembling multiple models

Topics:
- Hard vs Soft VotingClassifier
- StackingClassifier (Combining diverse base learners with a meta-estimator)
- Bagging and Boosting fundamentals
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def main():
    print("=" * 60)
    print(" Week 12: Ensembling Multiple Models ")
    print("=" * 60)

    cancer = load_breast_cancer(as_frame=True)
    X, y = cancer.data, cancer.target

    # Base estimators
    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipe_svc = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))

    estimators = [('lr', pipe_lr), ('rf', rf), ('svc', pipe_svc)]

    # 1. Soft Voting Classifier
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')

    # 2. Stacking Classifier with LogisticRegression Meta-Learner
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)

    print("\n--- 5-Fold Cross-Validation Performance ---")
    for name, model in [
        ('Logistic Regression', pipe_lr),
        ('Random Forest', rf),
        ('Support Vector Machine', pipe_svc),
        ('Ensemble: Soft Voting', voting_soft),
        ('Ensemble: Stacking', stacking)
    ]:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"{name:<25}: Mean Accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")

if __name__ == "__main__":
    main()
