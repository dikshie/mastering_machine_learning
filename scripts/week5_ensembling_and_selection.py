"""
Week 5: Ensembling, Feature Selection & Standardization
Covering Chapters 12, 13, and 14 from Master Machine Learning with scikit-learn.

Topics:
- VotingClassifier (hard vs soft voting ensembles)
- Feature Selection techniques (SelectFromModel with L1 penalty, RFE)
- Feature Standardization (StandardScaler) inside pipeline workflows
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel, RFE

def demo_ensembling_and_selection():
    print("--- 1. Feature Selection & Standardization ---")
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature selection using L1 regularized Logistic Regression inside a pipeline
    l1_selector_pipe = make_pipeline(
        StandardScaler(),
        SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)),
        LogisticRegression(random_state=42)
    )
    l1_selector_pipe.fit(X_train, y_train)
    
    # Check how many features were selected
    selector_step = l1_selector_pipe.named_steps['selectfrommodel']
    selected_mask = selector_step.get_support()
    print(f"Original number of features: {X.shape[1]}")
    print(f"Selected features count (L1): {selected_mask.sum()}")
    print(f"Pipeline Test Accuracy: {l1_selector_pipe.score(X_test, y_test):.4f}\n")

    print("--- 2. Ensembling Multiple Models (VotingClassifier) ---")
    # Base estimators
    clf1 = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)

    # Hard Voting Ensemble
    voting_hard = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2)],
        voting='hard'
    )

    # Soft Voting Ensemble (uses predicted probabilities)
    voting_soft = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2)],
        voting='soft'
    )

    voting_hard.fit(X_train, y_train)
    voting_soft.fit(X_train, y_train)

    print(f"Hard Voting Test Accuracy: {voting_hard.score(X_test, y_test):.4f}")
    print(f"Soft Voting Test Accuracy: {voting_soft.score(X_test, y_test):.4f}")

if __name__ == '__main__':
    demo_ensembling_and_selection()
