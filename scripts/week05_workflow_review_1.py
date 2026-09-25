"""
Week 5: Workflow Review #1
Book Chapter: 5 - Workflow review #1

Topics:
- Comprehensive review of end-to-end ML workflows
- Integrating feature scaling, One-Hot Encoding, and Model evaluation into a robust pipeline
- Benchmarking pipeline baseline performance across cross-validation folds
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def main():
    print("=" * 60)
    print(" Week 5: Workflow Review #1 ")
    print("=" * 60)

    # Load Titanic dataset from OpenML for realistic tabular data workflow review
    print("\nFetching Titanic dataset...")
    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = titanic.frame[['pclass', 'sex', 'age', 'fare', 'survived']].dropna()

    X = df[['pclass', 'sex', 'age', 'fare']]
    y = df['survived'].astype(int)

    numeric_features = ['age', 'fare']
    categorical_features = ['pclass', 'sex']

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    # Full modeling pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5-fold Stratified Cross-Validation benchmark
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring='accuracy')

    print(f"\nTitanic Baseline Pipeline 5-Fold Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    print("Scores per fold:", [round(s, 4) for s in scores])

if __name__ == "__main__":
    main()
