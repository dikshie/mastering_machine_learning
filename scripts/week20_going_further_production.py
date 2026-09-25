"""
Week 20: Going Further - Model Persistence, Deployment & Pipeline Validation
Book Chapter: 20 - Going further

Topics:
- Serialization & Deserialization of full Scikit-Learn Pipelines using joblib
- Production verification: asserting schema compatibility, handling incoming unseen data, predictions validation
- Pipeline inspectability: feature name tracking across pipeline transformers
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def main():
    print("=" * 60)
    print(" Week 20: Model Persistence, Deployment & Pipeline Validation ")
    print("=" * 60)

    # 1. Train production pipeline
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    prod_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    prod_pipeline.fit(X_train, y_train)
    score = prod_pipeline.score(X_test, y_test)
    print(f"\n1. Trained Production Pipeline (Test Accuracy: {score:.4f})")

    # 2. Serialize pipeline artifact using joblib
    model_filename = 'model_pipeline.joblib'
    joblib.dump(prod_pipeline, model_filename)
    print(f"2. Saved production pipeline artifact to '{model_filename}'")

    # 3. Load model artifact (Simulating inference service load)
    loaded_pipeline = joblib.load(model_filename)
    print("3. Loaded production pipeline artifact from disk.")

    # 4. Inference validation on new payload
    raw_payload = pd.DataFrame([{
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.4,
        'petal width (cm)': 0.2
    }])

    pred_class = loaded_pipeline.predict(raw_payload)[0]
    pred_name = iris.target_names[pred_class]
    pred_probs = loaded_pipeline.predict_proba(raw_payload)[0]

    print(f"\n4. Inference Result:")
    print(f"   Predicted Class: {pred_name} (Class ID: {pred_class})")
    print(f"   Class Probabilities: {dict(zip(iris.target_names, np.round(pred_probs, 4)))}")

    # Cleanup temporary artifact file
    if os.path.exists(model_filename):
        os.remove(model_filename)
        print(f"\nCleaned up temporary artifact file '{model_filename}'.")

if __name__ == "__main__":
    main()
