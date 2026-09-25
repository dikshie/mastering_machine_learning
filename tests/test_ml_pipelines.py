"""
Unit tests for pipeline utilities, imputation, and model persistence.
"""

import os
import tempfile
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def test_column_transformer_pipeline():
    df = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'gender': ['M', 'F', 'F', 'M'],
        'target': [0, 1, 0, 1]
    })
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['gender'])
    ])
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    pipeline.fit(df[['age', 'gender']], df['target'])
    preds = pipeline.predict(df[['age', 'gender']])
    assert len(preds) == 4

def test_pipeline_serialization():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    pipe.fit(X)
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        joblib.dump(pipe, tmp_path)
        loaded_pipe = joblib.load(tmp_path)
        transformed_original = pipe.transform(X)
        transformed_loaded = loaded_pipe.transform(X)
        np.testing.assert_array_almost_equal(transformed_original, transformed_loaded)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
