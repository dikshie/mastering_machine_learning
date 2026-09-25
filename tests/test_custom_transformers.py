"""
Unit tests for custom transformers in scripts/week15_custom_transformers.py and week16_workflow_review_3.py
"""

import numpy as np
import pandas as pd
import pytest
from scripts.week15_custom_transformers import OutlierCapper
from scripts.week16_workflow_review_3 import RatioAdder

def test_outlier_capper_fit_transform():
    df = pd.DataFrame({
        'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100] # 100 is outlier
    })
    capper = OutlierCapper(lower_quantile=0.1, upper_quantile=0.9)
    capper.fit(df)
    transformed = capper.transform(df)

    # Upper quantile (90th percentile) should cap 100
    assert transformed.max() < 100
    assert transformed.shape == (10, 1)

def test_ratio_adder():
    df = pd.DataFrame({
        'debt': [100, 200],
        'income': [500, 1000]
    })
    ratio_adder = RatioAdder(num_col='debt', denom_col='income', new_col_name='dti')
    result = ratio_adder.transform(df)

    assert 'dti' in result.columns
    assert pytest.approx(result['dti'].iloc[0], 0.001) == 0.2
    assert pytest.approx(result['dti'].iloc[1], 0.001) == 0.2
