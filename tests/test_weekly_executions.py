"""
Smoke tests for all 20 weekly curriculum scripts.
"""

import importlib
import pytest

WEEK_MODULES = [f"scripts.week{i:02d}_" for i in range(1, 21)]

@pytest.mark.parametrize("week_num", range(1, 21))
def test_weekly_script_execution(week_num):
    # Mapping of week numbers to full module names
    module_mapping = {
        1: "scripts.week01_ml_workflow_fundamentals",
        2: "scripts.week02_review_ml_workflow",
        3: "scripts.week03_categorical_encoding",
        4: "scripts.week04_column_transformer_pipelines",
        5: "scripts.week05_workflow_review_1",
        6: "scripts.week06_encoding_text_data",
        7: "scripts.week07_handling_missing_values",
        8: "scripts.week08_fixing_workflow_problems",
        9: "scripts.week09_workflow_review_2",
        10: "scripts.week10_evaluating_tuning_pipelines",
        11: "scripts.week11_linear_vs_nonlinear_models",
        12: "scripts.week12_ensembling_models",
        13: "scripts.week13_feature_selection",
        14: "scripts.week14_feature_standardization",
        15: "scripts.week15_custom_transformers",
        16: "scripts.week16_workflow_review_3",
        17: "scripts.week17_high_cardinality_features",
        18: "scripts.week18_class_imbalance",
        19: "scripts.week19_class_imbalance_walkthrough",
        20: "scripts.week20_going_further_production",
    }
    
    module_name = module_mapping[week_num]
    mod = importlib.import_module(module_name)
    
    # Execute main or primary demo function
    if hasattr(mod, "main"):
        mod.main()
    elif hasattr(mod, "demo_cross_validation"):
        mod.demo_cross_validation()
    else:
        pytest.fail(f"No executable main entry point found in {module_name}")
