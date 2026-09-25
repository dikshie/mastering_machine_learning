# 6-Week Machine Learning Study Plan
> **Based on *Master Machine Learning with scikit-learn* by Kevin Markham (Data School)**  
> Website: [https://mlbook.dataschool.io/](https://mlbook.dataschool.io/)

---

## 📌 Overview

This study plan translates the 20 chapters of *Master Machine Learning with scikit-learn* into a structured 6-week curriculum. Each week focuses on practical, production-grade workflows in `scikit-learn`, complete with executable Python scripts provided in the `scripts/` directory of this repository.

---

## 🛠 Prerequisites & Environment Setup

This repository contains a pre-configured Python virtual environment inside `.venv/`.

### Run Python Scripts
To execute any weekly script, use the project's virtual environment:
```bash
.venv/bin/python scripts/week1_workflow_and_encoding.py
.venv/bin/python scripts/week2_pipelines_and_text.py
.venv/bin/python scripts/week3_imputation_and_leakage.py
.venv/bin/python scripts/week4_tuning_and_model_comparison.py
.venv/bin/python scripts/week5_ensembling_and_selection.py
.venv/bin/python scripts/week6_custom_transformers_and_imbalance.py
```

---

## 📅 Weekly Curriculum

### 🗓️ Week 1: Fundamentals of Scikit-Learn & ML Workflow
- **Book Chapters**: 
  - [Chapter 1: Introduction](https://mlbook.dataschool.io/ch01.html)
  - [Chapter 2: Review of the Machine Learning workflow](https://mlbook.dataschool.io/ch02.html)
  - [Chapter 3: Encoding categorical features](https://mlbook.dataschool.io/ch03.html)
- **Key Concepts**:
  - Distinguishing `scikit-learn` workflow from Deep Learning frameworks.
  - The standard 4-step ML workflow: `fit`, `predict`, `predict_proba`.
  - Splitting data using `train_test_split` and stratified sampling.
  - Model evaluation with cross-validation (`cross_val_score`).
  - Nominal vs. Ordinal categorical features.
  - Categorical encoders: `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`.
- **Executable Script**: [`scripts/week1_workflow_and_encoding.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week1_workflow_and_encoding.py)

---

### 🗓️ Week 2: Advanced Preprocessing Pipelines & Text Encoding
- **Book Chapters**:
  - [Chapter 4: Improving your workflow with ColumnTransformer and Pipeline](https://mlbook.dataschool.io/ch04.html)
  - [Chapter 5: Workflow review #1](https://mlbook.dataschool.io/ch05.html)
  - [Chapter 6: Encoding text data](https://mlbook.dataschool.io/ch06.html)
- **Key Concepts**:
  - Bundling column-specific preprocessing using `ColumnTransformer` & `make_column_transformer`.
  - Automatic column selection using `make_column_selector` by data types or column names.
  - Chaining preprocessing steps with estimators using `Pipeline` & `make_pipeline`.
  - Vectorizing text columns with `CountVectorizer` and sparse matrix considerations.
- **Executable Script**: [`scripts/week2_pipelines_and_text.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week2_pipelines_and_text.py)

---

### 🗓️ Week 3: Handling Missing Values & Preventing Data Leakage
- **Book Chapters**:
  - [Chapter 7: Handling missing values](https://mlbook.dataschool.io/ch07.html)
  - [Chapter 8: Fixing common workflow problems](https://mlbook.dataschool.io/ch08.html)
  - [Chapter 9: Workflow review #2](https://mlbook.dataschool.io/ch09.html)
- **Key Concepts**:
  - Strategies for missing value imputation: `SimpleImputer` (mean, median, mode, constant), `KNNImputer`.
  - Treating missingness as a feature (`MissingIndicator`).
  - Building nested pipelines (`Pipeline` inside `ColumnTransformer`) for complex features.
  - Preventing **Data Leakage**: Why transformations should never be fitted on the entire dataset prior to splitting.
- **Executable Script**: [`scripts/week3_imputation_and_leakage.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week3_imputation_and_leakage.py)

---

### 🗓️ Week 4: Pipeline Tuning & Model Comparison
- **Book Chapters**:
  - [Chapter 10: Evaluating and tuning a Pipeline](https://mlbook.dataschool.io/ch10.html)
  - [Chapter 11: Comparing linear and non-linear models](https://mlbook.dataschool.io/ch11.html)
- **Key Concepts**:
  - Hyperparameter tuning across entire pipelines using `GridSearchCV` & `RandomizedSearchCV`.
  - Jointly tuning transformer parameters (e.g., imputation strategies) and model estimators.
  - Comparing linear models (`LogisticRegression`) and non-linear models (`RandomForestClassifier`) within a single grid search space.
  - Persisting trained pipelines to disk with `joblib`.
- **Executable Script**: [`scripts/week4_tuning_and_model_comparison.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week4_tuning_and_model_comparison.py)

---

### 🗓️ Week 5: Ensembling, Feature Selection & Standardization
- **Book Chapters**:
  - [Chapter 12: Ensembling multiple models](https://mlbook.dataschool.io/ch12.html)
  - [Chapter 13: Feature selection](https://mlbook.dataschool.io/ch13.html)
  - [Chapter 14: Feature standardization](https://mlbook.dataschool.io/ch14.html)
- **Key Concepts**:
  - Model Ensembling with `VotingClassifier` (Hard voting vs. Soft voting).
  - Feature Selection strategies:
    - Intrinsic & Filter methods: L1 regularization (`SelectFromModel`).
    - Wrapper methods: Recursive Feature Elimination (`RFE`).
  - Feature Standardization (`StandardScaler`): Understanding when scaling is required (linear models, distance-based algorithms vs. tree ensembles).
- **Executable Script**: [`scripts/week5_ensembling_and_selection.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week5_ensembling_and_selection.py)

---

### 🗓️ Week 6: Custom Transformers, High Cardinality & Class Imbalance
- **Book Chapters**:
  - [Chapter 15: Feature engineering with custom transformers](https://mlbook.dataschool.io/ch15.html)
  - [Chapter 16: Workflow review #3](https://mlbook.dataschool.io/ch16.html)
  - [Chapter 17: High-cardinality categorical features](https://mlbook.dataschool.io/ch17.html)
  - [Chapter 18: Class imbalance](https://mlbook.dataschool.io/ch18.html)
  - [Chapter 19: Class imbalance walkthrough](https://mlbook.dataschool.io/ch19.html)
  - [Chapter 20: Going further](https://mlbook.dataschool.io/ch20.html)
- **Key Concepts**:
  - Building custom transformers using `FunctionTransformer` and `BaseEstimator` / `TransformerMixin`.
  - Handling high-cardinality categorical variables using target encoding (`TargetEncoder`).
  - Dealing with Class Imbalance:
    - Cost-sensitive learning (`class_weight='balanced'`).
    - Evaluation metrics beyond accuracy: ROC AUC score, Confusion Matrix, Precision-Recall.
    - Decision threshold tuning via `predict_proba`.
- **Executable Script**: [`scripts/week6_custom_transformers_and_imbalance.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week6_custom_transformers_and_imbalance.py)

---

## 📊 Summary of Resources Created

| Week | Target Chapters | Python Example Script |
|---|---|---|
| Week 1 | Ch 1 - 3 | [`scripts/week1_workflow_and_encoding.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week1_workflow_and_encoding.py) |
| Week 2 | Ch 4 - 6 | [`scripts/week2_pipelines_and_text.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week2_pipelines_and_text.py) |
| Week 3 | Ch 7 - 9 | [`scripts/week3_imputation_and_leakage.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week3_imputation_and_leakage.py) |
| Week 4 | Ch 10 - 11 | [`scripts/week4_tuning_and_model_comparison.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week4_tuning_and_model_comparison.py) |
| Week 5 | Ch 12 - 14 | [`scripts/week5_ensembling_and_selection.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week5_ensembling_and_selection.py) |
| Week 6 | Ch 15 - 20 | [`scripts/week6_custom_transformers_and_imbalance.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week6_custom_transformers_and_imbalance.py) |
