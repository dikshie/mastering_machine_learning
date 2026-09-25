# Mastering Machine Learning with Scikit-Learn

[![CI/CD Pipeline](https://github.com/dikshie/mastering_machine_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/dikshie/mastering_machine_learning/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade, 20-week structured curriculum and hands-on repository based on ***Master Machine Learning with scikit-learn*** by Kevin Markham (Data School).  
> Official Website: [https://mlbook.dataschool.io/](https://mlbook.dataschool.io/)

---

## 📌 Overview

This repository translates all 20 chapters of Kevin Markham's *Master Machine Learning with scikit-learn* into a structured **20-week study plan**. Each week features a dedicated chapter guide, deep-dive theoretical concepts, edge-case analysis, and an executable Python script demonstrating real-world `scikit-learn` workflows.

Whether you are learning `scikit-learn` best practices, avoiding data leakage in cross-validation, building custom feature transformers, tuning complex pipelines, or persisting models for production deployment, this repository provides battle-tested code and patterns.

---

## 🚀 Quick Start

### 1. Repository Setup

Clone the repository and navigate into the project directory:

```bash
git clone git@github.com:dikshie/mastering_machine_learning.git
cd mastering_machine_learning
```

### 2. Environment Activation & Installation

Using the included pre-configured virtual environment or your own Python environment:

```bash
# Activate existing virtualenv (or create one using `python3 -m venv .venv`)
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 💻 Running the 20-Week Python Scripts

Each weekly module is located in the [`scripts/`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts) directory. You can execute any script directly via Python:

```bash
.venv/bin/python scripts/week01_ml_workflow_fundamentals.py
.venv/bin/python scripts/week02_review_ml_workflow.py
.venv/bin/python scripts/week03_categorical_encoding.py
.venv/bin/python scripts/week04_column_transformer_pipelines.py
.venv/bin/python scripts/week05_workflow_review_1.py
.venv/bin/python scripts/week06_encoding_text_data.py
.venv/bin/python scripts/week07_handling_missing_values.py
.venv/bin/python scripts/week08_fixing_workflow_problems.py
.venv/bin/python scripts/week09_workflow_review_2.py
.venv/bin/python scripts/week10_evaluating_tuning_pipelines.py
.venv/bin/python scripts/week11_linear_vs_nonlinear_models.py
.venv/bin/python scripts/week12_ensembling_models.py
.venv/bin/python scripts/week13_feature_selection.py
.venv/bin/python scripts/week14_feature_standardization.py
.venv/bin/python scripts/week15_custom_transformers.py
.venv/bin/python scripts/week16_workflow_review_3.py
.venv/bin/python scripts/week17_high_cardinality_features.py
.venv/bin/python scripts/week18_class_imbalance.py
.venv/bin/python scripts/week19_class_imbalance_walkthrough.py
.venv/bin/python scripts/week20_going_further_production.py
```

---

## 🧪 Unit Testing & CI/CD

This repository includes a comprehensive unit test suite written with `pytest` in [`tests/`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/tests), covering custom transformer logic, pipeline serialization, and execution validation across all 20 weeks.

### Run Unit Tests Locally

```bash
.venv/bin/pytest
```

### GitHub Actions Integration

Automated Continuous Integration runs via [GitHub Actions](.github/workflows/ci.yml) on every `push` and `pull_request` targeting the `main` branch across Python versions 3.10, 3.11, and 3.12.

---

## 📅 20-Week Curriculum Map

Full details for each week can be found in **[STUDY_PLAN.md](STUDY_PLAN.md)**.

| Week | Target Book Chapter | Focus & Key Concepts | Executable Python Script |
|---|---|---|---|
| **Week 1** | [Ch 1: Introduction](https://mlbook.dataschool.io/ch01.html) | Estimator API (`fit`, `predict`, `predict_proba`), Iris classification | [`scripts/week01_ml_workflow_fundamentals.py`](scripts/week01_ml_workflow_fundamentals.py) |
| **Week 2** | [Ch 2: Review ML Workflow](https://mlbook.dataschool.io/ch02.html) | Stratified K-Fold CV vs K-Fold Regression | [`scripts/week02_review_ml_workflow.py`](scripts/week02_review_ml_workflow.py) |
| **Week 3** | [Ch 3: Categorical Encoding](https://mlbook.dataschool.io/ch03.html) | `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder` | [`scripts/week03_categorical_encoding.py`](scripts/week03_categorical_encoding.py) |
| **Week 4** | [Ch 4: Preprocessing Pipelines](https://mlbook.dataschool.io/ch04.html) | `ColumnTransformer` & `make_column_selector` | [`scripts/week04_column_transformer_pipelines.py`](scripts/week04_column_transformer_pipelines.py) |
| **Week 5** | [Ch 5: Workflow Review #1](https://mlbook.dataschool.io/ch05.html) | End-to-end Titanic preprocessing & tree ensemble | [`scripts/week05_workflow_review_1.py`](scripts/week05_workflow_review_1.py) |
| **Week 6** | [Ch 6: Encoding Text Data](https://mlbook.dataschool.io/ch06.html) | `CountVectorizer`, `TfidfVectorizer` & multi-modal features | [`scripts/week06_encoding_text_data.py`](scripts/week06_encoding_text_data.py) |
| **Week 7** | [Ch 7: Missing Values](https://mlbook.dataschool.io/ch07.html) | `SimpleImputer`, `KNNImputer`, `IterativeImputer` (MICE) | [`scripts/week07_handling_missing_values.py`](scripts/week07_handling_missing_values.py) |
| **Week 8** | [Ch 8: Workflow Problems](https://mlbook.dataschool.io/ch08.html) | Data leakage prevention inside CV pipelines | [`scripts/week08_fixing_workflow_problems.py`](scripts/week08_fixing_workflow_problems.py) |
| **Week 9** | [Ch 9: Workflow Review #2](https://mlbook.dataschool.io/ch09.html) | Multi-step nested preprocessing architectures | [`scripts/week09_workflow_review_2.py`](scripts/week09_workflow_review_2.py) |
| **Week 10** | [Ch 10: Tuning Pipelines](https://mlbook.dataschool.io/ch10.html) | `GridSearchCV` hyperparameter tuning across pipelines | [`scripts/week10_evaluating_tuning_pipelines.py`](scripts/week10_evaluating_tuning_pipelines.py) |
| **Week 11** | [Ch 11: Linear vs Non-Linear](https://mlbook.dataschool.io/ch11.html) | Benchmarking Logistic Regression, SVM, Random Forest | [`scripts/week11_linear_vs_nonlinear_models.py`](scripts/week11_linear_vs_nonlinear_models.py) |
| **Week 12** | [Ch 12: Model Ensembling](https://mlbook.dataschool.io/ch12.html) | Hard/Soft `VotingClassifier` & `StackingClassifier` | [`scripts/week12_ensembling_models.py`](scripts/week12_ensembling_models.py) |
| **Week 13** | [Ch 13: Feature Selection](https://mlbook.dataschool.io/ch13.html) | Filter (`SelectKBest`), Embedded (`L1`), Wrapper (`RFE`) | [`scripts/week13_feature_selection.py`](scripts/week13_feature_selection.py) |
| **Week 14** | [Ch 14: Feature Scaling](https://mlbook.dataschool.io/ch14.html) | `StandardScaler` vs `MinMaxScaler` vs `RobustScaler` | [`scripts/week14_feature_standardization.py`](scripts/week14_feature_standardization.py) |
| **Week 15** | [Ch 15: Custom Transformers](https://mlbook.dataschool.io/ch15.html) | Subclassing `BaseEstimator` & `TransformerMixin` | [`scripts/week15_custom_transformers.py`](scripts/week15_custom_transformers.py) |
| **Week 16** | [Ch 16: Workflow Review #3](https://mlbook.dataschool.io/ch16.html) | Modular production feature engineering pipelines | [`scripts/week16_workflow_review_3.py`](scripts/week16_workflow_review_3.py) |
| **Week 17** | [Ch 17: High-Cardinality](https://mlbook.dataschool.io/ch17.html) | `TargetEncoder` & Feature Hashing | [`scripts/week17_high_cardinality_features.py`](scripts/week17_high_cardinality_features.py) |
| **Week 18** | [Ch 18: Class Imbalance](https://mlbook.dataschool.io/ch18.html) | Cost-sensitive weighting & PR-AUC evaluation | [`scripts/week18_class_imbalance.py`](scripts/week18_class_imbalance.py) |
| **Week 19** | [Ch 19: Imbalance Walkthrough](https://mlbook.dataschool.io/ch19.html) | Decision threshold tuning via Precision-Recall curves | [`scripts/week19_class_imbalance_walkthrough.py`](scripts/week19_class_imbalance_walkthrough.py) |
| **Week 20** | [Ch 20: Production Deployment](https://mlbook.dataschool.io/ch20.html) | Model persistence with `joblib` & payload validation | [`scripts/week20_going_further_production.py`](scripts/week20_going_further_production.py) |

---

## 📂 Directory Structure

```
mastering_machine_learning/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline configuration
├── scripts/
│   ├── __init__.py
│   ├── week01_ml_workflow_fundamentals.py
│   ├── week02_review_ml_workflow.py
│   ├── week03_categorical_encoding.py
│   ├── week04_column_transformer_pipelines.py
│   ├── week05_workflow_review_1.py
│   ├── week06_encoding_text_data.py
│   ├── week07_handling_missing_values.py
│   ├── week08_fixing_workflow_problems.py
│   ├── week09_workflow_review_2.py
│   ├── week10_evaluating_tuning_pipelines.py
│   ├── week11_linear_vs_nonlinear_models.py
│   ├── week12_ensembling_models.py
│   ├── week13_feature_selection.py
│   ├── week14_feature_standardization.py
│   ├── week15_custom_transformers.py
│   ├── week16_workflow_review_3.py
│   ├── week17_high_cardinality_features.py
│   ├── week18_class_imbalance.py
│   ├── week19_class_imbalance_walkthrough.py
│   └── week20_going_further_production.py
├── tests/
│   ├── test_custom_transformers.py
│   ├── test_ml_pipelines.py
│   └── test_weekly_executions.py
├── pytest.ini                     # Pytest configuration settings
├── requirements.txt               # Project dependency requirements
├── STUDY_PLAN.md                  # Comprehensive 20-week study plan document
└── README.md                      # Project documentation overview
```

---

## 📜 License

This repository is released under the [MIT License](LICENSE).
