# 20-Week Machine Learning Study Plan

> **Based on *Master Machine Learning with scikit-learn* by Kevin Markham (Data School)**  
> Website: [https://mlbook.dataschool.io/](https://mlbook.dataschool.io/)

---

## 📌 Overview

This 20-week study plan expands upon the 20 chapters of *Master Machine Learning with scikit-learn* to provide an in-depth, production-ready curriculum. Each week focuses on a single chapter, delivering a comprehensive exploration of `scikit-learn` workflows, best practices, preprocessing pipelines, model selection, ensembling, and deployment strategies.

Every week is accompanied by an executable, standalone Python script located in the [`scripts/`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts) directory.

---

## 🛠 Prerequisites & Environment Setup

This repository contains a pre-configured Python virtual environment inside `.venv/`.

### Executing Weekly Scripts
To run any of the weekly scripts, execute the command via the project's virtual environment:

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

## 📅 Table of Contents

- [Week 1: Fundamentals of Scikit-Learn & ML Workflow](#-week-1-fundamentals-of-scikit-learn--ml-workflow)
- [Week 2: Review of the Machine Learning Workflow](#-week-2-review-of-the-machine-learning-workflow)
- [Week 3: Encoding Categorical Features](#-week-3-encoding-categorical-features)
- [Week 4: Preprocessing Pipelines & ColumnTransformers](#-week-4-preprocessing-pipelines--columntransformers)
- [Week 5: Workflow Review #1](#-week-5-workflow-review-1)
- [Week 6: Encoding Text Data](#-week-6-encoding-text-data)
- [Week 7: Handling Missing Values](#-week-7-handling-missing-values)
- [Week 8: Fixing Common Workflow Problems & Data Leakage](#-week-8-fixing-common-workflow-problems--data-leakage)
- [Week 9: Workflow Review #2](#-week-9-workflow-review-2)
- [Week 10: Evaluating & Tuning Pipelines](#-week-10-evaluating--tuning-pipelines)
- [Week 11: Comparing Linear & Non-Linear Models](#-week-11-comparing-linear--non-linear-models)
- [Week 12: Ensembling Multiple Models](#-week-12-ensembling-multiple-models)
- [Week 13: Feature Selection Methods](#-week-13-feature-selection-methods)
- [Week 14: Feature Standardization & Scaling](#-week-14-feature-standardization--scaling)
- [Week 15: Custom Transformers & Feature Engineering](#-week-15-custom-transformers--feature-engineering)
- [Week 16: Workflow Review #3](#-week-16-workflow-review-3)
- [Week 17: High-Cardinality Categorical Features](#-week-17-high-cardinality-categorical-features)
- [Week 18: Class Imbalance Fundamentals & Metrics](#-week-18-class-imbalance-fundamentals--metrics)
- [Week 19: Class Imbalance Walkthrough & Threshold Tuning](#-week-19-class-imbalance-walkthrough--threshold-tuning)
- [Week 20: Going Further: Persistence, Deployment & Validation](#-week-20-going-further-persistence-deployment--validation)

---

## 📅 Detailed 20-Week Curriculum

### 🗓️ Week 1: Fundamentals of Scikit-Learn & ML Workflow
- **Book Chapter**: [Chapter 1: Introduction](https://mlbook.dataschool.io/ch01.html)
- **Core API Focus**: `LogisticRegression`, `train_test_split`, `accuracy_score`, `classification_report`
- **Key Concepts**:
  - The standard 4-step `scikit-learn` estimator API pattern: instantiate, `fit()`, `predict()`, and `predict_proba()`.
  - Understanding the difference between supervised learning tasks (classification vs regression).
  - Train/Test dataset partitioning, random seeds, and class stratification using `stratify=y`.
  - Interpreting class probability outputs (`predict_proba`) and output shapes.
- **Key Pitfalls**: Failing to set `stratify=y` on classification tasks with mild or strong class imbalance.
- **Executable Script**: [`scripts/week01_ml_workflow_fundamentals.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week01_ml_workflow_fundamentals.py)

---

### 🗓️ Week 2: Review of the Machine Learning Workflow
- **Book Chapter**: [Chapter 2: Review of the Machine Learning workflow](https://mlbook.dataschool.io/ch02.html)
- **Core API Focus**: `cross_val_score`, `KFold`, `StratifiedKFold`, `Ridge`
- **Key Concepts**:
  - Evaluating model variance across $K$-folds vs single train-test splits.
  - Stratified K-Fold cross-validation for classification vs standard K-Fold for regression.
  - Computing out-of-fold metrics (`accuracy`, `r2`, `neg_mean_squared_error`).
  - Standard deviation across CV folds as an indicator of model instability.
- **Key Pitfalls**: Performing feature selection or scaling prior to split inside cross-validation.
- **Executable Script**: [`scripts/week02_review_ml_workflow.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week02_review_ml_workflow.py)

---

### 🗓️ Week 3: Encoding Categorical Features
- **Book Chapter**: [Chapter 3: Encoding categorical features](https://mlbook.dataschool.io/ch03.html)
- **Core API Focus**: `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`
- **Key Concepts**:
  - Nominal vs Ordinal categorical features and selecting appropriate encoders.
  - `OneHotEncoder`: dummy variable creation, sparse vs dense matrix outputs (`sparse_output=False`), handling unseen levels via `handle_unknown='ignore'`.
  - `OrdinalEncoder`: explicitly mapping ordered categories (`categories=[['small', 'medium', 'large']]`).
  - `LabelEncoder`: encoding target vectors ($y$) vs feature matrices ($X$).
- **Key Pitfalls**: Using `LabelEncoder` on input feature matrices, which accidentally imposes false ordering on nominal categories.
- **Executable Script**: [`scripts/week03_categorical_encoding.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week03_categorical_encoding.py)

---

### 🗓️ Week 4: Preprocessing Pipelines & ColumnTransformers
- **Book Chapter**: [Chapter 4: Improving your workflow with ColumnTransformer and Pipeline](https://mlbook.dataschool.io/ch04.html)
- **Core API Focus**: `ColumnTransformer`, `make_column_transformer`, `make_column_selector`, `Pipeline`
- **Key Concepts**:
  - Combining different feature preprocessing steps into a unified `ColumnTransformer`.
  - Dynamic feature column selection based on data type (`make_column_selector(dtype_include=['int64', 'object'])`).
  - Constructing clean end-to-end pipelines using `make_pipeline`.
  - Inspecting transformed feature names using `get_feature_names_out()`.
- **Key Pitfalls**: Manually applying transformers to DataFrame subsets and concatenating arrays outside of a pipeline framework.
- **Executable Script**: [`scripts/week04_column_transformer_pipelines.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week04_column_transformer_pipelines.py)

---

### 🗓️ Week 5: Workflow Review #1
- **Book Chapter**: [Chapter 5: Workflow review #1](https://mlbook.dataschool.io/ch05.html)
- **Core API Focus**: `RandomForestClassifier`, `ColumnTransformer`, `Pipeline`, `cross_val_score`
- **Key Concepts**:
  - Comprehensive review of modular preprocessing pipelines on realistic tabular data.
  - End-to-end integration: scaling numeric columns, one-hot encoding categorical columns, and fitting tree ensembles.
  - Benchmark evaluation of baseline pipelines using 5-fold stratified cross-validation.
- **Key Pitfalls**: Missing drop parameters in `OneHotEncoder` when using linear models (multicollinearity).
- **Executable Script**: [`scripts/week05_workflow_review_1.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week05_workflow_review_1.py)

---

### 🗓️ Week 6: Encoding Text Data
- **Book Chapter**: [Chapter 6: Encoding text data](https://mlbook.dataschool.io/ch06.html)
- **Core API Focus**: `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`
- **Key Concepts**:
  - Bag-of-words text representations and tokenization.
  - `CountVectorizer` vs `TfidfVectorizer` (Term Frequency-Inverse Document Frequency weighting).
  - Stopword filtering, n-gram extraction (`ngram_range=(1, 2)`), and vocabulary size management.
  - Incorporating text vectorizers into multi-modal `ColumnTransformer` pipelines alongside numeric features.
- **Key Pitfalls**: Fitting text vectorization on test data or out-of-fold validation splits.
- **Executable Script**: [`scripts/week06_encoding_text_data.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week06_encoding_text_data.py)

---

### 🗓️ Week 7: Handling Missing Values
- **Book Chapter**: [Chapter 7: Handling missing values](https://mlbook.dataschool.io/ch07.html)
- **Core API Focus**: `SimpleImputer`, `KNNImputer`, `IterativeImputer`, `MissingIndicator`
- **Key Concepts**:
  - Missing data mechanisms: Missing Completely at Random (MCAR), Missing at Random (MAR), Missing Not at Random (MNAR).
  - Univariate imputation strategies: mean, median, most_frequent, constant.
  - Multivariate imputation strategies: nearest neighbors (`KNNImputer`) and MICE (`IterativeImputer`).
  - Tracking missingness patterns explicitly using `MissingIndicator`.
- **Key Pitfalls**: Imputing missing values on full datasets prior to splitting into train/test or cross-validation sets.
- **Executable Script**: [`scripts/week07_handling_missing_values.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week07_handling_missing_values.py)

---

### 🗓️ Week 8: Fixing Common Workflow Problems & Data Leakage
- **Book Chapter**: [Chapter 8: Fixing common workflow problems](https://mlbook.dataschool.io/ch08.html)
- **Core API Focus**: `make_pipeline`, `cross_val_score`, `StandardScaler`
- **Key Concepts**:
  - Defining **Data Leakage** and understanding how information leaks from validation/test sets into training steps.
  - Demonstrating artificially optimistic CV metrics resulting from pre-split transformation.
  - Enforcing strict boundary isolation between train and validation splits using `scikit-learn` `Pipeline`.
- **Key Pitfalls**: Applying feature scaling, target encoding, or dimensionality reduction before cross-validation splitting.
- **Executable Script**: [`scripts/week08_fixing_workflow_problems.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week08_fixing_workflow_problems.py)

---

### 🗓️ Week 9: Workflow Review #2
- **Book Chapter**: [Chapter 9: Workflow review #2](https://mlbook.dataschool.io/ch09.html)
- **Core API Focus**: `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `GradientBoostingClassifier`
- **Key Concepts**:
  - Designing multi-stage nested pipelines (e.g. `SimpleImputer` $\rightarrow$ `OneHotEncoder` inside `ColumnTransformer`).
  - Handling missing data in both numerical and categorical features simultaneously.
  - Benchmarking complex architectures against raw baseline models.
- **Key Pitfalls**: Mismatched feature index names during inference on unseen DataFrames containing missing values.
- **Executable Script**: [`scripts/week09_workflow_review_2.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week09_workflow_review_2.py)

---

### 🗓️ Week 10: Evaluating & Tuning Pipelines
- **Book Chapter**: [Chapter 10: Evaluating and tuning a Pipeline](https://mlbook.dataschool.io/ch10.html)
- **Core API Focus**: `GridSearchCV`, `RandomizedSearchCV`, `Pipeline`
- **Key Concepts**:
  - Jointly searching hyperparameter spaces across both transformers and estimators using double-underscore syntax (`step_name__param_name`).
  - Evaluating preprocessing choices (e.g. `imputer__strategy`: `['mean', 'median']`) alongside classifier hyperparameters.
  - Extracting best estimator configurations, validation curves, and inspecting `cv_results_`.
- **Key Pitfalls**: Grid searching without wrapping transformers inside a Pipeline, leading to repeated data leakage during evaluation.
- **Executable Script**: [`scripts/week10_evaluating_tuning_pipelines.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week10_evaluating_tuning_pipelines.py)

---

### 🗓️ Week 11: Comparing Linear & Non-Linear Models
- **Book Chapter**: [Chapter 11: Comparing linear and non-linear models](https://mlbook.dataschool.io/ch11.html)
- **Core API Focus**: `LogisticRegression`, `SVC`, `RandomForestClassifier`, `GradientBoostingClassifier`
- **Key Concepts**:
  - Evaluating model capacity, expressiveness, decision boundaries, and computational complexity.
  - Linear models with L1/L2 regularization vs non-linear Kernel SVMs vs Tree-based Ensembles.
  - Interpreting linear coefficients vs tree feature importances.
- **Key Pitfalls**: Neglecting feature scaling when benchmarking distance-sensitive algorithms (Linear models, SVM) against tree models.
- **Executable Script**: [`scripts/week11_linear_vs_nonlinear_models.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week11_linear_vs_nonlinear_models.py)

---

### 🗓️ Week 12: Ensembling Multiple Models
- **Book Chapter**: [Chapter 12: Ensembling multiple models](https://mlbook.dataschool.io/ch12.html)
- **Core API Focus**: `VotingClassifier`, `StackingClassifier`, `BaggingClassifier`
- **Key Concepts**:
  - Ensembling fundamentals: variance reduction via Bagging and bias reduction via Boosting.
  - `VotingClassifier`: Hard voting (majority rules) vs Soft voting (weighted class probabilities).
  - `StackingClassifier`: Combining heterogeneous base models via out-of-fold predictions fed into a second-stage meta-learner.
- **Key Pitfalls**: Using hard voting when estimators output well-calibrated class probabilities.
- **Executable Script**: [`scripts/week12_ensembling_models.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week12_ensembling_models.py)

---

### 🗓️ Week 13: Feature Selection Methods
- **Book Chapter**: [Chapter 13: Feature selection](https://mlbook.dataschool.io/ch13.html)
- **Core API Focus**: `SelectKBest`, `SelectFromModel`, `RFE`, `SequentialFeatureSelector`
- **Key Concepts**:
  - Three main taxonomy branches of feature selection: Filter, Wrapper, and Embedded methods.
  - Filter methods: `SelectKBest` with statistical tests (ANOVA F-test, Mutual Information).
  - Embedded methods: L1 regularization (Lasso / Sparse Logistic Regression) with `SelectFromModel`.
  - Wrapper methods: Recursive Feature Elimination (`RFE`) and Sequential Feature Selection (`SFS`).
- **Key Pitfalls**: Running wrapper methods on high-dimensional data without setting a reasonable feature step size.
- **Executable Script**: [`scripts/week13_feature_selection.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week13_feature_selection.py)

---

### 🗓️ Week 14: Feature Standardization & Scaling
- **Book Chapter**: [Chapter 14: Feature standardization](https://mlbook.dataschool.io/ch14.html)
- **Core API Focus**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`
- **Key Concepts**:
  - `StandardScaler` (z-score normalization: $\mu=0, \sigma=1$).
  - `MinMaxScaler` (bounding features between $[0, 1]$).
  - `RobustScaler` (using median and interquartile range to handle heavy outliers).
  - Sensitivity analysis across algorithms: Distance-based (KNN, SVM, K-Means), Gradient-based (Linear/Logistic Regression), and Tree-based models (Random Forest, XGBoost).
- **Key Pitfalls**: Using `StandardScaler` on data containing extreme outliers, which distorts the mean and variance calculation.
- **Executable Script**: [`scripts/week14_feature_standardization.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week14_feature_standardization.py)

---

### 🗓️ Week 15: Custom Transformers & Feature Engineering
- **Book Chapter**: [Chapter 15: Feature engineering with custom transformers](https://mlbook.dataschool.io/ch15.html)
- **Core API Focus**: `FunctionTransformer`, `BaseEstimator`, `TransformerMixin`
- **Key Concepts**:
  - Creating stateless custom transformers with `FunctionTransformer` (e.g. logarithmic or mathematical transformations).
  - Building stateful custom transformers by subclassing `BaseEstimator` and `TransformerMixin`.
  - Implementing `fit()`, `transform()`, and compatible `get_feature_names_out()` methods.
- **Key Pitfalls**: Mutating input DataFrames in-place inside `transform()` calls instead of returning fresh copies or arrays.
- **Executable Script**: [`scripts/week15_custom_transformers.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week15_custom_transformers.py)

---

### 🗓️ Week 16: Workflow Review #3
- **Book Chapter**: [Chapter 16: Workflow review #3](https://mlbook.dataschool.io/ch16.html)
- **Core API Focus**: `Pipeline`, `ColumnTransformer`, `BaseEstimator`, `HistGradientBoostingClassifier`
- **Key Concepts**:
  - Production-grade modular pipeline construction integrating domain-specific feature engineering.
  - Embedding custom ratio/interaction transformers, imputation, scaling, and categorical encoders into a unified pipeline.
  - End-to-end evaluation and cross-validation verification.
- **Key Pitfalls**: Hardcoding column indices inside custom transformers, making them fragile to raw input schema changes.
- **Executable Script**: [`scripts/week16_workflow_review_3.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week16_workflow_review_3.py)

---

### 🗓️ Week 17: High-Cardinality Categorical Features
- **Book Chapter**: [Chapter 17: High-cardinality categorical features](https://mlbook.dataschool.io/ch17.html)
- **Core API Focus**: `TargetEncoder`, `OneHotEncoder`, `FeatureHasher`
- **Key Concepts**:
  - The curse of dimensionality when applying `OneHotEncoder` to high-cardinality features (e.g., ZIP codes, IP addresses).
  - Target Encoding (`TargetEncoder`): Replacing categorical levels with smoothed out-of-fold target mean estimates.
  - Feature Hashing (`FeatureHasher` / `HashingEncoder`): Bounding output feature space using hash functions.
- **Key Pitfalls**: Naively computing target encodings on training sets without out-of-fold smoothing, causing severe target leakage.
- **Executable Script**: [`scripts/week17_high_cardinality_features.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week17_high_cardinality_features.py)

---

### 🗓️ Week 18: Class Imbalance Fundamentals & Metrics
- **Book Chapter**: [Chapter 18: Class imbalance](https://mlbook.dataschool.io/ch18.html)
- **Core API Focus**: `classification_report`, `roc_auc_score`, `average_precision_score`, `LogisticRegression(class_weight='balanced')`
- **Key Concepts**:
  - Why standard accuracy is misleading on imbalanced datasets (e.g. 99% accuracy on 1% positive rate).
  - Cost-sensitive learning: using `class_weight='balanced'` to adjust loss function penalties.
  - Evaluation metric selection: Precision, Recall, F1-Score, ROC AUC, and Precision-Recall AUC (PR-AUC).
- **Key Pitfalls**: Relying on ROC AUC instead of PR-AUC when evaluating datasets with extreme class imbalance.
- **Executable Script**: [`scripts/week18_class_imbalance.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week18_class_imbalance.py)

---

### 🗓️ Week 19: Class Imbalance Walkthrough & Threshold Tuning
- **Book Chapter**: [Chapter 19: Class imbalance walkthrough](https://mlbook.dataschool.io/ch19.html)
- **Core API Focus**: `precision_recall_curve`, `predict_proba`, `RandomForestClassifier`
- **Key Concepts**:
  - Post-processing classification decisions: shifting decision thresholds away from the default $0.5$.
  - Analyzing Precision-Recall trade-off curves.
  - Finding optimal decision thresholds to maximize domain-specific cost metrics or F1 scores.
- **Key Pitfalls**: Re-training models repeatedly when tuning classification thresholds instead of storing predicted probabilities.
- **Executable Script**: [`scripts/week19_class_imbalance_walkthrough.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week19_class_imbalance_walkthrough.py)

---

### 🗓️ Week 20: Going Further: Persistence, Deployment & Validation
- **Book Chapter**: [Chapter 20: Going further](https://mlbook.dataschool.io/ch20.html)
- **Core API Focus**: `joblib.dump`, `joblib.load`, `Pipeline`
- **Key Concepts**:
  - Model serialization and deserialization using `joblib`.
  - Exporting complete end-to-end preprocessing + model pipelines into a single binary artifact (`.joblib`).
  - Production inference verification: ensuring input DataFrame schemas match production expectations.
  - Validating predictions and probability scores post-deserialization.
- **Key Pitfalls**: Saving fitted models without including the associated preprocessing transformers in the serialized artifact.
- **Executable Script**: [`scripts/week20_going_further_production.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week20_going_further_production.py)

---

## 📊 Summary of Resources Created

| Week | Book Chapter | Python Implementation Script |
|---|---|---|
| Week 1 | [Chapter 1](https://mlbook.dataschool.io/ch01.html) | [`scripts/week01_ml_workflow_fundamentals.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week01_ml_workflow_fundamentals.py) |
| Week 2 | [Chapter 2](https://mlbook.dataschool.io/ch02.html) | [`scripts/week02_review_ml_workflow.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week02_review_ml_workflow.py) |
| Week 3 | [Chapter 3](https://mlbook.dataschool.io/ch03.html) | [`scripts/week03_categorical_encoding.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week03_categorical_encoding.py) |
| Week 4 | [Chapter 4](https://mlbook.dataschool.io/ch04.html) | [`scripts/week04_column_transformer_pipelines.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week04_column_transformer_pipelines.py) |
| Week 5 | [Chapter 5](https://mlbook.dataschool.io/ch05.html) | [`scripts/week05_workflow_review_1.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week05_workflow_review_1.py) |
| Week 6 | [Chapter 6](https://mlbook.dataschool.io/ch06.html) | [`scripts/week06_encoding_text_data.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week06_encoding_text_data.py) |
| Week 7 | [Chapter 7](https://mlbook.dataschool.io/ch07.html) | [`scripts/week07_handling_missing_values.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week07_handling_missing_values.py) |
| Week 8 | [Chapter 8](https://mlbook.dataschool.io/ch08.html) | [`scripts/week08_fixing_workflow_problems.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week08_fixing_workflow_problems.py) |
| Week 9 | [Chapter 9](https://mlbook.dataschool.io/ch09.html) | [`scripts/week09_workflow_review_2.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week09_workflow_review_2.py) |
| Week 10 | [Chapter 10](https://mlbook.dataschool.io/ch10.html) | [`scripts/week10_evaluating_tuning_pipelines.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week10_evaluating_tuning_pipelines.py) |
| Week 11 | [Chapter 11](https://mlbook.dataschool.io/ch11.html) | [`scripts/week11_linear_vs_nonlinear_models.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week11_linear_vs_nonlinear_models.py) |
| Week 12 | [Chapter 12](https://mlbook.dataschool.io/ch12.html) | [`scripts/week12_ensembling_models.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week12_ensembling_models.py) |
| Week 13 | [Chapter 13](https://mlbook.dataschool.io/ch13.html) | [`scripts/week13_feature_selection.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week13_feature_selection.py) |
| Week 14 | [Chapter 14](https://mlbook.dataschool.io/ch14.html) | [`scripts/week14_feature_standardization.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week14_feature_standardization.py) |
| Week 15 | [Chapter 15](https://mlbook.dataschool.io/ch15.html) | [`scripts/week15_custom_transformers.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week15_custom_transformers.py) |
| Week 16 | [Chapter 16](https://mlbook.dataschool.io/ch16.html) | [`scripts/week16_workflow_review_3.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week16_workflow_review_3.py) |
| Week 17 | [Chapter 17](https://mlbook.dataschool.io/ch17.html) | [`scripts/week17_high_cardinality_features.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week17_high_cardinality_features.py) |
| Week 18 | [Chapter 18](https://mlbook.dataschool.io/ch18.html) | [`scripts/week18_class_imbalance.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week18_class_imbalance.py) |
| Week 19 | [Chapter 19](https://mlbook.dataschool.io/ch19.html) | [`scripts/week19_class_imbalance_walkthrough.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week19_class_imbalance_walkthrough.py) |
| Week 20 | [Chapter 20](https://mlbook.dataschool.io/ch20.html) | [`scripts/week20_going_further_production.py`](file:///Users/dikshie/VIRTUAL/mastering_machine_learning/scripts/week20_going_further_production.py) |
