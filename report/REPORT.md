# CE888 — Offline Agentic Data Scientist: Technical Report

**Author:** Francisco Antonio Alamo Rios  
**Registration number:** 2508983  
**Module:** CE888 — Data Science and Decision Making  
**Academic Year:** 2025/2026  

---

## 1. Introduction

The goal of this project is to build an autonomous, offline data scientist agent capable of performing end-to-end machine learning pipelines without human intervention or reliance on Large Language Models. The agent must profile datasets, plan preprocessing and modelling strategies, train and evaluate candidate models, reflect on results, and learn from experience across runs — all driven by rule-based logic and heuristics.

The motivation for an agentic approach is that data science workflows are inherently conditional. The right preprocessing strategy depends on the data: a dataset with severe class imbalance needs different treatment than a balanced one; a dataset with highly correlated features calls for feature selection; a small dataset risks overfitting with complex models. A static pipeline cannot adapt. An agent that reads the data, reasons about it, and adjusts its plan accordingly is both more principled and more practical.

The design philosophy throughout this project was to make every decision traceable. Rather than applying all preprocessing steps by default, each step is gated behind a profile signal. This means the output plan — logged per run — acts as an explanation of why specific adaptations were made. For example, when `drop_leaky_features` appears in the plan, it is because the profiler measured a normalised mutual information score ≥ 0.9 between that feature and the target. This transparency distinguishes an agentic system from a fixed sklearn pipeline.

A secondary design goal was robustness. The agent includes a training retry loop (up to 3 attempts), memory-guided model prioritisation, and failed-target tracking. These mechanisms reduce the chance that a single bad run wastes time or misleads subsequent runs.

This report describes the design and implementation of each component — Planner, Executor, Modelling tools, Reflector, and Memory — alongside the signals that drive decisions, the adaptations made beyond the skeleton code, and the limitations identified through testing.

---

## 2. System Architecture

The agent follows a sense–plan–act loop with a reflection and memory feedback cycle:

```
Dataset → Profiler → Planner → Executor → Modelling → Evaluator
                        ↑                                    ↓
                     Memory  ←────────── Reflector ←────────
```

**Components:**

- **Profiler** (`tools/data_profiler.py`): Analyses the dataset and produces a profile dictionary containing shape, feature types, missing data percentages, class balance, correlation structure, leakage risk, scale mismatch, and near-constant columns.
- **Planner** (`agents/planner.py`): Reads the profile and generates an ordered list of plan steps. Each step is either a core pipeline action (`build_preprocessor`, `train_models`) or a configuration flag (`apply_regularization`, `drop_leaky_features`).
- **Executor** (`agentic_data_scientist.py`): Iterates over the plan, sets configuration flags on the profile, builds the preprocessor and model candidates, trains and evaluates, then reflects. If reflection recommends a replan, the cycle repeats up to `max_replans` times.
- **Modelling** (`tools/modelling.py`): Provides `build_preprocessor()` which constructs a `ColumnTransformer` from profile flags, and `select_models()` which returns a size-appropriate set of candidate models.
- **Reflector** (`agents/reflector.py`): Analyses evaluation metrics against dataset characteristics and returns a structured reflection containing issues, suggestions, and a `replan_recommended` flag.
- **Memory** (`agents/memory.py`): Persists outcomes per dataset fingerprint. Provides lookup by fingerprint, filename, or target+shape. Also tracks failed targets to avoid repeating them on subsequent runs.

---

## 3. Dataset Understanding — Signals and Their Meaning

The profiler extracts a rich set of signals from every dataset. The table below maps the most important signals to the decisions they drive.

| Signal | How detected | Downstream effect |
|--------|-------------|-------------------|
| `shape.rows` | Row count | Size bucket: small (<1000), medium (1000–9999), large (≥10000) |
| `imbalance_ratio` | majority/minority class count | ≥3.0 → class-weighted models; suggestion to consider SMOTE |
| `missing_pct` | % NaN per column | >20% → `handle_severe_missing_data`; columns above threshold dropped |
| `outlier_cols` | IQR-based outlier count >5% | `handle_outliers` → RobustScaler |
| `scale_mismatch` | max_range / median_range ≥ 50 | `apply_robust_scaling` → RobustScaler instead of StandardScaler |
| `leaky_cols` | Normalised mutual information ≥ 0.9 with target | `drop_leaky_features` → column excluded from all transformers |
| `high_corr_pairs` | Pairwise Pearson abs_corr ≥ 0.95 | `drop_correlated_features` → weaker of each pair dropped |
| `near_constant_cols` | One value ≥ 95% dominant | `drop_near_constant_features` → column excluded |
| `n_unique_by_col` | Unique value count per column | >50 unique + <10% of rows → `apply_target_encoding` |
| `is_classification` | Target cardinality and dtype | Routes to classification or regression models and metrics |

A key implementation decision was the **leakage detection fix**. The original skeleton passed the regression target through `pd.factorize()` before computing `mutual_info_regression`, converting continuous values into integer rank labels. This broke the MI normalisation — with 30,000 near-unique values, the target entropy denominator became enormous, collapsing all normalised MI scores below the 0.9 threshold. After the fix, regression leakage is detected correctly by using raw target values and normalising by the maximum MI score instead of target entropy. This allowed `final_price_usd` (normalised MI = 1.0) to be correctly flagged as leaky in the Sales dataset, where `revenue_usd = final_price_usd × units_sold` exactly.

---

## 4. Planning Logic — Conditional Decisions

The planner generates an ordered plan by applying a series of conditional checks on the profile. The core pipeline steps are fixed; additional steps are inserted or appended based on the signals detected.

**Size bucket logic** drives model selection and regularization:
- Rows < 1000 (small): `apply_regularization` (LogisticRegression C=0.1) and `use_simple_models_only` (Dummy + LR + RF only, no GradientBoosting or SVC)
- Rows 1000–9999 (medium): full model set including GradientBoosting
- Rows ≥ 10,000 (large): `use_ensemble_models` (GradientBoosting always included)

This bucket logic is consistent across the planner, modelling, and memory components — all three use the same thresholds of 1,000 and 10,000.

**Memory-guided planning** uses prior run outcomes. If a previous fingerprint match exists in memory and recorded a best-performing model, the planner adds `prioritize_model:<name>` to the plan. The modelling layer then places that model first in the candidate list, ensuring it is trained even if the run is later truncated. This was observed in practice on the Titanic dataset: the second run correctly prioritised `LogisticRegression` based on the first run's memory record.

**Target encoding** is triggered when the planner finds categorical or text columns with more than 50 unique values but fewer than 10% of rows (to distinguish true high-cardinality categoricals from free-text or ID columns). The preprocessing layer adds a `TargetEncoder` branch to the `ColumnTransformer` for those columns. This was important for the Sales dataset where `model_name` (899 unique values) was classified as `text` by the profiler but is meaningfully categorical.

---

## 5. Tool Use: Modelling and Evaluation

### Preprocessing

`build_preprocessor()` constructs a scikit-learn `ColumnTransformer` with separate branches for continuous, ordinal, and categorical features. Key adaptive decisions:

- **Scaler selection**: StandardScaler by default; RobustScaler when `handle_outliers` or `use_robust_scaling` is set; `RobustScaler(unit_variance=True)` for large datasets with outliers (quantile clamping)
- **Missing value strategy**: median imputation for numeric, most-frequent for categorical (threshold-based column dropping before this)
- **High-cardinality categoricals**: sklearn's `TargetEncoder` (v1.3+) fitted inside the pipeline to prevent leakage — target means are computed on training data only
- **Feature dropping**: near-constant, highly correlated, leaky, and high-missing columns are all excluded before the transformers are applied

### Model Selection

Models are selected based on the size bucket and profile flags:

| Condition | Models |
|-----------|--------|
| `simple_models_only` (small dataset) | DummyClassifier, LogisticRegression, RandomForest |
| Medium / Large (classification) | + GradientBoosting |
| `prefer_ensemble` (large) | GradientBoosting always included |
| Regression | DummyRegressor, LinearRegression, RandomForestRegressor, GradientBoostingRegressor |

A dummy baseline is always included. This is critical for the reflector — all performance assessments are relative to baseline improvement.

SVC was restricted to datasets with fewer than 1,000 rows and 50 columns, and `probability=True` was removed since Platt scaling adds significant training cost and probabilities are never used by the evaluation layer.

### Metrics

Classification uses balanced accuracy and macro-averaged F1 as primary metrics (to handle class imbalance). Regression uses R², MAE, and RMSE. The best model is selected by balanced accuracy + F1 for classification, R² for regression.

---

## 6. Reflection and Re-planning

The reflector analyses evaluation results against dataset characteristics and generates a structured output. Every reflection contains:
- `status`: `"ok"` or `"needs_attention"`
- `issues`: list of identified problems
- `suggestions`: list of improvement recommendations
- `replan_recommended`: boolean
- `training_warnings`: sklearn runtime warnings captured during training

### Classification Checks

| Condition | Issue | Replan? |
|-----------|-------|---------|
| Best model < 0.05 above dummy balanced accuracy | Weak signal or pipeline issues | Yes (if f1 < 0.60) |
| bal_acc > 0.90 and f1 < 0.70 | Overfitting suspected | Yes (if f1 < 0.60) |
| f1_macro < 0.60 | Modest F1 score | Yes |
| ≥2 non-dummy models with bal_acc ≥ 0.99 and f1 ≥ 0.99 | Suspicious near-perfect performance | No |
| All models F1 within 0.05 | Low model diversity | Yes (if f1 < 0.60) |

### Regression Checks

| Condition | Issue | Replan? |
|-----------|-------|---------|
| R² improvement < 0.05 over dummy | Weak regression signal | Yes |
| R² < 0.10 | Very low explanatory power | Yes |
| ≥2 non-dummy models R² ≥ 0.99 | Suspicious near-perfect R² | Yes |

### Training Warning Detection

sklearn runtime warnings (overflow, divide-by-zero, invalid value in matrix operations) are captured during training using `warnings.catch_warnings(record=True)` and stored in `reflection.json`. If numerical instability warnings are present, the reflector always adds a suggestion to check feature scales. If performance is also weak (f1 < 0.60 or R² < 0.1), the warning is escalated to an issue, since instability may be degrading results.

### Replanning

`should_replan()` defers entirely to `replan_recommended`. An earlier implementation also triggered replanning when multiple issues were present or status was `needs_attention` — this caused spurious replans on the penguins dataset where near-perfect performance is legitimate, not a problem. The fix was to make `replan_recommended` the single gate, set only when performance is genuinely poor.

`apply_replan_strategy()` modifies the plan based on issue keywords. If overfitting is detected, `apply_regularization` is added. If imbalance is flagged, `consider_imbalance_strategy` is added. If F1 is weak, `try_ensemble_methods` is appended (though this step is currently not handled by the executor — a known limitation discussed in Section 8).

---

## 7. Memory and Learning

The memory system (`agents/memory.py`) persists outcomes in a JSON file (`agent_memory.json`). Each record is keyed by a dataset fingerprint (hash of filename + target + column names) and stores:

- `dataset`: filename
- `target`: column name
- `target_source`: whether it was provided manually, inferred, or read from memory
- `shape`: rows and columns
- `is_classification`: task type
- `best_model`: name of the best-performing model
- `best_metrics`: full metrics dict
- `reflection_status`: `"ok"` or `"needs_attention"`
- `last_seen`: UTC timestamp

**Lookup** operates in three steps: exact fingerprint, then filename, then target+shape. This means a renamed copy of a dataset still gets a memory hit, enabling continuity across experiments.

**Model prioritisation**: When memory returns a best model, the planner adds `prioritize_model:<name>` to the plan. This was observed to improve second-run efficiency on the Titanic dataset, where memory correctly identified `LogisticRegression` as the best-performing model.

**Failed target tracking**: A dedicated `failed_targets` dict records targets that produced no useful results (best model is Dummy + status is `needs_attention`). On subsequent auto-detection runs, those targets are skipped. This was tested with `Sales.csv` where `category` as a target produced random-level predictions — the agent stored it as failed and skipped to the next candidate on the following run.

**Similarity matching** (`get_similar_record`) finds records from structurally similar datasets using three heuristics: same size bucket, similar imbalance level, similar missingness. This enables cross-dataset transfer of strategies when no exact fingerprint match exists.

---

## 8. Ethics and Limitations

### Data Leakage
The agent detects leakage through mutual information analysis but only at the individual feature level. Multiplicative relationships (e.g., `revenue_usd = final_price_usd × units_sold`) are not detected because neither component alone reaches the MI threshold. The reflector's near-perfect performance check acts as a secondary signal, correctly flagging this case as suspicious even when the feature-level detector misses it.

### Proxy Features and Fairness
The leakage detector flags features that are near-perfect proxies of the target. However, it does not assess whether features encode protected characteristics (e.g., gender, race). The agent makes no fairness-aware adjustments to model selection or thresholds.

### Imbalance Handling
Class weights are applied automatically when imbalance ratio ≥ 3.0, but SMOTE and oversampling are not implemented. For severe imbalance (ratio ≥ 10), class weights alone may be insufficient.

### Known Dead Flags
Three plan steps set profile flags that are not yet fully implemented:
- `robust_imputation`: the imputer always uses median regardless of this flag
- `use_feature_engineering`: no feature engineering is applied
- `try_ensemble_methods`: added by the replanner but has no executor handler

These represent known limitations that would be addressed in a production system.

### Replan Effectiveness
The replan mechanism works correctly for performance-based issues (weak F1, poor R²). For structural issues like multiplicative leakage, the replanner currently has no handler — it adds `replan_attempt` to the plan but changes nothing in the second run. The correct fix would be to progressively lower the MI leakage threshold during replanning, catching features that individually fall below 0.9 but together construct the target.

### Scope
The agent is designed for tabular, offline batch processing. It has no support for streaming data, time series, image or text-heavy datasets, or deployment.

---

## 9. Experimental Results

Four datasets were used to validate the agent across different scenarios:

| Dataset | Rows | Task | Best Model | Key Metric | Plan Adaptations |
|---------|------|------|-----------|-----------|-----------------|
| `penguins.csv` | 344 | Classification | LogisticRegression | bal_acc=1.000, f1=1.000 | regularization, simple_models_only, robust_scaling |
| `titanic.csv` | 784 | Classification | LogisticRegression | bal_acc=0.853, f1=0.855 | outliers, missing_data, robust_scaling, leaky `alive` dropped |
| `digits.csv` | 1797 | Classification | RandomForest | bal_acc=0.972, f1=0.972 | outliers, 14 near-constant pixel cols dropped |
| `Sales.csv` | 30000 | Regression | RandomForestRegressor | R²=1.000 (suspicious) | target_encoding, leaky `final_price_usd` dropped, ensemble |

**Penguins (species classification):** The agent correctly identified this as a small dataset (344 rows) and applied regularisation and simple model selection. Both LogisticRegression and RandomForest achieved near-perfect balanced accuracy of 1.000. The reflector flagged this as suspicious near-perfect performance, which is a false positive in this case — the penguin species target is genuinely well-separated by bill and flipper measurements, making it a trivially easy classification problem. This highlights a known limitation: the near-perfect check cannot distinguish genuine separability from data leakage without inspecting feature semantics.

**Titanic (survival classification):** The Titanic run demonstrates several adaptive behaviours working together. The `alive` column — a string encoding of whether the passenger survived — was correctly flagged as leaky via mutual information (normalised MI = 1.0) and dropped before training. The `deck` column (77% missing values) triggered `handle_severe_missing_data`. Scale mismatch between `fare` (0–512) and ordinal features like `pclass` (1–3) triggered RobustScaler. The final result — LogisticRegression with balanced accuracy 0.853 and macro F1 0.855 — is a strong outcome for this well-studied dataset, achieved through fully automated adaptive preprocessing. Memory was correctly updated and the second run prioritised LogisticRegression, improving run efficiency.

**Digits (handwritten digit classification):** The digits dataset contains 64 pixel features derived from 8×8 images. Border pixels are always or nearly always zero — 14 columns were flagged as near-constant (dominant value ≥ 95%) and dropped. RandomForest achieved balanced accuracy 0.972 and macro F1 0.972 on 10-class digit recognition, a strong result. The plan included `handle_outliers` and `drop_near_constant_features`, demonstrating that the profile correctly identified the dataset's structural characteristics without any human guidance.

**Sales (revenue regression):** The Sales dataset provided the most instructive run. `final_price_usd` was correctly dropped as leaky (normalised MI = 1.0 with `revenue_usd`) and `model_name` (899 unique product names) was target-encoded rather than dropped. Despite dropping the strongest leaky feature, RandomForestRegressor still achieved R² = 1.000 because `revenue_usd = base_price_usd × (1 - discount_percent/100) × units_sold` — all three components remain in the dataset. The reflector correctly raised a `needs_attention` warning with a suspicious near-perfect R² issue. This demonstrates both the strength of the leakage detector (caught the direct proxy) and its limitation (cannot detect multiplicative combinations of individually sub-threshold features).

---

## 10. Conclusion and Future Work

This project implements a functioning offline agentic data scientist that adapts its behaviour to dataset characteristics through a profile–plan–execute–reflect–remember loop. The key contributions beyond the skeleton are:

1. **Leakage detection via mutual information** with a regression-specific fix (raw target values rather than factorized ranks)
2. **Scale mismatch detection** driving RobustScaler selection
3. **Size-bucket model selection** consistent across planner, modelling, and memory
4. **Target encoding** for high-cardinality categoricals via sklearn's `TargetEncoder`
5. **Near-constant feature detection** surfacing in the plan for transparency
6. **Training warning capture** with reflector integration
7. **Failed target memory** preventing the agent from repeating unsuccessful targets
8. **Spurious replan fix** ensuring replanning only triggers when performance is genuinely poor

Future work would address the identified limitations: implementing progressive leakage threshold reduction during replanning; adding SMOTE for severe imbalance; implementing the dead flags (`robust_imputation`, `feature_engineering`); extending the reflector with per-class F1 breakdown; and adding fairness-aware checks for protected characteristics.

The test suite achieves 93% code coverage (125 tests), well above the 60% requirement, with integration-level smoke tests that exercise the full pipeline end-to-end using real datasets.

The most important lesson from this project is that the value of an agentic system lies not in any single technique but in the connections between components. Detecting scale mismatch is only useful if the planner acts on it; leakage detection is only useful if the preprocessor respects it; training warnings are only useful if the reflector surfaces them. Each signal–plan–action–reflection chain required careful wiring across multiple files, and maintaining a shared signal map (`AGENT_SIGNAL_MAP.md`) throughout development proved essential for tracking which signals were implemented end-to-end versus partially wired or dead. This documentation practice would be equally valuable in any production ML system where multiple engineers share responsibility for different pipeline stages.

---

*AI assistance (Claude) was used during the development and documentation of this project, in accordance with module guidelines.*
