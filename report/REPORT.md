# CE888 — Offline Agentic Data Scientist: Technical Report

**Author:** Francisco Antonio Alamo Rios  
**Registration number:** 2508983  
**Module:** CE888 — Data Science and Decision Making  
**Academic Year:** 2025/2026  

---

## 1. Introduction

The goal of this project is to build an autonomous, offline data scientist agent that can execute end-to-end machine learning workflows without human intervention or reliance on Large Language Models. The agent profiles datasets, plans preprocessing and modelling strategies, trains and evaluates candidate models, reflects on results, and reuses prior experience through rule-based logic and heuristics.

Data science workflows are inherently conditional. A dataset with severe class imbalance needs different treatment than a balanced one; highly correlated features call for feature selection; a small dataset risks overfitting with complex models. A static pipeline cannot adapt to those cases, so the system is designed to read the data profile and adjust its plan accordingly.

The design philosophy throughout the project was to keep every decision traceable. Rather than applying all preprocessing steps by default, each step is gated behind a profile signal, so the generated plan doubles as an explanation of why specific adaptations were made. For example, when `drop_leaky_features` appears in the plan, it is because the profiler measured a normalised mutual information score ≥ 0.9 between that feature and the target.

Robustness was a secondary design goal. The agent includes a training retry loop (up to 3 attempts), memory-guided model prioritisation, and failed-target tracking so that one poor run does not waste time or contaminate later runs.

This report summarises the main design decisions, extensions beyond the skeleton, and the limitations identified through testing.

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
- **Memory** (`agents/memory.py`): Persists outcomes per dataset fingerprint. Provides lookup by fingerprint, filename, or target+shape, reuses only reliable runs as priors, and tracks failed targets to avoid repeating them on subsequent runs.

---

## 3. Dataset Understanding — Signals and Their Meaning

The profiler extracts a rich set of signals from every dataset. The table below maps the most important signals to the decisions they drive.

| Signal | How detected | Downstream effect |
|--------|-------------|-------------------|
| `shape.rows` | Row count | Size bucket: small (<1000), medium (1000–9999), large (≥10000) |
| `imbalance_ratio` | majority/minority class count | ≥3.0 → class-weighted models; ≥5.0 → oversampling step |
| `missing_pct` | % NaN per column | >20% → `handle_severe_missing_data`; columns above threshold dropped |
| `outlier_cols` | IQR-based outlier count >5% | `handle_outliers` → RobustScaler |
| `scale_mismatch` | max_range / median_range ≥ 50 | `apply_robust_scaling` → RobustScaler instead of StandardScaler |
| `hard_leakage_cols` | Hard leakage evidence from mutual information analysis | `drop_leaky_features` → column excluded from all transformers |
| `soft_leakage_cols` | High-MI proxy suspicion | surfaced to verdict / human review rather than auto-drop |
| `high_corr_pairs` | Pairwise Pearson abs_corr ≥ 0.95 | `drop_correlated_features` → weaker of each pair dropped |
| `near_constant_cols` | One value ≥ 95% dominant | `drop_near_constant_features` → column excluded |
| `n_unique_by_col` | Unique value count per column | >50 unique + <10% of rows → `apply_target_encoding` |
| `sensitive_cols` | Column-name keyword match | `drop_sensitive_features` → excluded before training |
| `is_classification` | Target cardinality and dtype | Routes to classification or regression models and metrics |

A key implementation decision was the **leakage detection fix**. The original skeleton passed the regression target through `pd.factorize()` before computing `mutual_info_regression`, converting continuous values into integer rank labels. This broke the MI normalisation — with 30,000 near-unique values, the target entropy denominator became enormous, collapsing all normalised MI scores below the 0.9 threshold. After the fix, regression leakage is detected correctly by using raw target values and normalising by the maximum MI score instead of target entropy. This allowed `final_price_usd` (normalised MI = 1.0) to be correctly flagged as leaky in the Sales dataset, where `revenue_usd = final_price_usd × units_sold` exactly.

---

## 4. Planning Logic — Conditional Decisions

The planner generates an ordered plan by applying a series of conditional checks on the profile. The core pipeline steps are fixed; additional steps are inserted or appended based on the signals detected.

**Size bucket logic** drives model selection and regularization:
- Rows < 1000 (small): `apply_regularization` (LogisticRegression C=0.1) and `use_simple_models_only` (Dummy + LR + RF only, no GradientBoosting or SVC)
- Rows 1000–9999 (medium): full model set including GradientBoosting
- Rows ≥ 10,000 (large): `use_ensemble_models` (tree ensembles favoured unless the dataset is too wide)

This bucket logic is consistent across the planner, modelling, and memory components — all three use the same thresholds of 1,000 and 10,000.

Two additional planner extensions improve realism beyond a simple size bucket. First, **high-dimensional routing** treats wide datasets (`feature_cols >= 100` or `feature_cols / rows >= 0.25`) as overfitting-prone even when they are not row-small, so the planner biases toward regularisation and simpler candidate sets rather than blindly favouring ensembles. Second, **cost-aware planning** estimates workload as `rows * cols`; large workloads trigger `reduce_tuning_budget` so hyperparameter search uses fewer iterations and optional row sub-sampling, while extreme workloads can skip cross-validation entirely.

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
| `simple_models_only` (small / wide dataset) | DummyClassifier, LogisticRegression, RandomForest |
| Medium / Large (classification) | + GradientBoosting, HistGradientBoosting |
| `prefer_ensemble` (large) | tree ensembles explicitly favoured |
| Regression | DummyRegressor, LinearRegression, Ridge, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor |

A dummy baseline is always included. This is critical for the reflector — all performance assessments are relative to baseline improvement.

SVC was restricted to datasets with fewer than 1,000 rows and 50 columns, and `probability=True` was removed since Platt scaling adds significant training cost and probabilities are never used by the evaluation layer.

### Metrics

Classification uses balanced accuracy and macro-averaged F1 as primary metrics (to handle class imbalance). Regression uses R², MAE, and RMSE. The best model is selected by balanced accuracy + F1 for classification, R² for regression.

---

## 6. Reflection and Re-planning

The reflector analyses evaluation results against dataset characteristics and generates a structured output. Every reflection contains:
- `status`: `"ok"` or `"needs_attention"`
- `issues`: list of identified problems
- `suggestions`: list of improvement recommendations, ordered by priority (critical signals first)
- `replan_recommended`: boolean
- `training_warnings`: sklearn runtime warnings captured during training
- `significance_test`: paired t-test result comparing the top two models (when CV is enabled)

### Classification Checks

| Condition | Issue | Replan? |
|-----------|-------|---------|
| Best model < 0.05 above dummy balanced accuracy | Weak signal or pipeline issues | Yes (if f1 < threshold) |
| bal_acc > 0.90 and f1 < 0.70 | Overfitting suspected | Yes (if f1 < threshold) |
| f1_macro < adaptive threshold | Modest F1 score | Yes |
| ≥2 non-dummy models with bal_acc ≥ 0.99 and f1 ≥ 0.99 | Suspicious near-perfect performance | No |
| All models F1 within 0.05 | Low model diversity | Yes (if f1 < threshold) |

### Adaptive F1 Threshold

Rather than applying a fixed F1 threshold of 0.60 to all datasets, the reflector adjusts the threshold based on class imbalance. When the minority class is genuinely hard to learn, a flat threshold penalises the agent unfairly:

| Imbalance ratio | F1 threshold |
|----------------|-------------|
| < 3.0 (balanced) | 0.60 |
| ≥ 3.0 (moderate imbalance) | 0.50 |
| ≥ 5.0 (severe imbalance) | 0.45 |

This ensures that a model achieving F1 = 0.52 on a severely imbalanced dataset is not flagged as failing when that score may already represent meaningful learning on the minority class.

### Per-Class F1 Analysis

Beyond macro-averaged F1, the reflector inspects per-class F1 scores obtained from `classification_report(output_dict=True)`. If any class has an F1 score below 0.40, it is flagged as an issue with the specific class name and score. This is more informative than macro F1 alone — a high macro score can mask a single class the model consistently fails on. A per-class F1 bar chart is saved to `per_class_f1.png` in the output directory, with bars below 0.60 highlighted in red.

### Regression Checks

| Condition | Issue | Replan? |
|-----------|-------|---------|
| R² improvement < 0.05 over dummy | Weak regression signal | Yes |
| R² < 0.10 | Very low explanatory power | Yes |
| ≥2 non-dummy models R² ≥ 0.99 | Suspicious near-perfect R² | Yes |

### Statistical Significance Testing

When cross-validation is enabled, the reflector runs a paired t-test (scipy `ttest_rel`) between the per-fold primary metric scores of the best and runner-up models. This tests whether the performance gap between the two models is statistically reliable or could be explained by fold-to-fold variance.

The result is stored in `reflection.json` under `significance_test` and surfaces in the report:

```json
{
  "test": "paired t-test",
  "model_a": "LogisticRegression",
  "model_b": "RandomForest",
  "p_value": 0.6315,
  "significant": false,
  "alpha": 0.05,
  "note": "No significant difference — the simpler model may be equally reliable."
}
```

When the gap is not significant (p > 0.05), the reflector adds a suggestion to prefer the simpler model. This prevents the agent from recommending a more complex model when there is no statistical evidence it is actually better. On the Titanic dataset, LogisticRegression and RandomForest produced indistinguishable CV performance (p = 0.63), and the agent correctly flagged this.

### Leakage-Aware Verdict

An earlier version of the verdict logic marked any run with detected leaky columns as "Invalid due to leakage risk", even when those columns had been dropped before training. This produced misleading verdicts — a run that correctly detected and removed a leaky feature was penalised the same as one that trained on it. The fix distinguishes between two states: leaky columns detected but not dropped (genuine concern, "Invalid") versus leaky columns detected and dropped (handled, "Use with caution" if other signals warrant it, otherwise "Reliable result"). This change corrects verdicts for datasets like Titanic, where `alive` is a leaky proxy of the target `survived` and is correctly removed before any model sees the data.

### Training Warning Detection

sklearn runtime warnings (overflow, divide-by-zero, invalid value in matrix operations) are captured during training using `warnings.catch_warnings(record=True)` and stored in `reflection.json`. If numerical instability warnings are present, the reflector always adds a suggestion to check feature scales. If performance is also weak (f1 < threshold or R² < 0.1), the warning is escalated to an issue, since instability may be degrading results.

### Prioritised Suggestions

Suggestions are ordered by estimated impact before being returned. High-priority keywords (instability, leakage, scaling, overflow) surface first; medium-priority keywords (regularization, ensemble, feature engineering) come next; general advice appears last. This ensures the most actionable recommendations are visible at the top of the report without requiring the user to scan the full list.

### Replanning

`should_replan()` defers entirely to `replan_recommended`. An earlier implementation also triggered replanning when multiple issues were present or status was `needs_attention` — this caused spurious replans on clean, well-separated datasets where near-perfect performance is legitimate, not a problem. The fix was to make `replan_recommended` the single gate, set only when performance is genuinely poor.

`apply_replan_strategy()` modifies the plan based on issue keywords. If overfitting is detected, `apply_regularization` is added. If F1 is weak or the best model only marginally beats the baseline, `use_ensemble_models` is added — this step is handled by the executor, which sets `prefer_ensemble=True` in the profile, causing `select_models()` to include GradientBoosting-style ensembles. If training instability was flagged, `apply_robust_scaling` is added to address potential feature scale issues in the next pass. When the held-out score diverges materially from the cross-validation mean, the replanner also sets `increase_test_size=True`, widening the test split by 0.10 (capped at 0.35) for the next pass so the held-out estimate is more representative.

---

## 7. Memory and Learning

The memory system (`agents/memory.py`) persists outcomes in a JSON file (`agent_memory.json`). Each record is keyed by a dataset fingerprint (hash of filename + target + column names) and stores:

- `dataset`: filename
- `target`: column name
- `target_source` / `target_origin`: whether it was provided manually, inferred, or read from memory
- `shape`: rows and columns
- `is_classification`: task type
- `verdict_label` and `verdict_detail`: run-level trust outcome
- `reflection_status`: `"ok"` or `"needs_attention"`
- `review_required`: whether human review is explicitly recommended
- `best_model` + `best_metrics` for reliable runs, or `diagnostic_model` + `diagnostic_metrics` otherwise
- `cross_validation`: stored when CV was executed
- `last_seen`: UTC timestamp

**Lookup** operates in three steps: exact fingerprint, then filename, then target+shape. This means a renamed copy of a dataset still gets a memory hit, enabling continuity across experiments.

**Model prioritisation**: When memory returns a best model, the planner adds `prioritize_model:<name>` to the plan. This was observed to improve second-run efficiency on the Titanic dataset, where memory correctly identified `LogisticRegression` as the best-performing model.

**Failed target tracking**: A dedicated `failed_targets` dict records targets that produced no useful results (best model is Dummy + status is `needs_attention`) or were marked invalid because of leakage risk. On subsequent auto-detection runs, those targets are skipped. This was tested with `Sales.csv` where `category` as a target produced random-level predictions — the agent stored it as failed and skipped to the next candidate on the following run.

**Similarity matching** (`get_similar_record`) finds records from structurally similar datasets using three heuristics: same size bucket, similar imbalance level, similar missingness. This enables cross-dataset transfer of strategies when no exact fingerprint match exists.

---

## 8. Ethics and Limitations

### Data Leakage
The agent detects leakage through mutual information analysis but only at the individual feature level. Multiplicative relationships (e.g., `revenue_usd = final_price_usd × units_sold`) are not detected because neither component alone reaches the MI threshold. The reflector's near-perfect performance check acts as a secondary signal, correctly flagging this case as suspicious even when the feature-level detector misses it.

### Proxy Features and Fairness
The leakage detector flags features that are near-perfect proxies of the target. Additionally, the profiler actively scans for sensitive attributes (e.g., gender, race, age) and automatically drops them to prevent direct algorithmic bias. During the evaluation phase, the agent computes a Demographic Parity proxy to audit the model for disparate impacts across protected groups, ensuring fairness issues are surfaced to human reviewers.

### Imbalance Handling
Class weights are applied automatically when imbalance ratio ≥ 3.0. For severe imbalance (ratio ≥ 5.0), the planner dynamically injects an oversampling step that utilises SMOTE (Synthetic Minority Over-sampling Technique) to artificially balance the training data, provided the `imbalanced-learn` library is available.

### Feature Engineering
When the planner emits `apply_feature_engineering`, the executor dynamically injects a `PolynomialFeatures(degree=2)` step into the continuous data preprocessing pipeline. This automatically generates interaction terms and squared features, allowing simpler linear models to capture non-linear relationships without manual domain knowledge.

### Resolved Limitations
All previously dead flags from the original skeleton have been resolved: `robust_imputation` now correctly switches the imputer to median imputation with missing-value indicators, `use_ensemble_models` (the replacement for the earlier dead `try_ensemble_methods`) is handled by the executor and sets `prefer_ensemble=True` in the profile, `drop_sensitive_features` successfully drops sensitive attributes before training to prevent direct algorithmic bias, and `apply_oversampling` natively integrates SMOTE to handle severe imbalance.

### Replan Effectiveness
The replan mechanism works correctly for performance-based issues (weak F1, poor R²). For structural issues like multiplicative leakage, the replanner currently has no handler — it adds `replan_attempt` to the plan but changes nothing in the second run. The correct fix would be to progressively lower the MI leakage threshold during replanning, catching features that individually fall below 0.9 but together construct the target.

### Scope
The agent is designed for tabular, offline batch processing. It has no support for streaming data, time series, image or text-heavy datasets, or deployment.

---

## 9. Experimental Results

Four datasets were used to validate the agent across different scenarios:

| Dataset | Rows | Task | Best Model | Key Metric | Plan Adaptations |
|---------|------|------|-----------|-----------|-----------------|
| `titanic.csv` | 784 | Classification | LogisticRegression | bal_acc=0.825, f1=0.828 | outliers, missing_data, robust_scaling, leaky `alive` dropped, CV-gap replan |
| `digits.csv` | 1797 | Classification | RandomForest | bal_acc=0.972, f1=0.972 | outliers, 14 near-constant pixel cols dropped |
| `Sales.csv` | 30000 | Regression | RandomForestRegressor | R²=1.000 (suspicious) | target_encoding, sensitive feature dropped, reduced tuning budget, soft leakage review |
| `telco_churn.csv` | 7043 | Classification | LogisticRegression | bal_acc=0.723, f1=0.733 | robust_scaling, sensitive feature dropped, hyperparameter tuning |

**Titanic (survival classification):** The Titanic run demonstrates several adaptive behaviours working together. The `alive` column — a string encoding of whether the passenger survived — was correctly flagged as leaky via mutual information (normalised MI = 1.0) and dropped before training. The `deck` column (77% missing values) triggered `handle_severe_missing_data`. Scale mismatch between `fare` (0–512) and ordinal features like `pclass` (1–3) triggered RobustScaler. The held-out score (balanced accuracy 0.853) differed noticeably from the 5-fold cross-validation mean (0.770), triggering the CV-gap replan: the test split was widened from 20% to 30% to obtain a more representative held-out estimate. After the replan, LogisticRegression achieved balanced accuracy 0.825 and macro F1 0.828 — a strong outcome for this well-studied dataset, achieved through fully automated adaptive preprocessing. Memory was correctly updated and the second run prioritised LogisticRegression, improving run efficiency.

**Digits (handwritten digit classification):** The digits dataset contains 64 pixel features derived from 8×8 images. Border pixels are always or nearly always zero — 14 columns were flagged as near-constant (dominant value ≥ 95%) and dropped. RandomForest achieved balanced accuracy 0.972 and macro F1 0.972 on 10-class digit recognition, a strong result. The plan included `handle_outliers` and `drop_near_constant_features`, demonstrating that the profile correctly identified the dataset's structural characteristics without any human guidance.

**Sales (revenue regression):** The Sales dataset provided the most instructive run. `final_price_usd` was correctly flagged as a soft leakage signal (normalised MI = 1.0 with `revenue_usd`) and `model_name` (899 unique product names) was target-encoded rather than dropped. The planner also dropped the sensitive `gender` column and reduced the tuning budget because the workload was large (`30,000 × 18`). Soft leakage signals are not automatically removed — they surface as `Use with caution` verdicts requiring human review, because high MI alone is not proof that a feature is unavailable at prediction time. With `final_price_usd` still in the feature set (along with `units_sold`), RandomForestRegressor achieved R² = 1.000 trivially, since `revenue_usd = final_price_usd × units_sold` exactly. The reflector correctly raised a `needs_attention` warning with a suspicious near-perfect R² issue. This demonstrates both the strength of the leakage detector (caught the high-MI proxy and surfaced it) and the intentional design choice to prefer human review over silent feature removal for soft signals.

**Telco Churn (customer churn classification):** This dataset perfectly demonstrated the agent's ability to handle dirty data and class imbalance. The `TotalCharges` column contained blank spaces, which initially caused pandas to load it as a string. The agent's schema profiler successfully coerced the string column to numeric, recovering the data and triggering `RobustScaler` to handle its massive scale range. Because the dataset has a class imbalance, the agent automatically applied class weights. It safely bypassed benign math warnings during training to successfully tune the `LogisticRegression` model, achieving a reliable balanced accuracy of 0.723.

---

## 10. Conclusion and Future Work

This project implements a functioning offline agentic data scientist that adapts its behaviour to dataset characteristics through a profile–plan–execute–reflect–remember loop. The key contributions beyond the skeleton are:

1. **Leakage detection via mutual information** with a regression-specific fix (raw target values rather than factorized ranks)
2. **Leakage-aware verdict logic** distinguishing between leakage detected-and-dropped versus detected-and-untreated
3. **Scale mismatch detection** driving RobustScaler selection
4. **Size-bucket model selection** consistent across planner, modelling, and memory
5. **Target encoding** for high-cardinality categoricals via sklearn's `TargetEncoder`
6. **Near-constant feature detection** surfacing in the plan for transparency
7. **Training warning capture** with reflector integration
8. **Failed target memory** preventing the agent from repeating unsuccessful targets
9. **Spurious replan fix** ensuring replanning only triggers when performance is genuinely poor
10. **Statistical significance testing** via paired t-test on CV fold scores, comparing the best and runner-up model
11. **Adaptive F1 threshold** adjusted by class imbalance ratio so minority-class difficulty is not penalised unfairly
12. **Per-class F1 analysis** flagging specific underperforming classes beyond macro-averaged metrics
13. **Feature importance and per-class F1 visualisations** saved automatically for tree-based models and classification runs
14. **Centralised configuration** (`config.py`) making all thresholds and hyperparameters visible and adjustable in one place
15. **Algorithmic fairness audits** including sensitive attribute detection, exclusion, and demographic parity calculations
16. **SMOTE integration** for dynamically handling severely imbalanced datasets
17. **Low-dimensional polynomial feature engineering** for simple non-linear signal capture
18. **Cost-aware compute estimation** to scale down tuning budgets for massive datasets
19. **High-dimensional planning rules** that bias wide datasets toward simpler, more regularised search

Future work would address the identified limitations, such as implementing progressive leakage threshold reduction during replanning and broadening feature engineering beyond the current low-dimensional quadratic expansion.

The test suite exceeds the 60% coverage requirement and includes integration-level smoke tests that exercise the full pipeline end-to-end using real datasets.

The most important lesson from this project is that the value of an agentic system lies not in any single technique but in the connections between components. Detecting scale mismatch is only useful if the planner acts on it; leakage detection is only useful if the preprocessor respects it; training warnings are only useful if the reflector surfaces them. Each signal–plan–action–reflection chain required careful wiring across multiple files, and maintaining a shared signal map (`AGENT_SIGNAL_MAP.md`) throughout development proved essential for tracking which signals were implemented end-to-end versus only partially wired.

---

*AI assistance tools were used during the development and documentation of this project, in accordance with module guidelines.*
