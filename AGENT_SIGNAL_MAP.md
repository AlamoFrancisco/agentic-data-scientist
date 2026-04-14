# Agent Signal Map

Maps dataset profile signals → Planner decisions → Executor actions → Reflector checks.
Use this as a reference when extending `agents/planner.py` and `agents/reflector.py`.

---

## 1. Profile Fields Available

All fields that `profile_dataset()` currently returns.

| Field | Type | What it tells you |
|-------|------|-------------------|
| `shape.rows` | int | Dataset size |
| `shape.cols` | int | Dimensionality |
| `imbalance_ratio` | float | Majority/minority class ratio |
| `class_counts` | dict | Count per class |
| `missing_pct` | dict | % missing per column |
| `outlier_cols` | list | Columns with >5% IQR outliers |
| `near_constant_cols` | list | Columns where one value dominates (≥95%) |
| `high_corr_pairs` | list | Strongly correlated feature pairs (abs_corr ≥ 0.6) |
| `high_corr_present` | bool | Any pair with abs_corr ≥ 0.6 |
| `max_abs_corr` | float | Highest pairwise correlation |
| `corr_cols_to_drop` | list | Columns to drop (weaker of each highly correlated pair, abs_corr ≥ 0.95) |
| `scale_mismatch` | bool | True when max_range / median_range ≥ 50 across numeric cols |
| `scale_range_ratio` | float | max_range / median_range across numeric cols |
| `leaky_cols` | list | Column indices with normalised MI ≥ 0.9 with target |
| `leaky_col_names` | list | Column names with normalised MI ≥ 0.9 with target |
| `feature_types.text` | list | Text columns (not encoded) |
| `feature_types.categorical.binary` | list | Binary categorical columns |
| `feature_types.categorical.multiclass` | list | Multi-value categorical columns |
| `feature_types.numeric.continuous` | list | Continuous numeric columns |
| `feature_types.numeric.ordinal` | list | Integer-like low-cardinality columns |
| `n_unique_by_col` | dict | Unique value count per column |
| `duplicate_count` | int | Duplicate rows (already dropped before profiling) |
| `is_classification` | bool | Classification (True) vs regression (False) |
| `notes` | list | Human-readable warnings generated during profiling |

> **Not yet in profiler (from eda.ipynb):**
> - Per-column `skewness` — could trigger `apply_log_transform` for skew > 1.5

---

## 2. Planner Signal → Plan Step Mapping

### Implemented
| Profile signal | Condition | Plan step added |
|----------------|-----------|-----------------|
| `imbalance_ratio` | ≥ 3.0 | `consider_imbalance_strategy` |
| `shape.rows` | < 1000 | `apply_regularization` + `use_simple_models_only` |
| `shape.rows` | ≥ 10,000 | `use_ensemble_models` |
| `missing_pct` max | > 20% | `handle_severe_missing_data` |
| `outlier_cols` | not empty | `handle_outliers` |
| `scale_mismatch` | True | `apply_robust_scaling` |
| `high_corr_pairs` | any pair abs_corr ≥ 0.95 | `drop_correlated_features` |
| `leaky_cols` | not empty | `drop_leaky_features` |
| `feature_types.categorical.multiclass` | n_unique > 50 | `apply_target_encoding` |
| `memory_hint.best_model` | present | `prioritize_model:<name>` |

| `near_constant_cols` | not empty | `drop_near_constant_features` | Preprocessor already drops them unconditionally; plan step adds visibility in log/report |

### Not yet implemented
| Profile signal | Condition | Plan step to add | Why |
| `feature_types.text` | not empty | `encode_text_features` | Text silently ignored right now |
| `shape.cols` | > 100 | `apply_dimensionality_reduction` | High-dim → overfitting risk |
| `skewness` (not in profiler) | > 1.5 on any continuous col | `apply_log_transform` | Skew hurts linear/distance models |
| `imbalance_ratio` | ≥ 10.0 | `apply_oversampling` | Severe imbalance → class weights alone insufficient |

---

## 3. Executor — Plan Step → Profile Flag

The executor (`agentic_data_scientist.py` main loop) reads plan steps and sets flags on the profile dict. Those flags are then consumed by `build_preprocessor()` and `select_models()`.

| Plan step | Profile flag set | Downstream effect | Status |
|-----------|-----------------|-------------------|--------|
| `consider_imbalance_strategy` | `use_class_weights = True` | Redundant — `select_models` reads `imbalance_ratio` directly | Works (flag unused) |
| `apply_regularization` | `use_regularization = True` | `LogisticRegression(C=0.1, solver=saga)` | Yes |
| `handle_severe_missing_data` | `robust_imputation = True` | Dead flag — imputer always uses `median` | Dead flag |
| `apply_target_encoding` | `use_target_encoding = True` | `TargetEncoder` branch added to `ColumnTransformer` for high-cardinality cols (n_unique ≥ 50) | Yes |
| `apply_feature_engineering` | `use_feature_engineering = True` | Dead flag — feature engineering not implemented | Dead flag |
| `handle_outliers` | `handle_outliers = True` | `RobustScaler` (small) or `RobustScaler(unit_variance=True)` (≥1000 rows) | Yes |
| `apply_robust_scaling` | `use_robust_scaling = True` | `RobustScaler` instead of `StandardScaler` | Yes |
| `drop_correlated_features` | `drop_high_corr = True` | Drops `corr_cols_to_drop` in `build_preprocessor()` | Yes |
| `drop_leaky_features` | `drop_leaky = True` | Drops `leaky_col_names` in `build_preprocessor()` | Yes |
| `use_simple_models_only` | `simple_models_only = True` | Only `Dummy + LR + RF` — no GradientBoosting or SVC | Yes |
| `use_ensemble_models` | `prefer_ensemble = True` | Always includes `GradientBoosting` | Yes |
| `drop_near_constant_features` | logs columns | Preprocessor already drops `near_constant_cols` unconditionally — step adds plan visibility | Yes |
| `apply_log_transform` | `use_log_transform = True` | Not wired up yet | Not implemented |

---

## 4. Reflector Signal → Issue/Suggestion Mapping

### Implemented
| Signal | Issue raised | Suggestion | Triggers replan? |
|--------|-------------|------------|-----------------|
| Best model barely beats dummy (< 0.05 improvement) | "Weak signal or pipeline issues" | Check leakage / verify target quality | Yes (if f1 < 0.60) |
| `bal_acc > 0.90` and `f1_macro < 0.70` | "Overfitting suspected" | Regularization / reduce complexity | Yes (if f1 < 0.60) |
| `f1_macro < 0.60` | "Modest F1 score" | Try different models / tune hyperparameters | Yes |
| ≥2 non-dummy models near-perfect (bal_acc ≥ 0.99, f1 ≥ 0.99) | "Suspicious near-perfect performance" | Check for target proxies / leakage | No (f1 too high) |
| All models F1 within 0.05 | "Low model diversity" | Try more diverse models | Yes (if f1 < 0.60) |
| `imbalance_ratio ≥ 3.0` | — | Consider class_weight / threshold tuning / SMOTE | No (suggestion only) |
| Regression: R² barely beats dummy (< 0.05 improvement) | "Weak regression signal" | Check feature engineering / target quality | Yes |
| Regression: `r2 < 0.1` | "R² very low" | Try feature engineering / verify target is predictable | Yes |
| Regression: ≥2 non-dummy models with R² ≥ 0.99 | "Suspicious near-perfect R²" | Inspect features for target proxies or multiplicative combinations | Yes |

| `training_warnings` contains overflow/divide-by-zero | Suggestion always; escalates to issue if f1 < 0.60 (classification) or r2 < 0.1 (regression) | Apply robust scaling / check extreme feature values | Only if weak performance too |

### Not yet implemented
| Signal | Issue to raise | Suggestion |
|--------|---------------|------------|
| Per-class F1 breakdown | "Class X has low recall" | Investigate minority class / threshold tuning |
| `imbalance_ratio ≥ 10` | "Severe imbalance" | Try SMOTE or oversampling, not just class weights |
| `outlier_cols` present + bad R² | "Outliers may be distorting regression" | Use robust scaling |
| Replan count > 1, no improvement | "Replanning not helping" | Stop early |

---

## 5. Replan Logic

`should_replan()` defers entirely to `replan_recommended` from `reflect()`.

`replan_recommended` is `True` only when **both** of these hold:
- `issues` list is non-empty
- `f1_macro < 0.60` (classification) or `r2 improvement < 0.05` (regression)

This prevents spurious replans when near-perfect performance raises a leakage warning but f1 is already high (e.g. penguins/species).

`apply_replan_strategy()` modifies the plan based on issue text:
| Issue keyword | Plan step added |
|---------------|----------------|
| `"overfitting"` | `apply_regularization` |
| `"imbalance"` | `consider_imbalance_strategy` |
| `"f1"` (case-insensitive) | `try_ensemble_methods` (⚠️ ignored by executor) |
| `"baseline"` | `apply_feature_engineering` |

---

## 6. Training Warnings

Runtime warnings from sklearn are captured during `train_models()` and stored in `reflection.json`.

| Where captured | `tools/modelling.py` — `warnings.catch_warnings(record=True)` |
|---|---|
| Where stored | `reflection.json` → `training_warnings: [...]` |
| Scope | Unique warning strings across all trained models per run |
| Console | Suppressed — won't clutter output |

> **Future:** Reflector could inspect `training_warnings` and raise an issue if overflow/divide-by-zero warnings appear (suggests feature scale problems).

---

## 7. Memory

### What gets stored after each run (`agent_memory.json`)
| Field | What it is |
|-------|-----------|
| `dataset` | Filename of the dataset |
| `target` | Target column name |
| `target_source` | How target was found: `"manual"`, `"inferred"`, or `"memory"` |
| `shape` | `{rows, cols}` at time of run |
| `is_classification` | Whether this was a classification or regression run |
| `best_model` | Name of the best performing model |
| `best_metrics` | Full metrics dict for the best model |
| `reflection_status` | `"ok"` or `"needs_attention"` |
| `last_seen` | ISO timestamp of the last run |

### Failed targets (`failed_targets` key in `agent_memory.json`)
When a run produces no useful result (best model is Dummy + status = `needs_attention`), the target is saved as failed for that dataset filename. On the next auto-detection run, those targets are skipped.

| Method | What it does |
|--------|-------------|
| `add_failed_target(dataset_name, target)` | Appends target to failed list for that dataset |
| `get_failed_targets(dataset_name)` | Returns list of failed targets for that dataset |

### How memory is looked up (`get_dataset_record`)
1. Exact fingerprint match (hash of file + target + column names)
2. Same dataset filename
3. Same target + shape (handles renamed files)

### Similarity matching (`get_similar_record`)
Finds records from *different* datasets that are structurally similar.

| Check | Match condition | Score |
|-------|----------------|-------|
| Size bucket | Both in same bucket (small/medium/large) | +1 |
| Imbalance | Both imbalanced (≥3.0) or both not | +1 |
| Missing data | Both high (>20%) or both low | +1 |

Threshold: ≥ 0.5 (≥ 2 of 3 checks). Returns best match above threshold.

### What the Planner reads from `memory_hint`
| Field used | Plan step triggered |
|-----------|---------------------|
| `best_model` | `prioritize_model:<name>` — put that model first in candidate list |

### Could be extended
| What to store | Benefit |
|---------------|---------|
| `successful_plan_steps` | Planner re-uses steps that worked on similar datasets |
| `failed_strategies` | Planner avoids repeating strategies that didn't help |
| `issues_found` | Reflector can compare current issues against past trend |

---

## 8. Model Selection — Size Bucket Rules

| Bucket | Rows | Models included |
|--------|------|----------------|
| Small | < 1000 | Dummy + LR + RF + GradientBoosting + SVC (if cols ≤ 50) — overridden to Dummy + LR + RF when `simple_models_only=True` |
| Medium | 1000–9999 | Dummy + LR + RF + GradientBoosting |
| Large | ≥ 10000 | Dummy + LR + RF + GradientBoosting (via `prefer_ensemble`) |

**SVC:** `probability=False` (Platt scaling removed — unused in pipeline). Only added for small datasets with ≤ 50 raw columns. Currently unreachable because planner always sets `simple_models_only` for rows < 1000.

---

## 9. Known Issues / Technical Debt

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| `robust_imputation` flag set but unused | executor | Medium | `build_preprocessor` always uses `SimpleImputer(median)` regardless |
| ~~`use_target_encoding` flag set but unused~~ | executor | Fixed | `TargetEncoder` branch wired into `build_preprocessor()` for high-cardinality cols |
| `use_feature_engineering` flag set but unused | executor | Medium | Feature engineering not implemented |
| `try_ensemble_methods` added by replan but ignored | reflector/executor | Medium | No executor handler — silently skipped |
| `replan_attempt` marker added by replan | reflector | Low | Harmless but clutters plan output |
| `use_class_weights` flag redundant | executor | Low | `select_models` reads `imbalance_ratio` directly — flag not needed |
| ~~Spurious replan on near-perfect datasets~~ | reflector | Fixed | `should_replan()` now only returns `True` when `replan_recommended=True` |
| ~~Leakage detection broken for regression~~ | profiler | Fixed | Regression target was factorized before MI — now uses raw values, normalised by max MI |
| ~~Training warnings not acted on~~ | reflector | Fixed | Overflow/divide warnings now raise suggestion (and issue if performance is also weak) |
| SVC unreachable | modelling | Low | Planner always sets `simple_models_only` for rows < 1000, which returns before SVC is added |

---

## 10. Implementation Priority

| Priority | What | Why |
|----------|------|-----|
| 1 | Tests (>60% coverage required) | Assignment hard requirement — nothing else ships without this |
| 2 | Fix dead flag: `robust_imputation` | Advertised in plan but `build_preprocessor` ignores it |
| 3 | Fix dead flag: `use_feature_engineering` | Advertised in plan but not implemented |
| 4 | Report (3000–4000 words) | Assignment requirement — do last |
