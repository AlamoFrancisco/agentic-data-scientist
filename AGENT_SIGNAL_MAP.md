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
| `high_corr_pairs` | list | Strongly correlated feature pairs (abs_corr ≥ 0.8) |
| `high_corr_present` | bool | Any pair with abs_corr ≥ 0.8 |
| `max_abs_corr` | float | Highest pairwise correlation |
| `scale_mismatch` | bool | True when max_range / median_range ≥ 50 across numeric cols |
| `scale_range_ratio` | float | max_range / median_range across numeric cols |
| `leaky_cols` | list | Leakage evidence dicts returned by `leakage_report()` |
| `hard_leakage_cols` | list | Leakage signals treated as hard evidence |
| `soft_leakage_cols` | list | Leakage signals treated as proxy suspicion |
| `feature_types.text` | list | Text columns (not encoded) |
| `feature_types.categorical.binary` | list | Binary categorical columns |
| `feature_types.categorical.multiclass` | list | Multi-value categorical columns |
| `feature_types.numeric.continuous` | list | Continuous numeric columns |
| `feature_types.numeric.ordinal` | list | Integer-like low-cardinality columns |
| `n_unique_by_col` | dict | Unique value count per column |
| `sensitive_cols` | list | Columns whose names suggest protected / sensitive attributes |
| `duplicate_count` | int | Duplicate rows (already dropped before profiling) |
| `is_classification` | bool | Classification (True) vs regression (False) |
| `notes` | list | Human-readable warnings generated during profiling |

> **Not yet in profiler (from eda.ipynb):**
> - Per-column `skewness` — could trigger `apply_log_transform` for skew > 1.5

---

## 2. Planner Signal → Plan Step Mapping

### Implemented
| Profile signal | Condition | Plan step added / changed |
|----------------|-----------|---------------------------|
| `imbalance_ratio` | `>= 5.0` | `apply_oversampling` |
| `imbalance_ratio` | `>= 3.0` and `< 5.0` | `consider_imbalance_strategy` |
| `shape.rows` | `< 1000` | `apply_regularization` + `use_simple_models_only` |
| `shape.rows` + `shape.cols` | high-dimensional (`feature_cols >= 100` or `feature_cols / rows >= 0.25`) | `apply_regularization` + `use_simple_models_only` |
| `shape.rows` | `>= 10,000` and not high-dimensional | `use_ensemble_models` |
| `missing_pct` max | `> 20%` | `handle_severe_missing_data` |
| `outlier_cols` | not empty | `handle_outliers` |
| `shape.cols` | `feature_cols <= 15` | `apply_feature_engineering` |
| `scale_mismatch` | `True` | `apply_robust_scaling` |
| `high_corr_pairs` | any pair `abs_corr >= 0.95` | `drop_correlated_features` |
| `hard_leakage_cols` | not empty | `drop_leaky_features` |
| `near_constant_cols` | not empty | `drop_near_constant_features` |
| `sensitive_cols` | not empty | `drop_sensitive_features` |
| `feature_types.categorical.multiclass` / `feature_types.text` | high-cardinality values detected | `apply_target_encoding` |
| `memory_hint.best_model` | present | `prioritize_model:<name>` |
| `shape.rows` + workload | medium/large and not high-dimensional | `tune_hyperparameters` |
| workload (`rows * cols`) | `> COMPUTE_COST_THRESHOLD` | `reduce_tuning_budget` inserted before tuning |
| extreme workload | `rows * cols > PLANNER_CV_MAX_WORKLOAD` or `feature_cols >= PLANNER_CV_MAX_COLS` | `validate_with_cross_validation` removed |

### Still not implemented
| Profile signal | Condition | Plan step to add | Why |
|----------------|-----------|------------------|-----|
| `feature_types.text` | free-form text genuinely present | `encode_text_features` | Only categorical-like text is currently handled |
| `feature_types.datetime` | not empty | `time_aware_validation` / `temporal_features` | No time-series-specific planning yet |
| `skewness` (not in profiler) | `> 1.5` on any continuous column | `apply_log_transform` | Skew-aware preprocessing is still absent |

---

## 3. Executor — Plan Step → Profile Flag

The executor (`agentic_data_scientist.py` main loop) reads plan steps and sets flags on the profile dict. Those flags are then consumed by `build_preprocessor()` and `select_models()`.

| Plan step | Profile flag set | Downstream effect | Status |
|-----------|------------------|-------------------|--------|
| `consider_imbalance_strategy` | `use_class_weights = True` | `select_models()` prefers explicit class weights, with `imbalance_ratio` as fallback | Yes |
| `apply_oversampling` | `apply_oversampling = True` | `train_models()` enables SMOTE when `imbalanced-learn` is installed | Yes |
| `apply_regularization` | `use_regularization = True` | `LogisticRegression(C=0.1, solver=lbfgs)` | Yes |
| `handle_severe_missing_data` | `robust_imputation = True` | Adds indicator-aware numeric imputation and `__missing__` categorical fills | Yes |
| `apply_target_encoding` | `use_target_encoding = True` | `TargetEncoder` branch added to `ColumnTransformer` for high-cardinality categorical/text-like cols | Yes |
| `apply_feature_engineering` | `use_feature_engineering = True` | `build_preprocessor()` appends `PolynomialFeatures(degree=2, include_bias=False)` to the continuous pipeline | Yes |
| `handle_outliers` | `handle_outliers = True` | `RobustScaler` (small) or `RobustScaler(unit_variance=True)` (≥1000 rows) | Yes |
| `apply_robust_scaling` | `use_robust_scaling = True` | `RobustScaler` instead of `StandardScaler` | Yes |
| `drop_correlated_features` | `drop_high_corr = True` | Drops `corr_cols_to_drop` in `build_preprocessor()` | Yes |
| `drop_leaky_features` | `drop_leaky = True` | Drops `leaky_col_names` in `build_preprocessor()` | Yes |
| `drop_sensitive_features` | `drop_sensitive = True` | Drops `sensitive_cols` in `build_preprocessor()` | Yes |
| `use_simple_models_only` | `simple_models_only = True` | Restricts to simple candidates; currently `Dummy + LR + RF`, plus `SVC_RBF` when the dataset is small and `cols <= 50` | Yes |
| `use_ensemble_models` | `prefer_ensemble = True` | Favors tree ensembles; medium datasets can include GradientBoosting + HistGradientBoosting, very large datasets prefer HistGradientBoosting over classic GradientBoosting | Yes |
| `drop_near_constant_features` | logs columns | Preprocessor already drops `near_constant_cols` unconditionally — step adds plan visibility | Yes |
| `reduce_tuning_budget` | `reduce_tuning_budget = True` | `tune_best_model()` reduces search iterations and may sub-sample rows | Yes |
| `tune_hyperparameters` | — | Orchestrator calls `tune_best_model()` on the best candidate | Yes |
| `validate_with_cross_validation` | — | Orchestrator runs `cross_validate_top_models()` on top candidates | Yes |

---

## 4. Reflector Signal → Issue/Suggestion Mapping

### Implemented
| Signal | Issue raised | Suggestion | Triggers replan? |
|--------|-------------|------------|-----------------|
| Hard leakage evidence in profiler | "Profiler found hard target-leakage evidence..." | Remove flagged columns / confirm they are unavailable at prediction time | No automatic replan; forces review |
| Soft leakage evidence in profiler | "Profiler flagged soft target-proxy risk..." | Human review before trusting the run | No automatic replan; forces review |
| Best model barely beats dummy (< 0.05 improvement) | "Weak signal or pipeline issues" | Check leakage / verify target quality | Yes (if macro F1 is also below the active threshold) |
| `bal_acc > 0.90` and `f1_macro < 0.70` | "Overfitting suspected" | Regularization / reduce complexity | Yes (if macro F1 is also below the active threshold) |
| `f1_macro` below adaptive threshold | "Macro F1 below expected threshold" | Try different models / tuning / preprocessing | Yes |
| Per-class F1 breakdown | "Class X has very low F1" | Investigate class overlap / weights / sample coverage | Yes |
| ≥2 non-dummy models near-perfect (bal_acc ≥ 0.99, f1 ≥ 0.99) | "Suspicious near-perfect performance" | Check for target proxies / leakage | No (f1 too high) |
| All models F1 within 0.05 | "Low model diversity" | Try more diverse models | Yes (if macro F1 is also below the active threshold) |
| `imbalance_ratio ≥ 3.0` | — | Consider class_weight / threshold tuning | No (suggestion only) |
| `imbalance_ratio ≥ 5.0` + weak baseline margin | "Severe class imbalance..." | Consider class_weight / threshold tuning / oversampling | Yes |
| Sensitive columns detected | — | Run a fairness audit / inspect disparate impact | No |
| Regression: R² barely beats dummy (< 0.05 improvement) | "Weak regression signal" | Check feature engineering / target quality | Yes |
| Regression: `r2 < 0.1` | "R² very low" | Try feature engineering / verify target is predictable | Yes |
| Regression: ≥2 non-dummy models with R² ≥ 0.99 | "Suspicious near-perfect R²" | Inspect features for target proxies or multiplicative combinations | Yes |
| `training_warnings` contains overflow/divide-by-zero | Suggestion always; escalates to issue if performance is also weak | Apply robust scaling / check extreme feature values | Only if weak performance too |

### Still not implemented
| Signal | Issue to raise | Suggestion |
|--------|---------------|------------|
| `outlier_cols` present + bad R² | "Outliers may be distorting regression" | Use robust scaling / robust models |
| Replan count > 1, no improvement | "Replanning not helping" | Stop early / escalate to human review |

---

## 5. Replan Logic

`should_replan()` defers entirely to `replan_recommended` from `reflect()`.

`replan_recommended` is `True` only when **both** of these hold:
- `issues` list is non-empty
- `f1_macro < 0.60` (classification) or `r2 improvement < 0.05` (regression)

This prevents spurious replans when near-perfect performance raises a leakage warning but f1 is already high on a clean, well-separated dataset.

`apply_replan_strategy()` modifies the plan based on issue text:
| Issue keyword | Plan step added |
|---------------|----------------|
| `"overfitting"` | `apply_regularization` |
| `"imbalance"` | `consider_imbalance_strategy` |
| `"f1"` (case-insensitive) | `use_ensemble_models` |
| `"baseline"` | `use_ensemble_models` |
| `"instability"` / `"scaling"` | `apply_robust_scaling` |
| `"cross-validation mean"` | `apply_regularization` + `increase_test_size = True` |

---

## 6. Training Warnings

Runtime warnings from sklearn are captured during `train_models()` and stored in `reflection.json`.

| Where captured | `tools/modelling.py` — `warnings.catch_warnings(record=True)` |
|---|---|
| Where stored | `reflection.json` → `training_warnings: [...]` |
| Scope | Unique warning strings across all trained models per run |
| Console | Captured from training/CV; known non-actionable sklearn deprecation noise is filtered during tuning |

> **Current behavior:** Reflector already inspects `training_warnings` and adds a scaling-related suggestion. It escalates them to an issue when weak performance suggests the warnings are actually harming the run. The remaining limitation is that actionable warnings do not automatically downgrade the final verdict if held-out and CV performance still look strong.

### Explanation Outputs

Evaluation now has three explanation paths:

| Signal | Condition | Output / behavior |
|--------|-----------|-------------------|
| Native model importances | Best model exposes `feature_importances_` | `feature_importance.png` + `top_feature_importance` using model-native importances |
| Linear model coefficients | Best model exposes `coef_` | `feature_importance.png` + `top_feature_importance` using absolute coefficients |
| Unsupported model | No native importances but `X_test`/`y_test` available | Permutation-importance fallback (e.g. `SVC_RBF`) |
| Preprocessor name export fails | `get_feature_names_out()` unavailable or broken | Best-effort feature-name recovery from the fitted preprocessing graph |
| Final model changes across passes | Earlier candidate wrote a stale artefact | Old `feature_importance.png` removed if the final model does not support it |

> **Current behavior:** Explanation is now reliable across trees, linear models, and unsupported models via permutation importance, but it is still reported at the transformed feature level rather than always being aggregated back to the original raw feature.

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
| Small | < 1000 | Dummy + LR + RF, and `SVC_RBF` when `cols <= 50` |
| Medium | 1000–9999 | Dummy + LR + RF + GradientBoosting + HistGradientBoosting |
| Large | ≥ 10000 | Dummy + LR + RF + HistGradientBoosting when `prefer_ensemble=True` (classic GradientBoosting skipped on very large workloads) |

**SVC:** `probability=False` (Platt scaling removed — unused in pipeline). Added for small datasets with ≤ 50 raw columns. It remains reachable even under `simple_models_only`, because the candidate list is assembled before the early return.

---

## 9. Known Issues / Technical Debt

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| ~~`robust_imputation` flag set but unused~~ | executor | Fixed | `build_preprocessor()` now switches imputation strategy and adds missing indicators when requested |
| ~~`use_target_encoding` flag set but unused~~ | executor | Fixed | `TargetEncoder` branch wired into `build_preprocessor()` for high-cardinality cols |
| ~~`use_feature_engineering` flag set but unused~~ | executor/modelling | Fixed | `build_preprocessor()` now injects `PolynomialFeatures(degree=2, include_bias=False)` when requested |
| `replan_attempt` marker added by replan | reflector | Low | Harmless but clutters plan output |
| `use_class_weights` flag partly redundant | executor | Low | `select_models()` honours the flag, but can also infer the same behaviour from `imbalance_ratio` |
| ~~Spurious replan on near-perfect datasets~~ | reflector | Fixed | `should_replan()` now only returns `True` when `replan_recommended=True` |
| ~~Leakage detection broken for regression~~ | profiler | Fixed | Regression target was factorized before MI — now uses raw values, normalised by max MI |
| ~~Training warnings not acted on~~ | reflector | Fixed | Overflow/divide warnings now raise suggestion (and issue if performance is also weak) |
| ~~`apply_oversampling` advertised but not wired~~ | planner/executor/modelling | Fixed | Severe imbalance now routes to SMOTE-compatible training when available |
| ~~`drop_sensitive_features` advertised but not wired~~ | planner/executor/modelling | Fixed | Sensitive attributes can now be removed before preprocessing |
| Verdict remains optimistic when actionable warnings exist | evaluation/reflector | Medium | Numerical-instability warnings are surfaced, but can still coexist with a `Reliable result` verdict |
| Explanations still use transformed features | evaluation | Medium | Human-facing reports can still show encoded/expanded feature names instead of raw-feature groups |
| Proxy fairness remains unresolved | profiler/evaluation | Medium | Dropping sensitive columns does not remove proxy signals such as relationship-based features on `adult.csv` |

---

## 10. Implementation Priority

| Priority | What | Why |
|----------|------|-----|
| 1 | Tests (>60% coverage required) | Assignment hard requirement — nothing else ships without this |
| 2 | README / report / signal-map sync | Submission-facing documentation should match the implemented planner and test counts |
| 3 | Downgrade verdicts on actionable warnings | Current warning capture is stronger than the final verdict logic |
| 4 | Aggregate explanation back to raw features | Current reports are accurate but still overly tied to transformed feature names |
| 5 | Expand feature engineering breadth | Current implementation is limited to low-dimensional numeric polynomial expansion |
