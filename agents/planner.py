"""
Planner Agent

Analyses dataset characteristics and generates a tailored execution plan.

Implemented:
- Size-bucket routing: small (<1000) → regularization + simple_models_only;
  large (≥10000) → ensemble models
- High-dimensional routing: wide / p≈n datasets also bias toward
  regularization + simple models
- Scale mismatch detection → apply_robust_scaling
- Mutual-information leakage detection → drop_leaky_features
- Near-constant column detection → drop_near_constant_features
- High-correlation feature pruning (abs_corr ≥ 0.95) → drop_correlated_features
- High-cardinality categoricals and text columns → apply_target_encoding
- Class imbalance (ratio ≥ 3.0) → consider_imbalance_strategy
- Outlier-heavy columns → handle_outliers
- Severe missing data (>20%) → handle_severe_missing_data
- Memory-guided model prioritisation → prioritize_model:<name>
- Cost-aware planning: reduces tuning budget on large workloads and skips
  cross-validation on extreme workloads

TODO:
- Plan templates for unsupported scenarios (datetime-heavy / time-series, etc.)
"""

from typing import Any, Dict, List, Optional, Tuple

from config import (
    HIGH_CORR_DROP_THRESHOLD,
    HIGH_DIMENSIONAL_COL_RATIO,
    HIGH_DIMENSIONAL_MIN_COLS,
    IMBALANCE_THRESHOLD,
    IMBALANCE_VERY_SEVERE,
    LARGE_DATASET_ROWS,
    MAX_OHE_UNIQUE,
    MAX_TEXT_UNIQUE_FRAC,
    PLANNER_CV_MAX_COLS,
    PLANNER_CV_MAX_WORKLOAD,
    POLY_FEATURES_MAX_COLS,
    SEVERE_MISSING_THRESHOLD,
    SMALL_DATASET_ROWS,
    COMPUTE_COST_THRESHOLD,
)


def _insert_before_unique(plan: List[str], anchor: str, step: str) -> None:
    if step not in plan:
        plan.insert(plan.index(anchor), step)


def _append_unique(plan: List[str], step: str) -> None:
    if step not in plan:
        plan.append(step)


def _feature_count(cols: int) -> int:
    return max(int(cols) - 1, 1)


def _is_high_dimensional(rows: int, cols: int) -> bool:
    feature_cols = _feature_count(cols)
    return (
        feature_cols >= HIGH_DIMENSIONAL_MIN_COLS
        or (feature_cols / max(int(rows), 1)) >= HIGH_DIMENSIONAL_COL_RATIO
    )


def create_plan(
    dataset_profile: Dict[str, Any], 
    memory_hint: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate an execution plan based on dataset characteristics.

        Args:
            dataset_profile: Dictionary containing dataset metadata including:
                - shape: {rows: int, cols: int}
                - feature_types: {
                    numeric: {ordinal: List[str], continuous: List[str]},
                    categorical: {binary: List[str], multiclass: List[str]}
                  }
                - imbalance_ratio: float (majority/minority class ratio)
                - missing_pct: Dict[str, float] (missing % per column)
                - is_classification: bool
                - notes: List[str] (warnings/observations)
        memory_hint: Optional dict with info from previous runs on similar datasets
    
    Returns:
        List of task names representing the execution plan
        
    Example:
        >>> profile = {"shape": {"rows": 5000}, "imbalance_ratio": 4.5}
        >>> plan = create_plan(profile)
        >>> print(plan)
        ['profile_dataset', 'consider_imbalance_strategy', 'train_models', ...]
    """

    # Base execution plan; adaptive steps are inserted from profiler signals below.
    plan: List[str] = [
        "profile_dataset",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "validate_with_cross_validation",
        "reflect",
        "write_report",
    ]

    rows = int(dataset_profile["shape"]["rows"])
    cols = int(dataset_profile["shape"]["cols"])
    complexity = rows * cols
    feature_cols = _feature_count(cols)
    high_dimensional = _is_high_dimensional(rows, cols)

    imb = dataset_profile.get("imbalance_ratio") or 1.0
    if imb >= IMBALANCE_VERY_SEVERE:
        _insert_before_unique(plan, "train_models", "apply_oversampling")
    elif imb >= IMBALANCE_THRESHOLD:
        _insert_before_unique(plan, "train_models", "consider_imbalance_strategy")

    # Size bucket — drives model complexity and regularization.
    if rows < SMALL_DATASET_ROWS:
        _append_unique(plan, "apply_regularization")
        _append_unique(plan, "use_simple_models_only")
    elif rows >= LARGE_DATASET_ROWS and not high_dimensional:
        _append_unique(plan, "use_ensemble_models")
        
    if dataset_profile.get("feature_types", {}).get("datetime", []):
        _insert_before_unique(plan, "validate_with_cross_validation", "use_time_aware_validation")

    # Wide datasets are more prone to overfitting and make complex search
    # disproportionately expensive, even when the row count is moderate.
    if high_dimensional:
        _append_unique(plan, "apply_regularization")
        _append_unique(plan, "use_simple_models_only")
    
    # High-cardinality categoricals: check multiclass AND text columns.
    # Text cols are high-cardinality strings — those with n_unique < 10% of rows
    # are likely real categoricals (e.g. model_name, city) rather than free-form text or IDs.
    categorical_groups = dataset_profile.get("feature_types", {}).get("categorical", {})
    categorical_cols = categorical_groups.get("binary", []) + categorical_groups.get("multiclass", [])
    text_cols = dataset_profile.get("feature_types", {}).get("text", [])
    n_unique = dataset_profile.get("n_unique_by_col", {})
    high_card_cats = [c for c in categorical_cols if n_unique.get(c, 0) > MAX_OHE_UNIQUE]
    # Text cols that look categorical: more unique values than MAX_OHE_UNIQUE but under 10% of rows
    high_card_text = [
        c for c in text_cols
        if MAX_OHE_UNIQUE < n_unique.get(c, 0) < rows * MAX_TEXT_UNIQUE_FRAC
    ]
    if high_card_cats or high_card_text:
        _insert_before_unique(plan, "build_preprocessor", "apply_target_encoding")

    if memory_hint:
        if memory_hint.get("best_model"):
            _append_unique(plan, f"prioritize_model:{memory_hint['best_model']}")
            
        # Meta-learning: pre-emptively apply strategies that successfully fixed issues in prior runs
        if memory_hint.get("successful_plan"):
            for step in memory_hint["successful_plan"]:
                if step in {
                    "apply_robust_scaling", "handle_outliers", "handle_severe_missing_data", 
                    "apply_target_encoding", "drop_near_constant_features", "drop_correlated_features", 
                    "drop_leaky_features", "drop_sensitive_features", "apply_feature_engineering"
                }:
                    _insert_before_unique(plan, "build_preprocessor", step)
                elif step in {"consider_imbalance_strategy", "apply_oversampling", "apply_regularization"}:
                    _insert_before_unique(plan, "train_models", step)
                elif step in {"use_simple_models_only", "use_ensemble_models"}:
                    _insert_before_unique(plan, "select_models", step)
                elif step == "reduce_tuning_budget" and "tune_hyperparameters" in plan:
                    _insert_before_unique(plan, "tune_hyperparameters", step)
    
    # Add outlier handling step if the profiler detected outlier-heavy columns
    outlier_cols = dataset_profile.get("outlier_cols", [])
    if outlier_cols:
        _insert_before_unique(plan, "build_preprocessor", "handle_outliers")

    missing_pct = dataset_profile.get("missing_pct", {})
    if missing_pct:
        max_missing = max(missing_pct.values())
        if max_missing > SEVERE_MISSING_THRESHOLD:
            _insert_before_unique(plan, "build_preprocessor", "handle_severe_missing_data")

    # Hyperparameter tuning: worthwhile for medium/large datasets where the
    # extra compute is justified; skip for small or wide datasets to stay fast.
    if rows >= SMALL_DATASET_ROWS and not high_dimensional:
        _insert_before_unique(plan, "evaluate", "tune_hyperparameters")

        if complexity > COMPUTE_COST_THRESHOLD:
            _insert_before_unique(plan, "tune_hyperparameters", "reduce_tuning_budget")

    # Cross-validation is useful by default, but very wide / extreme workloads
    # should skip it to keep the plan compute-aware.
    if complexity > PLANNER_CV_MAX_WORKLOAD or feature_cols >= PLANNER_CV_MAX_COLS:
        if "validate_with_cross_validation" in plan:
            plan.remove("validate_with_cross_validation")

    # Use robust scaling when scale mismatch detected
    if dataset_profile.get("scale_mismatch"):
        _insert_before_unique(plan, "build_preprocessor", "apply_robust_scaling")

    # Automatically drop only hard leakage evidence. Soft leakage signals should
    # trigger review, not silent feature removal.
    hard_leaky_cols = [c["column"] for c in dataset_profile.get("hard_leakage_cols", [])]
    if hard_leaky_cols:
        dataset_profile["leaky_col_names"] = hard_leaky_cols
        _insert_before_unique(plan, "build_preprocessor", "drop_leaky_features")

    # Drop near-constant features — they carry no signal and inflate one-hot encoding
    near_constant_cols = dataset_profile.get("near_constant_cols", [])
    if near_constant_cols:
        _insert_before_unique(plan, "build_preprocessor", "drop_near_constant_features")

    # Drop highly correlated features (abs_corr >= 0.95) — likely redundant or leaky
    high_corr_pairs = dataset_profile.get("high_corr_pairs", [])
    target = dataset_profile.get("target")
    cols_to_drop = []
    for p in high_corr_pairs:
        if p.get("abs_corr", 0) >= HIGH_CORR_DROP_THRESHOLD and target not in (p["col_a"], p["col_b"]):
            cols_to_drop.append(p["col_b"])
    cols_to_drop = list(set(cols_to_drop))
    if cols_to_drop:
        dataset_profile["corr_cols_to_drop"] = cols_to_drop
        _insert_before_unique(plan, "build_preprocessor", "drop_correlated_features")
        
    # Drop sensitive features to mitigate direct algorithmic bias
    sensitive_cols = dataset_profile.get("sensitive_cols", [])
    if sensitive_cols:
        _insert_before_unique(plan, "build_preprocessor", "drop_sensitive_features")

    # Feature engineering: add polynomial features to capture non-linear relationships,
    # but only if the feature space is very small to avoid combinatorial explosion.
    if feature_cols <= POLY_FEATURES_MAX_COLS:
        _insert_before_unique(plan, "build_preprocessor", "apply_feature_engineering")
    
    return plan



def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Modify the plan and dataset profile based on reflection issues.

    Called by the orchestrator when the reflector recommends a replan.
    Each handler targets a specific issue string and makes a concrete change
    that the executor or preprocessor will act on in the next pass.

    Args:
        plan: Current execution plan
        dataset_profile: Current dataset profile
        reflection: Reflection output including issues list

    Returns:
        Tuple of (modified_plan, modified_profile)
    """
    new_plan = list(plan)
    new_profile = dict(dataset_profile)

    notes = list(new_profile.get("notes", []))
    notes.append("Replan: adjusting strategy after reflection.")
    new_profile["notes"] = notes

    issues = reflection.get("issues", [])
    issues_lower = " ".join(issues).lower()

    # Overfitting: strengthen regularization
    if "overfitting" in issues_lower:
        _insert_before_unique(new_plan, "train_models", "apply_regularization")

    # Severe imbalance with weak baseline margin: ensure class weights are applied
    if "imbalance" in issues_lower:
        _insert_before_unique(new_plan, "train_models", "consider_imbalance_strategy")

    # Low F1/R² or weak-vs-baseline: switch to ensemble models for more predictive power
    if "f1" in issues_lower or "r²" in issues_lower or "r2" in issues_lower or "baseline" in issues_lower:
        _append_unique(new_plan, "use_ensemble_models")

    # Numerical instability: apply robust scaling if not already planned
    if "instability" in issues_lower or "scaling" in issues_lower:
        _insert_before_unique(new_plan, "build_preprocessor", "apply_robust_scaling")

    # CV gap: held-out score diverges from cross-validation mean.
    # Strengthen regularization to reduce variance between splits, and widen
    # the test set so the held-out estimate is more representative.
    if "cross-validation mean" in issues_lower:
        _insert_before_unique(new_plan, "train_models", "apply_regularization")
        new_profile["increase_test_size"] = True

    _append_unique(new_plan, "replan_attempt")

    return new_plan, new_profile
