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
- Fallback strategies when initial plan produces no useful result
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
    PLANNER_CV_MAX_COLS,
    PLANNER_CV_MAX_WORKLOAD,
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

    imb = dataset_profile.get("imbalance_ratio") or 1.0
    if imb >= IMBALANCE_VERY_SEVERE:
        plan.insert(plan.index("train_models"), "apply_oversampling")
    elif imb >= IMBALANCE_THRESHOLD:
        plan.insert(plan.index("train_models"), "consider_imbalance_strategy")

    # Size bucket — drives model complexity and regularization
    rows = dataset_profile["shape"]["rows"]
    if rows < SMALL_DATASET_ROWS:
        plan.append("apply_regularization")
        plan.append("use_simple_models_only")
    elif rows >= LARGE_DATASET_ROWS:
        plan.append("use_ensemble_models")
    
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
        if MAX_OHE_UNIQUE < n_unique.get(c, 0) < rows * 0.10
    ]
    if high_card_cats or high_card_text:
        plan.insert(plan.index("build_preprocessor"), "apply_target_encoding")

    if memory_hint and memory_hint.get("best_model"):
        plan.append(f"prioritize_model:{memory_hint['best_model']}")
    
    # Add outlier handling step if the profiler detected outlier-heavy columns
    outlier_cols = dataset_profile.get("outlier_cols", [])
    if outlier_cols:
        plan.insert(plan.index("build_preprocessor"), "handle_outliers")

    missing_pct = dataset_profile.get("missing_pct", {})
    if missing_pct:
        max_missing = max(missing_pct.values())
        if max_missing > SEVERE_MISSING_THRESHOLD:
            plan.insert(plan.index("build_preprocessor"), "handle_severe_missing_data")

    # Hyperparameter tuning: worthwhile for medium/large datasets where the
    # extra compute is justified; skip for small datasets to stay fast.
    if rows >= SMALL_DATASET_ROWS:
        plan.insert(plan.index("evaluate"), "tune_hyperparameters")
        
        # Cost-aware planning: estimate compute cost (rows * cols)
        complexity = rows * dataset_profile["shape"]["cols"]
        if complexity > COMPUTE_COST_THRESHOLD:
            dataset_profile["reduce_tuning_budget"] = True
            plan.insert(plan.index("tune_hyperparameters"), "reduce_tuning_budget")

    # Use robust scaling when scale mismatch detected
    if dataset_profile.get("scale_mismatch"):
        plan.insert(plan.index("build_preprocessor"), "apply_robust_scaling")

    # Automatically drop only hard leakage evidence. Soft leakage signals should
    # trigger review, not silent feature removal.
    hard_leaky_cols = [c["column"] for c in dataset_profile.get("hard_leakage_cols", [])]
    if hard_leaky_cols:
        dataset_profile["leaky_col_names"] = hard_leaky_cols
        plan.insert(plan.index("build_preprocessor"), "drop_leaky_features")

    # Drop near-constant features — they carry no signal and inflate one-hot encoding
    near_constant_cols = dataset_profile.get("near_constant_cols", [])
    if near_constant_cols:
        plan.insert(plan.index("build_preprocessor"), "drop_near_constant_features")

    # Drop highly correlated features (abs_corr >= 0.95) — likely redundant or leaky
    high_corr_pairs = dataset_profile.get("high_corr_pairs", [])
    cols_to_drop = list({
        p["col_b"] for p in high_corr_pairs if p.get("abs_corr", 0) >= HIGH_CORR_DROP_THRESHOLD
    })
    if cols_to_drop:
        dataset_profile["corr_cols_to_drop"] = cols_to_drop
        plan.insert(plan.index("build_preprocessor"), "drop_correlated_features")
        
    # Drop sensitive features to mitigate direct algorithmic bias
    sensitive_cols = dataset_profile.get("sensitive_cols", [])
    if sensitive_cols:
        dataset_profile["drop_sensitive"] = True
        plan.insert(plan.index("build_preprocessor"), "drop_sensitive_features")
    
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
        if "apply_regularization" not in new_plan:
            new_plan.insert(new_plan.index("train_models"), "apply_regularization")

    # Severe imbalance with weak baseline margin: ensure class weights are applied
    if "imbalance" in issues_lower:
        if "consider_imbalance_strategy" not in new_plan:
            new_plan.insert(new_plan.index("train_models"), "consider_imbalance_strategy")

    # Low F1 or weak-vs-baseline: switch to ensemble models for more predictive power
    if "f1" in issues_lower or "baseline" in issues_lower:
        if "use_ensemble_models" not in new_plan:
            new_plan.append("use_ensemble_models")

    # Numerical instability: apply robust scaling if not already planned
    if "instability" in issues_lower or "scaling" in issues_lower:
        if "apply_robust_scaling" not in new_plan:
            new_plan.insert(new_plan.index("build_preprocessor"), "apply_robust_scaling")

    # CV gap: held-out score diverges from cross-validation mean.
    # Strengthen regularization to reduce variance between splits, and widen
    # the test set so the held-out estimate is more representative.
    if "cross-validation mean" in issues_lower:
        if "apply_regularization" not in new_plan:
            new_plan.insert(new_plan.index("train_models"), "apply_regularization")
        new_profile["increase_test_size"] = True

    new_plan.append("replan_attempt")

    return new_plan, new_profile
