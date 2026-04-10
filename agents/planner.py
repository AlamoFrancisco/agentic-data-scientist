"""
Planner Agent - Students must extend this significantly

The planner analyzes dataset characteristics and generates an execution plan.
Your task is to implement sophisticated planning logic that adapts to different
dataset types, sizes, and characteristics.

TODO: Extend this module with:
1. Sophisticated planning logic based on dataset profiles
2. Different plan templates for different scenarios
3. Memory-guided planning (use past successful strategies)
4. Dependency management (task ordering)
5. Conditional planning (if X then Y else Z)
6. Fallback strategies for edge cases
"""

from typing import Any, Dict, List, Optional


def create_plan(
    dataset_profile: Dict[str, Any], 
    memory_hint: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate an execution plan based on dataset characteristics.
    
    This is a basic implementation. Students should extend this significantly.
    
    Args:
        dataset_profile: Dictionary containing dataset metadata including:
            - shape: {rows: int, cols: int}
            - feature_types: {numeric: List[str], categorical: List[str]}
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
    
    TODO for students:
    - Implement conditional logic based on dataset size
    - Add different strategies for imbalanced datasets
    - Handle high-cardinality categorical features
    - Use memory hints to prioritize successful models
    - Create plan templates for common scenarios
    - Add preprocessing steps based on data quality
    """
    
    # Basic plan structure (students should make this much more sophisticated)
    plan: List[str] = [
        "profile_dataset",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]
    
    # TODO: Add sophisticated logic here
    # Example: Check for imbalance
    imb = dataset_profile.get("imbalance_ratio") or 1.0
    if imb >= 3.0:
        # TODO: Make this more sophisticated
        # Consider: SMOTE, class weights, threshold tuning, etc.
        plan.insert(plan.index("train_models"), "consider_imbalance_strategy")
    
    # TODO: Add logic for small datasets
    if dataset_profile["shape"]["rows"] < 1000:
        plan.append("apply_regularization")
    
    # TODO: Add logic for high-cardinality categoricals
    categorical_cols = dataset_profile.get("feature_types", {}).get("categorical", [])
    n_unique = dataset_profile.get("n_unique_by_col", {})
    high_card_cats = [c for c in categorical_cols if n_unique.get(c, 0) > 50]
    if high_card_cats:
        plan.insert(plan.index("build_preprocessor"), "apply_target_encoding")

    # TODO: Use memory hints
    if memory_hint and memory_hint.get("best_model"):
        plan.append(f"prioritize_model:{memory_hint['best_model']}")
    
    # Add outlier handling step if the profiler detected outlier-heavy columns
    outlier_cols = dataset_profile.get("outlier_cols", [])
    if outlier_cols:
        plan.insert(plan.index("build_preprocessor"), "handle_outliers")

    # TODO: Add logic based on missing data
    missing_pct = dataset_profile.get("missing_pct", {})
    if missing_pct:
        max_missing = max(missing_pct.values())
        if max_missing > 20:
            plan.insert(plan.index("build_preprocessor"), "handle_severe_missing_data")
    
    return plan



# TODO: Add helper functions for planning
# def create_small_dataset_plan(...):
# def create_imbalanced_dataset_plan(...):
# def create_high_dimensional_plan(...):
# def select_preprocessing_strategy(...):
# def estimate_plan_cost(...):  # For cost-aware planning
