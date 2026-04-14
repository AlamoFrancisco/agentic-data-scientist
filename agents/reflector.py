"""
Reflector Agent - Students must extend this significantly

The reflector evaluates execution results, identifies issues, and suggests improvements.
Your task is to implement sophisticated analysis that goes beyond simple threshold checks.

TODO: Extend this module with:
1. Statistical significance testing between models
2. Per-class performance analysis
3. Root cause diagnosis (data quality, preprocessing, model issues)
4. Actionable, prioritized suggestions
5. Learning from past reflections (meta-learning)
"""

from typing import Any, Dict, List, Optional, Tuple


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
    training_warnings: Optional[List[str]] = None,
    cv_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze results and generate reflection with issues and suggestions.
    
    This is a basic implementation. Students should extend this significantly.
    
    Args:
        dataset_profile: Dataset characteristics
        evaluation: Best model's metrics
        all_metrics: Metrics for all trained models
    
    Returns:
        Dictionary with:
            - status: str ("ok" or "needs_attention")
            - best_model: str (model name)
            - issues: List[str] (identified problems)
            - suggestions: List[str] (improvement recommendations)
            - replan_recommended: bool (should we replan?)
    
    TODO for students:
    - Implement statistical tests (paired t-tests, Wilcoxon tests)
    - Add per-class performance analysis
    - Detect overfitting vs underfitting
    - Analyze confusion matrix patterns
    - Check for data quality issues
    - Prioritize suggestions by expected impact
    - Learn which suggestions work from memory
    """
    
    best_model = evaluation.get("model")
    is_classification = dataset_profile.get("is_classification", True)
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    r2 = float(evaluation.get("r2", 0.0))
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)

    issues: List[str] = []
    suggestions: List[str] = []
    cv_concerns = False

    def check_cv_consistency(split_metric: float, metric_name: str) -> None:
        nonlocal cv_concerns
        if not cv_summary or not cv_summary.get("enabled"):
            return

        cv_models = cv_summary.get("models", [])
        if not cv_models:
            return

        cv_entry = next((m for m in cv_models if m.get("model") == best_model), None)
        if cv_entry is None:
            return

        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        cv_mean = float(cv_entry.get(mean_key, 0.0))
        cv_std = float(cv_entry.get(std_key, 0.0))
        gap = abs(split_metric - cv_mean)
        unstable_std = 0.05 if is_classification else 0.10
        unstable_gap = 0.08 if is_classification else 0.15

        if gap > unstable_gap:
            cv_concerns = True
            issues.append(
                f"Held-out {metric_name.replace('_', ' ')} ({split_metric:.3f}) differs noticeably "
                f"from cross-validation mean ({cv_mean:.3f})."
            )
            suggestions.append(
                "Treat the current split cautiously and prefer the cross-validation estimate "
                "when judging model quality."
            )

        if cv_std > unstable_std:
            cv_concerns = True
            issues.append(
                f"Cross-validation {metric_name.replace('_', ' ')} is unstable across folds "
                f"(std={cv_std:.3f})."
            )
            suggestions.append(
                "Performance appears split-sensitive. Consider more data, stronger regularization, "
                "or simpler models."
            )

        cv_best_model = cv_summary.get("best_model")
        if cv_best_model and cv_best_model != best_model:
            suggestions.append(
                f"Cross-validation ranked `{cv_best_model}` above the held-out winner `{best_model}`. "
                "Use the cross-validated ranking for the final decision."
            )

    # Route to regression or classification analysis
    if not is_classification:
        # Regression analysis
        dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
        if dummy is not None:
            dummy_r2 = float(dummy.get("r2", 0.0))
            improvement = r2 - dummy_r2
            if improvement < 0.05:
                issues.append(
                    f"Best model R² only {improvement:.3f} better than baseline. "
                    "Weak signal or pipeline issues."
                )
                suggestions.append(
                    "Check for target leakage, verify target quality, "
                    "or improve feature engineering."
                )
        if r2 < 0.1:
            issues.append(f"R² is very low ({r2:.3f}) — model explains little variance.")
            suggestions.append("Try feature engineering or check if target is predictable.")

        # Numerical instability warnings
        numerical_warning_keywords = ("overflow", "divide by zero", "invalid value")
        numerical_warnings = [
            w for w in (training_warnings or [])
            if any(kw in w.lower() for kw in numerical_warning_keywords)
        ]
        if numerical_warnings:
            suggestions.append(
                "Numerical instability detected during training (overflow/divide-by-zero). "
                "Consider applying robust scaling or checking for extreme feature values."
            )
            if r2 < 0.1:
                issues.append(
                    "Numerical instability warnings present alongside low R² — "
                    "scaling issues may be degrading model performance."
                )

        # Near-perfect R² across multiple real models is suspicious — likely leakage
        real_reg_models = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
        near_perfect_reg = [m for m in real_reg_models if float(m.get("r2", 0.0)) >= 0.99]
        if len(near_perfect_reg) >= 2:
            issues.append(
                "Near-perfect R² across multiple non-baseline models is suspicious."
            )
            suggestions.append(
                "Inspect features for target proxies or columns that deterministically "
                "derive the target (e.g. multiplicative combinations)."
            )

        check_cv_consistency(r2, "r2")

        return {
            "status": "needs_attention" if issues else "ok",
            "best_model": best_model,
            "issues": issues,
            "suggestions": suggestions,
            "replan_recommended": bool(issues and (r2 < 0.1 or cv_concerns)),
            "training_warnings": training_warnings or [],
        }

    # Classification analysis below
    # Basic comparison with dummy baseline
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)

    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba

        # TODO: Make this more sophisticated
        # Consider: confidence intervals, effect sizes, etc.
        if improvement < 0.05:
            issues.append(
                f"Best model only {improvement:.3f} better than baseline. "
                "Weak signal or pipeline issues."
            )
            suggestions.append(
                "Check for target leakage, verify target quality, "
                "or improve feature engineering."
            )
    
    # TODO: Add more sophisticated checks
    # Check for overfitting 
    if bal_acc > 0.90 and f1_macro < 0.70:
        issues.append("High balanced accuracy but low F1 macro suggests overfitting.")
        suggestions.append(
            "Try regularization, reduce model complexity, or add more data."
        )
    
    # Check F1 score
    # TODO: Make threshold adaptive based on problem difficulty
    if f1_macro < 0.60:
        issues.append("Macro F1 score is modest (<0.60).")
        suggestions.append(
            "Try different models, tune hyperparameters, "
            "or improve preprocessing."
        )
    
    # TODO: Add imbalance-specific analysis
    if imb >= 3.0:
        suggestions.append(
            "Imbalance detected: consider class_weight, "
            "threshold tuning, or SMOTE."
        )

    # Suspiciously strong performance across multiple real models can indicate
    # leakage, duplicate target encodings, or an overly trivial target.
    real_models = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
    near_perfect_models = [
        m for m in real_models
        if float(m.get("balanced_accuracy", 0.0)) >= 0.99
        and float(m.get("f1_macro", 0.0)) >= 0.99
    ]
    if len(near_perfect_models) >= 2:
        issues.append(
            "Near-perfect performance across multiple non-baseline models is suspicious."
        )
        suggestions.append(
            "Inspect features for target proxies, leakage, or columns that deterministically map to the target."
        )

    check_cv_consistency(bal_acc, "balanced_accuracy")
    
    # TODO: Add checks for:
    # - Model diversity (are all models performing similarly?)
    # Check for model diversity
    f1_scores = [m.get("f1_macro", 0.0) for m in all_metrics]
    if len(f1_scores) > 1:
        score_range = max(f1_scores) - min(f1_scores)
        if score_range < 0.05:
            issues.append("All models performing similarly — low model diversity.")
            suggestions.append(
                "Try more diverse models or investigate data/preprocessing issues."
            )

    # Numerical instability warnings from training
    # overflow/divide-by-zero in matmul = gradient computation hitting scale limits
    numerical_warning_keywords = ("overflow", "divide by zero", "invalid value")
    numerical_warnings = [
        w for w in (training_warnings or [])
        if any(kw in w.lower() for kw in numerical_warning_keywords)
    ]
    if numerical_warnings:
        suggestions.append(
            "Numerical instability detected during training (overflow/divide-by-zero). "
            "Consider applying robust scaling or checking for extreme feature values."
        )
        # Escalate to issue if performance is also weak — instability may be the cause
        if f1_macro < 0.60:
            issues.append(
                "Numerical instability warnings present alongside weak F1 — "
                "scaling issues may be degrading model performance."
            )

    # Determine status
    status = "needs_attention" if issues else "ok"
    
    # Simple replanning trigger
    # TODO: Make this more sophisticated
    replan_recommended = bool(issues and (f1_macro < 0.60 or cv_concerns))
    
    return {
        "status": status,
        "best_model": best_model,
        "issues": issues,
        "suggestions": suggestions,
        "replan_recommended": replan_recommended,
        "training_warnings": training_warnings or [],
    }


def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Decide whether to trigger replanning based on reflection.

    Only replan when the reflector explicitly sets replan_recommended=True.
    Using issue count or status alone causes spurious replans on datasets where
    near-perfect performance or expected warnings exist (e.g. penguins/species).
    The reflect() function already integrates issue severity and f1 thresholds
    before setting replan_recommended, so defer entirely to that signal.
    """
    return bool(reflection.get("replan_recommended", False))



def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Modify the plan and dataset profile based on reflection.
    
    This is a very basic implementation. Students should make this sophisticated.
    
    Args:
        plan: Current execution plan
        dataset_profile: Current dataset profile
        reflection: Reflection results
    
    Returns:
        Tuple of (modified_plan, modified_profile)
    
    TODO for students:
    - Implement specific strategies for specific issues
    - Add preprocessing steps based on identified problems
    - Modify model selection based on performance patterns
    - Adjust hyperparameters
    - Try ensemble methods
    - Implement different replan strategies (aggressive, conservative)
    """
    
    # Copy to avoid modifying originals
    new_plan = list(plan)
    new_profile = dict(dataset_profile)
    
    # Basic strategy: add a note
    notes = list(new_profile.get("notes", []))
    notes.append("Replan: adjusting strategy after reflection.")
    new_profile["notes"] = notes
    
    # Get issues from reflection to decide strategy
    issues = reflection.get("issues", [])
    
    # TODO: Implement sophisticated replan strategies:
    # If overfitting detected: add regularization
    if any("overfitting" in issue for issue in issues):
        if "apply_regularization" not in new_plan:
            new_plan.insert(new_plan.index("train_models"), "apply_regularization")
    
    # If imbalance issues: add SMOTE or adjust thresholds
    if any("imbalance" in issue.lower() for issue in issues):
        if "consider_imbalance_strategy" not in new_plan:
            new_plan.insert(new_plan.index("train_models"), "consider_imbalance_strategy")
    
    # If low performance: try ensemble methods
    if any("f1" in issue.lower() for issue in issues):
        if "try_ensemble_methods" not in new_plan:
            new_plan.append("try_ensemble_methods")
    
    # If weak baseline: add feature engineering
    if any("baseline" in issue for issue in issues):
        if "apply_feature_engineering" not in new_plan:
            new_plan.insert(new_plan.index("build_preprocessor"), "apply_feature_engineering")
    
    new_plan.append("replan_attempt")
    
    return new_plan, new_profile



# TODO: Add helper functions for reflection
# def compare_models_statistically(...):
# def analyze_per_class_performance(...):
# def detect_overfitting(...):
# def detect_data_quality_issues(...):
# def prioritize_suggestions(...):
# def generate_explanation(...):
