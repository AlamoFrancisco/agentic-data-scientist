"""
Reflector Agent

The reflector evaluates execution results, identifies issues, and suggests improvements.

Implemented:
- Per-class F1 analysis: flags classes where the model consistently underperforms
- Prioritized suggestions: critical signals (instability, leakage) surface first
- Adaptive F1 threshold: adjusts expectation based on class imbalance
- Imbalance escalation: severe imbalance with weak performance is raised as an issue
- CV consistency checks: flags splits that diverge from cross-validation estimates
- Near-perfect leakage detection: suspicious R²/balanced_accuracy across multiple models
- Statistical significance testing: paired t-test between best and runner-up CV fold
  scores; flags non-significant gaps so the simpler model can be preferred

TODO: Still to add:
- Meta-learning from past reflections (read reflection_status from memory)
"""

from typing import Any, Dict, List, Optional

from scipy.stats import ttest_rel

from config import (
    F1_THRESHOLD_BALANCED,
    F1_THRESHOLD_IMBALANCED,
    F1_THRESHOLD_SEVERE_IMBALANCE,
    IMBALANCE_THRESHOLD,
    IMBALANCE_VERY_SEVERE,
    CV_GAP_THRESHOLD_CLS,
    CV_GAP_THRESHOLD_REG,
    CV_STD_THRESHOLD_CLS,
    CV_STD_THRESHOLD_REG,
    BASELINE_MIN_IMPROVEMENT_CLS,
    BASELINE_MIN_IMPROVEMENT_REG,
    BASELINE_MIN_IMPROVEMENT_SEVERE_IMBALANCE,
    NEAR_PERFECT_THRESHOLD,
    DIVERSITY_GAP_THRESHOLD,
    SIGNIFICANCE_ALPHA,
    R2_LOW_THRESHOLD,
)


def _compare_models_statistically(
    cv_summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Paired t-test between the best and runner-up model's CV fold scores.

    Uses the per-fold primary metric (balanced_accuracy for classification, R²
    for regression) stored in each model's 'fold_scores' list. Returns a result
    dict, or None if there is insufficient data to run the test.
    """
    models = cv_summary.get("models", [])
    if len(models) < 2:
        return None

    best = models[0]
    runner_up = models[1]
    scores_a = best.get("fold_scores", [])
    scores_b = runner_up.get("fold_scores", [])

    if len(scores_a) < 2 or len(scores_b) < 2:
        return None

    try:
        t_stat, p_value = ttest_rel(scores_a, scores_b)
    except Exception:
        return None

    significant = float(p_value) <= SIGNIFICANCE_ALPHA

    if significant:
        note = (
            f"`{best['model']}` is significantly better than `{runner_up['model']}` "
            f"across CV folds (paired t-test: p={p_value:.3f})."
        )
    else:
        note = (
            f"No significant difference between `{best['model']}` and "
            f"`{runner_up['model']}` across CV folds (paired t-test: p={p_value:.3f}). "
            "The simpler model may be equally reliable."
        )

    return {
        "test": "paired t-test",
        "model_a": best["model"],
        "model_b": runner_up["model"],
        "p_value": round(float(p_value), 4),
        "t_statistic": round(float(t_stat), 4),
        "significant": significant,
        "alpha": SIGNIFICANCE_ALPHA,
        "note": note,
    }


def _prioritize_suggestions(suggestions: List[str]) -> List[str]:
    """Order suggestions by estimated impact: critical signals first, general advice last."""
    high_priority_keywords = ["instability", "leakage", "scaling", "overflow", "suspicious", "proxy"]
    medium_priority_keywords = ["regularization", "class weight", "threshold", "ensemble", "feature", "cross-validation"]

    def _priority(s: str) -> int:
        lower = s.lower()
        if any(kw in lower for kw in high_priority_keywords):
            return 0
        if any(kw in lower for kw in medium_priority_keywords):
            return 1
        return 2

    return sorted(suggestions, key=_priority)


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
    training_warnings: Optional[List[str]] = None,
    cv_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze results and generate reflection with issues and suggestions.

    Args:
        dataset_profile: Dataset characteristics
        evaluation: Best model's metrics (includes per_class_f1 for classification)
        all_metrics: Metrics for all trained models
        training_warnings: Warnings captured during model training
        cv_summary: Cross-validation results, if available

    Returns:
        Dictionary with:
            - status: str ("ok" or "needs_attention")
            - best_model: str (model name)
            - issues: List[str] (identified problems)
            - suggestions: List[str] (improvement recommendations, prioritized)
            - replan_recommended: bool (should we replan?)
            - review_required: bool (should a human review the result before trusting it?)
            - training_warnings: List[str] (warnings passed through)
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
    review_required = False
    hard_leakage = dataset_profile.get("hard_leakage_cols", [])
    soft_leakage = dataset_profile.get("soft_leakage_cols", [])

    if hard_leakage and not dataset_profile.get("drop_leaky"):
        evidence = []
        for item in hard_leakage[:5]:
            if not item.get("column"):
                continue
            reason = item.get("reason")
            if reason == "exact_target_copy":
                evidence.append(f"`{item['column']}` (exact target copy)")
            elif reason == "deterministic_target_mapping":
                evidence.append(f"`{item['column']}` (deterministic target mapping)")
            else:
                evidence.append(f"`{item['column']}`")
        flagged = ", ".join(evidence)
        issues.append(f"Profiler found hard target-leakage evidence in: {flagged}.")
        suggestions.append(
            "Human review required before trusting this run. Remove the flagged columns "
            "or confirm they are unavailable at prediction time."
        )
        review_required = True
    elif hard_leakage and dataset_profile.get("drop_leaky"):
        evidence = []
        for item in hard_leakage[:5]:
            if not item.get("column"):
                continue
            reason = item.get("reason")
            if reason == "exact_target_copy":
                evidence.append(f"`{item['column']}` (exact target copy)")
            elif reason == "deterministic_target_mapping":
                evidence.append(f"`{item['column']}` (deterministic target mapping)")
            else:
                evidence.append(f"`{item['column']}`")
        flagged = ", ".join(evidence)
        suggestions.append(
            f"Hard-leakage columns ({flagged}) were detected and dropped before training. "
            "Verify the remaining features are genuinely predictive."
        )
    if soft_leakage and not dataset_profile.get("drop_leaky"):
        evidence = []
        for item in soft_leakage[:5]:
            if not item.get("column"):
                continue
            norm_mi = item.get("normalised_mi")
            if norm_mi is not None:
                evidence.append(f"`{item['column']}` (normalised MI {float(norm_mi):.2f})")
            else:
                evidence.append(f"`{item['column']}`")
        flagged = ", ".join(evidence)
        issues.append(f"Profiler flagged soft target-proxy risk in: {flagged}.")
        suggestions.append(
            "Human review recommended before trusting this result. "
            "These are soft profiler signals, not proof of leakage."
        )
        review_required = True

    def check_cv_consistency(split_metric: float, metric_name: str) -> None:
        nonlocal cv_concerns, review_required
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
        unstable_std = CV_STD_THRESHOLD_CLS if is_classification else CV_STD_THRESHOLD_REG
        unstable_gap = CV_GAP_THRESHOLD_CLS if is_classification else CV_GAP_THRESHOLD_REG

        if gap > unstable_gap:
            cv_concerns = True
            review_required = True
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
            review_required = True
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
            if improvement < BASELINE_MIN_IMPROVEMENT_REG:
                issues.append(
                    f"Best model R² only {improvement:.3f} better than baseline. "
                    "Weak signal or pipeline issues."
                )
                suggestions.append(
                    "Check for target leakage, verify target quality, "
                    "or improve feature engineering."
                )
        if r2 < R2_LOW_THRESHOLD:
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
            if r2 < R2_LOW_THRESHOLD:
                review_required = True
                issues.append(
                    "Numerical instability warnings present alongside low R² — "
                    "scaling issues may be degrading model performance."
                )

        # Near-perfect R² across multiple real models is suspicious, but not proof of leakage
        real_reg_models = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
        near_perfect_reg = [m for m in real_reg_models if float(m.get("r2", 0.0)) >= NEAR_PERFECT_THRESHOLD]
        if len(near_perfect_reg) >= 2:
            review_required = True
            issues.append(
                "Near-perfect R² across multiple non-baseline models is suspicious, but not proof of leakage."
            )
            suggestions.append(
                "Inspect features for target proxies or columns that deterministically "
                "derive the target (e.g. multiplicative combinations)."
            )

        check_cv_consistency(r2, "r2")

        significance = None
        if cv_summary and cv_summary.get("enabled"):
            significance = _compare_models_statistically(cv_summary)
            if significance and not significance["significant"]:
                suggestions.append(
                    f"No significant difference between `{significance['model_a']}` and "
                    f"`{significance['model_b']}` (p={significance['p_value']:.3f}) — "
                    "consider the simpler model."
                )

        return {
            "status": "needs_attention" if issues else "ok",
            "best_model": best_model,
            "issues": issues,
            "suggestions": _prioritize_suggestions(suggestions),
            "replan_recommended": bool(issues and (r2 < R2_LOW_THRESHOLD or cv_concerns)),
            "review_required": review_required,
            "training_warnings": training_warnings or [],
            "significance_test": significance,
        }

    # Classification analysis below
    # Basic comparison with dummy baseline
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)

    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba

        if improvement < BASELINE_MIN_IMPROVEMENT_CLS:
            issues.append(
                f"Best model only {improvement:.3f} better than baseline. "
                "Weak signal or pipeline issues."
            )
            suggestions.append(
                "Check for target leakage, verify target quality, "
                "or improve feature engineering."
            )
    
    # Check for overfitting: high balanced accuracy but poor F1 suggests the model
    # is learning the majority class well but failing on minority classes
    if bal_acc > 0.90 and f1_macro < 0.70:
        issues.append("High balanced accuracy but low F1 macro suggests overfitting.")
        suggestions.append(
            "Try regularization, reduce model complexity, or add more data."
        )

    # Adaptive F1 threshold: lower expectation when class imbalance makes minority
    # classes genuinely hard — a flat 0.60 threshold penalises imbalanced datasets unfairly
    if imb >= IMBALANCE_VERY_SEVERE:
        f1_threshold = F1_THRESHOLD_SEVERE_IMBALANCE
    elif imb >= IMBALANCE_THRESHOLD:
        f1_threshold = F1_THRESHOLD_IMBALANCED
    else:
        f1_threshold = F1_THRESHOLD_BALANCED

    if f1_macro < f1_threshold:
        issues.append(f"Macro F1 ({f1_macro:.3f}) is below the expected threshold ({f1_threshold:.2f}).")
        suggestions.append(
            "Try different models, tune hyperparameters, "
            "or improve preprocessing."
        )

    # Imbalance-specific analysis: escalate to an issue when imbalance is severe
    # and the model is not beating the dummy by a meaningful margin
    if imb >= IMBALANCE_VERY_SEVERE:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0)) if dummy else 0.0
        if bal_acc - dummy_ba < BASELINE_MIN_IMPROVEMENT_SEVERE_IMBALANCE:
            issues.append(
                f"Severe class imbalance (ratio {imb:.1f}x) and weak improvement over baseline "
                f"({bal_acc - dummy_ba:.3f}) — the model may be ignoring minority classes."
            )
        suggestions.append(
            "Severe imbalance detected: consider class_weight='balanced', "
            "threshold tuning, or oversampling the minority class."
        )
    elif imb >= IMBALANCE_THRESHOLD:
        suggestions.append(
            "Imbalance detected: consider class_weight='balanced' or threshold tuning."
        )

    # Suspiciously strong performance across multiple real models can indicate
    # leakage, duplicate target encodings, or an overly trivial target, but is
    # not proof by itself.
    real_models = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
    near_perfect_models = [
        m for m in real_models
        if float(m.get("balanced_accuracy", 0.0)) >= NEAR_PERFECT_THRESHOLD
        and float(m.get("f1_macro", 0.0)) >= NEAR_PERFECT_THRESHOLD
    ]
    if len(near_perfect_models) >= 2:
        review_required = True
        issues.append(
            "Near-perfect performance across multiple non-baseline models is suspicious, but not proof of leakage."
        )
        suggestions.append(
            "Inspect features for target proxies, leakage, or columns that deterministically map to the target."
        )

    # Per-class analysis: flag any class the model consistently struggles with
    per_class_f1 = evaluation.get("per_class_f1", {})
    if per_class_f1:
        weak_classes = [(cls, f1) for cls, f1 in per_class_f1.items() if f1 < 0.4]
        if weak_classes:
            worst_cls, worst_f1 = min(weak_classes, key=lambda x: x[1])
            issues.append(
                f"Class '{worst_cls}' has very low F1 ({worst_f1:.3f}) — "
                "model struggles to identify this class."
            )
            suggestions.append(
                f"Class '{worst_cls}' is underperforming. Consider collecting more samples, "
                "adjusting class weights, or reviewing its feature overlap with other classes."
            )

    check_cv_consistency(bal_acc, "balanced_accuracy")

    # Check for model diversity
    f1_scores = [m.get("f1_macro", 0.0) for m in all_metrics]
    if len(f1_scores) > 1:
        score_range = max(f1_scores) - min(f1_scores)
        if score_range < DIVERSITY_GAP_THRESHOLD:
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
        if f1_macro < F1_THRESHOLD_BALANCED:
            review_required = True
            issues.append(
                "Numerical instability warnings present alongside weak F1 — "
                "scaling issues may be degrading model performance."
            )

    # Determine status
    status = "needs_attention" if issues else "ok"

    replan_recommended = bool(issues and (f1_macro < f1_threshold or cv_concerns))

    significance = None
    if cv_summary and cv_summary.get("enabled"):
        significance = _compare_models_statistically(cv_summary)
        if significance and not significance["significant"]:
            suggestions.append(
                f"No significant difference between `{significance['model_a']}` and "
                f"`{significance['model_b']}` (p={significance['p_value']:.3f}) — "
                "consider the simpler model."
            )

    return {
        "status": status,
        "best_model": best_model,
        "issues": issues,
        "suggestions": _prioritize_suggestions(suggestions),
        "replan_recommended": replan_recommended,
        "review_required": review_required,
        "training_warnings": training_warnings or [],
        "significance_test": significance,
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



