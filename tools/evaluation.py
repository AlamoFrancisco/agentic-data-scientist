"""
Evaluation Tools

Metrics computation, reporting, and run-verdict logic.

Implemented:
- evaluate_best: computes classification (accuracy, balanced accuracy, macro F1/precision/recall)
  and regression (R², MAE, RMSE) metrics; saves confusion matrix or predicted-vs-actual plot;
  per_class_f1 added to every entry in all_metrics (classification) for consistent schema;
  saves feature importance bar chart (tree-based models) and per-class F1 bar chart (classification)
- derive_run_verdict: classifies each run as "Reliable result", "Use with caution", or
  "Invalid due to leakage risk" based on hard leakage evidence, reflection issues, and CV stability
- write_markdown_report: full human-readable run summary including dataset profile, adaptive
  plan, metrics table, cross-validation section, reflection issues, and confidence-aware summary;
  Artefacts section includes confusion matrix, predicted-vs-actual, feature importance, per-class F1
- cross_validation_section: tabulates per-model fold means and standard deviations
"""

import os
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels

from config import (
    CV_GAP_THRESHOLD_CLS,
    CV_GAP_THRESHOLD_REG,
    CV_STD_THRESHOLD_CLS,
    CV_STD_THRESHOLD_REG,
    F1_THRESHOLD_BALANCED,
    FEATURE_IMPORTANCE_THRESHOLD,
    HIGH_CORR_THRESHOLD,
)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(y_test: Any, y_pred: Any, out_path: str, title: str) -> None:
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")
    # Diagonal line — perfect predictions lie on this
    min_val = min(float(min(y_test)), float(min(y_pred)))
    max_val = max(float(max(y_test)), float(max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipeline: Any, out_path: str, title: str, top_n: int = 15) -> Optional[str]:
    """
    Horizontal bar chart of the top-N feature importances from the best model.
    Returns the saved path, or None if the model does not support feature_importances_.
    """
    try:
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_
    except (AttributeError, KeyError):
        return None

    try:
        preprocessor = pipeline.named_steps["preprocess"]
        raw_names = list(preprocessor.get_feature_names_out())
        # Strip transformer prefix: "cont__alcohol" → "alcohol", "cat__sex_Male" → "sex_Male"
        feature_names = [n.split("__", 1)[-1] for n in raw_names]
    except Exception:
        feature_names = [f"f{i}" for i in range(len(importances))]

    n = min(top_n, len(importances))
    indices = np.argsort(importances)[-n:]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.45)))
    colors = ["#d95f02" if v >= FEATURE_IMPORTANCE_THRESHOLD else "#1b9e77" for v in values]
    ax.barh(range(n), values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_per_class_f1(per_class_f1: Dict[str, float], out_path: str, title: str) -> str:
    """
    Horizontal bar chart of per-class F1 scores for the best model.
    Bars below F1_THRESHOLD_BALANCED are shown in red to highlight weak classes.
    """
    classes = list(per_class_f1.keys())
    scores = [per_class_f1[c] for c in classes]

    fig, ax = plt.subplots(figsize=(6, max(3, len(classes) * 0.55)))
    colors = ["#d62728" if s < F1_THRESHOLD_BALANCED else "#2ca02c" for s in scores]
    ax.barh(classes, scores, color=colors)
    ax.axvline(x=F1_THRESHOLD_BALANCED, color="grey", linestyle="--", linewidth=0.8, label="F1_THRESHOLD_BALANCED threshold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def _ordered_class_labels(y_true: Any, y_pred: Any) -> List[Any]:
    """Return one canonical class order for both metrics and plotting."""
    return list(unique_labels(y_true, y_pred))


def _humanize_plan_step(step: str) -> Optional[str]:
    if step == "replan_attempt":
        return None
    if step.startswith("prioritize_model:"):
        _, _, model_name = step.partition(":")
        return f"Prioritize `{model_name}` based on prior successful runs."

    mapping = {
        "profile_dataset": "Profile the dataset and detect risk signals.",
        "build_preprocessor": "Build the preprocessing pipeline.",
        "select_models": "Select candidate models.",
        "train_models": "Train candidate models.",
        "evaluate": "Evaluate model performance.",
        "validate_with_cross_validation": "Validate the strongest candidate models with cross-validation.",
        "reflect": "Review the run and diagnose issues.",
        "write_report": "Generate the final report.",
        "consider_imbalance_strategy": "Use an imbalance-aware training strategy.",
        "apply_regularization": "Bias toward stronger regularization for a smaller dataset.",
        "handle_severe_missing_data": "Use robust preprocessing for higher missingness.",
        "apply_target_encoding": "Use target encoding for high-cardinality categoricals.",
        "apply_feature_engineering": "Add feature engineering before training.",
        "apply_robust_scaling": "Use robust scaling because feature scales differ substantially.",
        "handle_outliers": "Use outlier-aware preprocessing.",
        "drop_correlated_features": "Drop highly correlated features to reduce redundancy.",
        "drop_leaky_features": "Drop suspected leaky features before training.",
        "use_simple_models_only": "Restrict the search to simpler models.",
        "use_ensemble_models": "Favor ensemble models for a larger dataset.",
        "tune_hyperparameters": "Tune the best model's hyperparameters with randomized search.",
    }
    return mapping.get(step, step.replace("_", " ").capitalize() + ".")


def _plan_sections(plan: List[str]) -> Dict[str, Any]:
    core_step_ids = {
        "profile_dataset",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "validate_with_cross_validation",
        "reflect",
        "write_report",
    }

    core_steps: List[str] = []
    adaptive_steps: List[str] = []
    replan_attempted = False

    for step in plan:
        if step == "replan_attempt":
            replan_attempted = True
            continue
        human = _humanize_plan_step(step)
        if not human:
            continue
        if step in core_step_ids:
            if human not in core_steps:
                core_steps.append(human)
        else:
            if human not in adaptive_steps:
                adaptive_steps.append(human)

    return {
        "core": core_steps,
        "adaptive": adaptive_steps,
        "replan_attempted": replan_attempted,
    }


def _report_data_quality(profile: Dict[str, Any]) -> List[str]:
    dup = profile.get("duplicate_count", 0)
    near_const = profile.get("near_constant_cols", [])
    outlier_cols = profile.get("outlier_cols", [])
    missing_pct = profile.get("missing_pct", {})
    hard_leakage = profile.get("hard_leakage_cols", [])
    soft_leakage = profile.get("soft_leakage_cols", [])

    lines: List[str] = []

    if dup > 0:
        lines.append(f"Removed **{dup}** duplicate rows before training.")

    top_missing = [
        (col, pct) for col, pct in sorted(missing_pct.items(), key=lambda item: item[1], reverse=True)
        if pct > 0
    ][:3]
    if top_missing:
        missing_text = ", ".join(f"`{col}` {pct:.1f}%" for col, pct in top_missing)
        lines.append(f"Highest missingness: {missing_text}.")

    if near_const:
        lines.append(f"Excluded **{len(near_const)}** near-constant column(s): `{'`, `'.join(near_const[:5])}`.")

    if outlier_cols:
        lines.append(f"Outlier-heavy columns (>5% IQR): `{'`, `'.join(outlier_cols[:5])}`.")

    strongest_corr = _strongest_high_corr_pair(profile)
    if strongest_corr:
        corr_text = (
            "High correlation detected between "
            f"`{strongest_corr['col_a']}` and `{strongest_corr['col_b']}` "
            f"(r={strongest_corr.get('corr', 0.0):.2f}"
        )
        if strongest_corr.get("n") is not None:
            corr_text += f", n={strongest_corr['n']}"
        p_value = strongest_corr.get("p_value")
        if p_value is not None:
            corr_text += ", p<0.001" if p_value < 0.001 else f", p={p_value:.3f}"
        corr_text += "). Correlation does not imply causation."
        lines.append(corr_text)

    if profile.get("scale_mismatch"):
        lines.append(
            f"Large scale mismatch detected across numeric features "
            f"(range ratio {profile.get('scale_range_ratio', 'N/A')}x)."
        )

    if hard_leakage:
        evidence = []
        for item in hard_leakage[:5]:
            column = item.get("column")
            if not column:
                continue
            reason = item.get("reason")
            if reason == "exact_target_copy":
                evidence.append(f"`{column}` (exact target copy)")
            elif reason == "deterministic_target_mapping":
                evidence.append(f"`{column}` (deterministic target mapping)")
            else:
                evidence.append(f"`{column}`")
        flagged = ", ".join(evidence)
        lines.append(f"Hard leakage evidence flagged for: {flagged}.")
    elif soft_leakage:
        evidence = []
        for item in soft_leakage[:5]:
            column = item.get("column")
            if not column:
                continue
            norm_mi = item.get("normalised_mi")
            if norm_mi is not None:
                evidence.append(f"`{column}` (normalised MI {float(norm_mi):.2f})")
            else:
                evidence.append(f"`{column}`")
        flagged = ", ".join(evidence)
        lines.append(f"Potential leakage flagged for: {flagged}. Human review recommended before trusting the result.")

    if not lines:
        lines.append("No major data quality risks were detected by the profiler.")

    return lines


def _report_profiler_notes(profile: Dict[str, Any], replan_attempted: bool) -> List[str]:
    notes = [str(note) for note in profile.get("notes", [])]
    report_notes = [note for note in notes if not note.startswith("Replan:")]
    if replan_attempted:
        report_notes.append("A replan attempt was triggered after reflection.")
    return report_notes


def _strongest_high_corr_pair(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return next(
        (pair for pair in profile.get("high_corr_pairs", []) if pair.get("abs_corr", 0.0) >= HIGH_CORR_THRESHOLD),
        None,
    )


def _contains_keyword(items: List[str], keywords: List[str]) -> bool:
    lowered = " ".join(str(item).lower() for item in items)
    return any(keyword in lowered for keyword in keywords)


def _cv_alert(eval_payload: Dict[str, Any]) -> Optional[str]:
    cv_payload = eval_payload.get("cross_validation") or {}
    if not cv_payload.get("enabled"):
        return None

    best = eval_payload.get("best_metrics", {})
    best_model = best.get("model")
    cv_models = cv_payload.get("models", [])
    cv_entry = next((item for item in cv_models if item.get("model") == best_model), None)
    if not cv_entry:
        return None

    if "balanced_accuracy_mean" in cv_entry:
        gap = abs(float(best.get("balanced_accuracy", 0.0)) - float(cv_entry.get("balanced_accuracy_mean", 0.0)))
        std = float(cv_entry.get("balanced_accuracy_std", 0.0))
        if gap > CV_GAP_THRESHOLD_CLS:
            return "Held-out performance differs materially from the cross-validation estimate."
        if std > CV_STD_THRESHOLD_CLS:
            return "Cross-validation performance is unstable across folds."
        return None

    gap = abs(float(best.get("r2", 0.0)) - float(cv_entry.get("r2_mean", 0.0)))
    std = float(cv_entry.get("r2_std", 0.0))
    if gap > CV_GAP_THRESHOLD_REG:
        return "Held-out regression performance differs materially from the cross-validation estimate."
    if std > CV_STD_THRESHOLD_REG:
        return "Cross-validation regression performance is unstable across folds."
    return None


def derive_run_verdict(
    dataset_profile: Dict[str, Any],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Dict[str, str]:
    best = eval_payload.get("best_metrics", {})
    training_warnings = reflection.get("training_warnings", []) if reflection else []
    best_model = str(best.get("model", ""))
    hard_leakage = dataset_profile.get("hard_leakage_cols", [])
    soft_leakage = dataset_profile.get("soft_leakage_cols", [])

    if hard_leakage and not dataset_profile.get("drop_leaky"):
        evidence = []
        for item in hard_leakage[:5]:
            column = item.get("column")
            if not column:
                continue
            reason = item.get("reason")
            if reason == "exact_target_copy":
                evidence.append(f"`{column}` (exact target copy)")
            elif reason == "deterministic_target_mapping":
                evidence.append(f"`{column}` (deterministic target mapping)")
            else:
                evidence.append(f"`{column}`")
        detail = "The profiler found hard target-leakage evidence."
        if evidence:
            detail = (
                "The profiler found hard target-leakage evidence in "
                + ", ".join(evidence)
                + "."
            )
        return {
            "label": "Invalid due to leakage risk",
            "detail": detail,
        }

    if soft_leakage and not dataset_profile.get("drop_leaky"):
        evidence = []
        for item in soft_leakage[:5]:
            column = item.get("column")
            if not column:
                continue
            norm_mi = item.get("normalised_mi")
            if norm_mi is not None:
                evidence.append(f"`{column}` (normalised MI {float(norm_mi):.2f})")
            else:
                evidence.append(f"`{column}`")
        detail = "The profiler raised soft leakage suspicion. Human review is recommended before trusting this result."
        if evidence:
            detail = (
                "The profiler raised soft leakage suspicion in "
                + ", ".join(evidence)
                + ". Human review is recommended before trusting this result."
            )
        return {
            "label": "Use with caution",
            "detail": detail,
        }

    cv_warning = _cv_alert(eval_payload)
    if cv_warning:
        return {
            "label": "Use with caution",
            "detail": f"{cv_warning} Human review is recommended before trusting this result.",
        }

    if (
        reflection.get("review_required", False)
        or reflection.get("status") == "needs_attention"
        or reflection.get("replan_recommended", False)
        or "Dummy" in best_model
        or bool(training_warnings)
    ):
        detail = "The run completed, but the reflection step or training process raised signals that warrant follow-up."
        if reflection.get("review_required", False):
            detail = "The run completed, but a human should review the result before it is trusted."
        return {
            "label": "Use with caution",
            "detail": detail,
        }

    return {
        "label": "Reliable result",
        "detail": "No major leakage, instability, or reflection alerts were raised for this run.",
    }


def _confidence_aware_summary(
    best: Dict[str, Any],
    dataset_profile: Dict[str, Any],
    verdict: Dict[str, str],
    eval_payload: Optional[Dict[str, Any]] = None,
) -> str:
    is_cls = dataset_profile.get("is_classification")
    model_name = best.get("model", "Unknown")
    cv_payload = (eval_payload or {}).get("cross_validation") or {}

    if is_cls:
        key_metric = (
            f"balanced accuracy of {best.get('balanced_accuracy', 0):.1%} "
            f"and macro F1 of {best.get('f1_macro', 0):.1%}"
        )
    else:
        key_metric = (
            f"R² of {best.get('r2', 0):.3f} "
            f"(explains {best.get('r2', 0):.1%} of variance)"
        )

    summary = (
        f"On this run, the strongest model on the held-out split was **{model_name}** "
        f"with a {key_metric}."
    )

    cv_entry = None
    if cv_payload.get("enabled"):
        cv_entry = next(
            (item for item in cv_payload.get("models", []) if item.get("model") == model_name),
            None,
        )

    if verdict["label"] == "Reliable result":
        if cv_entry:
            metric_name = "balanced accuracy" if is_cls else "R²"
            mean_value = cv_entry.get("balanced_accuracy_mean" if is_cls else "r2_mean", 0.0)
            std_value = cv_entry.get("balanced_accuracy_std" if is_cls else "r2_std", 0.0)
            summary += (
                f" Cross-validation also looked stable for this model "
                f"({metric_name} {mean_value:.3f} +/- {std_value:.3f})."
            )
        else:
            summary += " This looks credible for this split, but it should still be confirmed across additional splits or cross-validation."
    elif verdict["label"] == "Use with caution":
        if cv_entry:
            metric_name = "balanced accuracy" if is_cls else "R²"
            mean_value = cv_entry.get("balanced_accuracy_mean" if is_cls else "r2_mean", 0.0)
            std_value = cv_entry.get("balanced_accuracy_std" if is_cls else "r2_std", 0.0)
            summary += (
                f" Cross-validation was weaker or less stable "
                f"({metric_name} {mean_value:.3f} +/- {std_value:.3f}), so the held-out split may be optimistic."
            )
        else:
            summary += " This result should be treated cautiously because performance may be unstable or sensitive to the current split."
    else:
        summary += " This apparent performance should not be treated as trustworthy because the run shows leakage risk."

    return summary


def _cross_validation_section(eval_payload: Dict[str, Any], is_classification: bool) -> str:
    cv_payload = eval_payload.get("cross_validation") or {}
    if not cv_payload.get("enabled"):
        reason = cv_payload.get("reason") or "Cross-validation was not run."
        return f"- {reason}"

    models = cv_payload.get("models", [])
    if not models:
        return "- Cross-validation was requested, but no model summaries were produced."

    if is_classification:
        header = "| Model | Folds | Balanced Acc (mean +/- std) | Macro F1 (mean +/- std) |"
        sep = "|---|---|---|---|"
        rows = "\n".join(
            f"| {m.get('model')} | {cv_payload.get('n_splits', 0)} | "
            f"{m.get('balanced_accuracy_mean', 0):.3f} +/- {m.get('balanced_accuracy_std', 0):.3f} | "
            f"{m.get('f1_macro_mean', 0):.3f} +/- {m.get('f1_macro_std', 0):.3f} |"
            for m in models
        )
    else:
        header = "| Model | Folds | R² (mean +/- std) | MAE (mean +/- std) | RMSE (mean +/- std) |"
        sep = "|---|---|---|---|---|"
        rows = "\n".join(
            f"| {m.get('model')} | {cv_payload.get('n_splits', 0)} | "
            f"{m.get('r2_mean', 0):.3f} +/- {m.get('r2_std', 0):.3f} | "
            f"{m.get('mae_mean', 0):.3f} +/- {m.get('mae_std', 0):.3f} | "
            f"{m.get('rmse_mean', 0):.3f} +/- {m.get('rmse_std', 0):.3f} |"
            for m in models
        )

    warning_lines = cv_payload.get("warnings", [])
    if warning_lines:
        warning_block = "\n\nCross-validation warnings:\n" + "\n".join(f"- {warning}" for warning in warning_lines)
    else:
        warning_block = ""

    return f"{header}\n{sep}\n{rows}{warning_block}"


def evaluate_best(training_payload: Dict[str, Any], output_dir: str, is_classification: bool = True) -> Dict[str, Any]:
    best = training_payload["best"]
    all_metrics = training_payload["all_metrics"]

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    per_class_f1: Dict[str, float] = {}

    if is_classification:
        # Confusion matrix — classification only
        labels = _ordered_class_labels(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, [str(label) for label in labels], cm_path, f"Confusion Matrix: {best['name']}")
        cls_report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
        cls_report_dict = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)
        per_class_f1 = {
            str(label): float(cls_report_dict.get(str(label), {}).get("f1-score", 0.0))
            for label in labels
        }
        # Per-class F1 bar chart
        pcf_path = plot_per_class_f1(
            per_class_f1,
            os.path.join(output_dir, "per_class_f1.png"),
            f"Per-Class F1: {best['name']}",
        )
        # Add per_class_f1 to every model entry so all_metrics has a consistent schema
        enriched_all_metrics = []
        for result in training_payload["results"]:
            m = dict(result["metrics"])
            r = classification_report(result["y_test"], result["y_pred"], labels=labels, zero_division=0, output_dict=True)
            m["per_class_f1"] = {
                str(label): float(r.get(str(label), {}).get("f1-score", 0.0))
                for label in labels
            }
            enriched_all_metrics.append(m)
    else:
        # Regression — predicted vs actual scatter plot
        cm_path = None
        cls_report = None
        pcf_path = None
        reg_plot_path = os.path.join(output_dir, "predicted_vs_actual.png")
        plot_predicted_vs_actual(y_test, y_pred, reg_plot_path, f"Predicted vs Actual: {best['name']}")
        enriched_all_metrics = list(all_metrics)

    # Feature importance bar chart — tree-based models only; returns None otherwise
    pipeline = best.get("pipeline")
    fi_path = (
        plot_feature_importance(
            pipeline,
            os.path.join(output_dir, "feature_importance.png"),
            f"Feature Importance: {best['name']}",
        )
        if pipeline is not None
        else None
    )

    best_metrics = dict(best["metrics"])
    if per_class_f1:
        best_metrics["per_class_f1"] = per_class_f1

    return {
        "best_metrics": best_metrics,
        "all_metrics": enriched_all_metrics,
        "confusion_matrix_path": cm_path,
        "classification_report": cls_report,
        "regression_plot_path": reg_plot_path if not is_classification else None,
        "feature_importance_path": fi_path,
        "per_class_f1_path": pcf_path if is_classification else None,
    }


def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: List[str],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
) -> None:
    best = eval_payload["best_metrics"]

    def short_list(xs: List[str], n: int = 12) -> str:
        return ", ".join(xs[:n]) + (" ..." if len(xs) > n else "")

    def feature_summary(profile: Dict[str, Any]) -> str:
        feature_types = profile.get("feature_types", {})
        numeric_groups = feature_types.get("numeric", {})
        categorical_groups = feature_types.get("categorical", {})
        if isinstance(numeric_groups, dict) or isinstance(categorical_groups, dict):
            ordinal = numeric_groups.get("ordinal", []) if isinstance(numeric_groups, dict) else []
            continuous = numeric_groups.get("continuous", []) if isinstance(numeric_groups, dict) else []
            binary = categorical_groups.get("binary", []) if isinstance(categorical_groups, dict) else []
            multiclass = categorical_groups.get("multiclass", []) if isinstance(categorical_groups, dict) else []
            text = feature_types.get("text", [])
            datetime_cols = feature_types.get("datetime", [])
            all_missing = feature_types.get("all_missing", [])
            parts = []
            if ordinal:
                parts.append(f"{len(ordinal)} ordinal ({short_list(ordinal)})")
            if continuous:
                parts.append(f"{len(continuous)} continuous ({short_list(continuous)})")
            if binary:
                parts.append(f"{len(binary)} binary categorical ({short_list(binary)})")
            if multiclass:
                parts.append(f"{len(multiclass)} multiclass categorical ({short_list(multiclass)})")
            if text:
                parts.append(f"{len(text)} text ({short_list(text)})")
            if datetime_cols:
                parts.append(f"{len(datetime_cols)} datetime ({short_list(datetime_cols)})")
            if all_missing:
                parts.append(f"{len(all_missing)} all-missing ({short_list(all_missing)})")
            if parts:
                return ", ".join(parts)

        schema = profile.get("schema", {})
        target = profile.get("target", getattr(ctx, "target", None))
        if schema:
            feature_schema = {col: kind for col, kind in schema.items() if col != target}
            ordered_groups = [
                ("ordinal", "ordinal"),
                ("continuous", "continuous"),
                ("categorical", "categorical"),
                ("text", "text"),
                ("boolean", "boolean"),
                ("datetime", "datetime"),
                ("all-missing", "all_missing"),
            ]
            parts = []
            for label, schema_key in ordered_groups:
                cols = [col for col, kind in feature_schema.items() if kind == schema_key]
                if cols:
                    parts.append(f"{len(cols)} {label} ({short_list(cols)})")
            if parts:
                return ", ".join(parts)

        numeric = feature_types.get("numeric", [])
        categorical = feature_types.get("categorical", [])
        return f"{len(numeric)} numeric ({short_list(numeric)}), {len(categorical)} categorical ({short_list(categorical)})"

    plan_sections = _plan_sections(plan)
    quality_lines = _report_data_quality(dataset_profile)
    profiler_notes = _report_profiler_notes(dataset_profile, plan_sections["replan_attempted"])
    reflection_status = str(reflection.get("status", "unknown")).replace("_", " ").title() if reflection else "Unknown"
    reflection_issues = reflection.get("issues", []) if reflection else []
    reflection_suggestions = reflection.get("suggestions", []) if reflection else []
    training_warnings = reflection.get("training_warnings", []) if reflection else []
    sig_test = reflection.get("significance_test") if reflection else None
    significance_row = ""
    significance_section = "- No formal model-comparison test was produced."
    if sig_test:
        sig_label = "Yes" if sig_test["significant"] else "No"
        significance_row = f"\n| Model comparison (paired t-test) | {sig_test['model_a']} vs {sig_test['model_b']}: p={sig_test['p_value']:.3f}, significant={sig_label} |"
        significance_section = f"- {sig_test['note']}"
    verdict = derive_run_verdict(dataset_profile, eval_payload, reflection or {})

    # Confidence-aware summary sentence
    is_cls = dataset_profile.get("is_classification")
    summary = _confidence_aware_summary(best, dataset_profile, verdict, eval_payload)

    # Metrics table for best model
    if is_cls:
        metrics_table = f"""| Metric | Score |
|---|---|
| Accuracy | {best.get("accuracy", 0):.3f} |
| Balanced Accuracy | {best.get("balanced_accuracy", 0):.3f} |
| Macro F1 | {best.get("f1_macro", 0):.3f} |
| Macro Precision | {best.get("precision_macro", 0):.3f} |
| Macro Recall | {best.get("recall_macro", 0):.3f} |"""
    else:
        metrics_table = f"""| Metric | Score |
|---|---|
| R² | {best.get("r2", 0):.3f} |
| MAE | {best.get("mae", 0):.3f} |
| RMSE | {best.get("rmse", 0):.3f} |"""

    # All candidates table
    all_metrics = eval_payload.get("all_metrics", [])
    if is_cls:
        candidates_header = "| Model | Accuracy | Balanced Acc | Macro F1 |"
        candidates_sep    = "|---|---|---|---|"
        candidates_rows   = "\n".join(
            f"| {m.get('model')} | {m.get('accuracy', 0):.3f} | {m.get('balanced_accuracy', 0):.3f} | {m.get('f1_macro', 0):.3f} |"
            for m in all_metrics
        )
    else:
        candidates_header = "| Model | R² | MAE | RMSE |"
        candidates_sep    = "|---|---|---|---|"
        candidates_rows   = "\n".join(
            f"| {m.get('model')} | {m.get('r2', 0):.3f} | {m.get('mae', 0):.3f} | {m.get('rmse', 0):.3f} |"
            for m in all_metrics
        )
    candidates_table = f"{candidates_header}\n{candidates_sep}\n{candidates_rows}"
    cv_section = _cross_validation_section(eval_payload, is_cls)

    quality_section = "\n".join(f"- {line}" for line in quality_lines)
    profiler_notes_section = (
        "\n".join(f"- {note}" for note in profiler_notes)
        if profiler_notes
        else "- No additional profiler notes."
    )
    core_plan_section = (
        "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_sections["core"]))
        if plan_sections["core"]
        else "1. Standard pipeline executed."
    )
    adaptive_plan_section = (
        "\n".join(f"- {step}" for step in plan_sections["adaptive"])
        if plan_sections["adaptive"]
        else "- No adaptive strategy changes were applied."
    )
    reflection_issues_section = (
        "\n".join(f"- {issue}" for issue in reflection_issues)
        if reflection_issues
        else "- No major issues were identified."
    )
    reflection_suggestions_section = (
        "\n".join(f"- {suggestion}" for suggestion in reflection_suggestions)
        if reflection_suggestions
        else "- No follow-up actions were suggested."
    )
    training_warnings_section = (
        "\n".join(f"- {warning}" for warning in training_warnings)
        if training_warnings
        else "- No training warnings were captured."
    )

    md = f"""# _Agentic Data Scientist Report_
    
**Author:** Francisco Antonio Alamo Rios  
**Registration number:** 2508983
---

- **Run ID:** `{ctx.run_id}`
- **Started (UTC):** {ctx.started_at}
- **Dataset:** `{ctx.data_path}`
- **Target:** `{ctx.target}`
- **Fingerprint:** `{fingerprint}`

## Summary
{summary}

## Verdict
> **{verdict["label"]}**
>
> {verdict["detail"]}

## Dataset Profile
| Property | Value |
|---|---|
| Rows | {dataset_profile["shape"]["rows"]} |
| Columns | {dataset_profile["shape"]["cols"]} |
| Task type | {"Classification" if is_cls else "Regression"} |
| Imbalance ratio | {dataset_profile.get("imbalance_ratio") or "N/A"} |

**Features:** {feature_summary(dataset_profile)}

## Data Quality
{quality_section}

## Profiler Notes
{profiler_notes_section}

## Execution Strategy
### Core Pipeline
{core_plan_section}

### Adaptive Decisions
{adaptive_plan_section}

## Results (Best Model)
{metrics_table}

### All Candidates
{candidates_table}

### Cross-Validation Check
{cv_section}

## Reflection
| Field | Value |
|---|---|
| Status | {reflection_status} |
| Replan Recommended | {reflection.get("replan_recommended", False) if reflection else False} |{significance_row}
| Review Required | {reflection.get("review_required", False) if reflection else False} |

### Issues
{reflection_issues_section}

### Suggested Next Steps
{reflection_suggestions_section}

### Model Comparison Check
{significance_section}

### Training Warnings
{training_warnings_section}

## Artefacts
{f"![Confusion Matrix](confusion_matrix.png)" if eval_payload.get("confusion_matrix_path") else ""}
{f"![Predicted vs Actual](predicted_vs_actual.png)" if eval_payload.get("regression_plot_path") else ""}
{f"![Feature Importance](feature_importance.png)" if eval_payload.get("feature_importance_path") else ""}
{f"![Per-Class F1](per_class_f1.png)" if eval_payload.get("per_class_f1_path") else ""}

"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
