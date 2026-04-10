import os
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


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


def evaluate_best(training_payload: Dict[str, Any], output_dir: str, is_classification: bool = True) -> Dict[str, Any]:
    best = training_payload["best"]
    all_metrics = training_payload["all_metrics"]

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    if is_classification:
        # Confusion matrix — classification only
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted([str(x) for x in y_test.dropna().unique().tolist()])
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {best['name']}")
        cls_report = classification_report(y_test, y_pred, zero_division=0)
    else:
        # Regression — predicted vs actual scatter plot
        cm_path = None
        cls_report = None
        reg_plot_path = os.path.join(output_dir, "predicted_vs_actual.png")
        plot_predicted_vs_actual(y_test, y_pred, reg_plot_path, f"Predicted vs Actual: {best['name']}")

    return {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "confusion_matrix_path": cm_path,
        "classification_report": cls_report,
        "regression_plot_path": reg_plot_path if not is_classification else None,
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

    numeric = dataset_profile.get("feature_types", {}).get("numeric", [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes = dataset_profile.get("notes", [])

    # Plain English summary sentence
    is_cls = dataset_profile.get("is_classification")
    model_name = best.get("model", "Unknown")
    if is_cls:
        key_metric = f"balanced accuracy of {best.get('balanced_accuracy', 0):.1%} and macro F1 of {best.get('f1_macro', 0):.1%}"
    else:
        key_metric = f"R² of {best.get('r2', 0):.3f} (explains {best.get('r2', 0):.1%} of variance)"
    summary = f"The best model was **{model_name}** with a {key_metric}."

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

    # Data quality summary
    dup = dataset_profile.get("duplicate_count", 0)
    near_const = dataset_profile.get("near_constant_cols", [])
    outlier_cols = dataset_profile.get("outlier_cols", [])
    quality_lines = []
    if dup > 0:
        quality_lines.append(f"- **{dup}** duplicate rows removed before training.")
    if near_const:
        quality_lines.append(f"- **{len(near_const)}** near-constant column(s) excluded: `{'`, `'.join(near_const[:5])}`")
    if outlier_cols:
        quality_lines.append(f"- Outlier-heavy columns (>5% IQR): `{'`, `'.join(outlier_cols[:5])}`")
    if not quality_lines:
        quality_lines.append("- No data quality issues detected.")
    quality_section = "\n".join(quality_lines)

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

## Dataset Profile
| Property | Value |
|---|---|
| Rows | {dataset_profile["shape"]["rows"]} |
| Columns | {dataset_profile["shape"]["cols"]} |
| Task type | {"Classification" if is_cls else "Regression"} |
| Imbalance ratio | {dataset_profile.get("imbalance_ratio") or "N/A"} |

**Features:** {len(numeric)} numeric ({short_list(numeric)}), {len(categorical)} categorical ({short_list(categorical)})

## Data Quality
{quality_section}

## Plan
{chr(10).join([f"{i+1}. `{t}`" for i, t in enumerate(plan)])}

## Results (Best Model)
{metrics_table}

### All Candidates
{candidates_table}

## Reflection
{chr(10).join([f"- {s}" for s in reflection.get("suggestions", [])]) if reflection and reflection.get("suggestions") else "- No issues detected — results look good."}

## Artefacts
{f"![Confusion Matrix](confusion_matrix.png)" if eval_payload.get("confusion_matrix_path") else ""}
{f"![Predicted vs Actual](predicted_vs_actual.png)" if eval_payload.get("regression_plot_path") else ""}

"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
