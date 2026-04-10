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
        # Regression — no confusion matrix
        cm_path = None
        cls_report = None

    return {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "confusion_matrix_path": cm_path,
        "classification_report": cls_report,
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

    # Build metrics section based on task type
    if dataset_profile.get("is_classification"):
        metrics_section = f"""- Accuracy: **{best.get("accuracy", 0):.3f}**
- Balanced accuracy: **{best.get("balanced_accuracy", 0):.3f}**
- Macro F1: **{best.get("f1_macro", 0):.3f}**
- Macro Precision: **{best.get("precision_macro", 0):.3f}**
- Macro Recall: **{best.get("recall_macro", 0):.3f}**"""
    else:
        metrics_section = f"""- R²: **{best.get("r2", 0):.3f}**
- MAE: **{best.get("mae", 0):.3f}**
- RMSE: **{best.get("rmse", 0):.3f}**"""

    md = f"""# Agentic Data Scientist Report

**Run ID:** `{ctx.run_id}`  
**Started (UTC):** {ctx.started_at}  
**Dataset:** `{ctx.data_path}`  
**Target:** `{ctx.target}`  
**Fingerprint:** `{fingerprint}`  

## Dataset Profile
- Rows: **{dataset_profile["shape"]["rows"]}**
- Columns: **{dataset_profile["shape"]["cols"]}**
- Task type: **{"classification" if dataset_profile.get("is_classification") else "regression"}**
- Imbalance ratio: **{dataset_profile.get("imbalance_ratio")}**

**Feature Types**
- Numeric ({len(numeric)}): {short_list(numeric)}
- Categorical ({len(categorical)}): {short_list(categorical)}

**Notes**
{chr(10).join([f"- {n}" for n in notes]) if notes else "- (none)"}

## Plan
{chr(10).join([f"- {t}" for t in plan])}

## Results (Best Model)
**Model:** `{best.get("model")}`

{metrics_section}

Top metrics (all candidates):
```json
{json.dumps(eval_payload.get("all_metrics", []), indent=2)}
```

## Reflection
{chr(10).join([f"- {s}" for s in reflection.get("suggestions", [])]) if reflection else "- (none)"}

# Artefacts
- Confusion matrix: {eval_payload.get("confusion_matrix_path")}

"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
