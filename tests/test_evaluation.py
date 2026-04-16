"""
Tests for tools/evaluation.py

Covers: evaluate_best (classification → confusion matrix saved, regression → no
matrix), write_markdown_report (classification and regression reports written
with correct content).
"""
import os
import numpy as np
import pandas as pd
import pytest

from tools import evaluation as evaluation_module
from tools.evaluation import evaluate_best, write_markdown_report


# ── fake context ──────────────────────────────────────────────────────────────

class FakeCtx:
    run_id = "test_run_001"
    started_at = "2026-01-01T00:00:00Z"
    data_path = "data/test.csv"
    target = "label"


# ── payloads ──────────────────────────────────────────────────────────────────

def cls_payload():
    y_test = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = [0, 1, 0, 1, 0, 1, 0, 0]
    best = {
        "name": "RandomForest",
        "metrics": {
            "model": "RandomForest",
            "accuracy": 0.875,
            "balanced_accuracy": 0.857,
            "f1_macro": 0.857,
            "precision_macro": 0.875,
            "recall_macro": 0.857,
        },
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return {
        "best": best,
        "results": [best],
        "all_metrics": [{"model": "RandomForest", "accuracy": 0.875, "f1_macro": 0.857}],
    }


def reg_payload():
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = [1.1, 2.1, 2.9, 4.0, 4.8]
    return {
        "best": {
            "name": "LinearRegression",
            "metrics": {
                "model": "LinearRegression",
                "r2": 0.99,
                "mae": 0.10,
                "rmse": 0.12,
            },
            "y_test": y_test,
            "y_pred": y_pred,
        },
        "all_metrics": [{"model": "LinearRegression", "r2": 0.99}],
    }


# ── evaluate_best ─────────────────────────────────────────────────────────────

def test_evaluate_best_classification_returns_metrics(tmp_path):
    result = evaluate_best(cls_payload(), str(tmp_path), is_classification=True)
    assert "best_metrics" in result
    assert result["best_metrics"]["model"] == "RandomForest"


def test_evaluate_best_classification_saves_confusion_matrix(tmp_path):
    result = evaluate_best(cls_payload(), str(tmp_path), is_classification=True)
    assert result["confusion_matrix_path"] is not None
    assert os.path.exists(result["confusion_matrix_path"])


def test_evaluate_best_classification_has_report(tmp_path):
    result = evaluate_best(cls_payload(), str(tmp_path), is_classification=True)
    assert result["classification_report"] is not None


def test_evaluate_best_uses_same_label_order_for_confusion_matrix_and_plot(tmp_path, monkeypatch):
    payload = cls_payload()
    payload["best"]["y_test"] = pd.Series([1, 10, 2, 10, 1, 2])
    payload["best"]["y_pred"] = [1, 2, 2, 10, 10, 2]

    recorded: dict = {}

    def fake_confusion_matrix(y_true, y_pred, labels=None):
        recorded["cm_labels"] = list(labels)
        return np.eye(len(labels), dtype=int)

    def fake_plot_confusion_matrix(cm, labels, out_path, title):
        recorded["plot_labels"] = list(labels)

    monkeypatch.setattr(evaluation_module, "confusion_matrix", fake_confusion_matrix)
    monkeypatch.setattr(evaluation_module, "plot_confusion_matrix", fake_plot_confusion_matrix)

    evaluate_best(payload, str(tmp_path), is_classification=True)

    assert recorded["cm_labels"] == [1, 2, 10]
    assert recorded["plot_labels"] == ["1", "2", "10"]


def test_evaluate_best_regression_no_confusion_matrix(tmp_path):
    result = evaluate_best(reg_payload(), str(tmp_path), is_classification=False)
    assert result["confusion_matrix_path"] is None


def test_evaluate_best_regression_no_cls_report(tmp_path):
    result = evaluate_best(reg_payload(), str(tmp_path), is_classification=False)
    assert result["classification_report"] is None


def test_evaluate_best_regression_returns_metrics(tmp_path):
    result = evaluate_best(reg_payload(), str(tmp_path), is_classification=False)
    assert result["best_metrics"]["model"] == "LinearRegression"


# ── write_markdown_report ─────────────────────────────────────────────────────

def _cls_profile():
    return {
        "shape": {"rows": 100, "cols": 5},
        "is_classification": True,
        "imbalance_ratio": 1.0,
        "target": "label",
        "schema": {
            "a": "continuous",
            "b": "ordinal",
            "c": "categorical",
            "label": "ordinal",
        },
        "feature_types": {
            "numeric": {"ordinal": ["b"], "continuous": ["a"]},
            "categorical": {"binary": [], "multiclass": ["c"]},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        "notes": [],
    }


def _reg_profile():
    return {
        "shape": {"rows": 100, "cols": 3},
        "is_classification": False,
        "imbalance_ratio": None,
        "target": "label",
        "schema": {
            "a": "ordinal",
            "b": "continuous",
            "label": "continuous",
        },
        "feature_types": {
            "numeric": {"ordinal": ["a"], "continuous": ["b"]},
            "categorical": {"binary": [], "multiclass": []},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        "notes": ["Regression target detected."],
    }


def test_write_markdown_report_creates_file(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_12345",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"suggestions": ["Try regularization"]},
    )
    assert os.path.exists(out)


def test_write_markdown_report_classification_contains_accuracy(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_12345",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"suggestions": []},
    )
    content = open(out).read()
    assert "Accuracy" in content
    assert "RandomForest" in content


def test_write_markdown_report_uses_confidence_aware_summary_language(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_summary",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"status": "ok", "issues": [], "suggestions": [], "replan_recommended": False, "training_warnings": []},
    )
    content = open(out).read()
    assert "strongest model on the held-out split" in content
    assert "confirmed across additional splits or cross-validation" in content


def test_write_markdown_report_surfaces_cross_validation_results(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    eval_result["cross_validation"] = {
        "enabled": True,
        "reason": "",
        "n_splits": 5,
        "models": [
            {
                "model": "RandomForest",
                "balanced_accuracy_mean": 0.812,
                "balanced_accuracy_std": 0.021,
                "f1_macro_mean": 0.804,
                "f1_macro_std": 0.018,
            }
        ],
        "warnings": [],
    }
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_cv",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models", "evaluate", "validate_with_cross_validation", "reflect"],
        eval_payload=eval_result,
        reflection={"status": "ok", "issues": [], "suggestions": [], "replan_recommended": False, "training_warnings": []},
    )
    content = open(out).read()
    assert "### Cross-Validation Check" in content
    assert "Balanced Acc (mean +/- std)" in content
    assert "0.812 +/- 0.021" in content


def test_write_markdown_report_uses_cross_validation_in_caution_summary(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    eval_result["cross_validation"] = {
        "enabled": True,
        "reason": "",
        "n_splits": 5,
        "models": [
            {
                "model": "RandomForest",
                "balanced_accuracy_mean": 0.650,
                "balanced_accuracy_std": 0.120,
                "f1_macro_mean": 0.640,
                "f1_macro_std": 0.110,
            }
        ],
        "warnings": [],
    }
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_cv_caution",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models", "evaluate", "validate_with_cross_validation", "reflect"],
        eval_payload=eval_result,
        reflection={"status": "ok", "issues": [], "suggestions": [], "replan_recommended": False, "training_warnings": []},
    )
    content = open(out).read()
    assert "Use with caution" in content
    assert "held-out split may be optimistic" in content


def test_write_markdown_report_regression_contains_r2(tmp_path):
    out = str(tmp_path / "report.md")
    payload = reg_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=False)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_99999",
        dataset_profile=_reg_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"suggestions": []},
    )
    content = open(out).read()
    assert "R²" in content


def test_write_markdown_report_uses_schema_feature_types(tmp_path):
    out = str(tmp_path / "report.md")
    payload = reg_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=False)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_schema",
        dataset_profile=_reg_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"suggestions": []},
    )
    content = open(out).read()
    assert "1 ordinal (a), 1 continuous (b)" in content


def test_write_markdown_report_contains_run_id(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_abc",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset"],
        eval_payload=eval_result,
        reflection={"suggestions": []},
    )
    assert "test_run_001" in open(out).read()


def test_write_markdown_report_shows_reflection_suggestions(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_abc",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset"],
        eval_payload=eval_result,
        reflection={"suggestions": ["Add more data", "Try feature engineering"]},
    )
    content = open(out).read()
    assert "Add more data" in content


def test_write_markdown_report_includes_reflection_status_issues_and_warnings(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_warn",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "build_preprocessor", "train_models", "evaluate", "reflect", "write_report"],
        eval_payload=eval_result,
        reflection={
            "status": "needs_attention",
            "issues": ["Best model only marginally beats baseline."],
            "suggestions": ["Try stronger feature engineering."],
            "replan_recommended": True,
            "training_warnings": ["RuntimeWarning: overflow encountered in matmul"],
        },
    )
    content = open(out).read()
    assert "Needs Attention" in content
    assert "Best model only marginally beats baseline." in content
    assert "RuntimeWarning: overflow encountered in matmul" in content


def test_write_markdown_report_shows_reliable_result_verdict(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_reliable",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"status": "ok", "issues": [], "suggestions": [], "replan_recommended": False, "training_warnings": []},
    )
    content = open(out).read()
    assert "Reliable result" in content


def test_write_markdown_report_shows_use_with_caution_verdict(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    payload["best"]["metrics"]["model"] = "DummyMostFrequent"
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_caution",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"status": "needs_attention", "issues": ["Weak performance."], "suggestions": [], "replan_recommended": False, "training_warnings": []},
    )
    content = open(out).read()
    assert "Use with caution" in content
    assert "treated cautiously" in content


def test_write_markdown_report_shows_invalid_due_to_leakage_risk_verdict(tmp_path):
    out = str(tmp_path / "report.md")
    payload = reg_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=False)
    profile = _reg_profile()
    profile["hard_leakage_cols"] = [{"column": "b", "reason": "exact_target_copy", "evidence_level": "hard"}]
    profile["leaky_cols"] = [{"column": "b", "reason": "exact_target_copy", "evidence_level": "hard"}]
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_invalid",
        dataset_profile=profile,
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"status": "ok", "issues": [], "suggestions": [], "replan_recommended": False, "training_warnings": []},
    )
    content = open(out).read()
    assert "Invalid due to leakage risk" in content
    assert "should not be treated as trustworthy" in content
    assert "exact target copy" in content


def test_write_markdown_report_uses_caution_for_soft_leakage_and_review_required(tmp_path):
    out = str(tmp_path / "report.md")
    payload = reg_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=False)
    profile = _reg_profile()
    profile["soft_leakage_cols"] = [{"column": "bmi", "normalised_mi": 1.0, "evidence_level": "soft"}]
    profile["leaky_cols"] = [{"column": "bmi", "normalised_mi": 1.0, "evidence_level": "soft"}]
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_soft_leak",
        dataset_profile=profile,
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={"status": "needs_attention", "issues": ["Profiler flagged soft target-proxy risk in: `bmi` (normalised MI 1.00)."], "suggestions": ["Human review recommended before trusting this result."], "replan_recommended": False, "review_required": True, "training_warnings": []},
    )
    content = open(out).read()
    assert "Use with caution" in content
    assert "soft leakage suspicion" in content
    assert "Human review is recommended" in content
    assert "| Review Required | True |" in content


def test_write_markdown_report_uses_caution_for_suspicious_performance_without_profiler_leakage(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    payload["best"]["metrics"]["accuracy"] = 1.0
    payload["best"]["metrics"]["balanced_accuracy"] = 1.0
    payload["best"]["metrics"]["f1_macro"] = 1.0
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_suspicious",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models"],
        eval_payload=eval_result,
        reflection={
            "status": "needs_attention",
            "issues": ["Near-perfect performance across multiple non-baseline models is suspicious."],
            "suggestions": ["Inspect features for target proxies, leakage, or columns that deterministically map to the target."],
            "replan_recommended": False,
            "training_warnings": [],
        },
    )
    content = open(out).read()
    assert "Use with caution" in content
    assert "Invalid due to leakage risk" not in content


def test_write_markdown_report_filters_internal_replan_step_and_surfaces_quality_risks(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    profile = _cls_profile()
    profile["high_corr_pairs"] = [{"col_a": "a", "col_b": "b", "corr": 0.94, "abs_corr": 0.94, "n": 100, "p_value": 0.0002}]
    profile["high_corr_present"] = True
    profile["notes"] = ["Ordinal-like numeric columns detected."]
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_plan",
        dataset_profile=profile,
        plan=["profile_dataset", "build_preprocessor", "train_models", "evaluate", "reflect", "write_report", "use_ensemble_models", "replan_attempt"],
        eval_payload=eval_result,
        reflection={"status": "ok", "suggestions": [], "issues": [], "replan_recommended": False, "training_warnings": []},
    )
    content = open(out).read()
    assert "replan_attempt" not in content
    assert "High correlation detected between" in content
    assert "n=100" in content
    assert "p<0.001" in content
    assert "Correlation does not imply causation." in content
    assert "A replan attempt was triggered after reflection." in content


def test_write_markdown_report_surfaces_model_comparison_note(tmp_path):
    out = str(tmp_path / "report.md")
    payload = cls_payload()
    eval_result = evaluate_best(payload, str(tmp_path), is_classification=True)
    write_markdown_report(
        out_path=out, ctx=FakeCtx(), fingerprint="fp_sig",
        dataset_profile=_cls_profile(),
        plan=["profile_dataset", "train_models", "evaluate", "validate_with_cross_validation", "reflect"],
        eval_payload=eval_result,
        reflection={
            "status": "ok",
            "issues": [],
            "suggestions": [],
            "replan_recommended": False,
            "training_warnings": [],
            "significance_test": {
                "model_a": "RandomForest",
                "model_b": "LogisticRegression",
                "p_value": 0.012,
                "significant": True,
                "note": "`RandomForest` is significantly better than `LogisticRegression` across CV folds (paired t-test: p=0.012).",
            },
        },
    )
    content = open(out).read()
    assert "### Model Comparison Check" in content
    assert "significantly better than" in content
