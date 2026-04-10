"""
Tests for tools/evaluation.py

Covers: evaluate_best (classification → confusion matrix saved, regression → no
matrix), write_markdown_report (classification and regression reports written
with correct content).
"""
import os
import pandas as pd
import pytest

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
    return {
        "best": {
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
        },
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
        "feature_types": {"numeric": ["a", "b"], "categorical": ["c"]},
        "notes": [],
    }


def _reg_profile():
    return {
        "shape": {"rows": 100, "cols": 3},
        "is_classification": False,
        "imbalance_ratio": None,
        "feature_types": {"numeric": ["a", "b"], "categorical": []},
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
