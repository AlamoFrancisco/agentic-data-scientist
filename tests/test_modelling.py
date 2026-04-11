"""
Tests for tools/modelling.py

Covers: build_preprocessor (numeric-only, with categoricals, high-cardinality drop),
select_models (classification vs regression, size thresholds),
train_models (classification metrics, regression metrics, invalid target).
"""
import numpy as np
import pandas as pd
import pytest

from tools.modelling import build_preprocessor, select_models, train_models


# ── fixtures / helpers ────────────────────────────────────────────────────────

def make_cls_df(n=120):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "num1":   rng.standard_normal(n),
        "num2":   rng.standard_normal(n),
        "cat1":   rng.choice(["A", "B", "C"], n),
        "target": rng.choice([0, 1], n),
    })


def make_reg_df(n=120):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "num1":   rng.standard_normal(n),
        "num2":   rng.standard_normal(n),
        "target": rng.standard_normal(n) * 10,
    })


def make_profile(df, target="target", is_classification=True):
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "feature_types": {
            "numeric": {
                "ordinal": [],
                "continuous": numeric_cols,
            },
            "categorical": {
                "binary": [c for c in categorical_cols if X[c].nunique(dropna=True) <= 2],
                "multiclass": [c for c in categorical_cols if X[c].nunique(dropna=True) > 2],
            },
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        "n_unique_by_col": {c: int(df[c].nunique()) for c in df.columns},
        "imbalance_ratio": 1.0,
        "is_classification": is_classification,
    }


# ── build_preprocessor ────────────────────────────────────────────────────────

def test_build_preprocessor_numeric_only():
    df = make_reg_df()
    profile = make_profile(df, is_classification=False)
    pp = build_preprocessor(profile)
    assert pp is not None


def test_build_preprocessor_with_categoricals():
    df = make_cls_df()
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    transformer_names = [t[0] for t in pp.transformers]
    assert "cat" in transformer_names


def test_build_preprocessor_drops_high_cardinality_categoricals():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "num":    rng.standard_normal(n),
        "hi_cat": [f"val_{i}" for i in range(n)],   # 200 unique → should be dropped
        "lo_cat": rng.choice(["X", "Y", "Z"], n),
        "target": rng.choice([0, 1], n),
    })
    profile = {
        "shape": {"rows": n, "cols": 4},
        "feature_types": {
            "numeric": {"ordinal": [], "continuous": ["num"]},
            "categorical": {"binary": [], "multiclass": ["hi_cat", "lo_cat"]},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        "n_unique_by_col": {"num": 200, "hi_cat": 200, "lo_cat": 3, "target": 2},
        "imbalance_ratio": 1.0,
        "is_classification": True,
    }
    pp = build_preprocessor(profile)
    # After filtering, cat transformer should only encode lo_cat
    # transformers is a list of (name, transformer, columns) 3-tuples
    cat_entry = next(t for t in pp.transformers if t[0] == "cat")
    encoded_cols = cat_entry[2]
    assert "hi_cat" not in encoded_cols
    assert "lo_cat" in encoded_cols


# ── select_models ─────────────────────────────────────────────────────────────

def test_select_models_classification_contains_dummy():
    profile = make_profile(make_cls_df())
    names = [n for n, _ in select_models(profile)]
    assert "DummyMostFrequent" in names


def test_select_models_classification_contains_random_forest():
    profile = make_profile(make_cls_df())
    names = [n for n, _ in select_models(profile)]
    assert "RandomForest" in names


def test_select_models_regression_contains_dummy():
    profile = make_profile(make_reg_df(), is_classification=False)
    names = [n for n, _ in select_models(profile)]
    assert "DummyMean" in names


def test_select_models_regression_contains_random_forest():
    profile = make_profile(make_reg_df(), is_classification=False)
    names = [n for n, _ in select_models(profile)]
    assert "RandomForestRegressor" in names


def test_select_models_large_dataset_excludes_gradient_boosting():
    profile = make_profile(make_cls_df())
    profile["shape"]["rows"] = 100_000   # above 50k threshold
    names = [n for n, _ in select_models(profile)]
    assert "GradientBoosting" not in names


def test_select_models_small_dataset_includes_svc():
    profile = make_profile(make_cls_df())
    profile["shape"]["rows"] = 1000
    profile["shape"]["cols"] = 10
    names = [n for n, _ in select_models(profile)]
    assert "SVC_RBF" in names


def test_select_models_respects_regression_priority():
    profile = make_profile(make_reg_df(), is_classification=False)
    names = [n for n, _ in select_models(profile, preferred_model="GradientBoostingRegressor")]
    assert names[0] == "GradientBoostingRegressor"


def test_select_models_respects_classification_priority():
    profile = make_profile(make_cls_df())
    names = [n for n, _ in select_models(profile, preferred_model="RandomForest")]
    assert names[0] == "RandomForest"


def test_select_models_ignores_unknown_priority():
    profile = make_profile(make_reg_df(), is_classification=False)
    names = [n for n, _ in select_models(profile, preferred_model="DoesNotExist")]
    assert names[0] == "DummyMean"


# ── train_models ─────────────────────────────────────────────────────────────

def test_train_models_classification_returns_best():
    df = make_cls_df()
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    # Use only fast models for the test
    candidates = [(n, m) for n, m in select_models(profile)
                  if n in ("DummyMostFrequent", "LogisticRegression")]
    result = train_models(df, "target", pp, candidates, seed=42,
                          test_size=0.2, output_dir=".", is_classification=True)
    assert "best" in result
    assert "all_metrics" in result
    assert "accuracy" in result["best"]["metrics"]
    assert "f1_macro" in result["best"]["metrics"]


def test_train_models_regression_returns_r2():
    df = make_reg_df()
    profile = make_profile(df, is_classification=False)
    pp = build_preprocessor(profile)
    candidates = [(n, m) for n, m in select_models(profile)
                  if n in ("DummyMean", "LinearRegression")]
    result = train_models(df, "target", pp, candidates, seed=42,
                          test_size=0.2, output_dir=".", is_classification=False)
    assert "r2" in result["best"]["metrics"]
    assert "mae" in result["best"]["metrics"]
    assert "rmse" in result["best"]["metrics"]


def test_train_models_sorted_best_first_classification():
    df = make_cls_df()
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    candidates = [(n, m) for n, m in select_models(profile)
                  if n in ("DummyMostFrequent", "LogisticRegression")]
    result = train_models(df, "target", pp, candidates, seed=42,
                          test_size=0.2, output_dir=".", is_classification=True)
    # Best model should have balanced_accuracy >= any other
    best_ba = result["best"]["metrics"]["balanced_accuracy"]
    for m in result["all_metrics"]:
        assert m["balanced_accuracy"] <= best_ba + 1e-9


def test_train_models_sorted_best_first_regression():
    df = make_reg_df()
    profile = make_profile(df, is_classification=False)
    pp = build_preprocessor(profile)
    candidates = [(n, m) for n, m in select_models(profile)
                  if n in ("DummyMean", "LinearRegression")]
    result = train_models(df, "target", pp, candidates, seed=42,
                          test_size=0.2, output_dir=".", is_classification=False)
    best_r2 = result["best"]["metrics"]["r2"]
    for m in result["all_metrics"]:
        assert m["r2"] <= best_r2 + 1e-9


def test_train_models_invalid_target_raises():
    df = make_cls_df()
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    candidates = select_models(profile)[:1]
    with pytest.raises(ValueError, match="not found"):
        train_models(df, "nonexistent", pp, candidates, seed=42,
                     test_size=0.2, output_dir=".")
