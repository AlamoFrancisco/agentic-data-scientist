"""
Tests for tools/data_profiler.py

Covers: infer_schema, infer_target_column, is_classification_target,
dataset_fingerprint, profile_dataset (classification + regression paths,
imbalance detection, small/high-dim notes, missing target error).
"""
import numpy as np
import pandas as pd
import pytest

from tools.data_profiler import (
    correlation_report,
    dataset_fingerprint,
    infer_schema,
    infer_target_column,
    is_classification_target,
    ordinal_report,
    profile_dataset,
)


# ── infer_schema ──────────────────────────────────────────────────────────────

def test_infer_schema_low_cardinality_integer_like_numeric_is_ordinal():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert infer_schema(df)["a"] == "ordinal"


def test_infer_schema_continuous_float_is_continuous():
    df = pd.DataFrame({"b": [1.1, 2.2, 3.3]})
    assert infer_schema(df)["b"] == "continuous"


def test_infer_schema_all_missing():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    assert infer_schema(df)["x"] == "all_missing"


def test_infer_schema_low_cardinality_string_is_categorical():
    df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
    assert infer_schema(df)["cat"] == "categorical"


def test_infer_schema_high_cardinality_string_is_text():
    df = pd.DataFrame({"txt": [f"unique_{i}" for i in range(30)]})
    assert infer_schema(df)["txt"] == "text"


def test_infer_schema_bool_is_boolean():
    df = pd.DataFrame({"flag": pd.array([True, False, True], dtype="boolean")})
    schema = infer_schema(df)
    assert schema["flag"] == "boolean"


# ── infer_target_column ───────────────────────────────────────────────────────

def test_infer_target_column_named_target():
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    assert infer_target_column(df) == "target"


def test_infer_target_column_named_label():
    df = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
    assert infer_target_column(df) == "label"


def test_infer_target_column_named_y():
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    assert infer_target_column(df) == "y"


def test_infer_target_column_falls_back_to_last_col():
    df = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [5, 6, 7, 8], "outcome": [0, 1, 0, 1]})
    assert infer_target_column(df) == "outcome"


def test_infer_target_column_skips_id_like_last_col():
    # Each value in last col is unique → looks like an ID → skip to fallback
    data = {f"feat{i}": range(20) for i in range(3)}
    data["id_col"] = range(20)   # all unique — should be skipped
    df = pd.DataFrame(data)
    # Should NOT return "id_col"; fallback picks a numeric col with low cardinality
    result = infer_target_column(df)
    assert result != "id_col"


# ── is_classification_target ──────────────────────────────────────────────────

def test_is_classification_string_target():
    s = pd.Series(["cat", "dog", "cat"])
    assert is_classification_target(s) is True


def test_is_classification_float_target_is_regression():
    s = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    assert is_classification_target(s) is False


def test_is_classification_int_low_cardinality():
    s = pd.Series([0, 1, 0, 1, 0] * 20)
    assert is_classification_target(s) is True


def test_is_classification_int_high_cardinality_is_regression():
    # 100 unique values in 100 rows → (uniq/n) > 0.05, uniq > 50 → regression
    s = pd.Series(range(100))
    assert is_classification_target(s) is False


# ── dataset_fingerprint ───────────────────────────────────────────────────────

def test_fingerprint_is_stable():
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    assert dataset_fingerprint(df, "target") == dataset_fingerprint(df, "target")


def test_fingerprint_starts_with_fp():
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    assert dataset_fingerprint(df, "target").startswith("fp_")


def test_fingerprint_differs_for_different_targets():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [0, 1]})
    assert dataset_fingerprint(df, "b") != dataset_fingerprint(df, "c")


# ── ordinal_report ────────────────────────────────────────────────────────────

def test_ordinal_report_detects_integer_like_low_cardinality_numeric_column():
    df = pd.DataFrame(
        {
            "rating": [1, 2, 3, 4, 5, 1, 2, 3],
            "price": [10.5, 10.7, 10.8, 10.9, 11.1, 11.2, 11.4, 11.5],
        }
    )
    schema = infer_schema(df)
    nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    result = ordinal_report(df, schema, nunique)

    assert ("rating", 5, 1.0) in result
    assert all(col != "price" for col, _, _ in result)


def test_correlation_report_detects_high_correlation_pairs():
    df = pd.DataFrame(
        {
            "a": np.arange(30),
            "b": np.arange(30) * 2,
            "c": np.arange(30)[::-1],
        }
    )
    schema = infer_schema(df)

    result = correlation_report(df, schema)

    assert result["corr"] is not None
    assert any(
        {pair["col_a"], pair["col_b"]} == {"a", "b"} and pair["abs_corr"] == 1.0
        for pair in result["high_corr_pairs"]
    )


# ── profile_dataset ───────────────────────────────────────────────────────────

def test_profile_dataset_basic_shape():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"], "target": [0, 1, 0, 1]})
    profile = profile_dataset(df, "target")
    assert profile["shape"]["rows"] == 4
    assert profile["shape"]["cols"] == 3


def test_profile_dataset_classification_flag():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    profile = profile_dataset(df, "target")
    assert profile["is_classification"] is True


def test_profile_dataset_regression_flag():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [10.5, 20.5, 30.5]})
    profile = profile_dataset(df, "target")
    assert profile["is_classification"] is False
    assert profile["class_counts"] is None
    assert profile["imbalance_ratio"] is None


def test_profile_dataset_regression_note():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [10.5, 20.5, 30.5]})
    profile = profile_dataset(df, "target")
    assert any("egression" in n for n in profile["notes"])


def test_profile_dataset_numeric_features_detected():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "target": [0, 1, 0, 1]})
    profile = profile_dataset(df, "target")
    assert "a" in profile["feature_types"]["numeric"]["ordinal"]
    assert "b" in profile["feature_types"]["numeric"]["ordinal"]


def test_profile_dataset_binary_and_multiclass_categorical_features_detected():
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green"],
            "flag": ["yes", "no", "yes", "no"],
            "is_active": pd.array([True, False, True, False], dtype="boolean"),
            "target": [0, 1, 0, 1],
        }
    )
    profile = profile_dataset(df, "target")
    assert "flag" in profile["feature_types"]["categorical"]["binary"]
    assert "is_active" in profile["feature_types"]["categorical"]["binary"]
    assert "color" in profile["feature_types"]["categorical"]["multiclass"]


def test_profile_dataset_imbalance_detected():
    # 9 zeros, 1 one → ratio = 9.0
    df = pd.DataFrame({"a": range(10), "target": [0] * 9 + [1]})
    profile = profile_dataset(df, "target")
    assert profile["imbalance_ratio"] >= 3.0
    assert any("mbalance" in n for n in profile["notes"])


def test_profile_dataset_small_dataset_note():
    df = pd.DataFrame({"a": range(50), "target": [0, 1] * 25})
    profile = profile_dataset(df, "target")
    assert any("1000" in n or "mall" in n for n in profile["notes"])


def test_profile_dataset_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        profile_dataset(df, "nonexistent_target")


def test_profile_dataset_missing_pct_keys_are_strings():
    df = pd.DataFrame({"a": [1, None, 3], "target": [0, 1, 0]})
    profile = profile_dataset(df, "target")
    assert all(isinstance(k, str) for k in profile["missing_pct"].keys())


def test_profile_dataset_adds_ordinal_signals_and_note():
    df = pd.DataFrame(
        {
            "bedrooms": [1, 2, 3, 2, 4, 3, 2, 1],
            "price": [100000.5, 120000.7, 140000.2, 130000.1, 160000.4, 150000.6, 125000.3, 110000.8],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    profile = profile_dataset(df, "target")

    assert profile["has_ordinal"] is True
    assert profile["schema"]["bedrooms"] == "ordinal"
    assert profile["schema"]["price"] == "continuous"
    assert "bedrooms" in profile["ordinal_cols"]
    assert "price" in profile["continuous_cols"]
    assert any("Ordinal-like numeric columns detected" in note for note in profile["notes"])


def test_profile_dataset_adds_correlation_signals():
    df = pd.DataFrame(
        {
            "a": np.arange(30),
            "b": np.arange(30) * 3,
            "target": ["A", "B"] * 15,
        }
    )

    profile = profile_dataset(df, "target")

    assert profile["correlation"] is not None
    assert profile["high_corr_present"] is True
    assert profile["max_abs_corr"] == 1.0
    assert any(
        {pair["col_a"], pair["col_b"]} == {"a", "b"}
        for pair in profile["high_corr_pairs"]
    )
