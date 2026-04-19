"""
Tests for agents/planner.py

Covers: create_plan with all conditional branches — imbalance, small dataset,
high-cardinality categoricals, memory hints, and missing data handling.
"""
from agents.planner import create_plan


# ── helpers ───────────────────────────────────────────────────────────────────

def make_profile(**overrides):
    """Build a minimal dataset profile dict; override any field as needed."""
    base = {
        "shape": {"rows": 5000, "cols": 10},
        "feature_types": {
            "numeric": {"ordinal": [], "continuous": ["a", "b"]},
            "categorical": {"binary": [], "multiclass": []},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        "imbalance_ratio": 1.0,
        "missing_pct": {},
        "is_classification": True,
        "notes": [],
        "n_unique_by_col": {},
    }
    base.update(overrides)
    return base


# ── core steps ────────────────────────────────────────────────────────────────

def test_basic_plan_contains_all_core_steps():
    plan = create_plan(make_profile())
    for step in ["profile_dataset", "build_preprocessor", "select_models",
                 "train_models", "evaluate", "reflect", "write_report"]:
        assert step in plan, f"Expected '{step}' in plan"


def test_plan_returns_a_list():
    assert isinstance(create_plan(make_profile()), list)


# ── imbalance branch ──────────────────────────────────────────────────────────

def test_imbalanced_dataset_adds_strategy():
    plan = create_plan(make_profile(imbalance_ratio=4.0))
    assert "consider_imbalance_strategy" in plan


def test_strategy_is_before_train_models():
    plan = create_plan(make_profile(imbalance_ratio=4.0))
    assert plan.index("consider_imbalance_strategy") < plan.index("train_models")


def test_no_imbalance_strategy_when_balanced():
    plan = create_plan(make_profile(imbalance_ratio=2.0))
    assert "consider_imbalance_strategy" not in plan


# ── small dataset branch ──────────────────────────────────────────────────────

def test_small_dataset_adds_regularization():
    plan = create_plan(make_profile(shape={"rows": 500, "cols": 10}))
    assert "apply_regularization" in plan


def test_large_dataset_no_regularization():
    plan = create_plan(make_profile(shape={"rows": 5000, "cols": 10}))
    assert "apply_regularization" not in plan


# ── high-cardinality categorical branch ──────────────────────────────────────

def test_high_cardinality_categorical_adds_target_encoding():
    profile = make_profile(
        feature_types={
            "numeric": {"ordinal": [], "continuous": []},
            "categorical": {"binary": [], "multiclass": ["model_name"]},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        n_unique_by_col={"model_name": 900},
    )
    plan = create_plan(profile)
    assert "apply_target_encoding" in plan


def test_high_cardinality_encoding_is_before_preprocessor():
    profile = make_profile(
        feature_types={
            "numeric": {"ordinal": [], "continuous": []},
            "categorical": {"binary": [], "multiclass": ["model_name"]},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        n_unique_by_col={"model_name": 900},
    )
    plan = create_plan(profile)
    assert plan.index("apply_target_encoding") < plan.index("build_preprocessor")


def test_low_cardinality_categorical_no_target_encoding():
    profile = make_profile(
        feature_types={
            "numeric": {"ordinal": [], "continuous": []},
            "categorical": {"binary": [], "multiclass": ["color"]},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        n_unique_by_col={"color": 5},
    )
    plan = create_plan(profile)
    assert "apply_target_encoding" not in plan


# ── memory hint branch ────────────────────────────────────────────────────────

def test_memory_hint_appends_prioritize_step():
    plan = create_plan(make_profile(), memory_hint={"best_model": "RandomForest"})
    assert "prioritize_model:RandomForest" in plan


def test_no_memory_hint_no_prioritize_step():
    plan = create_plan(make_profile(), memory_hint=None)
    assert not any(s.startswith("prioritize_model:") for s in plan)


def test_memory_hint_without_best_model_key_no_prioritize():
    plan = create_plan(make_profile(), memory_hint={"other_key": "value"})
    assert not any(s.startswith("prioritize_model:") for s in plan)


# ── severe missing data branch ────────────────────────────────────────────────

def test_severe_missing_data_adds_handler():
    profile = make_profile(missing_pct={"col_a": 25.0, "col_b": 5.0})
    plan = create_plan(profile)
    assert "handle_severe_missing_data" in plan


def test_mild_missing_no_handler():
    profile = make_profile(missing_pct={"col_a": 10.0})
    plan = create_plan(profile)
    assert "handle_severe_missing_data" not in plan


def test_no_missing_no_handler():
    plan = create_plan(make_profile(missing_pct={}))
    assert "handle_severe_missing_data" not in plan


def test_hard_leakage_adds_drop_leaky_features():
    profile = make_profile(hard_leakage_cols=[{"column": "alive", "reason": "deterministic_target_mapping"}])
    plan = create_plan(profile)
    assert "drop_leaky_features" in plan
    assert profile["leaky_col_names"] == ["alive"]


def test_soft_leakage_does_not_auto_drop_features():
    profile = make_profile(soft_leakage_cols=[{"column": "bmi", "normalised_mi": 1.0, "evidence_level": "soft"}])
    plan = create_plan(profile)
    assert "drop_leaky_features" not in plan
    assert "leaky_col_names" not in profile


def test_high_dimensional_dataset_prefers_regularized_simple_models():
    profile = make_profile(shape={"rows": 5_000, "cols": 150})
    plan = create_plan(profile)
    assert "apply_regularization" in plan
    assert "use_simple_models_only" in plan
    assert "tune_hyperparameters" not in plan


def test_high_dimensional_large_dataset_does_not_force_ensemble_models():
    profile = make_profile(shape={"rows": 15_000, "cols": 150})
    plan = create_plan(profile)
    assert "use_simple_models_only" in plan
    assert "use_ensemble_models" not in plan


def test_cost_aware_planning_reduces_tuning_budget_on_large_workload():
    profile = make_profile(shape={"rows": 30_000, "cols": 20})
    plan = create_plan(profile)
    assert "tune_hyperparameters" in plan
    assert "reduce_tuning_budget" in plan


def test_extreme_workload_skips_cross_validation():
    profile = make_profile(shape={"rows": 30_000, "cols": 100})
    plan = create_plan(profile)
    assert "validate_with_cross_validation" not in plan
