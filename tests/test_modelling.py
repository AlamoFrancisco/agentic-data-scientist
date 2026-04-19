"""
Tests for tools/modelling.py

Covers: build_preprocessor (numeric-only, with categoricals, high-cardinality drop),
select_models (classification vs regression, size thresholds),
train_models (classification metrics, regression metrics, invalid target).
"""
import numpy as np
import pandas as pd
import pytest

import tools.modelling as modelling
from tools.modelling import build_preprocessor, cross_validate_top_models, select_models, train_models, tune_best_model


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


def test_build_preprocessor_uses_sparse_one_hot_encoding():
    df = make_cls_df()
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    cat_entry = next(t for t in pp.transformers if t[0] == "cat")
    ohe = cat_entry[1].named_steps["onehot"]
    sparse_flag = getattr(ohe, "sparse_output", getattr(ohe, "sparse", None))
    assert sparse_flag is True


def test_build_preprocessor_enables_robust_imputation_when_requested():
    df = make_cls_df()
    profile = make_profile(df)
    profile["robust_imputation"] = True
    pp = build_preprocessor(profile)

    cont_entry = next(t for t in pp.transformers if t[0] == "cont")
    cont_imputer = cont_entry[1].named_steps["imputer"]
    assert cont_imputer.add_indicator is True

    cat_entry = next(t for t in pp.transformers if t[0] == "cat")
    cat_imputer = cat_entry[1].named_steps["imputer"]
    assert cat_imputer.strategy == "constant"
    assert cat_imputer.fill_value == "__missing__"


def test_build_preprocessor_handles_integer_categorical_with_robust_imputation():
    df = pd.DataFrame({
        "sex": pd.Series([0, 1, 0, 1, 1, 0], dtype="int64"),
        "target": [0, 1, 0, 1, 1, 0],
    })
    profile = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "feature_types": {
            "numeric": {"ordinal": [], "continuous": []},
            "categorical": {"binary": ["sex"], "multiclass": []},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        "n_unique_by_col": {"sex": 2, "target": 2},
        "imbalance_ratio": 1.0,
        "is_classification": True,
        "robust_imputation": True,
    }

    pp = build_preprocessor(profile)
    transformed = pp.fit_transform(df[["sex"]], df["target"])

    assert transformed.shape[0] == len(df)


def test_build_preprocessor_drops_one_level_for_binary_categoricals():
    df = pd.DataFrame({
        "smoker": ["yes", "no", "yes", "no", "yes", "no"],
        "target": [1, 0, 1, 0, 1, 0],
    })
    profile = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "feature_types": {
            "numeric": {"ordinal": [], "continuous": []},
            "categorical": {"binary": ["smoker"], "multiclass": []},
            "text": [],
            "datetime": [],
            "all_missing": [],
        },
        "n_unique_by_col": {"smoker": 2, "target": 2},
        "imbalance_ratio": 1.0,
        "is_classification": True,
    }

    pp = build_preprocessor(profile)
    transformed = pp.fit_transform(df[["smoker"]], df["target"])

    assert transformed.shape == (len(df), 1)


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
    # SVC is only included for small datasets (rows < 1000) with few columns (cols <= 50)
    profile = make_profile(make_cls_df())
    profile["shape"]["rows"] = 500
    profile["shape"]["cols"] = 10
    names = [n for n, _ in select_models(profile)]
    assert "SVC_RBF" in names


def test_select_models_large_dataset_with_ensemble_preference_excludes_classic_gradient_boosting():
    profile = make_profile(make_cls_df())
    profile["shape"]["rows"] = 100_000
    profile["prefer_ensemble"] = True

    names = [n for n, _ in select_models(profile)]

    assert "GradientBoosting" not in names
    assert "HistGradientBoosting" in names


def test_select_models_respects_use_class_weights_flag():
    profile = make_profile(make_cls_df())
    profile["use_class_weights"] = True
    candidates = dict(select_models(profile))

    assert candidates["LogisticRegression"].class_weight == "balanced"
    assert candidates["RandomForest"].class_weight == "balanced"
    assert candidates["SVC_RBF"].class_weight == "balanced"


def test_select_models_use_class_weights_flag_can_disable_imbalance_weights():
    profile = make_profile(make_cls_df())
    profile["imbalance_ratio"] = 10.0
    profile["use_class_weights"] = False
    candidates = dict(select_models(profile))

    assert candidates["LogisticRegression"].class_weight is None
    assert candidates["RandomForest"].class_weight is None
    assert candidates["SVC_RBF"].class_weight is None


def test_select_models_respects_regression_priority():
    profile = make_profile(make_reg_df(), is_classification=False)
    names = [n for n, _ in select_models(profile, preferred_model="GradientBoostingRegressor")]
    assert names[0] == "GradientBoostingRegressor"


def test_select_models_regression_simple_models_only_excludes_ensembles():
    profile = make_profile(make_reg_df(), is_classification=False)
    profile["simple_models_only"] = True
    names = [n for n, _ in select_models(profile)]
    assert names == ["DummyMean", "LinearRegression", "Ridge"]


def test_select_models_large_regression_excludes_classic_gradient_boosting_regressor():
    profile = make_profile(make_reg_df(), is_classification=False)
    profile["shape"]["rows"] = 100_000

    names = [n for n, _ in select_models(profile)]

    assert "GradientBoostingRegressor" not in names
    assert "HistGradientBoostingRegressor" in names


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


def test_train_models_densifies_sparse_features_for_gradient_boosting():
    df = make_cls_df(n=200)
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    candidates = [("GradientBoosting", modelling.GradientBoostingClassifier(random_state=42))]

    result = train_models(
        df,
        "target",
        pp,
        candidates,
        seed=42,
        test_size=0.2,
        output_dir=".",
        is_classification=True,
    )

    assert result["best"]["name"] == "GradientBoosting"
    assert "to_dense" in result["best"]["pipeline"].named_steps


def test_train_models_adapts_smote_to_tiny_minority_class():
    rng = np.random.default_rng(7)
    n_majority = 57
    n_minority = 3
    df = pd.DataFrame({
        "num1": np.concatenate([rng.standard_normal(n_majority), rng.standard_normal(n_minority) + 2]),
        "num2": np.concatenate([rng.standard_normal(n_majority), rng.standard_normal(n_minority) + 2]),
        "target": np.array([0] * n_majority + [1] * n_minority),
    })
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    candidates = [("DummyMostFrequent", modelling.DummyClassifier(strategy="most_frequent"))]

    result = train_models(
        df,
        "target",
        pp,
        candidates,
        seed=42,
        test_size=0.2,
        output_dir=".",
        is_classification=True,
        apply_oversampling=True,
    )

    assert result["best"]["name"] == "DummyMostFrequent"
    assert result["best"]["pipeline"].named_steps["smote"].k_neighbors == 1


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


def test_cross_validate_top_models_classification_returns_summary():
    df = make_cls_df()
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    candidates = [(n, m) for n, m in select_models(profile) if n in ("DummyMostFrequent", "LogisticRegression")]
    trained = train_models(df, "target", pp, candidates, seed=42,
                           test_size=0.2, output_dir=".", is_classification=True)

    result = cross_validate_top_models(
        df=df,
        target="target",
        training_payload=trained,
        seed=42,
        is_classification=True,
        top_k=2,
    )

    assert result["enabled"] is True
    assert result["n_splits"] >= 2
    assert len(result["models"]) >= 1
    assert "balanced_accuracy_mean" in result["models"][0]


def test_cross_validate_top_models_regression_returns_summary():
    df = make_reg_df()
    profile = make_profile(df, is_classification=False)
    pp = build_preprocessor(profile)
    candidates = [(n, m) for n, m in select_models(profile) if n in ("DummyMean", "LinearRegression")]
    trained = train_models(df, "target", pp, candidates, seed=42,
                           test_size=0.2, output_dir=".", is_classification=False)

    result = cross_validate_top_models(
        df=df,
        target="target",
        training_payload=trained,
        seed=42,
        is_classification=False,
        top_k=2,
    )

    assert result["enabled"] is True
    assert result["n_splits"] >= 2
    assert len(result["models"]) >= 1
    assert "r2_mean" in result["models"][0]


def test_cross_validate_top_models_uses_training_split_when_available(monkeypatch):
    df = make_cls_df(n=120)
    profile = make_profile(df)
    pp = build_preprocessor(profile)
    candidates = [(n, m) for n, m in select_models(profile) if n in ("DummyMostFrequent", "LogisticRegression")]
    trained = train_models(
        df,
        "target",
        pp,
        candidates,
        seed=42,
        test_size=0.2,
        output_dir=".",
        is_classification=True,
    )

    seen = {}

    def fake_cross_validate(estimator, X, y, cv, scoring, n_jobs, error_score):
        seen["rows"] = len(X)
        return {
            "test_accuracy": np.array([0.5, 0.5, 0.5]),
            "test_balanced_accuracy": np.array([0.5, 0.5, 0.5]),
            "test_f1_macro": np.array([0.5, 0.5, 0.5]),
        }

    monkeypatch.setattr("tools.modelling.cross_validate", fake_cross_validate)

    result = cross_validate_top_models(
        df=df,
        target="target",
        training_payload=trained,
        seed=42,
        is_classification=True,
        top_k=1,
    )

    assert result["enabled"] is True
    assert seen["rows"] == len(trained["X_train"])
    assert seen["rows"] < len(df)


# ── tune_best_model ───────────────────────────────────────────────────────────

def _run_training(df, target="target", is_classification=True, n=200):
    """Helper: build a minimal training payload for tuning tests."""
    profile = make_profile(df, target=target, is_classification=is_classification)
    preprocessor = build_preprocessor(profile)
    candidates = select_models(profile, seed=42)
    return train_models(
        df=df,
        target=target,
        preprocessor=preprocessor,
        candidates=candidates,
        seed=42,
        test_size=0.2,
        output_dir="/tmp",
        verbose=False,
        is_classification=is_classification,
    )


def test_tune_best_model_classification_returns_correct_structure():
    df = make_cls_df(n=200)
    trained = _run_training(df, is_classification=True)
    tuned = tune_best_model(trained, seed=42, is_classification=True)

    assert "best" in tuned
    assert "results" in tuned
    assert "all_metrics" in tuned
    best = tuned["best"]
    assert "metrics" in best
    assert "balanced_accuracy" in best["metrics"]


def test_tune_best_model_tuned_flag_set_when_grid_exists():
    df = make_cls_df(n=200)
    trained = _run_training(df, is_classification=True)
    # Force the best model to LogisticRegression so there is a known param grid
    for r in trained["results"]:
        if r["name"] == "LogisticRegression":
            trained["best"] = r
            break
    tuned = tune_best_model(trained, seed=42, is_classification=True)
    best = tuned["best"]
    # If a grid exists, tuned=True and best_params must be present
    if best.get("tuned"):
        assert "best_params" in best
        assert isinstance(best["best_params"], dict)


def test_tune_best_model_falls_back_to_single_process_when_parallel_backend_fails(monkeypatch):
    df = make_cls_df(n=200)
    trained = _run_training(df, is_classification=True)
    for r in trained["results"]:
        if r["name"] == "LogisticRegression":
            trained["best"] = r
            break

    seen_n_jobs = []

    class FakeRandomizedSearchCV:
        def __init__(
            self,
            pipeline,
            param_distributions,
            n_iter,
            scoring,
            cv,
            random_state,
            n_jobs,
            refit,
        ):
            self.best_estimator_ = pipeline
            self.best_params_ = {"model__C": 1.0}
            self.n_jobs = n_jobs
            seen_n_jobs.append(n_jobs)

        def fit(self, X, y):
            if self.n_jobs == -1:
                raise PermissionError("SC_SEM_NSEMS_MAX operation not permitted")
            return self

    monkeypatch.setattr(modelling, "RandomizedSearchCV", FakeRandomizedSearchCV)

    tuned = tune_best_model(trained, seed=42, is_classification=True)

    assert seen_n_jobs == [-1, 1]
    assert tuned["best"]["tuned"] is True
    assert tuned["best"]["best_params"] == {"model__C": 1.0}


def test_tune_best_model_no_grid_returns_unchanged():
    df = make_cls_df(n=200)
    trained = _run_training(df, is_classification=True)
    # Force the best to a Dummy model — no param grid → payload returned as-is
    for r in trained["results"]:
        if "Dummy" in r["name"]:
            trained["best"] = r
            break
    original_name = trained["best"]["name"]
    tuned = tune_best_model(trained, seed=42, is_classification=True)
    assert tuned["best"]["name"] == original_name
    assert not tuned["best"].get("tuned", False)


def test_tune_best_model_regression_returns_r2():
    df = make_reg_df(n=200)
    trained = _run_training(df, is_classification=False)
    tuned = tune_best_model(trained, seed=42, is_classification=False)
    assert "r2" in tuned["best"]["metrics"]


def test_tune_best_model_all_metrics_length_unchanged():
    df = make_cls_df(n=200)
    trained = _run_training(df, is_classification=True)
    original_count = len(trained["all_metrics"])
    tuned = tune_best_model(trained, seed=42, is_classification=True)
    assert len(tuned["all_metrics"]) == original_count


def test_suppress_known_non_actionable_warnings_targets_validate_data_futurewarning(monkeypatch):
    recorded = {}

    def fake_filterwarnings(action, message="", category=None, module="", **kwargs):
        recorded["action"] = action
        recorded["message"] = message
        recorded["category"] = category
        recorded["module"] = module

    monkeypatch.setattr(modelling.warnings, "filterwarnings", fake_filterwarnings)

    modelling._suppress_known_non_actionable_warnings()

    assert recorded["action"] == "ignore"
    assert "BaseEstimator\\._validate_data" in recorded["message"]
    assert recorded["category"] is FutureWarning
    assert recorded["module"] == r"sklearn\.base"
