from typing import Any, Dict, List, Optional, Tuple

import warnings
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, TargetEncoder
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

# Classification models and metrics
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Regression models and metrics 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def build_preprocessor(profile: Dict[str, Any]) -> ColumnTransformer:
    numeric_groups = profile["feature_types"]["numeric"]
    categorical_groups = profile["feature_types"]["categorical"]

    ord_cols = numeric_groups.get("ordinal", [])
    cont_cols = numeric_groups.get("continuous", [])
    bin_cat_cols = categorical_groups.get("binary", [])
    multi_cat_cols = categorical_groups.get("multiclass", [])

    # Drop near-constant columns — they carry no signal and hurt one-hot encoding
    near_const = profile.get("near_constant_cols", [])
    ord_cols = [c for c in ord_cols if c not in near_const]
    cont_cols = [c for c in cont_cols if c not in near_const]
    bin_cat_cols = [c for c in bin_cat_cols if c not in near_const]
    multi_cat_cols = [c for c in multi_cat_cols if c not in near_const]

    # Drop highly correlated features identified by the planner
    corr_drop = profile.get("corr_cols_to_drop", []) if profile.get("drop_high_corr") else []
    ord_cols = [c for c in ord_cols if c not in corr_drop]
    cont_cols = [c for c in cont_cols if c not in corr_drop]
    bin_cat_cols = [c for c in bin_cat_cols if c not in corr_drop]
    multi_cat_cols = [c for c in multi_cat_cols if c not in corr_drop]

    # Drop leaky features identified by mutual information analysis
    leaky_drop = profile.get("leaky_col_names", []) if profile.get("drop_leaky") else []
    ord_cols = [c for c in ord_cols if c not in leaky_drop]
    cont_cols = [c for c in cont_cols if c not in leaky_drop]
    bin_cat_cols = [c for c in bin_cat_cols if c not in leaky_drop]
    multi_cat_cols = [c for c in multi_cat_cols if c not in leaky_drop]

    # Drop columns with too many missing values — threshold adapts to dataset size
    rows = profile["shape"]["rows"]
    if rows < 1000:
        missing_threshold = 60.0   # small dataset: keep more columns
    elif rows < 10000:
        missing_threshold = 50.0   # medium dataset
    else:
        missing_threshold = 40.0   # large dataset: stricter
    missing_pct = profile.get("missing_pct", {})
    ord_cols = [c for c in ord_cols if missing_pct.get(c, 0) <= missing_threshold]
    cont_cols = [c for c in cont_cols if missing_pct.get(c, 0) <= missing_threshold]
    bin_cat_cols = [c for c in bin_cat_cols if missing_pct.get(c, 0) <= missing_threshold]
    multi_cat_cols = [c for c in multi_cat_cols if missing_pct.get(c, 0) <= missing_threshold]

    # Separate low-cardinality (OHE) from high-cardinality (target encoding or drop)
    n_unique = profile.get("n_unique_by_col", {})
    multi_cat_cols_before_card_filter = list(multi_cat_cols)
    multi_cat_cols = [
        c for c in multi_cat_cols
        if n_unique.get(c, 0) < 50 and (n_unique.get(c, 0) / max(rows, 1)) < 0.05
    ]
    # Cols that failed the cardinality filter — candidates for target encoding
    high_card_cols = [
        c for c in multi_cat_cols_before_card_filter if c not in multi_cat_cols
    ]
    # Also include text cols that look categorical (50 < n_unique < 10% of rows)
    # These are classified as "text" by the profiler but are actually high-cardinality categoricals
    text_cols = profile.get("feature_types", {}).get("text", [])
    text_cols = [c for c in text_cols if c not in leaky_drop and c not in corr_drop and c not in near_const]
    text_cols = [c for c in text_cols if missing_pct.get(c, 0) <= missing_threshold]
    high_card_cols += [
        c for c in text_cols
        if 50 < n_unique.get(c, 0) < rows * 0.10
    ]
    
    use_robust_scaling = profile.get("use_robust_scaling", False)
    handle_outliers = profile.get("handle_outliers", False)

    # Small datasets: RobustScaler only — clipping risks losing too much information
    # Large datasets: RobustScaler with quantile clamping (unit_variance clips extremes)
    if handle_outliers and rows >= 1000:
        scaler = RobustScaler(unit_variance=True)
    elif use_robust_scaling or handle_outliers:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler(with_mean=True)

    continuous_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # scikit-learn renamed `sparse` -> `sparse_output` (v1.2+). Support both.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    transformers = [
        ("cont", continuous_transformer, cont_cols),
        ("ord", ordinal_transformer, ord_cols),
        ("cat", categorical_transformer, bin_cat_cols + multi_cat_cols),
    ]

    # Target encoding for high-cardinality categoricals (only when flag is set)
    if profile.get("use_target_encoding") and high_card_cols:
        target_enc_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", TargetEncoder()),
        ])
        transformers.append(("high_card", target_enc_transformer, high_card_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def prioritize_candidates(
    candidates: List[Tuple[str, Any]],
    preferred_model: Optional[str] = None,
) -> List[Tuple[str, Any]]:
    if not preferred_model:
        return candidates

    preferred = [(name, model) for name, model in candidates if name == preferred_model]
    others = [(name, model) for name, model in candidates if name != preferred_model]
    return preferred + others


def select_models(
    profile: Dict[str, Any],
    seed: int = 42,
    preferred_model: Optional[str] = None,
) -> List[Tuple[str, Any]]:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    imb = float(profile.get("imbalance_ratio") or 1.0)
    class_weight = "balanced" if imb >= 3.0 else None
    
    # Regression models if not a classification task
    if not profile.get("is_classification", True):
        candidates = [
            ("DummyMean", DummyRegressor(strategy="mean")),
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
            ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=seed)),
        ]
        return prioritize_candidates(candidates, preferred_model)

    use_regularization = profile.get("use_regularization", False)
    simple_models_only = profile.get("simple_models_only", False)
    prefer_ensemble = profile.get("prefer_ensemble", False)
    lr_C = 0.1 if use_regularization else 1.0

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(C=lr_C, max_iter=2000, class_weight=class_weight, solver="saga", tol=1e-3, random_state=seed)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight
        )),
    ]

    # Small datasets: skip complex models to avoid overfitting
    if simple_models_only:
        return prioritize_candidates(candidates, preferred_model)

    # Large datasets: always include ensemble models (bucket: >= 10000)
    # Medium datasets: also include GradientBoosting (bucket: 1000–9999)
    if prefer_ensemble or rows < 10000:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(random_state=seed)))

    # SVC: small datasets only (bucket: < 1000) — O(n²–n³) cost, unused probability estimates dropped
    if rows < 1000 and cols <= 50:
        candidates.append(("SVC_RBF", SVC(kernel="rbf", probability=False, class_weight=class_weight)))

    return prioritize_candidates(candidates, preferred_model)


def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
    is_classification: bool = True,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop missing target rows
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Stratify if possible
    stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        if verbose:
            print(f"[Modelling] Training: {name}")

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        model_warnings = list({
            f"{w.category.__name__}: {w.message}" for w in caught
        })

        if is_classification:
            metrics = {
                "model": name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            }
        else:
            metrics = {
                "model": name,
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }

        results.append({
            "name": name,
            "pipeline": pipe,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "warnings": model_warnings,
        })

    # Sort by appropriate metric — R² for regression, balanced accuracy + F1 for classification
    if is_classification:
        results.sort(key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]), reverse=True)
    else:
        results.sort(key=lambda r: r["metrics"]["r2"], reverse=True)

    all_warnings = list({w for r in results for w in r.get("warnings", [])})

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results],
        "training_warnings": all_warnings,
    }


def _cv_splitter(
    y: pd.Series,
    seed: int,
    rows: int,
    is_classification: bool,
):
    if rows < 4:
        return None

    desired_splits = 5 if rows < 1000 else 3

    if is_classification:
        min_class_count = int(y.value_counts(dropna=False).min())
        n_splits = min(desired_splits, min_class_count)
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n_splits = min(desired_splits, rows)
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _cv_scoring(is_classification: bool) -> Dict[str, str]:
    if is_classification:
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        }
    return {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_mean_squared_error",
    }


def _summarize_cv_metrics(
    model_name: str,
    scores: Dict[str, Any],
    is_classification: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"model": model_name}

    if is_classification:
        for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
            values = np.asarray(scores[f"test_{metric}"], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
        summary["primary_metric"] = "balanced_accuracy"
        summary["primary_metric_mean"] = summary["balanced_accuracy_mean"]
        summary["primary_metric_std"] = summary["balanced_accuracy_std"]
        return summary

    values = np.asarray(scores["test_r2"], dtype=float)
    summary["r2_mean"] = float(values.mean())
    summary["r2_std"] = float(values.std(ddof=0))

    mae_values = -np.asarray(scores["test_mae"], dtype=float)
    summary["mae_mean"] = float(mae_values.mean())
    summary["mae_std"] = float(mae_values.std(ddof=0))

    rmse_values = np.sqrt(-np.asarray(scores["test_rmse"], dtype=float))
    summary["rmse_mean"] = float(rmse_values.mean())
    summary["rmse_std"] = float(rmse_values.std(ddof=0))

    summary["primary_metric"] = "r2"
    summary["primary_metric_mean"] = summary["r2_mean"]
    summary["primary_metric_std"] = summary["r2_std"]
    return summary


def cross_validate_top_models(
    df: pd.DataFrame,
    target: str,
    training_payload: Dict[str, Any],
    seed: int,
    is_classification: bool = True,
    top_k: int = 2,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    splitter = _cv_splitter(y, seed=seed, rows=len(X), is_classification=is_classification)
    if splitter is None:
        return {
            "enabled": False,
            "reason": "Not enough data to run cross-validation safely.",
            "n_splits": 0,
            "models": [],
            "warnings": [],
        }

    scoring = _cv_scoring(is_classification=is_classification)
    selected = training_payload.get("results", [])[:max(1, top_k)]
    summaries: List[Dict[str, Any]] = []
    all_warnings: List[str] = []

    for result in selected:
        pipe = result["pipeline"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores = cross_validate(
                pipe,
                X,
                y,
                cv=splitter,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )

        summaries.append(
            _summarize_cv_metrics(
                model_name=result["name"],
                scores=scores,
                is_classification=is_classification,
            )
        )
        all_warnings.extend(f"{w.category.__name__}: {w.message}" for w in caught)

    summaries.sort(key=lambda item: item["primary_metric_mean"], reverse=True)
    deduped_warnings = list(dict.fromkeys(all_warnings))

    return {
        "enabled": True,
        "reason": "",
        "n_splits": splitter.get_n_splits(),
        "models": summaries,
        "best_model": summaries[0]["model"] if summaries else None,
        "warnings": deduped_warnings,
    }
