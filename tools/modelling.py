"""
Modelling Tools

Preprocessing pipeline construction, model selection, training, and cross-validation.

Implemented:
- build_preprocessor: ColumnTransformer with adaptive imputation, StandardScaler /
  RobustScaler (outlier or scale-mismatch flag), OneHotEncoder for low-cardinality
  categorical, TargetEncoder (sklearn ≥1.3) for high-cardinality columns; drops
  near-constant, correlated, and leaky features before fitting
- select_models: size-bucket selection (small / medium / large); class_weight='balanced'
  for imbalanced datasets; SVC only for small datasets (rows <1000, cols ≤50) to avoid
  O(n²–n³) cost; preferred model from memory hint placed first
- train_models: stratified train/test split, per-model warning capture
  (overflow / divide-by-zero surfaced to Reflector), classification + regression support
- cross_validate_top_models: StratifiedKFold / KFold CV for the top-k candidates;
  returns per-model mean ± std for primary metric
- tune_best_model: RandomizedSearchCV over a per-model param grid on the best
  candidate; replaces the best entry in the training payload with the tuned version
"""

from typing import Any, Dict, List, Optional, Tuple

from config import (
    SMALL_DATASET_ROWS,
    LARGE_DATASET_ROWS,
    MISSING_THRESHOLD_SMALL,
    MISSING_THRESHOLD_MEDIUM,
    MISSING_THRESHOLD_LARGE,
    MAX_OHE_UNIQUE,
    MAX_OHE_UNIQUE_FRAC,
    MAX_TEXT_UNIQUE_FRAC,
    OUTLIER_CLIP_MIN_ROWS,
    SVC_MAX_COLS,
    N_ESTIMATORS,
    LR_C_DEFAULT,
    LR_C_REGULARISED,
    IMBALANCE_THRESHOLD,
    CV_SPLITS_SMALL,
    CV_SPLITS_DEFAULT,
    CV_TOP_K,
    TUNE_N_ITER,
    TUNE_CV_SPLITS,
    TUNE_REDUCED_N_ITER,
    TUNE_MAX_ROWS,
)

import warnings
import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    TargetEncoder,
    PolynomialFeatures,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)

# Classification models and metrics
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

# Regression models and metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Conditionally import imbalanced-learn so the agent doesn't crash if it's missing
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


_DENSE_MODEL_NAMES = {
    "GradientBoosting",
    "GradientBoostingRegressor",
    "HistGradientBoosting",
    "HistGradientBoostingRegressor",
}


def _to_dense_if_sparse(X: Any) -> Any:
    return X.toarray() if sparse.issparse(X) else X


def _coerce_categorical_values(X: Any) -> Any:
    """
    Route semantic categoricals through an object-typed container so string
    imputers work even when the raw data is integer-coded (for example 0/1).
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        casted = X.astype("object")
        return casted.where(pd.notna(casted), np.nan)

    arr = np.asarray(X, dtype=object).copy()
    arr[pd.isna(arr)] = np.nan
    return arr


def _resolve_smote_k_neighbors(y: pd.Series) -> Optional[int]:
    """
    SMOTE needs at least k_neighbors + 1 samples in the smallest class.
    Adapt k to the actual training split so rare classes do not crash fitting.
    """
    class_counts = y.value_counts(dropna=False)
    if class_counts.empty:
        return None

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return None

    return min(5, min_class_count - 1)


def _build_training_pipeline(
    *,
    preprocessor: ColumnTransformer,
    model_name: str,
    model: Any,
    seed: int,
    apply_oversampling: bool,
    is_classification: bool,
    smote_k_neighbors: Optional[int] = None,
) -> Any:
    steps: List[Tuple[str, Any]] = [("preprocess", preprocessor)]
    needs_dense = model_name in _DENSE_MODEL_NAMES or (apply_oversampling and is_classification)

    if needs_dense:
        steps.append(("to_dense", FunctionTransformer(_to_dense_if_sparse, accept_sparse=True)))

    if apply_oversampling and is_classification:
        if HAS_IMBLEARN and smote_k_neighbors is not None:
            steps.extend([
                ("smote", SMOTE(random_state=seed, k_neighbors=smote_k_neighbors)),
                ("model", model),
            ])
            return ImbPipeline(steps=steps)

        if HAS_IMBLEARN:
            warnings.warn(
                "Skipping SMOTE oversampling because the smallest training class has fewer than 2 samples.",
                UserWarning,
            )
        else:
            warnings.warn("imbalanced-learn is not installed. Skipping SMOTE oversampling.", UserWarning)

    steps.append(("model", model))
    return Pipeline(steps=steps)

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

    # Drop sensitive features identified by the planner
    sensitive_drop = profile.get("sensitive_cols", []) if profile.get("drop_sensitive") else []
    ord_cols = [c for c in ord_cols if c not in sensitive_drop]
    cont_cols = [c for c in cont_cols if c not in sensitive_drop]
    bin_cat_cols = [c for c in bin_cat_cols if c not in sensitive_drop]
    multi_cat_cols = [c for c in multi_cat_cols if c not in sensitive_drop]

    # Drop columns with too many missing values — threshold adapts to dataset size
    rows = profile["shape"]["rows"]
    if rows < SMALL_DATASET_ROWS:
        missing_threshold = MISSING_THRESHOLD_SMALL
    elif rows < LARGE_DATASET_ROWS:
        missing_threshold = MISSING_THRESHOLD_MEDIUM
    else:
        missing_threshold = MISSING_THRESHOLD_LARGE
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
        if n_unique.get(c, 0) < MAX_OHE_UNIQUE and (n_unique.get(c, 0) / max(rows, 1)) < MAX_OHE_UNIQUE_FRAC
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
    text_cols = [c for c in text_cols if c not in sensitive_drop]
    high_card_cols += [
        c for c in text_cols
        if MAX_OHE_UNIQUE < n_unique.get(c, 0) < rows * MAX_TEXT_UNIQUE_FRAC
    ]
    
    use_robust_scaling = profile.get("use_robust_scaling", False)
    handle_outliers = profile.get("handle_outliers", False)
    robust_imputation = profile.get("robust_imputation", False)

    # Small datasets: RobustScaler only — clipping risks losing too much information
    # Large datasets: RobustScaler with quantile clamping (unit_variance clips extremes)
    if handle_outliers and rows >= OUTLIER_CLIP_MIN_ROWS:
        scaler = RobustScaler(unit_variance=True)
    elif use_robust_scaling or handle_outliers:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler(with_mean=True)

    if robust_imputation:
        continuous_imputer = SimpleImputer(strategy="median", add_indicator=True)
        ordinal_imputer = SimpleImputer(strategy="median", add_indicator=True)
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="__missing__")
        high_card_imputer = SimpleImputer(strategy="constant", fill_value="__missing__")
    else:
        continuous_imputer = SimpleImputer(strategy="median")
        ordinal_imputer = SimpleImputer(strategy="median")
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        high_card_imputer = SimpleImputer(strategy="most_frequent")

    continuous_steps = [
        ("imputer", continuous_imputer),
    ]
    if profile.get("use_feature_engineering"):
        continuous_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    continuous_steps.append(("scaler", scaler))

    continuous_transformer = Pipeline(steps=continuous_steps)

    ordinal_transformer = Pipeline(steps=[
        ("imputer", ordinal_imputer),
    ])

    # Keep one-hot output sparse to avoid densifying wide categorical spaces.
    # scikit-learn renamed `sparse` -> `sparse_output` (v1.2+). Support both.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_transformer = Pipeline(steps=[
        ("coerce_categorical", FunctionTransformer(_coerce_categorical_values, validate=False)),
        ("imputer", categorical_imputer),
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
            ("coerce_categorical", FunctionTransformer(_coerce_categorical_values, validate=False)),
            ("imputer", high_card_imputer),
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
    # The executor sets use_class_weights when the plan includes
    # consider_imbalance_strategy. Fall back to the raw profile heuristic for
    # direct callers that bypass the executor and invoke select_models() alone.
    use_class_weights = profile.get("use_class_weights")
    if use_class_weights is None:
        use_class_weights = imb >= IMBALANCE_THRESHOLD
    class_weight = "balanced" if use_class_weights else None
    simple_models_only = profile.get("simple_models_only", False)
    
    # Regression models if not a classification task
    if not profile.get("is_classification", True):
        candidates = [
            ("DummyMean", DummyRegressor(strategy="mean")),
            ("LinearRegression", LinearRegression()),
            ("Ridge", Ridge()),
        ]
        if not simple_models_only:
            candidates.extend([
                ("RandomForestRegressor", RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)),
                ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=seed)),
                ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=seed)),
            ])
        return prioritize_candidates(candidates, preferred_model)

    use_regularization = profile.get("use_regularization", False)
    prefer_ensemble = profile.get("prefer_ensemble", False)
    lr_C = LR_C_REGULARISED if use_regularization else LR_C_DEFAULT

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(C=lr_C, max_iter=2000, class_weight=class_weight, solver="saga", tol=1e-3, random_state=seed)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1, class_weight=class_weight
        )),
    ]

    # SVC: small datasets only — O(n²–n³) cost, unused probability estimates dropped
    if rows < SMALL_DATASET_ROWS and cols <= SVC_MAX_COLS:
        candidates.append(("SVC_RBF", SVC(kernel="rbf", probability=False, class_weight=class_weight)))

    # Small datasets: skip complex models to avoid overfitting
    if simple_models_only:
        return prioritize_candidates(candidates, preferred_model)

    # Large datasets: always include ensemble models (bucket: >= 10000)
    # Medium datasets: also include GradientBoosting (bucket: 1000–9999)
    if prefer_ensemble or rows < LARGE_DATASET_ROWS:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(random_state=seed)))
        candidates.append(("HistGradientBoosting", HistGradientBoostingClassifier(random_state=seed)))

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
    apply_oversampling: bool = False,
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
    smote_k_neighbors = _resolve_smote_k_neighbors(y_train) if (apply_oversampling and is_classification) else None

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        if verbose:
            print(f"[Modelling] Training: {name}")
        pipe = _build_training_pipeline(
            preprocessor=preprocessor,
            model_name=name,
            model=model,
            seed=seed,
            apply_oversampling=apply_oversampling,
            is_classification=is_classification,
            smote_k_neighbors=smote_k_neighbors,
        )

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
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _cv_splitter(
    y: pd.Series,
    seed: int,
    rows: int,
    is_classification: bool,
    time_aware: bool = False,
):
    if rows < 4:
        return None

    desired_splits = CV_SPLITS_SMALL if rows < SMALL_DATASET_ROWS else CV_SPLITS_DEFAULT

    if time_aware:
        n_splits = min(desired_splits, rows)
        if n_splits < 2:
            return None
        return TimeSeriesSplit(n_splits=n_splits)

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
        summary["fold_scores"] = np.asarray(scores["test_balanced_accuracy"], dtype=float).tolist()
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
    summary["fold_scores"] = values.tolist()
    return summary


def cross_validate_top_models(
    df: pd.DataFrame,
    target: str,
    training_payload: Dict[str, Any],
    seed: int,
    is_classification: bool = True,
    top_k: int = CV_TOP_K,
    time_aware: bool = False,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X = training_payload.get("X_train")
    y = training_payload.get("y_train")

    if X is None or y is None:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]

    splitter = _cv_splitter(
        y, 
        seed=seed, 
        rows=len(X), 
        is_classification=is_classification, 
        time_aware=time_aware
    )
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


def _param_grid(model_name: str) -> Dict[str, List[Any]]:
    """Return a RandomizedSearchCV param grid keyed with the 'model__' pipeline prefix."""
    grids: Dict[str, Dict[str, List[Any]]] = {
        "RandomForest": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "RandomForestRegressor": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 1.0],
        },
        "GradientBoostingRegressor": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 1.0],
        },
        "HistGradientBoosting": {
            "model__max_iter": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [20, 50, 100],
        },
        "HistGradientBoostingRegressor": {
            "model__max_iter": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [20, 50, 100],
        },
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l1", "l2"],
        },
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    }
    return grids.get(model_name, {})


def _build_random_search(
    pipeline: Any,
    param_grid: Dict[str, List[Any]],
    scoring: str,
    cv: Any,
    seed: int,
    n_jobs: int,
    n_iter: int = TUNE_N_ITER,
) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
    )


def _is_parallel_backend_error(exc: BaseException) -> bool:
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, (PermissionError, NotImplementedError)):
            return True
        if isinstance(current, OSError) and any(
            token in message for token in ("operation not permitted", "loky", "joblib", "sc_sem_nsems_max")
        ):
            return True
        if any(token in message for token in ("loky", "joblib", "sc_sem_nsems_max", "operation not permitted")):
            return True
        current = current.__cause__ or current.__context__
    return False


def tune_best_model(
    training_payload: Dict[str, Any],
    seed: int = 42,
    is_classification: bool = True,
    reduce_tuning_budget: bool = False,
    time_aware: bool = False,
) -> Dict[str, Any]:
    """
    Tune the best model from training_payload using RandomizedSearchCV.

    Looks up a parameter grid for the best model by name, runs a randomised
    search over that grid on the training data, then re-evaluates on the held-out
    test set.  If no grid exists for the model (e.g. Dummy, SVC) the payload is
    returned unchanged.

    Args:
        training_payload: dict returned by train_models()
        seed: random seed for reproducibility
        is_classification: True for classification tasks

    Returns:
        Updated training_payload with 'best' replaced by the tuned model.
    """
    best = training_payload.get("best", {})
    model_name = best.get("name", "")
    pipeline = best.get("pipeline")
    X_train = training_payload.get("X_train")
    y_train = training_payload.get("y_train")
    X_test = training_payload.get("X_test")
    y_test = training_payload.get("y_test")

    if pipeline is None or X_train is None or y_train is None:
        return training_payload

    param_grid = _param_grid(model_name)
    if not param_grid:
        return training_payload

    # Guard: need at least TUNE_CV_SPLITS × 2 samples to run CV safely
    if len(y_train) < TUNE_CV_SPLITS * 2:
        return training_payload

    n_iter = TUNE_REDUCED_N_ITER if reduce_tuning_budget else TUNE_N_ITER

    if reduce_tuning_budget and len(y_train) > TUNE_MAX_ROWS:
        stratify = y_train if is_classification and y_train.nunique() > 1 and y_train.value_counts().min() >= 2 else None
        try:
            X_train_tune, _, y_train_tune, _ = train_test_split(
                X_train, y_train, train_size=TUNE_MAX_ROWS, random_state=seed, stratify=stratify
            )
            X_train, y_train = X_train_tune, y_train_tune
        except ValueError:
            X_train_tune, _, y_train_tune, _ = train_test_split(
                X_train, y_train, train_size=TUNE_MAX_ROWS, random_state=seed
            )
            X_train, y_train = X_train_tune, y_train_tune

    if time_aware:
        cv = TimeSeriesSplit(n_splits=TUNE_CV_SPLITS)
    elif is_classification:
        cv = StratifiedKFold(n_splits=TUNE_CV_SPLITS, shuffle=True, random_state=seed)
        scoring = "balanced_accuracy"
    else:
        cv = KFold(n_splits=TUNE_CV_SPLITS, shuffle=True, random_state=seed)
        scoring = "r2"

    search = _build_random_search(
        pipeline=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        seed=seed,
        n_jobs=-1,
        n_iter=n_iter,
    )

    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            search.fit(X_train, y_train)
    except Exception as exc:
        if not _is_parallel_backend_error(exc):
            raise
        search = _build_random_search(
            pipeline=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            seed=seed,
            n_jobs=1,
            n_iter=n_iter,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            search.fit(X_train, y_train)

    tuned_pipeline = search.best_estimator_
    y_pred = tuned_pipeline.predict(X_test)

    if is_classification:
        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        }
    else:
        metrics = {
            "model": model_name,
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

    tuned_best = {
        **best,
        "pipeline": tuned_pipeline,
        "metrics": metrics,
        "y_pred": y_pred,
        "tuned": True,
        "best_params": search.best_params_,
    }

    # Reflect the tuned result back into the results list
    updated_results = []
    for r in training_payload.get("results", []):
        if r.get("name") == model_name:
            updated_results.append({**r, **tuned_best})
        else:
            updated_results.append(r)

    return {
        **training_payload,
        "best": tuned_best,
        "results": updated_results,
        "all_metrics": [r["metrics"] for r in updated_results],
    }
