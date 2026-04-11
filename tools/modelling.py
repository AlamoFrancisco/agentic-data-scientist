import profile
from typing import Any, Dict, List, Optional, Tuple

import os
from numpy.random import seed
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

    # Drop high cardinality categoricals
    n_unique = profile.get("n_unique_by_col", {})
    multi_cat_cols = [
        c for c in multi_cat_cols
        if n_unique.get(c, 0) < 50 and (n_unique.get(c, 0) / max(rows, 1)) < 0.05
    ]
    
    continuous_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
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

    return ColumnTransformer(
        transformers=[
            ("cont", continuous_transformer, cont_cols),
            ("ord", ordinal_transformer, ord_cols),
            ("cat", categorical_transformer, bin_cat_cols + multi_cat_cols),
        ],
        remainder="drop",
    )

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

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight
        )),
    ]

    if rows <= 50000:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(random_state=seed)))

    # SVC can be expensive after one-hot; keep for smaller problems
    if rows <= 20000 and cols <= 200:
        candidates.append(("SVC_RBF", SVC(kernel="rbf", probability=True, class_weight=class_weight)))

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
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

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
        })

    # Sort by appropriate metric — R² for regression, balanced accuracy + F1 for classification
    if is_classification:
        results.sort(key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]), reverse=True)
    else:
        results.sort(key=lambda r: r["metrics"]["r2"], reverse=True)

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results],
    }
