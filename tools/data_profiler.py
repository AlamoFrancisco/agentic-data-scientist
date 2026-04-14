import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

NUMERIC_SCHEMA_TYPES = {"ordinal", "continuous"}


def _integer_like_fraction(series: pd.Series, int_tol: float = 1e-6) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(np.mean(np.abs(numeric - np.round(numeric)) <= int_tol))


def _infer_numeric_schema_type(
    series: pd.Series,
    max_unique: int = 20,
    int_tol: float = 1e-6,
) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "continuous"

    n_unique = int(numeric.nunique(dropna=True))
    frac_int = _integer_like_fraction(numeric, int_tol=int_tol)
    if 0 < n_unique <= max_unique and frac_int >= 0.95:
        return "ordinal"
    return "continuous"


def _is_boolean_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    if pd.api.types.is_bool_dtype(non_null):
        return True

    numeric = pd.to_numeric(non_null, errors="coerce")
    if not numeric.isna().any():
        values = set(float(v) for v in numeric.unique().tolist())
        if values <= {0.0, 1.0} and values:
            return True

    normalized = {str(value).strip().lower() for value in non_null.unique().tolist()}
    boolean_pairs = (
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
        {"t", "f"},
    )
    return any(normalized == pair for pair in boolean_pairs)


def infer_schema(df: pd.DataFrame, cat_max_unique: int = 20) -> Dict[str, str]:
    """
    Classify each column as ordinal, continuous, categorical, boolean, datetime,
    text, or all_missing.
    """
    out = {}
    for col in df.columns:
        s = df[col]
        kind = s.dtype.kind
        if s.isna().all():
            out[col] = "all_missing"
        elif kind == "b" or _is_boolean_like(s):     # boolean / boolean-like
            out[col] = "boolean"
        elif kind in "ifc":   # int, float, complex
            out[col] = _infer_numeric_schema_type(s, max_unique=cat_max_unique)
        elif kind == "M":     # datetime
            out[col] = "datetime"
        else:
            n_unique = s.dropna().nunique()
            out[col] = "categorical" if n_unique <= cat_max_unique else "text"
    return out

def _score_target_candidate(
    col: str,
    df: pd.DataFrame,
    schema: Dict[str, str],
    nunique: Dict[str, int],
    last_col: str,
) -> float:
    """
    Score a column as a target candidate. Higher is better.
    """
    TARGET_NAMES = {
        "target", "label", "y", "outcome"
    }
    score = 0.0
    n = len(df)
    u = nunique.get(col, 0)

    # Name match — strongest signal
    if col.lower() in TARGET_NAMES:
        score += 3

    # Cardinality signals
    if 2 <= u <= 20:
        score += 2
    if 2 <= u <= 5:
        score += 1  # bonus for very low cardinality

    # Last column — weak positional signal
    if col == last_col:
        score += 1

    # Penalties
    if n > 0 and (u / n) > 0.5:
        score -= 3  # ID-like: too many unique values
    if schema.get(col) == "text":
        score -= 2
    if schema.get(col) == "all_missing":
        score -= 5

    return score


def infer_target_column(df: pd.DataFrame, return_scores: bool = False):
    """
    Score every column as a target candidate and return the best column name.
    When return_scores=True, also returns the scores dict as a second value.
    Returns None if no column scores above 0.
    """
    schema = infer_schema(df)
    nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    last_col = df.columns[-1]

    scores = {
        col: _score_target_candidate(col, df, schema, nunique, last_col)
        for col in df.columns
    }

    best = max(scores, key=lambda c: scores[c])
    result = best if scores[best] > 0 else None

    if return_scores:
        return result, scores
    return result


def is_classification_target(series: pd.Series) -> bool:
    # String or category columns are always classification
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True
    
    # Float columns are usually continuous — but if all values are whole numbers
    # and cardinality is low, they are likely class labels stored as floats
    if series.dtype == "float64":
        non_null = series.dropna()
        if len(non_null) > 0 and (non_null % 1 == 0).all() and non_null.nunique() <= 20:
            return True
        return False
    
    # Integer columns with few unique values are classification
    uniq = series.nunique(dropna=True)
    n = len(series)
    if n > 0 and (uniq / n) <= 0.05:
        return True
    
    return uniq <= 50

def dataset_fingerprint(df: pd.DataFrame, target: str, file_path: str = "") -> str:
    # Stable hash using filename + target + column names
    # Row count excluded — changes after deduplication
    cols = ",".join(df.columns.astype(str).tolist())
    base = f"{file_path}|{target}|{cols}"
    return "fp_" + hashlib.md5(base.encode()).hexdigest()[:12]

def detect_near_constant(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Return columns where a single value appears in >= threshold fraction of rows.
    These carry almost no signal and can cause issues with one-hot encoding.
    Adapted from EDA notebook near-constant detection logic.
    """
    near_const = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq >= threshold:
            near_const.append(col)
    return near_const


def detect_outliers(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.05) -> List[str]:
    """
    Return numeric columns where more than `threshold` fraction of values fall
    outside the IQR fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
    Adapted from EDA notebook IQR outlier detection logic.
    """
    outlier_cols = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n_out / len(s) > threshold:
            outlier_cols.append(col)
    return outlier_cols


def correlation_report(
    df: pd.DataFrame,
    schema: Dict[str, str],
    top_n: int = 20,
    min_abs_corr: float = 0.0,
) -> Dict[str, Any]:
    """
    Return the numeric correlation matrix and the strongest pairwise correlations.
    Adapted from the EDA notebook correlation_report() logic.
    """
    numeric_cols = [c for c, t in schema.items() if t in NUMERIC_SCHEMA_TYPES]

    if len(numeric_cols) < 2:
        return {"corr": None, "high_corr_pairs": []}

    corr = df[numeric_cols].corr(numeric_only=True)

    pairs: List[Dict[str, Any]] = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a = numeric_cols[i]
            b = numeric_cols[j]
            v = float(corr.loc[a, b])
            av = abs(v)

            if av < min_abs_corr:
                continue

            pairs.append({
                "col_a": a,
                "col_b": b,
                "corr": float(round(v, 4)),
                "abs_corr": float(round(av, 4)),
            })

    pairs.sort(key=lambda d: d["abs_corr"], reverse=True)
    return {"corr": corr, "high_corr_pairs": pairs[:top_n]}


def serialize_correlation_matrix(corr: Optional[pd.DataFrame]) -> Optional[Dict[str, Dict[str, float]]]:
    if corr is None:
        return None

    return {
        str(row): {
            str(col): float(round(corr.loc[row, col], 4))
            for col in corr.columns
        }
        for row in corr.index
    }

def ordinal_report(
    df: pd.DataFrame,
    schema: Dict[str, str],
    nunique_map: Dict[str, int],
    max_unique: int = 20,
    int_tol: float = 1e-6,
) -> List[tuple]:
    out = []

    for col, t in schema.items():
        if t != "ordinal" or col not in df.columns:
            continue

        u = int(nunique_map.get(col, 0))
        if u == 0 or u > max_unique:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        frac_int = _integer_like_fraction(s, int_tol=int_tol)
        out.append((col, u, round(frac_int, 3)))

    out.sort(key=lambda x: x[1])
    return out


def build_feature_types(
    schema: Dict[str, str],
    nunique_map: Dict[str, int],
    target: str,
) -> Dict[str, Any]:
    feature_schema = {col: kind for col, kind in schema.items() if col != target}

    ordinal_cols = [col for col, kind in feature_schema.items() if kind == "ordinal"]
    continuous_cols = [col for col, kind in feature_schema.items() if kind == "continuous"]
    categorical_cols = [col for col, kind in feature_schema.items() if kind == "categorical"]
    boolean_cols = [col for col, kind in feature_schema.items() if kind == "boolean"]
    binary_cats = boolean_cols + [col for col in categorical_cols if nunique_map.get(col, 0) <= 2]
    multiclass_cats = [col for col in categorical_cols if nunique_map.get(col, 0) > 2]

    return {
        "numeric": {
            "ordinal": ordinal_cols,
            "continuous": continuous_cols,
        },
        "categorical": {
            "binary": binary_cats,
            "multiclass": multiclass_cats,
        },
        "text": [col for col, kind in feature_schema.items() if kind == "text"],
        "datetime": [col for col, kind in feature_schema.items() if kind == "datetime"],
        "all_missing": [col for col, kind in feature_schema.items() if kind == "all_missing"],
    }


def leakage_report(
    df: pd.DataFrame,
    target: str,
    schema: Dict[str, str],
    is_classification: bool = True,
    threshold: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Detect potentially leaky features using mutual information.
    Normalises MI by target entropy so threshold is scale-invariant.
    Flags features where normalised MI >= threshold (default 0.9).
    """
    feature_cols = [c for c in df.columns if c != target]
    y = df[target].copy()
    X = df[feature_cols].copy()

    # Encode categoricals and text as integers for mutual_info
    for col in feature_cols:
        if schema.get(col) in ("categorical", "text", "boolean", "all_missing"):
            X[col] = pd.factorize(X[col])[0].astype(float)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.median(numeric_only=True))

    # Encode target — classification uses label-encoded integers;
    # regression uses raw values so MI reflects actual continuous relationships
    if is_classification:
        y_encoded = pd.factorize(y)[0]
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        # Normalise by target entropy (meaningful for discrete targets)
        vc = pd.Series(y_encoded).value_counts(normalize=True)
        target_entropy = float(-np.sum(vc * np.log(vc + 1e-10)))
        if target_entropy == 0:
            return []
        normaliser = target_entropy
    else:
        y_encoded = y.astype(float).values
        mi_scores = mutual_info_regression(X, y_encoded, random_state=42)
        # Normalise by max MI score — keeps threshold scale-invariant for regression
        max_mi = float(np.max(mi_scores)) if np.max(mi_scores) > 0 else 1.0
        normaliser = max_mi

    leaky = []
    for col, score in zip(feature_cols, mi_scores):
        normalised = float(score) / normaliser
        if normalised >= threshold:
            leaky.append({
                "column": col,
                "mi_score": round(float(score), 4),
                "normalised_mi": round(normalised, 4),
            })

    leaky.sort(key=lambda x: x["normalised_mi"], reverse=True)
    return leaky


def scale_range_report(df: pd.DataFrame, schema: Dict[str, str]) -> Dict[str, Any]:
    """
    Compute min/max range per numeric column and detect scale mismatches.
    A mismatch is flagged when the largest range is >= 50x the median range.
    Adapted from EDA notebook scale_range_report() logic.
    """
    ranges = []
    for col, t in schema.items():
        if t not in NUMERIC_SCHEMA_TYPES or col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mn, mx = float(s.min()), float(s.max())
        rg = mx - mn
        ranges.append({"column": col, "min": round(mn, 6), "max": round(mx, 6), "range": round(rg, 6)})

    ranges.sort(key=lambda d: d["range"], reverse=True)

    range_values = [d["range"] for d in ranges if d["range"] > 0]
    if len(range_values) < 2:
        scale_mismatch = False
        scale_range_ratio = 1.0
    else:
        max_r = max(range_values)
        med_r = float(np.median(range_values))
        ratio = max_r / med_r if med_r else float("inf")
        scale_range_ratio = round(ratio, 2)
        scale_mismatch = ratio >= 50

    return {
        "scale_range": ranges,
        "scale_range_ratio": scale_range_ratio,
        "scale_mismatch": scale_mismatch,
    }


def profile_dataset(df: pd.DataFrame, target: str, target_source: str = "inferred", target_candidate_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    profile: Dict[str, Any] = {}
    profile["target_source"] = target_source
    if target_candidate_scores is not None:
        profile["target_candidate_scores"] = {str(k): round(float(v), 2) for k, v in target_candidate_scores.items()}

    profile["schema"] = infer_schema(df)

    profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    missing = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

    profile["target"] = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))

    profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}
    profile["feature_types"] = build_feature_types(profile["schema"], profile["n_unique_by_col"], target)

    notes = []
    if profile["shape"]["rows"] < 1000:
        notes.append("Small dataset (<1000 rows): prefer simpler models / guard against overfitting.")
    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 columns): watch one-hot expansion and overfitting.")

    ord_cols = ordinal_report(
        df,
        profile["schema"],
        profile["n_unique_by_col"],
        max_unique=20,
    )
    continuous_cols = profile["feature_types"]["numeric"]["continuous"]

    profile["ordinal"] = ord_cols
    profile["has_ordinal"] = len(ord_cols) > 0
    profile["ordinal_cols"] = [c for c, _, _ in ord_cols]
    profile["continuous_cols"] = continuous_cols
    if profile["has_ordinal"]:
        notes.append(
            f"Ordinal-like numeric columns detected: {profile['ordinal_cols'][:5]}. "
            "These may represent ordered levels rather than continuous measurements."
        )

    # Duplicate detection — adapted from EDA notebook
    dup_count = int(df.duplicated().sum())
    profile["duplicate_count"] = dup_count
    profile["duplicate_pct"] = round(dup_count / max(len(df), 1) * 100, 2)
    if dup_count > 0:
        notes.append(f"Found {dup_count} duplicate rows ({profile['duplicate_pct']}%): dropped before training.")

    # Near-constant column detection — adapted from EDA notebook
    X = df.drop(columns=[target])
    near_const = detect_near_constant(X)
    profile["near_constant_cols"] = near_const
    if near_const:
        notes.append(f"Near-constant columns ({len(near_const)}): {near_const[:5]}. Excluded from features.")

    # Outlier detection (IQR method) — adapted from EDA notebook
    numeric_feature_cols = profile["feature_types"]["numeric"]["ordinal"] + profile["feature_types"]["numeric"]["continuous"]
    outlier_cols = detect_outliers(df, numeric_feature_cols)
    profile["outlier_cols"] = outlier_cols
    if outlier_cols:
        notes.append(f"Outliers detected (>5% IQR) in: {outlier_cols[:5]}. Consider robust scaling.")

    leaky = leakage_report(df, target, profile["schema"], is_classification=profile["is_classification"])
    profile["leaky_cols"] = leaky
    if leaky:
        names = [c["column"] for c in leaky]
        notes.append(f"Potential leakage detected in: {names}. Check before training.")

    sr = scale_range_report(df, profile["schema"])
    profile["scale_range"] = sr["scale_range"]
    profile["scale_range_ratio"] = sr["scale_range_ratio"]
    profile["scale_mismatch"] = sr["scale_mismatch"]
    if sr["scale_mismatch"]:
        notes.append(f"Scale mismatch detected (ratio={sr['scale_range_ratio']}x): consider robust scaling.")

    cr = correlation_report(df, profile["schema"])
    profile["correlation"] = serialize_correlation_matrix(cr["corr"])
    profile["high_corr_pairs"] = cr["high_corr_pairs"]
    max_corr = max((p.get("abs_corr", 0.0) for p in profile["high_corr_pairs"]), default=0.0)
    profile["max_abs_corr"] = round(float(max_corr), 4)
    profile["high_corr_present"] = profile["max_abs_corr"] >= 0.6

    profile["notes"] = notes

    # Class balance if classification
    if profile["is_classification"]:
        vc = y.value_counts(dropna=False)
        profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            ratio = float(vc.max() / max(vc.min(), 1))
        else:
            ratio = 1.0
        profile["imbalance_ratio"] = round(ratio, 3)
        if ratio >= 3.0:
            profile["notes"].append("Imbalance detected (ratio >= 3.0): prioritise macro metrics / balanced accuracy.")
    else:
        profile["class_counts"] = None
        profile["imbalance_ratio"] = None
        profile["notes"].append("Regression target detected: using regression models and metrics.")

    return profile
