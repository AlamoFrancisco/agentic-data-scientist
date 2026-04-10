from typing import Any, Dict, List, Optional
import pandas as pd


def infer_schema(df: pd.DataFrame, cat_max_unique: int = 20) -> Dict[str, str]:
    """
    Classify each column as numeric, categorical, boolean, datetime, text, or all_missing.
    Adapted from EDA notebook infer_schema() function.
    """
    out = {}
    for col in df.columns:
        s = df[col]
        kind = s.dtype.kind
        if s.isna().all():
            out[col] = "all_missing"
        elif kind in "ifc":   # int, float, complex
            out[col] = "numeric"
        elif kind == "M":     # datetime
            out[col] = "datetime"
        elif kind == "b":     # boolean
            out[col] = "boolean"
        else:
            n_unique = s.dropna().nunique()
            out[col] = "categorical" if n_unique <= cat_max_unique else "text"
    return out


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic target inference — adapted from EDA notebook infer_target_column().
      - prefer common target-like column names
      - else last column if it has relatively low cardinality
      - fallback: pick numeric column with lowest cardinality
    """
    schema = infer_schema(df)
    nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    n = len(df)

    # Step 1: check for common target names
    candidates = ["target", "label", "class", "y", "outcome"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]

    # Step 2: check last column
    last = df.columns[-1]
    uniq = nunique[last]

    # Skip if almost all values are unique — likely an ID (true IDs are ~100% unique)
    if n > 0 and (uniq / n) > 0.95:
        pass
    # Skip text columns
    elif schema.get(last) == "text":
        pass
    else:
        return last

    # Step 3: fallback — adapted from EDA: pick numeric column with lowest cardinality
    numeric_cols = [c for c, t in schema.items() if t == "numeric"]
    if numeric_cols:
        return min(numeric_cols, key=lambda c: nunique.get(c, float("inf")))

    return None


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



def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = abs(hash(base)) % (10**12)
    return f"fp_{h}"


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


def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    profile: Dict[str, Any] = {}

    # Store schema — adapted from EDA notebook
    profile["schema"] = infer_schema(df)

    profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    missing = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

    profile["target"] = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))

    # Feature types
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
    cat_cols = [c for c in X.columns.astype(str).tolist() if c not in numeric_cols]

    profile["feature_types"] = {"numeric": numeric_cols, "categorical": cat_cols}
    profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}

    notes = []
    if profile["shape"]["rows"] < 1000:
        notes.append("Small dataset (<1000 rows): prefer simpler models / guard against overfitting.")
    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 columns): watch one-hot expansion and overfitting.")

    # Duplicate detection — adapted from EDA notebook
    dup_count = int(df.duplicated().sum())
    profile["duplicate_count"] = dup_count
    profile["duplicate_pct"] = round(dup_count / max(len(df), 1) * 100, 2)
    if dup_count > 0:
        notes.append(f"Found {dup_count} duplicate rows ({profile['duplicate_pct']}%): dropped before training.")

    # Near-constant column detection — adapted from EDA notebook
    near_const = detect_near_constant(X)
    profile["near_constant_cols"] = near_const
    if near_const:
        notes.append(f"Near-constant columns ({len(near_const)}): {near_const[:5]}. Excluded from features.")

    # Outlier detection (IQR method) — adapted from EDA notebook
    outlier_cols = detect_outliers(df, numeric_cols)
    profile["outlier_cols"] = outlier_cols
    if outlier_cols:
        notes.append(f"Outliers detected (>5% IQR) in: {outlier_cols[:5]}. Consider robust scaling.")

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
