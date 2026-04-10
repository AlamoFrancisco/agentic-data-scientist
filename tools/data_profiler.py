from typing import Any, Dict, Optional
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

    # Skip if almost all values are unique — likely an ID
    if n > 0 and (uniq / n) > 0.9:
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
    
    # Float columns are continuous — regression, not classification
    if series.dtype == "float64":
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
