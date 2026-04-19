"""
Agent Configuration

All tuneable thresholds and parameters in one place.
Change a value here and it takes effect everywhere in the pipeline.

Sections:
- Dataset size buckets    — what counts as small / medium / large
- Preprocessing           — missing data, cardinality, scaling
- Model training          — tree size, regularisation, CV folds
- Reflector thresholds    — what performance levels trigger issues
- Profiler thresholds     — what signals the profiler flags
"""

# ── Dataset size buckets ──────────────────────────────────────────────────────
# Used by: select_models, build_preprocessor, _cv_splitter

SMALL_DATASET_ROWS = 1_000    # below this: simpler models, more CV folds, looser missing threshold
LARGE_DATASET_ROWS = 10_000   # above this: always include ensemble models

# Planner scenario thresholds
HIGH_DIMENSIONAL_MIN_COLS = 100       # wide datasets benefit from simpler, more regularised plans
HIGH_DIMENSIONAL_COL_RATIO = 0.25     # also treat p/n-heavy datasets as high-dimensional
PLANNER_CV_MAX_WORKLOAD = 2_000_000   # rows * cols budget above which CV is skipped
PLANNER_CV_MAX_COLS = 250             # very wide datasets skip CV to control compute

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Used by: build_preprocessor

# Maximum % of missing values allowed before a column is dropped entirely
MISSING_THRESHOLD_SMALL  = 60.0   # small datasets  (< SMALL_DATASET_ROWS rows)
MISSING_THRESHOLD_MEDIUM = 50.0   # medium datasets
MISSING_THRESHOLD_LARGE  = 40.0   # large datasets  (>= LARGE_DATASET_ROWS rows)

# % missing above which a column triggers the handle_severe_missing_data plan step
SEVERE_MISSING_THRESHOLD = 20.0

# Categorical cardinality limits for OneHotEncoder vs TargetEncoder
MAX_OHE_UNIQUE       = 50     # columns with more unique values go to TargetEncoder
MAX_OHE_UNIQUE_FRAC  = 0.05   # or if unique values > 5% of rows

# Outlier detection: minimum row count before RobustScaler clips extremes
OUTLIER_CLIP_MIN_ROWS = 1_000

# Maximum columns for SVC — avoids O(n²–n³) cost on wide datasets
SVC_MAX_COLS = 50

# Feature importance coloring threshold — bars above this value are highlighted
FEATURE_IMPORTANCE_THRESHOLD = 0.10

# ── Model training ────────────────────────────────────────────────────────────
# Used by: select_models, cross_validate_top_models

N_ESTIMATORS          = 300   # trees in RandomForest / GradientBoosting
LR_C_DEFAULT          = 1.0   # LogisticRegression regularisation (higher = less regularised)
LR_C_REGULARISED      = 0.1   # used when apply_regularization is in the plan
IMBALANCE_THRESHOLD   = 3.0   # imbalance ratio above which class_weight='balanced' is applied

CV_SPLITS_SMALL       = 5     # folds for small datasets (< SMALL_DATASET_ROWS rows)
CV_SPLITS_DEFAULT     = 3     # folds for medium / large datasets (>= SMALL_DATASET_ROWS rows)
CV_TOP_K              = 2     # how many top models get cross-validated (medium / large datasets)
CV_TOP_K_SMALL        = 3     # more candidates for small datasets — each fold is fast

# ── Reflector thresholds ──────────────────────────────────────────────────────
# Used by: reflect, derive_run_verdict

# F1 thresholds — adjusted down when class imbalance makes minority classes hard
F1_THRESHOLD_BALANCED          = 0.60   # default: balanced dataset
F1_THRESHOLD_IMBALANCED        = 0.50   # imbalance ratio >= IMBALANCE_THRESHOLD
F1_THRESHOLD_SEVERE_IMBALANCE  = 0.45   # imbalance ratio >= IMBALANCE_VERY_SEVERE

IMBALANCE_VERY_SEVERE = 5.0   # ratio above which F1 threshold is relaxed further

# Cross-validation consistency — how much the held-out split can differ from CV mean
CV_GAP_THRESHOLD_CLS  = 0.08   # balanced accuracy gap that triggers a warning
CV_GAP_THRESHOLD_REG  = 0.15   # R² gap that triggers a warning
CV_STD_THRESHOLD_CLS  = 0.05   # balanced accuracy std that flags instability
CV_STD_THRESHOLD_REG  = 0.10   # R² std that flags instability

# Baseline comparison — minimum improvement over dummy model to be meaningful
BASELINE_MIN_IMPROVEMENT_CLS             = 0.05   # balanced accuracy delta
BASELINE_MIN_IMPROVEMENT_REG             = 0.05   # R² delta
BASELINE_MIN_IMPROVEMENT_SEVERE_IMBALANCE = 0.10  # stricter bar when imbalance ratio >= IMBALANCE_VERY_SEVERE

# Near-perfect suspicion — models above this are flagged if multiple hit it
NEAR_PERFECT_THRESHOLD = 0.99

# Model diversity — if all models score within this range, flag low diversity
DIVERSITY_GAP_THRESHOLD = 0.05

# Statistical significance alpha level for paired t-test
SIGNIFICANCE_ALPHA = 0.05

# R² below this is considered too low to be useful
R2_LOW_THRESHOLD = 0.10

# ── Hyperparameter tuning ─────────────────────────────────────────────────────
# Used by: tune_best_model

TUNE_N_ITER    = 20   # number of random parameter combinations to try
TUNE_CV_SPLITS = 3    # CV folds used inside RandomizedSearchCV

# ── Cost-Awareness ────────────────────────────────────────────────────────────
COMPUTE_COST_THRESHOLD = 500_000 # rows * cols above which tuning is scaled down
TUNE_REDUCED_N_ITER    = 5       # reduced iterations for large datasets
TUNE_MAX_ROWS          = 10_000  # maximum rows used for tuning if budget is reduced

# ── Profiler thresholds ───────────────────────────────────────────────────────
# Used by: data_profiler

# Mutual information threshold for leakage detection (normalised, 0–1 scale)
LEAKAGE_MI_THRESHOLD = 0.90

# Fraction of rows a single value must appear in to flag a near-constant column
NEAR_CONSTANT_THRESHOLD = 0.95

# Fraction of rows with IQR outliers above which a column is flagged
OUTLIER_FRACTION_THRESHOLD = 0.05

# Minimum absolute correlation to flag a high-correlation pair
HIGH_CORR_THRESHOLD = 0.80

# Correlation above which the planner drops the weaker of a correlated pair
HIGH_CORR_DROP_THRESHOLD = 0.95

# ── Ethics & Fairness ─────────────────────────────────────────────────────────
# Keywords used to flag potentially sensitive attributes for fairness audits
SENSITIVE_COLUMN_KEYWORDS = {
    "age", "gender", "sex", "race", "ethnicity", "religion",
    "disability", "nationality", "marital_status", "pregnancy"
}
