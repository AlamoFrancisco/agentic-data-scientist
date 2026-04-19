# CE888 Agentic Data Scientist

**Name:** Francisco Antonio Alamo Rios  
**Assignment:** Offline Agentic AI for Data Science  
**Module:** CE888  
**Academic Year:** 2025/2026

---

## Overview

An **Offline Agentic Data Scientist** — an autonomous agent that performs end-to-end data science tasks (classification and regression) without relying on Large Language Models.

The agent uses **rule-based reasoning, heuristics, and persistent memory** to autonomously:
- Profile datasets (scale, leakage, imbalance, missing data, near-constant features)
- Plan execution workflows adapted to dataset characteristics
- Train and evaluate classification and regression models (Dummy, Linear/Logistic, Ridge, Random Forest, Gradient Boosting, HistGradientBoosting, SVC)
- Reflect on results and trigger targeted replanning
- Learn from experience across runs via persistent memory

---

## Quick Start

```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt

# Run on a dataset
python3 run_agent.py --data data/penguins.csv --target species
python3 run_agent.py --data data/titanic.csv --target survived
python3 run_agent.py --data data/Sales.csv --target revenue_usd

# Auto-detect target column
python3 run_agent.py --data data/penguins.csv --target auto
```

Outputs are written to `outputs/[timestamp]/`.

---

## Project Structure

```
ce888-agentic-data-scientist/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies (scikit-learn>=1.3.0 required)
├── AGENT_SIGNAL_MAP.md             # Reference: signals → plan → executor → reflector
├── .gitignore
│
├── agentic_data_scientist.py      # Core agent (Executor + orchestration)
├── run_agent.py                   # CLI entry point
│
├── agents/
│   ├── planner.py                 # Dataset-adaptive planning logic
│   ├── reflector.py               # Performance analysis and replanning
│   └── memory.py                  # Persistent experience store (agent_memory.json)
│
├── tools/
│   ├── data_profiler.py          # Dataset profiling (MI leakage, scale, correlations)
│   ├── modelling.py              # Model training with preprocessing pipeline
│   └── evaluation.py             # Metrics and reporting
│
├── data/
│   ├── README.md                 # Dataset documentation and trigger conditions
│   ├── penguins.csv              # Small classification (344 rows)
│   ├── titanic.csv               # Small classification with leakage (891 rows)
│   ├── digits.csv                # Medium classification, near-constant features (1797 rows)
│   ├── Sales.csv                 # Large regression with target encoding (30000 rows)
│   ├── WineQuality.csv           # Medium regression (1700 rows)
│   └── demo.csv                  # 20-row smoke test dataset
│
├── outputs/                       # Generated outputs (gitignored)
│   └── .gitkeep
│
├── report/
│   └── REPORT.md                 # Technical report (3000-4000 words)
│
└── tests/
    ├── test_planner.py
    ├── test_reflector.py
    ├── test_data_profiler.py
    ├── test_modelling.py
    ├── test_memory.py
    ├── test_smoke_run.py          # End-to-end smoke tests
    └── sanity_check.py
```

---

## Features Implemented

### 1. Planner (`agents/planner.py`)
Dataset-adaptive planning based on profiler signals:

| Signal | Plan step |
|--------|-----------|
| rows < 1000 | `apply_regularization` + `use_simple_models_only` |
| high-dimensional data (`feature_cols >= 100` or `feature_cols / rows >= 0.25`) | `apply_regularization` + `use_simple_models_only` |
| rows ≥ 10,000 and not high-dimensional | `use_ensemble_models` |
| feature_cols ≤ 15 | `apply_feature_engineering` |
| scale mismatch (max/median range ≥ 50) | `apply_robust_scaling` |
| hard leakage evidence | `drop_leaky_features` |
| high-cardinality categoricals (n_unique > 50) | `apply_target_encoding` |
| correlated features (abs_corr ≥ 0.95) | `drop_correlated_features` |
| near-constant columns (≥ 95% dominant value) | `drop_near_constant_features` |
| imbalance ratio ≥ 5.0 | `apply_oversampling` |
| imbalance ratio ≥ 3.0 and < 5.0 | `consider_imbalance_strategy` |
| sensitive columns detected | `drop_sensitive_features` |
| large tuning workload (`rows * cols` high) | `reduce_tuning_budget` |
| extreme workload (`rows * cols` very high or feature space very wide) | skip `validate_with_cross_validation` |
| datetime columns detected | `use_time_aware_validation` (TimeSeriesSplit) |
| memory hint (prior best model) | `prioritize_model:<name>` |
| memory hint (successful prior plan) | pre-emptively apply prior successful adaptations (Meta-learning) |

### 2. Reflector (`agents/reflector.py`)
Performance-aware reflection with targeted replanning:
- Flags models barely beating dummy (< 0.05 margin)
- Detects suspected overfitting (bal_acc > 0.90 but f1 < 0.70)
- Raises near-perfect performance warnings (≥ 2 non-dummy models with score ≥ 0.99) — both classification and regression
- Acts on numerical instability warnings (overflow/divide-by-zero) from training
- `replan_recommended` gate prevents spurious replans on genuinely well-performing datasets

### 3. Executor (`agentic_data_scientist.py`)
- Handlers for all plan flags from the planner
- Training retry loop (3 attempts)
- Saves failed targets to memory to avoid re-running them

### 4. Memory (`agents/memory.py`)
- Stores best model, metrics, and reflection status per dataset+target
- Failed target tracking — skips targets that previously produced no useful result
- Similarity matching across datasets (size bucket, imbalance, missingness)
- Memory hint passed to Planner to prioritise previously successful models
- Meta-learning — stores successful plan adaptations to pre-emptively fix issues on similar future runs

### 5. Data Profiler (`tools/data_profiler.py`)
Extended profiling signals:
- Scale mismatch detection (`scale_range_report`)
- Mutual information leakage detection — classification uses entropy normalisation, regression uses max-MI normalisation (`leakage_report`)
- Near-constant column detection (≥ 95% dominant value)
- Automatic target column inference with failed-target awareness

### 6. Modelling (`tools/modelling.py`)
- `TargetEncoder` (sklearn ≥ 1.3) for high-cardinality categoricals — fitted inside Pipeline to prevent leakage
- `PolynomialFeatures(degree=2)` for low-dimensional continuous feature engineering
- `RobustScaler` for outlier handling
- `SMOTE` oversampling for severe class imbalance (via `imbalanced-learn`)
- Reduced tuning budget for large workloads
- Size-bucket model selection (small/medium/large)
- Runtime warnings captured and surfaced to Reflector

---

## Running the Agent

### Basic Usage

```bash
python3 run_agent.py --data data/penguins.csv --target species
```

### All Arguments

```bash
python3 run_agent.py \
    --data data/your_dataset.csv \
    --target target_column_name \
    --output_root outputs \
    --seed 42 \
    --test_size 0.2 \
    --max_replans 2 \
    --quiet
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to CSV dataset | required |
| `--target` | Target column or `auto` | required |
| `--output_root` | Output directory | `outputs` |
| `--seed` | Random seed | 42 |
| `--test_size` | Test fraction | 0.2 |
| `--max_replans` | Max replanning attempts | 1 |
| `--quiet` | Reduce logging | False |

### Output Files

Each run creates `outputs/[timestamp]/`:

| File | Contents |
|------|----------|
| `report.md` | Human-readable run summary |
| `eda_summary.json` | Dataset profile |
| `plan.json` | Generated execution plan |
| `metrics.json` | Model performance metrics |
| `reflection.json` | Issues, suggestions, replan status |
| `confusion_matrix.png` | Confusion matrix (classification only) |

---

## Datasets

See [data/README.md](data/README.md) for full documentation of all datasets and the profiler signals each one triggers.

| Dataset | Rows | Task | Key triggers |
|---------|------|------|-------------|
| `penguins.csv` | 344 | Classification | `apply_regularization`, `apply_robust_scaling` |
| `titanic.csv` | 891 | Classification | `drop_leaky_features` (`alive` col), `handle_outliers` |
| `digits.csv` | 1797 | Classification | `drop_near_constant_features` (14 border pixels) |
| `Sales.csv` | 30000 | Regression | `apply_target_encoding`, `drop_sensitive_features`, `reduce_tuning_budget`, soft leakage warning |
| `WineQuality.csv` | 1700 | Regression | Baseline regression pipeline |

---

## Testing

```bash
# Run all tests
python3 -m pytest tests/

# With coverage report
python3 -m pytest --cov=agents --cov=tools --cov=agentic_data_scientist --cov-report=term tests/
```

Current coverage: **87%** total (177 tests, 0 failing).

---

## Submission Checklist

- [x] All code runs without errors
- [x] README.md updated with actual implementation details
- [x] requirements.txt includes all dependencies (`scikit-learn>=1.3.0`, `imbalanced-learn`)
- [x] 5 datasets documented in `data/README.md` (4 evaluation + 1 smoke test)
- [x] Technical report completed — `report/REPORT.md` (~4,000 words)
- [x] Test coverage > 60% (87% achieved)
- [x] All core components extended significantly
- [x] At least 3 advanced features implemented
- [x] Repository is organised for submission

---

## Academic Integrity

This is individual work. AI assistance was used and is disclosed in the technical report.

---

## Important Deadlines

- **Final Project Demonstration:** 21 April 2026, 13:59:59
- **Final Project Code:** 21 April 2026, 13:59:59
