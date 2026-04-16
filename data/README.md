## Datasets

### Datasets used in this project

| File | Rows | Cols | Task | Target | Source |
|------|------|------|------|--------|--------|
| `penguins.csv` | 344 | 7 | Classification | `species` (3 classes) | Palmer Penguins dataset — bill/flipper measurements of 3 penguin species |
| `titanic.csv` | 891 | 15 | Classification | `survived` (binary) | Titanic passenger manifest — survival prediction |
| `digits.csv` | 1797 | 65 | Classification | `target` (10 classes) | MNIST-derived 8×8 pixel digit images (sklearn digits dataset) |
| `wine_sklearn.csv` | 178 | 14 | Classification | `wine_class` (3 classes) | sklearn Wine dataset — chemical analysis of 3 cultivars |
| `Sales.csv` | 30000 | 18 | Regression | `revenue_usd` | Synthetic e-commerce sales records |
| `WineQuality.csv` | 1700 | 12 | Regression | `quality` | UCI Wine Quality dataset — physicochemical properties |
| `diabetes_sklearn.csv` | 442 | 11 | Regression | `disease_progression` | sklearn Diabetes dataset — baseline medical features and disease progression |
| `demo.csv` | 20 | 6 | Classification | `label` | Tiny synthetic dataset for smoke testing only |

### Dataset notes

**penguins.csv** — Small classification dataset. Triggers: `apply_regularization`, `use_simple_models_only`, `apply_robust_scaling` (scale mismatch between bill measurements and binary sex column). Species is genuinely well-separated, resulting in near-perfect model performance.

**titanic.csv** — Small classification dataset with real-world data quality issues. Triggers: `handle_outliers` (fare distribution), `handle_severe_missing_data` (deck: 77% missing), `apply_robust_scaling`, `drop_leaky_features` (the `alive` column is a direct text encoding of the target and is dropped via mutual information detection).

**digits.csv** — Medium classification dataset with 64 pixel features. Triggers: `handle_outliers`, `drop_near_constant_features` (14 border pixels with ≥95% zero values). A good test for the near-constant column detection.

**wine_sklearn.csv** — Small multiclass classification dataset. Good for testing the agent on clean numeric chemistry features without obvious leakage or missing-data complications.

**Sales.csv** — Large regression dataset. Triggers: `apply_target_encoding` (model_name: 899 unique values), `drop_leaky_features` (final_price_usd: normalised MI = 1.0 with revenue_usd), `use_ensemble_models`. Illustrates the limitation of per-feature leakage detection: even after dropping final_price_usd, the combination of base_price_usd × discount_percent × units_sold still perfectly reconstructs revenue_usd.

**WineQuality.csv** — Medium regression dataset. Used in smoke tests to verify regression pipeline. All-numeric features, no leakage.

**diabetes_sklearn.csv** — Small-to-medium regression dataset with standardized numeric features. Useful for checking regression behavior on a real dataset without high-cardinality categoricals.

### Notes on demo.csv
`demo.csv` is a 20-row synthetic dataset included only to verify the code runs. It is too small for meaningful model training and should not be used as an evaluation dataset.
