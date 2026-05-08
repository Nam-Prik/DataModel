# Tabular Classification Pipeline — Usage Guide

## Project Structure

```
.
├── data/                              # Place your datasets here
├── docs/
│   └── USAGE.md                       # This file
├── models/                            # Auto-saved trained models (.joblib)
├── notebooks/
│   ├── training_pipeline.ipynb        # Run multi-model GridSearch comparison
│   ├── evaluation.ipynb               # Visualize confusion matrix, ROC, feature importance
│   └── optuna_tuning.ipynb            # Advanced Bayesian hyperparameter tuning
├── requirements.txt
└── src/
    ├── config.py                      # All model configs, param grids, Optuna config
    ├── data.py                        # Load and split data
    ├── evaluate.py                    # Evaluation plots (confusion matrix, ROC, importance)
    ├── model.py                       # Generic train-tune-evaluate logic (GridSearchCV)
    ├── pipeline.py                    # High-level multi-model comparison wrapper
    ├── tuner.py                       # Optuna Bayesian tuning for XGBoost and RandomForest
    └── utils.py                       # save_model / load_model (joblib)
```

---

## Quickstart — Compare All Models

```python
import pandas as pd
from src.pipeline import run_multi_model_pipeline

df = pd.read_csv('data/your_data.csv')         # Clean data ready to train
results = run_multi_model_pipeline(df, target_col='your_target')

# Ranked comparison table (Accuracy, Precision, Recall, F1)
print(results['comparison_df'])

# Best model (auto-saved to models/)
best_model = results['best_model']
print(results['best_model_name'], results['best_params'])
```

---

## Evaluate the Best Model

```python
from src.utils import load_model
from src.evaluate import full_evaluation_report

model = load_model('XGBoost')   # Name matches what was saved
full_evaluation_report(model, X_test, y_test, feature_names, 'XGBoost')
# → Confusion Matrix + ROC Curve + Feature Importance plots
```

---

## Advanced Tuning with Optuna

```python
from src.tuner import optuna_tune_xgboost, optuna_tune_rf

xgb_model, xgb_params, study = optuna_tune_xgboost(X_train, y_train)
rf_model, rf_params, study   = optuna_tune_rf(X_train, y_train)
```

Optuna uses Bayesian optimization — significantly fewer trials to find better results than GridSearch.

---

## Configuration — `src/config.py`

This is the **single place** to control everything:

| Setting | What it does |
|---|---|
| `MODELS` | Add / remove models to compare |
| `PARAM_GRIDS` | Define GridSearch hyperparameter ranges per model |
| `TRAIN_CONFIG` | `test_size`, `n_splits`, `random_state` |
| `OPTUNA_CONFIG` | `n_trials`, `direction`, `scoring` for Optuna tuning |

**Never** change the pipeline code to adjust hyperparameters. Always modify `config.py` only.

---

## Model Persistence

- The **best model** from `run_multi_model_pipeline` is **automatically saved** to `models/<ModelName>.joblib`.
- Optuna-tuned models can be manually saved: `save_model(model, 'XGBoost_optuna')`.
- Load any saved model: `load_model('XGBoost')`.

---

## Recommended Workflow

```
1. Run training_pipeline.ipynb
   → Compares all models, saves best one

2. Run evaluation.ipynb
   → Deep-dive with plots on the saved best model

3. (Optional) Run optuna_tuning.ipynb
   → Squeeze out better performance with Bayesian search

4. Load the final model anywhere:
   model = load_model('XGBoost_optuna')
   predictions = model.predict(new_data)
```
