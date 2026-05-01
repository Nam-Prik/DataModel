# Multi-Model Tabular Data Pipeline Usage Guide

Welcome to the data modeling pipeline! This pipeline is designed around the best practices of keeping code modular, configurations separate, and evaluating multiple machine learning algorithms simultaneously.

## Project Structure

Here is what every file in the project does:

```
.
├── data/                               # Directory for saving raw or processed data
│   └── sample_data.csv                 # (Generated) sample dataset for testing
├── docs/
│   └── USAGE.md                        # This documentation file
├── notebooks/
│   └── training_pipeline.ipynb         # Interactive notebook showing how to use the pipeline
├── requirements.txt                    # Project dependencies
└── src/
    ├── __init__.py                     # Makes src a Python module
    ├── config.py                       # Hyperparameter configurations and train settings
    ├── data.py                         # Helper functions to load data and split it
    ├── model.py                        # Core functions for training, tuning (GridSearchCV), and evaluation
    └── pipeline.py                     # The high-level wrapper to run comparisons easily
```

## How to use the Pipeline

The primary entry point of this codebase is the `run_multi_model_pipeline` wrapper located in `src/pipeline.py`.

### 1. Basic Usage

To easily train and compare multiple algorithms (e.g., XGBoost, Random Forest, Logistic Regression), use the following Python snippet in a script or notebook:

```python
import pandas as pd
from src.pipeline import run_multi_model_pipeline

# 1. Load your DataFrame
# Ensure your DataFrame is pre-processed (e.g., categorical variables encoded)
df = pd.read_csv('path_to_your_dataset.csv')

# 2. Run the multi-model pipeline wrapper
# Provide your DataFrame and the name of your target (label) column
results = run_multi_model_pipeline(df, target_col='your_target_column_name')

# 3. View the Model Comparison DataFrame
# This dataframe contains Accuracy, Precision, Recall, and F1 Score for all evaluated models
print(results['comparison_df'])

# 4. Access the Best Model Details (determined by F1 Score)
print(f"Best Model Name: {results['best_model_name']}")
print(f"Best Model Parameters: {results['best_params']}")

# You can now use the best model to predict new data
best_model = results['best_model']
# predictions = best_model.predict(new_data)
```

### 2. Customizing Hyperparameters and Models

**Best Practice:** Do not modify the pipeline logic to change hyperparameter ranges.

If you want to modify what models are tested or their hyperparameter grids, open `src/config.py`.

- **`MODELS`**: Add or remove algorithms you want to test here.
- **`PARAM_GRIDS`**: Update the list of hyperparameter values for GridSearchCV to explore.
- **`TRAIN_CONFIG`**: Modify test size, random seeds, and K-Fold splits.

By keeping these settings in `src/config.py`, the main training code remains clean, making it easier to track configurations and share experiments.
