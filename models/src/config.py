from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODELS = {
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    'RandomForest': RandomForestClassifier(class_weight='balanced'),
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced')
}

PARAM_GRIDS = {
    'XGBoost': {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    },
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0]
    }
}

TRAIN_CONFIG = {
    'test_size': 0.2,
    'n_splits': 5,
    'random_state': 42
}

OPTUNA_CONFIG = {
    'n_trials': 50,
    'direction': 'maximize',
    'n_splits': 5,
    'random_state': 42,
    'scoring': 'f1_weighted'
}
