import optuna
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.config import OPTUNA_CONFIG

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _xgboost_objective(trial, X, y, cv, scoring):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()

def _rf_objective(trial, X, y, cv, scoring):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'class_weight': 'balanced'
    }
    model = RandomForestClassifier(**params)
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()

def optuna_tune_xgboost(X_train, y_train):
    cfg = OPTUNA_CONFIG
    cv = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=cfg['random_state'])
    study = optuna.create_study(direction=cfg['direction'])
    study.optimize(lambda t: _xgboost_objective(t, X_train, y_train, cv, cfg['scoring']), n_trials=cfg['n_trials'])
    best_model = XGBClassifier(**study.best_params, eval_metric='logloss', use_label_encoder=False)
    best_model.fit(X_train, y_train)
    return best_model, study.best_params, study

def optuna_tune_rf(X_train, y_train):
    cfg = OPTUNA_CONFIG
    cv = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=cfg['random_state'])
    study = optuna.create_study(direction=cfg['direction'])
    study.optimize(lambda t: _rf_objective(t, X_train, y_train, cv, cfg['scoring']), n_trials=cfg['n_trials'])
    best_model = RandomForestClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    return best_model, study.best_params, study
