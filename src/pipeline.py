import pandas as pd
from src.data import split_data
from src.model import train_and_tune, evaluate_model
from src.config import MODELS, PARAM_GRIDS, TRAIN_CONFIG
from src.utils import save_model

def run_multi_model_pipeline(df, target_col, auto_save_best=True, save_path='../models/best_model.joblib'):
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target_col, 
        test_size=TRAIN_CONFIG['test_size'], 
        random_state=TRAIN_CONFIG['random_state']
    )
    
    results = []
    best_overall_model = None
    best_overall_score = -1
    best_overall_name = ""
    best_overall_params = {}
    best_y_pred = None
    
    for model_name, estimator in MODELS.items():
        print(f"Training and tuning {model_name}...")
        param_grid = PARAM_GRIDS.get(model_name, {})
        
        best_model, best_params = train_and_tune(
            estimator,
            X_train, 
            y_train, 
            param_grid, 
            n_splits=TRAIN_CONFIG['n_splits'], 
            random_state=TRAIN_CONFIG['random_state']
        )
        
        metrics = evaluate_model(best_model, X_test, y_test)
        
        results.append({
            'Model': model_name,
            **metrics
        })
        
        if metrics['F1 Score'] > best_overall_score:
            best_overall_score = metrics['F1 Score']
            best_overall_model = best_model
            best_overall_name = model_name
            best_overall_params = best_params
            best_y_pred = best_model.predict(X_test)
            
    comparison_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
    
    if auto_save_best and best_overall_model is not None:
        save_model(best_overall_model, save_path)
        print(f"Best model saved to {save_path}")
    
    return {
        'comparison_df': comparison_df,
        'best_model': best_overall_model,
        'best_model_name': best_overall_name,
        'best_params': best_overall_params,
        'feature_names': list(X_train.columns),
        'y_test': y_test,
        'best_y_pred': best_y_pred
    }
