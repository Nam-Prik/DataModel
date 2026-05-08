import os
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f'{name}.joblib')
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path

def load_model(name):
    path = os.path.join(MODELS_DIR, f'{name}.joblib')
    return joblib.load(path)
