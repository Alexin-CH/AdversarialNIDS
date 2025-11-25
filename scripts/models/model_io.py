import os
import joblib

def save_sklearn_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_sklearn_model(path):
    return joblib.load(path)