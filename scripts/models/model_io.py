import os
import joblib

def save_sklearn_model(model, root_dir, model_name):
    path = os.path.join(root_dir, "results", "saved_models", f"{model_name}.sklearn")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_sklearn_model(path):
    return joblib.load(path)