# Imports and Configuration
import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# Structure: AdversarialNIDS / NIDS_attacks / knn_attack / demo_knn_attack.py
root_path = os.path.abspath(os.path.join(current_dir, "../..")) 

# Add the root to import 'UNSWNB15' and 'scripts'
if root_path not in sys.path:
    sys.path.append(root_path)

print(f"Project root set to: {root_path}")

# Import custom modules
from UNSWNB15.preprocessing.dataset import UNSWNB15

try:
    from pipeline import run_attack_pipeline
except ImportError:
    # Fallback if execute as module
    from NIDS_attacks.knn_attack.pipeline import run_attack_pipeline

def load_data(dataset_class, dataset_size="small"):
    """Loads, preprocesses, and splits the dataset."""
    print(f"Loading {dataset_class.__name__} dataset...")
    
    # 1. Initialization and Download
    dataset = dataset_class(dataset_size=dataset_size)
    
    # 2. Optimization and Encoding
    dataset.optimize_memory()
    dataset.encode(attack_encoder="label")

    # 3. Scaling (Normalization)
    try:
        dataset.scale(scaler="minmax")
    except:
        print("Warning: 'minmax' scaler not supported, falling back to standard scaler.")
        dataset.scale(scaler="standard")

    # 4. Split Train/Test
    X_train, X_test, y_train, y_test = dataset.split(
        test_size=0.25, 
        multiclass=False
    )

    # 5. Type Conversion for PyTorch/ART
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # Use a small subset for quick attack testing
    X_sub = X_test[:100]
    y_sub = y_test[:100]
    
    return X_train, y_train, X_sub, y_sub

def main():
    # --- Test 1: UNSW-NB15 ---
    X_train_unsw, y_train_unsw, X_sub_unsw, y_sub_unsw = load_data(UNSWNB15, dataset_size="small")
    
    run_attack_pipeline(
        X_train=X_train_unsw, 
        y_train=y_train_unsw, 
        X_sub=X_sub_unsw, 
        y_sub=y_sub_unsw, 
        dataset_name="UNSW-NB15",
        eps=0.3,            # Optimized perturbation for transfer
        surrogate_epochs=30 # Optimized training for fidelity
    )
    
    # --- Test 2: CICIDS2017 ---
    # from CICIDS2017.preprocessing.dataset import CICIDS2017
    # X_train_cicids, y_train_cicids, X_sub_cicids, y_sub_cicids = load_data(CICIDS2017, dataset_size=None)
    # run_attack_pipeline(
    #     X_train=X_train_cicids, 
    #     y_train=y_train_cicids, 
    #     X_sub=X_sub_cicids, 
    #     y_sub=y_sub_cicids, 
    #     dataset_name="CICIDS2017",
    #     eps=0.3,            
    #     surrogate_epochs=30 
    # )

if __name__ == "__main__":
    main()