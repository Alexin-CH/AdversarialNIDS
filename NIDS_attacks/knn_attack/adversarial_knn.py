# Imports and Configuration
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import sys
import os

# ART Imports (Adversarial Robustness Toolbox)
from art.estimators.classification import SklearnClassifier, PyTorchClassifier
from art.attacks.evasion import HopSkipJump, FastGradientMethod

# Path adjustment to find custom scripts (assuming this file is in a subdirectory like 'scripts/attacks/')
sys.path.append(os.path.abspath("../..")) 

# Import custom modules
from UNSWNB15.preprocessing.dataset import UNSWNB15

try:
    from scripts.models.knn import train_knn 
except ImportError:
    # Fallback for standalone testing if script is missing
    from sklearn.neighbors import KNeighborsClassifier
    def train_knn(X, y, n_neighbors=5, cv=5, logger=None):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X, y)
        return knn, []

class SurrogateModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SurrogateModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def load_data(dataset_class, dataset_size="small"):
    """Loads, preprocesses, and splits the dataset."""
    print(f"Loading {dataset_class.__name__} dataset...")
    
    # Initialization and Download
    dataset = dataset_class(dataset_size=dataset_size)
    
    # Optimization and Encoding
    dataset.optimize_memory()
    dataset.encode(attack_encoder="label")

    # Scaling (Normalization)
    try:
        dataset.scale(scaler="minmax")
    except:
        print("Warning: 'minmax' scaler not supported, falling back to standard scaler.")
        dataset.scale(scaler="standard")

    # Split Train/Test
    X_train, X_test, y_train, y_test = dataset.split(
        test_size=0.25, 
        multiclass=False
    )

    # Type Conversion for PyTorch/ART
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # Use a small subset for quick attack testing
    X_sub = X_test[:100]
    y_sub = y_test[:100]
    
    return X_train, y_train, X_sub, y_sub

def run_attack_pipeline(X_train, y_train, X_sub, y_sub, dataset_name, eps=0.3, surrogate_epochs=30):
    """
    Trains the KNN model and executes both HopSkipJump (Black Box) 
    and Surrogate FGSM (Transfer) attacks.
    """
    print(f"\nStarting attack pipeline on: {dataset_name}")

    # Train Target Model (KNN)
    knn_model, _ = train_knn(X_train, y_train, n_neighbors=5)
    print("KNN Model trained.")
    
    # Baseline accuracy on the subset
    baseline_acc = accuracy_score(y_sub, knn_model.predict(X_sub))
    print(f"Baseline accuracy on subset: {baseline_acc:.2f}")

    # Black Box Attack - HopSkipJump
    art_knn = SklearnClassifier(model=knn_model, clip_values=(0.0, 1.0))
    attack_hsj = HopSkipJump(classifier=art_knn, targeted=False, max_iter=50, max_eval=1000, init_eval=10) 
    print("Generating adversarial examples using HopSkipJump (Black Box)...")
    X_adv_hsj = attack_hsj.generate(x=X_sub)
    
    preds_clean = knn_model.predict(X_sub)
    preds_adv_hsj = knn_model.predict(X_adv_hsj)
    acc_clean = accuracy_score(y_sub, preds_clean)
    acc_adv_hsj = accuracy_score(y_sub, preds_adv_hsj)
    
    print(f"\n--- HopSkipJump Results (Black Box) ---")
    print(f"Accuracy on clean data: {acc_clean:.2f}")
    print(f"Accuracy on adversarial data: {acc_adv_hsj:.2f}")
    hsj_drop = (acc_clean - acc_adv_hsj) * 100
    print(f"Performance drop: {hsj_drop:.1f}%")
    
    # White Box Strategy - Surrogate Setup/Train
    # Train surrogate on KNN's predictions
    y_train_surrogate = knn_model.predict(X_train)
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.LongTensor(y_train_surrogate)
    dataset_tensor = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset_tensor, batch_size=64, shuffle=True)
    
    surrogate = SurrogateModel(input_size=X_train.shape[1], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate.parameters(), lr=0.01)
    
    print(f"\nTraining surrogate model to mimic KNN ({surrogate_epochs} epochs)...")
    for epoch in range(surrogate_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = surrogate(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    print("Surrogate training complete.")

    # Transfer Attack (FGSM via Surrogate)
    art_surrogate = PyTorchClassifier(
        model=surrogate, loss=criterion, optimizer=optimizer,
        input_shape=(X_train.shape[1],), nb_classes=2, clip_values=(0.0, 1.0)
    )
    
    attack_fgsm = FastGradientMethod(estimator=art_surrogate, eps=eps) 
    
    print("Generating adversarial examples using FGSM (on Surrogate)...")
    X_adv_surr = attack_fgsm.generate(x=X_sub)
    
    preds_clean_knn = knn_model.predict(X_sub)
    preds_adv_transfer = knn_model.predict(X_adv_surr)
    acc_clean = accuracy_score(y_sub, preds_clean_knn)
    acc_transf = accuracy_score(y_sub, preds_adv_transfer)
    
    print(f"\n--- Transfer Attack Results (White Box Surrogate) ---")
    print(f"KNN Accuracy (Clean): {acc_clean:.2f}")
    print(f"KNN Accuracy (Transferred Attack): {acc_transf:.2f}")
    fgsm_drop = (acc_clean - acc_transf) * 100
    print(f"Attack Success (Drop): {fgsm_drop:.1f}%")
    
    return {
        'dataset': dataset_name, 
        'hsj_drop': hsj_drop, 
        'fgsm_drop': fgsm_drop
    }

def main():
    # Test 1: UNSW-NB15
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
    
    # Test 2: CICIDS2017 
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