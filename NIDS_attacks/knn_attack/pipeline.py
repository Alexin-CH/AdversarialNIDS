import numpy as np
from sklearn.metrics import accuracy_score
from art.estimators.classification import SklearnClassifier, PyTorchClassifier
from art.attacks.evasion import HopSkipJump, FastGradientMethod

try:
    from surrogate import train_surrogate_model
except ImportError:
    from NIDS_attacks.knn_attack.surrogate import train_surrogate_model

# Managing KNN import (which is located in scripts/models/)
# This works because demo_knn_attack.py added the root directory to sys.path
try:
    from scripts.models.knn import train_knn 
except ImportError:
    from sklearn.neighbors import KNeighborsClassifier
    def train_knn(X, y, n_neighbors=5, cv=5, logger=None):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X, y)
        return knn, []

def run_attack_pipeline(X_train, y_train, X_sub, y_sub, dataset_name, eps=0.3, surrogate_epochs=30):
    """
    Trains the KNN model and executes both HopSkipJump (Black Box) 
    and Surrogate FGSM (Transfer) attacks.
    """
    print(f"\nStarting attack pipeline on: {dataset_name}")

    # 1. Train Target Model (KNN)
    knn_model, _ = train_knn(X_train, y_train, n_neighbors=5)
    print("KNN Model trained.")
    
    # Baseline accuracy
    baseline_acc = accuracy_score(y_sub, knn_model.predict(X_sub))
    print(f"Baseline accuracy on subset: {baseline_acc:.2f}")

    # 2. Black Box Attack - HopSkipJump
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
    
    # 3. White Box Strategy - Surrogate Setup/Train
    # Important: Surrogate learn on PREDICTIONS of KNN
    y_train_surrogate = knn_model.predict(X_train)
    
    surrogate, criterion, optimizer = train_surrogate_model(
        X_train=X_train,
        y_targets=y_train_surrogate,
        input_shape=X_train.shape[1],
        epochs=surrogate_epochs
    )

    # 4. Transfer Attack (FGSM via Surrogate)
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