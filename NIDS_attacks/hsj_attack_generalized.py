"""Generalized HopSkipJump attack with realistic constraints for NIDS datasets."""

import sys
import os

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

import numpy as np
import torch

from CICIDS2017.dataset import CICIDS2017
from UNSWNB15.dataset import UNSWNB15

from scripts.logger import LoggerManager

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier, PyTorchClassifier

from NIDS_attacks.bounds_constrains import apply_bounds_constraints
from NIDS_attacks.integers_constrains import apply_integer_constraints


def hsj_attack_generalized(model, dataset="CICIDS2017", nb_samples=10, ds_train_size=10000, 
                          is_multi_class=False, per_sample_visualization=False,
                          integer_indices=None, modifiable_indices=None, 
                          apply_constraints=True, logger=None):
    """
    Generalized HopSkipJump attack with realistic constraints.
    
    Args:
        model: Trained model (scikit-learn or PyTorch)
        dataset: Dataset name ("CICIDS2017" or "UNSWNB15")
        nb_samples: Number of samples to attack
        ds_train_size: Training dataset size
        is_multi_class: Whether to use multi-class classification
        per_sample_visualization: Show detailed per-sample results
        integer_indices: List of feature indices that should be integers
        modifiable_indices: List of feature indices that can be modified
        apply_constraints: Whether to apply realistic constraints
        logger: Logger instance (if None, creates new one)
    
    Returns:
        Dictionary with attack results
    """
    
    if logger is None:
        logger_mgr = LoggerManager(
            root_dir=root_dir,
            log_name=f"hsj_attack_generalized_{dataset}"
        )
        logger = logger_mgr.get_logger()
    
    logger.info("Starting Generalized HopSkipJump attack")

    # Load dataset
    if dataset == "CICIDS2017":
        logger.info("Loading CICIDS2017 dataset...")
        ds = CICIDS2017(logger=logger).optimize_memory().encode()
        ds = ds.subset(size=ds_train_size, multi_class=is_multi_class)
        targeted_class = 0
    else:
        logger.info("Loading UNSWNB15 dataset...")
        ds = UNSWNB15(dataset_size="small", logger=logger).optimize_memory().encode()
        ds = ds.subset(size=ds_train_size, multi_class=is_multi_class)
        targeted_class = 3

    if not is_multi_class:
        targeted_class = 0

    X_train, X_test, y_train, y_test = ds.split(test_size=0.2, apply_smote=True)
    
    # Calculate bounds and constraints from attack samples only
    attack_mask_train = y_train != targeted_class
    X_attacks_train = X_train[attack_mask_train]
    
    if apply_constraints:
        min_vals = torch.tensor(X_attacks_train, dtype=torch.float32).min(axis=0).values
        max_vals = torch.tensor(X_attacks_train, dtype=torch.float32).max(axis=0).values
        
        # Default indices if not provided
        if modifiable_indices is None:
            modifiable_indices = list(range(X_train.shape[1]))
        if integer_indices is None:
            integer_indices = []
            
        logger.info(f"Applying constraints with {len(modifiable_indices)} modifiable features")
        logger.info(f"Integer constraints on {len(integer_indices)} features")
    
    # Evaluate initial accuracy
    if hasattr(model, 'score'):
        initial_acc = model.score(X_test, y_test)
    else:  # PyTorch model
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            predictions = model(X_test_tensor).argmax(dim=1).numpy()
            initial_acc = (predictions == y_test).mean()
    
    logger.info(f"Initial accuracy: {initial_acc:.3f}")
    
    # Setup ART classifier
    if hasattr(model, 'predict'):  # Scikit-learn
        art_classifier = SklearnClassifier(model=model)
    else:  # PyTorch
        art_classifier = PyTorchClassifier(
            model=model,
            input_shape=X_train.shape[1:],
            nb_classes=len(np.unique(y_train))
        )
    
    # Custom attack with constraints
    class ConstrainedHopSkipJump(HopSkipJump):
        def __init__(self, classifier, apply_constraints=False, **kwargs):
            super().__init__(classifier, **kwargs)
            self.apply_constraints = apply_constraints
            
        def generate(self, x, y=None, **kwargs):
            # Generate base adversarial examples
            x_adv = super().generate(x, y, **kwargs)
            
            if self.apply_constraints and apply_constraints:
                x_original = torch.tensor(x, dtype=torch.float32)
                x_adv_tensor = torch.tensor(x_adv, dtype=torch.float32)
                
                # Apply bounds constraints
                x_adv_tensor = apply_bounds_constraints(
                    x_adv_tensor, x_original, modifiable_indices, min_vals, max_vals
                )
                
                # Apply integer constraints
                if integer_indices:
                    x_adv_tensor = apply_integer_constraints(x_adv_tensor, integer_indices)
                
                x_adv = x_adv_tensor.numpy()
            
            return x_adv
    
    attack = ConstrainedHopSkipJump(
        classifier=art_classifier,
        apply_constraints=apply_constraints,
        batch_size=32,
        max_iter=50,
        max_eval=1000,
        init_eval=100
    )
    
    # Select attack samples
    attack_mask = y_test != targeted_class
    if attack_mask.sum() == 0:
        logger.info("No attack samples found for attack")
        return None
        
    X_attacks = X_test[attack_mask][:nb_samples]
    y_attacks = y_test[attack_mask][:nb_samples]
    
    logger.info(f"Attacking {len(X_attacks)} attack samples to make them appear benign...")
    logger.info(f"Target classes being attacked: {np.unique(y_attacks)}")
    
    y_target = np.zeros(len(X_attacks), dtype=int)  # Target: benign class
    
    # Generate adversarial examples
    X_adv = attack.generate(x=X_attacks, y=y_target)
    
    # Evaluate attack success
    if hasattr(model, 'predict'):  # Scikit-learn
        y_pred_original = model.predict(X_attacks)
        y_pred_adversarial = model.predict(X_adv)
    else:  # PyTorch
        model.eval()
        with torch.no_grad():
            X_orig_tensor = torch.tensor(X_attacks, dtype=torch.float32)
            X_adv_tensor = torch.tensor(X_adv, dtype=torch.float32)
            y_pred_original = model(X_orig_tensor).argmax(dim=1).numpy()
            y_pred_adversarial = model(X_adv_tensor).argmax(dim=1).numpy()
    
    # Calculate metrics
    original_acc = (y_pred_original == y_attacks).mean()
    adversarial_acc = (y_pred_adversarial == y_attacks).mean()
    attack_success_rate = (y_pred_adversarial == targeted_class).mean()
    
    perturbation = np.linalg.norm(X_adv - X_attacks, axis=1).mean()
    
    logger.info("=== Attack Results ===")
    logger.info(f"Original accuracy on attack samples: {original_acc:.3f}")
    logger.info(f"Adversarial accuracy on attack samples: {adversarial_acc:.3f}")
    logger.info(f"Attack success rate (attacks -> benign): {attack_success_rate:.3f}")
    logger.info(f"Average L2 perturbation: {perturbation:.6f}")
    
    if apply_constraints:
        logger.info(f"Constraints applied: bounds + {len(integer_indices)} integer features")
    
    if per_sample_visualization:
        logger.info("=== Per-Sample Analysis ===")
        for i in range(len(X_attacks)):
            logger.info(f"Sample {i+1}: Class {y_attacks[i]} -> Original pred: {y_pred_original[i]} -> Adversarial pred: {y_pred_adversarial[i]}")

    logger.info(f"\nSummary: Attack succeeded {attack_success_rate:.1%} of the time")
    
    return {
        'original_accuracy': original_acc,
        'adversarial_accuracy': adversarial_acc,
        'attack_success_rate': attack_success_rate,
        'perturbation_l2': perturbation,
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'X_adv': X_adv,
        'y_attacks': y_attacks,
        'y_adversarial_pred': y_pred_adversarial,
        'attack_indices': attack_mask,
        'constraints_applied': apply_constraints,
        'min_vals': min_vals if apply_constraints else None,
        'max_vals': max_vals if apply_constraints else None
    }


if __name__ == "__main__":
    # Example usage with Decision Tree
    from scripts.models.decision_tree.decision_tree import train_decision_tree
    
    # Load dataset and train model
    ds = CICIDS2017().optimize_memory().encode()
    ds = ds.subset(size=10000, multi_class=False)
    X_train, X_test, y_train, y_test = ds.split(test_size=0.2, apply_smote=True)
    
    model = train_decision_tree(X_train, y_train, max_depth=10)
    
    # Example integer features (ports, packet counts, etc.)
    integer_indices = [0, 1, 2, 5, 10, 15]  # Example indices
    modifiable_indices = list(range(20))  # First 20 features modifiable
    
    results = hsj_attack_generalized(
        model=model,
        dataset="CICIDS2017",
        nb_samples=20,
        integer_indices=integer_indices,
        modifiable_indices=modifiable_indices,
        apply_constraints=True,
        per_sample_visualization=True
    )

# save X_adv to csv
    import pandas as pd
    X_adv_df = pd.DataFrame(results['X_adv'], columns=[f'feature_{i}' for i in range(results['X_adv'].shape[1])])
    X_adv_df.to_csv('X_adv_hsj_generalized.csv', index=False)