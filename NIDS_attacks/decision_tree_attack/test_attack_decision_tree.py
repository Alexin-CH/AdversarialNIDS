"""Test HopSkipJump attack on Decision Tree classifier for NIDS datasets."""

import sys
import os

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

import numpy as np

from CICIDS2017.preprocessing.dataset import CICIDS2017
from UNSWNB15.preprocessing.dataset import UNSWNB15

from scripts.models.decision_tree.decision_tree import train_decision_tree
from scripts.logger import LoggerManager

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier

def simple_hopskipjump_attack(dataset="CICIDS2017",nb_samples=10,per_sample_visualization=False):

    logger_mgr = LoggerManager(log_dir='logs', log_name='attack_test')
    logger = logger_mgr.get_logger()
    logger.info("Starting HopSkipJump attack on Decision Tree")
    
    if dataset == "CICIDS2017":
        logger.info("Loading CICIDS2017 dataset...")
        ds = CICIDS2017(logger=logger).optimize_memory().encode()
        ds, multi_class = ds.subset(size=10000, multi_class=True)
    else:
        logger.info("Loading UNSWNB15 dataset...")
        ds = UNSWNB15(dataset_size="small").optimize_memory().encode()
        ds, multi_class = ds.subset(size=10000, multi_class=True)

    X_train, X_test, y_train, y_test = ds.split(test_size=0.2, apply_smote=True)
    
    logger.info("Training Decision Tree...")
    model, cv_scores = train_decision_tree(X_train, y_train, max_depth=10, logger=logger)
    
    initial_acc = model.score(X_test, y_test)
    logger.info(f"Initial accuracy: {initial_acc:.3f}")
    
    art_classifier = SklearnClassifier(model=model)
    
    attack = HopSkipJump(classifier=art_classifier,batch_size=32,max_iter=50,max_eval=1000,init_eval=100)
    
    attack_mask = y_test != 0  # All non-benign classes (network attacks)
    if attack_mask.sum() == 0:
        logger.info("No attack samples found for attack")
        return
        
    X_attacks = X_test[attack_mask][:nb_samples]  # Attack first nb_samples attack samples / may be changed to random
    y_attacks = y_test[attack_mask][:nb_samples]
    
    logger.info(f"Attacking {len(X_attacks)} attack samples to make them appear benign...")
    logger.info(f"Target classes being attacked: {np.unique(y_attacks)}")
    
    y_target = np.zeros(len(X_attacks), dtype=int)  # Target: benign class
    
    # Generate adversarial examples (targeted attack to make attacks look benign)
    X_adv = attack.generate(x=X_attacks, y=y_target)
    
    # Evaluate attack success
    y_pred_original = model.predict(X_attacks)
    y_pred_adversarial = model.predict(X_adv)
    
    # Calculate metrics
    original_acc = (y_pred_original == y_attacks).mean()
    adversarial_acc = (y_pred_adversarial == y_attacks).mean()  # We want it as low as possible (good for attacker)
    attack_success_rate = (y_pred_adversarial == 0).mean()  # Success = predicted as benign
    

    perturbation = np.linalg.norm(X_adv - X_attacks, axis=1).mean() # Mean perturbation (L2 norm)
    
    logger.info("=== Attack Results ===")
    logger.info(f"Original accuracy on attack samples: {original_acc:.3f}")
    logger.info(f"Adversarial accuracy on attack samples: {adversarial_acc:.3f}")
    logger.info(f"Attack success rate (attacks -> benign): {attack_success_rate:.3f}")
    logger.info(f"Average L2 perturbation: {perturbation:.6f}")
    
    if per_sample_visualization:
        logger.info("=== Per-Sample Analysis ===")
        for i in range(len(X_attacks)):
            logger.info(f"Sample {i+1}: Class {y_attacks[i]} -> Original pred: {y_pred_original[i]} -> Adversarial pred: {y_pred_adversarial[i]}")

    logger.info(f"\nSummary: Attack succeeded {attack_success_rate:.1%} of the time")
    return {
        'original_accuracy': original_acc,
        'adversarial_accuracy': adversarial_acc,
        'attack_success_rate': attack_success_rate,
        'perturbation_l2': perturbation
    }


if __name__ == "__main__":
    results = simple_hopskipjump_attack(nb_samples=25,per_sample_visualization=True)
