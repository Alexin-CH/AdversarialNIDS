import os
import sys

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.analysis.pytorch_prediction import get_pytorch_predictions
from scripts.logger import SimpleLogger


def roc_analysis_binary(model, X_test, y_test, root_dir=root_dir, logger=SimpleLogger(), title="model",
                       plot=True, save_fig=True, device=None):
    """
    Perform ROC analysis for binary classification models.
    Works with both PyTorch and scikit-learn models.
    
    Args:
        model: Trained binary classification model (PyTorch nn.Module or sklearn model)
        X_test: Test features (numpy array, pandas DataFrame, or torch Tensor)
        y_test: True labels (numpy array, pandas Series, or torch Tensor)
        root_dir: Root directory for saving figures
        logger: Logger instance for recording results
        title: Name of the model for titles and logs (default: "model")
        plot: Whether to display the plot (default: True)
        save_fig: Whether to save the figure (default: True)
        device: Device for PyTorch models ('cuda' or 'cpu', default: auto-detect)
    
    Returns:
        tuple: (fpr, tpr, roc_auc) - False positive rate, true positive rate, and AUC score
    """
    is_pytorch = isinstance(model, nn.Module)
    
    # Get predictions and probabilities
    if is_pytorch:
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running ROC analysis for PyTorch model: {title} on device: {device}")
        
        model.eval()
        model = model.to(device)
        
        with torch.no_grad():
            if isinstance(X_test, torch.Tensor):
                X_test_tensor = X_test.to(device)
            else:
                X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            # Get probabilities (assume sigmoid output for binary classification)
            outputs = model(X_test_tensor)
            if outputs.shape[1] == 1:
                # Single output with sigmoid
                y_prob = torch.sigmoid(outputs).cpu().numpy().flatten()
            else:
                # Two outputs with softmax, take positive class probability
                y_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        # Convert true labels
        if isinstance(y_test, torch.Tensor):
            y_true = y_test.cpu().numpy()
        else:
            y_true = np.asarray(y_test)
            
        # If y_true is one-hot encoded, convert to class labels
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = y_true.argmax(axis=1)
            
    else:
        logger.info(f"Running ROC analysis for scikit-learn model: {title}")
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]  # Positive class probability
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            raise ValueError("Model must have predict_proba or decision_function method")
        
        # Convert true labels
        y_true = np.asarray(y_test)
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = y_true.argmax(axis=1)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    logger.info(f"ROC AUC for {title}: {roc_auc:.4f}")
    
    # Create ROC plot
    if plot or save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2%})', color='#2E86AB', linewidth=2)
        ax.plot([0, 1], [0, 1], color='#F24236', linestyle='--', linewidth=1)
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {title}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            # Save figure to dir
            save_dir = os.path.join(root_dir, "results", "roc_analysis")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{title.replace(' ', '_')}_roc_analysis.png"
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to: {os.path.join(save_dir, filename)}")
        
        if plot:
            plt.show()
        else:
            plt.close(fig)
    
    return fpr, tpr, roc_auc



