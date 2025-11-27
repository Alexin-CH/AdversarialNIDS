import os
import sys

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.analysis.pytorch_prediction import get_pytorch_predictions
from scripts.analysis.classification_report import plot_classification_report

from scripts.logger import SimpleLogger


def perform_model_analysis(model, X_test, y_test, root_dir=root_dir, logger=SimpleLogger(), title="model",
                          plot=True, save_fig=True, device=None):
    """
    Perform complete classification analysis with confusion matrix and report visualization.
    Works with both PyTorch and scikit-learn models.
    
    Args:
        model: Trained classification model (PyTorch nn.Module or sklearn model)
        X_test: Test features (numpy array, pandas DataFrame, or torch Tensor)
        y_test: True labels (numpy array, pandas Series, or torch Tensor)
        logger: Logger instance for recording results
        title: Name of the model for titles and logs (default: "Model")
        device: Device for PyTorch models ('cuda' or 'cpu', default: auto-detect)
    
    Outputs:
        - Logs classification report to logger
        - Displays side-by-side confusion matrix and classification report table
    """
    is_pytorch = isinstance(model, nn.Module)
    
    # Get predictions
    if is_pytorch:
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running analysis for PyTorch model: {title} on device: {device}")
        y_pred, y_true = get_pytorch_predictions(model, X_test, y_test, device)
    else:
        logger.info(f"Running analysis for scikit-learn model: {title}")
        y_true = np.asarray(y_test)
        y_pred = model.predict(X_test)
        
        # Handle both 1D (class indices) and 2D (one-hot encoded) formats
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = y_true.argmax(axis=1)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1)
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, 
                                       digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Log classification report
    log_header = f"Classification Report for {title}"
    logger.debug(f"\n{log_header}\n{report}\n")
    
    # Create visualization with confusion matrix and report table
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
        
    # Plot confusion matrix (left)
    ax_cm = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix', fontsize=12, weight='bold')
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    
    # Plot classification report table (right)
    ax_report = fig.add_subplot(gs[0, 1])
    plot_classification_report(report_dict, ax_report, 'Classification Report')
    
    fig.suptitle(f'Model Analysis - {title}', fontsize=16, weight='bold')

    if save_fig:
        # Save figure to dir
        dir = os.path.join(root_dir, "results", "model_analysis")
        os.makedirs(dir, exist_ok=True)
        filename = f"{title.replace(' ', '_')}_model_analysis.png"
        fig.savefig(os.path.join(dir, filename))
    
    if plot:
        plt.show()
    plt.close(fig)

    return cm, report_dict