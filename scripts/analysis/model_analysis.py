import os
import sys

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.analysis.pytorch_prediction import get_pytorch_predictions
from scripts.analysis.classification_report import plot_classification_report

from scripts.logger import SimpleLogger


def perform_model_analysis(model, X_test, y_test, num_classes, root_dir=root_dir, logger=SimpleLogger(), title="model",
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
        y_true = np.asarray(y_test).argmax(axis=-1)
        y_pred = model.predict(X_test).argmax(axis=-1)

    if num_classes == 2:
        if is_pytorch:
            y_score = torch.softmax(torch.FloatTensor(y_pred), dim=0).cpu().numpy()
        else:
            y_score = np.asarray(model.predict_proba(X_test))[1].argmax(axis=-1)
                

    # print(type(y_true), type(y_pred), type(y_score) if num_classes == 2 else "N/A")
    # print(y_true.shape, y_pred.shape, y_score.shape if num_classes == 2 else "N/A")

    # Calculate metrics
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)

        # print(type(fpr), type(tpr))
        # print(fpr.shape, tpr.shape)

        roc_auc = auc(fpr, tpr)
        logger.info(f"AUC: {roc_auc:.4f}")

    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Log classification report
    log_header = f"Classification Report for {title}"
    logger.debug(f"\n{log_header}\n{report_dict}\n")
    
    # Create visualization with confusion matrix, report table and ROC curve
    fig, axes = plt.subplots(1, 3 if num_classes == 2 else 2, figsize=(18, 6))
    fig.suptitle(f"Model Analysis: {title}", fontsize=16)
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    # Classification Report Table
    plot_classification_report(report_dict, ax=axes[1], title='Classification Report')
    # ROC Curve for non-PyTorch models
    if num_classes == 2:
        axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:4f})')
        axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].set_title('Receiver Operating Characteristic')
        axes[2].legend(loc="lower right")

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