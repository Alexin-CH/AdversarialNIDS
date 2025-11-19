import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def _get_pytorch_predictions(model, X_test, y_test, device, batch_size=32):
    model.to(device)
    model.eval()
    
    # --- Data to Tensor Conversion ---
    if isinstance(X_test, (np.ndarray, pd.DataFrame)):
        X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    else:
        X_test_tensor = X_test.clone().detach().to(dtype=torch.float32)

    # Handle y_test which might be a tuple/list for multi-output models
    if isinstance(y_test, (tuple, list)):
        # Multi-output case
        y_test_tensors = []
        for y in y_test:
            if isinstance(y, (np.ndarray, pd.Series)):
                y_test_tensors.append(torch.tensor(np.array(y), dtype=torch.long))
            else:
                y_test_tensors.append(y.clone().detach().to(dtype=torch.long))
        
        test_dataset = TensorDataset(X_test_tensor, *y_test_tensors)
    else:
        # Single output case
        if isinstance(y_test, (np.ndarray, pd.Series)):
            y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)
        else:
            y_test_tensor = y_test.clone().detach().to(dtype=torch.long)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine number of outputs by running a test batch
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        features = sample_batch[0].to(device)
        outputs = model(features)
        
        # Check if model has multiple outputs
        if isinstance(outputs, (tuple, list)):
            num_outputs = len(outputs)
            is_multi_output = True
        else:
            num_outputs = 1
            is_multi_output = False
    
    # Initialize storage for predictions and labels
    if is_multi_output:
        all_preds = [[] for _ in range(num_outputs)]
        all_labels = [[] for _ in range(num_outputs)]
    else:
        all_preds = []
        all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch[0].to(device)
            labels = batch[1:]  # All remaining items are labels
            
            outputs = model(features)
            
            if is_multi_output:
                # Handle multiple outputs
                for i, output in enumerate(outputs):
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    
                    # Handle one-hot encoded labels
                    if labels[i].dim() > 1 and labels[i].shape[1] > 1:
                        true_labels = torch.argmax(labels[i], dim=1).cpu().numpy()
                    else:
                        true_labels = labels[i].cpu().numpy()
                    
                    all_preds[i].extend(preds)
                    all_labels[i].extend(true_labels)
            else:
                # Handle single output
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Handle one-hot encoded labels
                if labels[0].dim() > 1 and labels[0].shape[1] > 1:
                    true_labels = torch.argmax(labels[0], dim=1).cpu().numpy()
                else:
                    true_labels = labels[0].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(true_labels)
    
    if is_multi_output:
        return [np.array(preds) for preds in all_preds], [np.array(labels) for labels in all_labels]
    else:
        return np.array(all_preds), np.array(all_labels)

def _plot_classification_report(report_dict, ax, title):
    """
    Helper function to plot a classification report as a table.
    
    Args:
        report_dict: Dictionary returned by classification_report with output_dict=True
        ax: Matplotlib axis to plot on
        title: Title for the table
    """
    # Extract the relevant metrics (exclude support for cleaner visualization)
    metrics = ['precision', 'recall', 'f1-score', 'support']
    
    # Get class names and overall metrics
    class_names = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Build data for the table
    data = []
    for class_name in class_names:
        row = [class_name] + [f"{report_dict[class_name][m]:.4f}" if m != 'support' 
                               else str(int(report_dict[class_name][m])) 
                               for m in metrics]
        data.append(row)
    
    # Add separator
    data.append(['---'] * 5)
    
    # Add macro average
    if 'macro avg' in report_dict:
        row = ['macro avg'] + [f"{report_dict['macro avg'][m]:.4f}" if m != 'support' 
                               else str(int(report_dict['macro avg'][m])) 
                               for m in metrics]
        data.append(row)
    
    # Add weighted average
    if 'weighted avg' in report_dict:
        row = ['weighted avg'] + [f"{report_dict['weighted avg'][m]:.4f}" if m != 'support' 
                                  else str(int(report_dict['weighted avg'][m])) 
                                  for m in metrics]
        data.append(row)
    
    # Add accuracy
    if 'accuracy' in report_dict:
        acc_row = ['accuracy', '', '', f"{report_dict['accuracy']:.4f}", 
                   str(int(report_dict['weighted avg']['support']))]
        data.append(acc_row)
    
    # Create the table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data,
                    colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator row
    separator_idx = len(class_names) + 1
    for i in range(5):
        table[(separator_idx, i)].set_facecolor('#E7E6E6')
    
    ax.set_title(title, fontsize=12, weight='bold', pad=20)


def perform_model_analysis(model, X_test, y_test, logger, model_name="Model", 
                          class_names=None, output_names=None, device=None):
    """
    Performs a complete analysis of a classification model, logs the results,
    and plots confusion matrices and classification reports. Supports both single and multi-output models.

    Args:
        model: The trained model to evaluate (PyTorch or scikit-learn).
        X_test: Test features (NumPy array, pandas DataFrame, or torch Tensor).
        y_test: True labels for the test set. For multi-output models, pass a list/tuple of label arrays.
        logger: An active logging object to record the output.
        model_name (str, optional): Name of the model for titles and logs. Defaults to "Model".
        class_names (list or list of lists, optional): Class names for plot labels. 
                     For multi-output models, pass a list of lists. Defaults to None.
        output_names (list, optional): Names for each output (e.g., ["Task 1", "Task 2"]). 
                                       Only used for multi-output models. Defaults to None.
        device (str, optional): The device to run PyTorch models on ('cuda' or 'cpu').
                                If None, auto-detects GPU. Defaults to None.
    """
    is_pytorch = isinstance(model, nn.Module)
    
    # --- 1. Get Predictions ---
    if is_pytorch:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running analysis for PyTorch model: {model_name} on device: {device}")
        y_pred, y_true = _get_pytorch_predictions(model, X_test, y_test, device)
    else:
        logger.info(f"Running analysis for scikit-learn model: {model_name}")
        y_true = np.asarray(y_test)
        y_pred = model.predict(X_test)
    
    # Determine if this is a multi-output model
    is_multi_output = isinstance(y_pred, list)
    
    if is_multi_output:
        num_outputs = len(y_pred)
        logger.info(f"Detected multi-output model with {num_outputs} outputs")
        
        # Set default output names if not provided
        if output_names is None:
            output_names = [f"Output {i+1}" for i in range(num_outputs)]
        
        # Ensure class_names is a list of lists
        if class_names is None:
            class_names = [[str(j) for j in sorted(np.unique(y_true[i]))] for i in range(num_outputs)]
        elif not isinstance(class_names[0], list):
            # If single list provided, use it for all outputs
            class_names = [class_names for _ in range(num_outputs)]
        
        # --- 2. Calculate Reports and Confusion Matrices ---
        reports = []
        report_dicts = []
        cms = []
        for i in range(num_outputs):
            report = classification_report(y_true[i], y_pred[i], 
                                          target_names=class_names[i], digits=4)
            report_dict = classification_report(y_true[i], y_pred[i], 
                                               target_names=class_names[i], 
                                               digits=4, output_dict=True)
            cm = confusion_matrix(y_true[i], y_pred[i])
            reports.append(report)
            report_dicts.append(report_dict)
            cms.append(cm)
            
            # Log each report
            log_header = f"--- Classification Report for {model_name} - {output_names[i]} ---"
            logger.info(f"\n{log_header}\n{report}\n")
        
        # --- 3. Plot Confusion Matrices and Classification Reports ---
        # Create a grid: 2 rows x num_outputs columns
        fig = plt.figure(figsize=(8*num_outputs, 12))
        gs = fig.add_gridspec(2, num_outputs, hspace=0.3, wspace=0.3)
        
        for i in range(num_outputs):
            # Confusion Matrix (top row)
            ax_cm = fig.add_subplot(gs[0, i])
            sns.heatmap(cms[i], annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names[i], yticklabels=class_names[i], ax=ax_cm)
            ax_cm.set_title(f'{output_names[i]} - Confusion Matrix', fontsize=12, weight='bold')
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_ylabel('True Label')
            
            # Classification Report (bottom row)
            ax_report = fig.add_subplot(gs[1, i])
            _plot_classification_report(report_dicts[i], ax_report, 
                                       f'{output_names[i]} - Classification Report')
        
        fig.suptitle(f'Model Analysis - {model_name}', fontsize=16, weight='bold', y=0.98)
        plt.show()
        
    else:
        # Single output case
        if class_names is None:
            class_names = [str(i) for i in sorted(np.unique(y_true))]
        
        # --- 2. Calculate Report and Confusion Matrix ---
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        report_dict = classification_report(y_true, y_pred, target_names=class_names, 
                                           digits=4, output_dict=True)
        log_header = f"--- Classification Report for {model_name} ---"
        logger.info(f"\n{log_header}\n{report}\n")
        
        # --- 3. Plot Confusion Matrix and Classification Report ---
        cm = confusion_matrix(y_true, y_pred)
        
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        
        # Confusion Matrix (left)
        ax_cm = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix', fontsize=12, weight='bold')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        
        # Classification Report (right)
        ax_report = fig.add_subplot(gs[0, 1])
        _plot_classification_report(report_dict, ax_report, 'Classification Report')
        
        fig.suptitle(f'Model Analysis - {model_name}', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()