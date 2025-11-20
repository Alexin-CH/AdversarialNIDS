import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

def get_pytorch_predictions(model, X_test, y_test, device, batch_size=32):
    """
    Get predictions from a PyTorch model.
    
    Args:
        model: PyTorch model
        X_test: Test features
        y_test: True labels
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for inference
    
    Returns:
        tuple: (predictions, true_labels) as numpy arrays
    """
    model.to(device)
    model.eval()
    
    # Convert data to tensors
    if isinstance(X_test, (np.ndarray, pd.DataFrame)):
        X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    else:
        X_test_tensor = X_test.clone().detach().to(dtype=torch.float32)

    if isinstance(y_test, (np.ndarray, pd.Series)):
        y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)
    else:
        y_test_tensor = y_test.clone().detach().to(dtype=torch.long)
    
    # Create DataLoader for batch processing
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    # Run inference
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            
            # Get class predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Handle one-hot encoded labels
            if labels.dim() > 1 and labels.shape[1] > 1:
                true_labels = torch.argmax(labels, dim=1).cpu().numpy()
            else:
                true_labels = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(true_labels)
    
    return np.array(all_preds), np.array(all_labels)
