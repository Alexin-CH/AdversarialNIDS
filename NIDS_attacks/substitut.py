import os
import torch
import torch.nn as nn
import numpy as np
import sys 
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from scripts.logger import SimpleLogger
from scripts.analysis.pytorch_prediction import get_pytorch_predictions
from scripts.models.pytorch.train import train
from scripts.models.pytorch.visualization import display_loss
from scripts.analysis.model_analysis import perform_model_analysis
from scripts.models.pytorch.MLP import NetworkIntrusionMLP


def attack_substitut(num_classes,input_size,model,X_test,y_test,root_dir=root_dir, logger=SimpleLogger(), model_name="Model", 
                          plot_analysis=False,plot_loss=True,save_fig=True, device=None):
    """
    Args:   num_classes : number of classes in the dataset
            input_size : size of the input
            model : model to attack
            X_test : test data      
            y_test : test labels
            root_dir : root directory to save results
            logger : logger to use
            model_name : name of the model attacked
            plot_analysis : whether to plot the analysis results
            plot_loss : whether to plot the loss curves
            save_fig : whether to save the figures
            device : device to use
        Returns:
            sub : substitute model trained
            cm : confusion matrix of the substitute model
            cr : classification report of the substitute model"""
    is_pytorch = isinstance(model, nn.Module)
    """Train a substitute model to mimic the behavior of a given model."""
    # Get predictions
    if is_pytorch:
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running analysis for PyTorch model: {model_name} on device: {device}")
        y_pred, y_true = get_pytorch_predictions(model, X_test, y_test, device)
    #y_true is not used right now but could be useful in the future
    else:
        logger.info(f"Running analysis for scikit-learn model: {model_name}")
        y_true = np.asarray(y_test)
        y_pred = model.predict(X_test)
        X_test = torch.tensor(X_test)
        y_pred = torch.tensor(y_pred)
        
    # Partition of the data gotten from the attacked model
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_pred, test_size=0.2)
    
    X_train  = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train  = torch.tensor(y_train, dtype=torch.long).to(device)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    #Initialisation of the sub-MLP
    sub = NetworkIntrusionMLP(
        input_size=input_size,
        num_classes=num_classes,
        device=device,
        layer_features=[200, 150],
        layer_classifier=[40, 40],
    )
    
    #Training of the sub-MLP
    num_epochs_sub = 100
    mlp_title = f"MLPS_mc_{model_name}_{num_epochs_sub}"
    title = "sub"
    
    optimizer_sub = optim.AdamW(sub.parameters(), lr=0.01)
    scheduler_sub = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sub, mode='min', factor=0.9, patience=8, min_lr=1e-8)
    sub,train_losses_sub, val_losses_sub = train(
        model=sub,
        optimizer=optimizer_sub,
        scheduler=scheduler_sub,
        criterion=nn.CrossEntropyLoss(),
        num_epochs=50,
        train_loader=train_loader,
        val_loader=val_loader,
        title=f"{title}_{mlp_title}",
        root_dir=f"{root_dir}/results/weights",
        device=device,
        logger=logger
    )
    
    # Loss plotting
    display_loss(
        list_epoch_loss=train_losses_sub,
        list_val_loss=val_losses_sub,
        title=f"{title}_{mlp_title}",
        root_dir=f"{root_dir}/results/plots",
        plot=plot_loss,
        logger=logger,
        epoch_min=2,
    )

    #Analysys
    cm, cr = perform_model_analysis(
        model=sub,
        X_test=X_val,
        y_test=y_val,
        logger=logger,
        title=f"{title}_{mlp_title}",
        root_dir=f"{root_dir}/results/analysis",
        plot=plot_analysis,
        device=device,
        save_fig=save_fig,
    )
    return sub, cm, cr
    