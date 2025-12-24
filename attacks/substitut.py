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


def attack_substitut(model, X_test, y_test, root_dir=root_dir, logger=SimpleLogger(),
                        model_name="model", plot_analysis=False, plot_loss=True, save_fig=True, device=None):
    """Train a substitute model to mimic the behavior of a given model.
    
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
            cr : classification report of the substitute model
    """
    is_pytorch = isinstance(model, nn.Module)

    # Get predictions
    if is_pytorch:
        logger.info(f"Running analysis for PyTorch model: {model_name} on device: {device}")

        # Get predictions from the PyTorch model
        y_pred, _ = get_pytorch_predictions(model, X_test, y_test, device)
    else:
        logger.info(f"Running analysis for scikit-learn model: {model_name}")

        # Get predictions from the scikit-learn model
        y_pred = model.predict(X_test)
        y_pred = torch.FloatTensor(y_pred)

    X_test = torch.FloatTensor(X_test).to(device)
    y_pred = torch.FloatTensor(y_pred).to(device)
        
    # Partition of the data gotten from the attacked model
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_pred, test_size=0.2)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    input_size = train_loader.dataset.tensors[0].shape[1]
    num_classes = train_loader.dataset.tensors[1].shape[1]
    
    #Initialisation of the sub-MLP
    substitute_model = NetworkIntrusionMLP(
        input_size=input_size,
        num_classes=num_classes,
        layer_features=[64,64],
        layer_classifier=[32],
        scaling_method="minmax",
        device=device,
    ).fit_scalers(X_train=X_train)
    
    #Training of the sub-MLP
    num_epochs_sub = 100
    learning_rate_sub = 0.01

    criterion = nn.CrossEntropyLoss()

    title = "Substitute"
    mlp_title = model_name
    
    optimizer_sub = optim.AdamW(substitute_model.parameters(), lr=learning_rate_sub)
    scheduler_sub = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sub, mode='min', factor=0.9, patience=8, min_lr=1e-8)
    
    substitute_model, train_losses_sub, val_losses_sub = train(
        model=substitute_model,
        optimizer=optimizer_sub,
        scheduler=scheduler_sub,
        criterion=criterion,
        num_epochs=num_epochs_sub,
        train_loader=train_loader,
        val_loader=val_loader,
        title=f"{title}_{mlp_title}",
        root_dir=root_dir,
        device=device,
        logger=logger
    )
    
    # Loss plotting
    display_loss(
        list_epoch_loss=train_losses_sub,
        list_val_loss=val_losses_sub,
        title=f"{title}_{mlp_title}",
        root_dir=root_dir,
        plot=plot_loss,
        logger=logger,
        epoch_min=2,
    )

    #Analysys
    cm, cr = perform_model_analysis(
        model=substitute_model,
        X_test=X_val,
        y_test=y_val,
        num_classes=num_classes,
        logger=logger,
        title=f"{title}_{mlp_title}",
        root_dir=root_dir,
        plot=plot_analysis,
        device=device,
        save_fig=save_fig,
    )
    return substitute_model, cm, cr
    