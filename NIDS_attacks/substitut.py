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

class Sub_MLP(nn.Module):
    def __init__(self, input_size, num_classes, scaling_method=None, device='cpu'):
        super(Sub_MLP, self).__init__()

        self.device = device

        assert scaling_method in [None, 'standard', 'minmax'], "scaling_method must be None, 'standard', or 'minmax'"
        self.scaling_method = scaling_method
        self.scaler_is_fitted = False

        self.activation = nn.Mish()

        self.features = nn.Sequential(
            nn.Linear(input_size, 200),
            self.activation,
            nn.Linear(200, 200),
            self.activation,
            nn.Linear(200, 40),
            self.activation
        )

        self.classifier = nn.Sequential(
            nn.Linear(40,40),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(40, num_classes),
        )

        self.to(device)

    def forward(self, x):

        # Apply scaling if specified
        if self.scaling_method == 'standard' and self.scaler_is_fitted:
            x = (x - self.mean) / self.std
        elif self.scaling_method == 'minmax' and self.scaler_is_fitted:
            x = (x - self.min) / (self.max - self.min)

        features = self.features(x)
        out = self.classifier(features)
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        to_save = {
            'model_state_dict': self.state_dict(),
            'scaler_is_fitted': self.scaler_is_fitted,
            'scaling_method': self.scaling_method,
            'mean': self.mean if self.scaler_is_fitted else None,
            'std': self.std if self.scaler_is_fitted else None,
            'min': self.min if self.scaler_is_fitted else None,
            'max': self.max if self.scaler_is_fitted else None,
        }
        torch.save(to_save, path)

    def load_model(self, path):
        saved_model = torch.load(path, map_location=self.device)
        self.scaler_is_fitted = saved_model['scaler_is_fitted']
        self.scaling_method = saved_model['scaling_method']
        if self.scaler_is_fitted:
            self.mean = saved_model['mean'].to(self.device)
            self.std = saved_model['std'].to(self.device)
            self.min = saved_model['min'].to(self.device)
            self.max = saved_model['max'].to(self.device)

        self.load_state_dict(saved_model['model_state_dict'])
        self.to(self.device)
        return self
    
    def fit_scalers(self, X_train):
        self.mean = X_train.mean(dim=0).to(self.device)
        self.std = X_train.std(dim=0).to(self.device)
        self.min = X_train.min(dim=0).values.to(self.device)
        self.max = X_train.max(dim=0).values.to(self.device)
        self.scaler_is_fitted = True
        return self

def attack_substitut(num_classes,input_size,model,X_test,y_test,dir=root_dir, logger=SimpleLogger(), model_name="Model", 
                          plot_analysis=False,plot_loss=True,save_fig=True, device=None):
    is_pytorch = isinstance(model, nn.Module)
    
    # Get predictions
    if is_pytorch:
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running analysis for PyTorch model: {model_name} on device: {device}")
        y_pred, y_true = get_pytorch_predictions(model, X_test, y_test, device)
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
    sub = Sub_MLP(
        input_size=input_size,
        num_classes=num_classes,
        device=device
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
        num_epochs=100,
        train_loader=train_loader,
        val_loader=val_loader,
        title=f"{title}_{mlp_title}",
        dir=f"{root_dir}/results/weights",
        device=device,
        logger=logger
    )
    
    # Loss plotting
    display_loss(
        list_epoch_loss=train_losses_sub,
        list_val_loss=val_losses_sub,
        title=f"{title}_{mlp_title}",
        dir=f"{root_dir}/results/plots",
        plot=plot_loss,
        logger=logger,
        epoch_min=2
    )

    #Analysys
    cm, cr = perform_model_analysis(
        model=sub,
        X_test=X_val,
        y_test=y_val,
        logger=logger,
        model_name=f"{title}_{mlp_title}",
        dir=f"{root_dir}/results/analysis",
        plot=plot_analysis,
        device=device
    )
    return sub, cm, cr
    