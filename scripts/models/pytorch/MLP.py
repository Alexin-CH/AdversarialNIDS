import os
import torch
import torch.nn as nn

class NetworkIntrusionMLP(nn.Module):
    def __init__(self, input_size, num_classes, scaling_method=None):
        super(NetworkIntrusionMLP, self).__init__()

        assert scaling_method in [None, 'standard', 'minmax'], "scaling_method must be None, 'standard', or 'minmax'"
        self.scaling_method = scaling_method
        self.scaler_is_fitted = False

        self.activation = nn.Mish()

        self.features = nn.Sequential(
            nn.Linear(input_size, 256//2),
            nn.BatchNorm1d(256//2),
            self.activation,
            nn.Linear(256//2, 128//2),
            nn.BatchNorm1d(128//2),
            self.activation,
            nn.Linear(128//2, 64//2),
            nn.BatchNorm1d(64//2),
            self.activation
        )

        self.classifier = nn.Sequential(
            nn.Linear(64//2, 32//2),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(32//2, num_classes),
        )

    def forward(self, x):
        # Apply scaling if specified
        if self.scaling_method == 'standard' and self.scaler_is_fitted:
            x = (x - self.mean) / self.std
        elif self.scaling_method == 'minmax' and self.scaler_is_fitted:
            x = (x - self.min) / (self.max - self.min)

        features = self.features(x)
        out = self.classifier(features)
        return torch.softmax(out, dim=1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        to_save = {
            'model_state_dict': self.state_dict(),
            'scaler_is_fitted': self.scaler_is_fitted,
            'mean': self.mean if self.scaler_is_fitted else None,
            'std': self.std if self.scaler_is_fitted else None,
            'min': self.min if self.scaler_is_fitted else None,
            'max': self.max if self.scaler_is_fitted else None,
        }
        torch.save(to_save, path)

    def load_model(self, path, device='cpu'):
        saved_model = torch.load(path, map_location=device)
        self.scaler_is_fitted = saved_model['scaler_is_fitted']
        if self.scaler_is_fitted:
            self.mean = saved_model['mean']
            self.std = saved_model['std']
            self.min = saved_model['min']
            self.max = saved_model['max']

        self.load_state_dict(saved_model['model_state_dict'])
        self.to(device)
        return self
    
    def fit_scalers(self, X_train):
        self.mean = nn.Parameter(X_train.mean(axis=0).values, requires_grad=False)
        self.std = nn.Parameter(X_train.std(axis=0).values, requires_grad=False)
        self.min = nn.Parameter(X_train.min(axis=0).values, requires_grad=False)
        self.max = nn.Parameter(X_train.max(axis=0).values, requires_grad=False)
        self.scaler_is_fitted = True
        return self
    