import os
import torch
import torch.nn as nn

class NetworkIntrusionMLP(nn.Module):
    def __init__(self, input_size,layer_features,layer_classifier, num_classes, scaling_method=None, device='cpu'):
        super(NetworkIntrusionMLP, self).__init__()

        self.device = device

        assert scaling_method in [None, 'standard', 'minmax'], "scaling_method must be None, 'standard', or 'minmax'"
        self.scaling_method = scaling_method
        self.scaler_is_fitted = False

        self.activation = nn.Mish()

        layers = []        
        layers.append(nn.Linear(input_size, layer_features[0]))
        layers.append(nn.BatchNorm1d(layer_features[0]))
        layers.append(self.activation)

        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.BatchNorm1d(out_f))
            layers.append(self.activation)
        #Wrap feature layers
        self.features = nn.Sequential(*layers)

        classifier = []
        classifier.append(nn.Linear(layer_features[-1], layer_classifier[0]))
        classifier.append(self.activation)
        
        for in_c, out_c in zip(layer_classifier[:-1], layer_classifier[1:]):
            classifier.append(nn.Linear(in_c, out_c))
            classifier.append(self.activation)
            nn.Dropout(0.1),
            
        #Add final output layer
        classifier.append(nn.Linear(layer_classifier[-1], num_classes))
        
        #Wrap classifier layers
        self.classifier = nn.Sequential(*classifier)
        

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
    