import os
import torch
import torch.nn as nn

class NetworkIntrusionMLP(nn.Module):
    """ Multi-Layer Perceptron (MLP) for Network Intrusion Detection. """
    def __init__(self, input_size, layer_features, layer_classifier, num_classes, scaling_method=None, device='cpu'):
        """
        Initialize the MLP model.

        Args:
            input_size (int): Number of input features.
            layer_features (list of int): List specifying the number of neurons in each hidden layer of the feature extractor.
            layer_classifier (list of int): List specifying the number of neurons in each hidden layer of the classifier.
            num_classes (int): Number of output classes.
            scaling_method (str or None): Feature scaling method ('standard', 'minmax', or None).
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(NetworkIntrusionMLP, self).__init__()

        self.device = device

        assert scaling_method in [None, 'standard', 'minmax'], "scaling_method must be None, 'standard', or 'minmax'"
        self.scaling_method = scaling_method
        self.scaler_is_fitted = False

        self.activation = nn.Mish()

        layers = [
            nn.Linear(input_size, layer_features[0]),
            nn.BatchNorm1d(layer_features[0]),
            self.activation
        ]

        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.extend([
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                self.activation
            ])

        # Wrap feature layers
        self.features = nn.Sequential(*layers)

        classifier = []
        layer_classifier = [layer_features[-1]] + layer_classifier
        
        for in_c, out_c in zip(layer_classifier[:-1], layer_classifier[1:]):
            classifier.extend([
                nn.Linear(in_c, out_c),
                self.activation,
                nn.Dropout(0.1)
            ])
            
        #Add final output layer
        classifier.append(nn.Linear(layer_classifier[-1], num_classes))
        
        #Wrap classifier layers
        self.classifier = nn.Sequential(*classifier)

        self.to(device)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Apply scaling if specified
        if self.scaling_method == 'standard' and self.scaler_is_fitted:
            x = (x - self.mean) / self.std
        elif self.scaling_method == 'minmax' and self.scaler_is_fitted:
            x = (x - self.min) / (self.max - self.min)

        features = self.features(x)
        out = self.classifier(features)
        return out

    def num_params(self):
        """ Return the number of trainable parameters in the model. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, root_dir, model_name):
        """
        Save the model state and scaler parameters.

        Args:
            root_dir (str): Root directory to save the model.
        """
        dir = os.path.join(root_dir, "results", "saved_models")
        os.makedirs(dir, exist_ok=True)
        to_save = {
            'model_state_dict': self.state_dict(),
            'scaler_is_fitted': self.scaler_is_fitted,
            'scaling_method': self.scaling_method,
            'mean': self.mean if self.scaler_is_fitted else None,
            'std': self.std if self.scaler_is_fitted else None,
            'min': self.min if self.scaler_is_fitted else None,
            'max': self.max if self.scaler_is_fitted else None,
        }
        torch.save(to_save, os.path.join(dir, model_name))

    def load_model(self, path):
        """
        Load the model state and scaler parameters.

        Args:
            path (str): Path to the saved model file.
        """
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
        """
        Fit the scaler parameters based on the training data.

        Args:
            X_train (torch.Tensor): Training data tensor of shape (num_samples, num_features).
        """
        self.mean = X_train.mean(dim=0).to(self.device)
        self.std = X_train.std(dim=0).to(self.device)
        self.min = X_train.min(dim=0).values.to(self.device)
        self.max = X_train.max(dim=0).values.to(self.device)
        self.scaler_is_fitted = True
        return self
    