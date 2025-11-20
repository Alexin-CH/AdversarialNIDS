import os
import torch
import torch.nn as nn

class NetworkIntrustionCNN(nn.Module):
    def __init__(self, input_size, input_channels, num_classes):
        super(NetworkIntrustionCNN, self).__init__()

        self.activation = nn.Mish()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            self.activation,
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            self.activation,
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 *  (input_size // 4), 32),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # Reshape input for 1D convolution
        x = x.unsqueeze(1)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return torch.softmax(out, dim=1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        return self
    