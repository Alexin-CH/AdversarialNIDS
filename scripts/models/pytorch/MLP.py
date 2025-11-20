import os
import torch
import torch.nn as nn

class NetworkIntrusionMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NetworkIntrusionMLP, self).__init__()

        self.activation = nn.Mish()

        self.features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.activation
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
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
    