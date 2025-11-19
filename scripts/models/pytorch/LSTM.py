import torch
import torch.nn as nn

class NetworkIntrusionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(NetworkIntrusionLSTM, self).__init__()

        self.activation = nn.Mish()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # LSTM expects (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use the last time step
        out = lstm_out
        out = self.classifier(out)
        return torch.softmax(out, dim=1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
