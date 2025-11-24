import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SurrogateModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SurrogateModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def train_surrogate_model(X_train, y_targets, input_shape, num_classes=2, epochs=30, lr=0.01):
    """
    Initializes and trains the surrogate model on the provided targets.
    Returns the trained model, criterion, and optimizer (needed for ART).
    """
    # Prepare data
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.LongTensor(y_targets)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model components
    model = SurrogateModel(input_size=input_shape, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining surrogate model to mimic KNN ({epochs} epochs)...")
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    print("Surrogate training complete.")
    return model, criterion, optimizer