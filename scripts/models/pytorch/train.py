import time
import torch
from tqdm import tqdm

def train(model, optimizer, scheduler, criterion, num_epochs, train_loader, val_loader, title, dir, device, logger):
    epoch_losses = []
    epoch_val_losses = []
    # Training loop
    tqdm_epochs = tqdm(range(int(num_epochs)), desc="Training Progress")
    for epoch in tqdm_epochs:
        model.train()
        losses = []
        for X_train, y_train in train_loader:
            # Forward pass
            output = model(X_train)
            loss = criterion(output, y_train)
            losses.append(loss)

        epoch_loss = sum(losses) / len(losses)
        epoch_losses.append(epoch_loss.cpu().detach().numpy())
            
        # Backward pass and optimization
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        scheduler.step(epoch_loss.item())
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    val_losses.append(val_loss)

                epoch_val_loss = sum(val_losses) / len(val_losses)
                epoch_val_losses.append(epoch_val_loss.cpu().detach().numpy())

            # Save model checkpoint
            model.save_model(root_dir=root_dir, model_name=f"{title}_epoch_{epoch+1}.pt")
        else:
            epoch_val_losses.append(epoch_val_losses[-1] if epoch_val_losses else 0)
            
        tqdm_epochs.set_description(f"Loss: {epoch_loss.item():.4f}, Val Loss: {epoch_val_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Final model save
    model.save_model(root_dir=root_dir, model_name=f"{title}_final.pt")
    
    return model, epoch_losses, epoch_val_losses