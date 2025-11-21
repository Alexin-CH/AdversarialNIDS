import time
import torch
from tqdm import tqdm

def train(model, optimizer, scheduler, criterion, num_epochs, train_loader, val_loader, title, dir, device, logger):
    list_epoch_loss = []
    list_val_loss = []
    count = len(str(int(num_epochs-1)))
    logger.info("Starting model training...")
    logger.info(f"Number of epochs: {num_epochs}")

    t0 = time.time()
    model.train()
    tqdm_epochs = tqdm(range(int(num_epochs)), desc="Training")
    for epoch in tqdm_epochs:
        logger.debug(f"Epoch {epoch + 1}/{num_epochs} started.")
        losses = []

        show_progress = True if epoch % max(num_epochs // 10, 1) == 0 else False
        tqdm_batchs = tqdm(train_loader, desc="Processing batches", leave=show_progress)
        for batch_idx, batch in enumerate(tqdm_batchs):
            X_train, y_train = batch

            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            losses.append(loss)
            loss.backward()

            optimizer.step()

            counter = str(epoch)
            zeros = count - len(counter)
            counter = '0' * zeros + counter
            current_lr = scheduler.get_last_lr()[0]
            tqdm_batchs.set_description(f"[{counter}] Batch Loss: {loss:.6f}, LR: {current_lr:.6f}, Processing")
            logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

        epoch_loss = sum(losses) / len(losses)

        scheduler.step(epoch_loss.item())

        list_epoch_loss.append(epoch_loss.item())
        logger.debug(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {epoch_loss:.6f}")
        tqdm_epochs.set_description(f"Epoch Loss: {epoch_loss:.6f}, Training")

        # Validation
        if (epoch) % 2 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                tqdm_val = tqdm(val_loader, desc="Processing validation batches", leave=False)
                for val_batch in tqdm_val:
                    X_val, y_val = val_batch

                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)

                    val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            list_val_loss.append(avg_val_loss)
            logger.debug(f"Validation Loss at epoch {epoch + 1}: {avg_val_loss:.6f}")

            # Save the model
            model.save_model(f"{dir}/saved_models/{title}_epoch{epoch + 1}.pt")

            model.train()
        else:
            list_val_loss.append(avg_val_loss)

    t1 = time.time() - t0
    th = int(t1 // 3600)
    tm = int((t1 % 3600) // 60)
    ts = int(t1 % 60)
    logger.info(f"Training completed in {th}h {tm}m {ts}s")
    return model, list_epoch_loss, list_val_loss

# end of file
