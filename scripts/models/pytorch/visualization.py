import os
import matplotlib.pyplot as plt

def display_loss(list_epoch_loss, list_val_loss, title, root_dir, logger, plot=True, epoch_min=2):
    """
    Display and save the training and validation loss curves.

    Args:
        list_epoch_loss (list): List of training loss values per epoch.
        list_val_loss (list): List of validation loss values per epoch.
        title (str): Title for the plot and filename.
        root_dir (str): Root directory to save the plot.
        logger: Logger instance for logging information.
        plot (bool): Whether to display the plot interactively.
        epoch_min (int): Minimum epoch to start plotting from.
    """
    logger.info("Plotting loss curve...")
    # Plotting loss curve with linear scale
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(list_epoch_loss[epoch_min:], label='Training Loss')
    plt.plot(list_val_loss[epoch_min:], '-r', label='Validation Loss')
    plt.title(f"Loss Curve - {title}")  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Plotting loss curve with logarithmic scale
    plt.subplot(2, 1, 2)
    plt.plot(list_epoch_loss[epoch_min:], label='Training Loss')
    plt.plot(list_val_loss[epoch_min:], '-r', label='Validation Loss') 
    plt.xlabel('Epoch')
    plt.xscale('log')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Save the plot
    dir = os.path.join(root_dir, "results", "loss_plots")
    os.makedirs(dir, exist_ok=True)
    loss_plot_path = f"{dir}/{title}_loss.png"
    plt.savefig(loss_plot_path, bbox_inches='tight', dpi=300)
    logger.info(f"Loss curve saved as {loss_plot_path}")

    if plot:
        plt.show()
    plt.close()
    