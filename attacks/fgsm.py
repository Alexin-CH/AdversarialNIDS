import os
import sys
import torch

root_dir = os.getcwd().split("AdversarialNIDS")[0] + "AdversarialNIDS"
sys.path.append(root_dir)

from CICIDS2017.analysis.features import recompute_features

from attacks.bounds_constrains import apply_bounds_constraints
from attacks.integers_constrains import apply_integer_constraints

def attack_fgsm(model, criterion, x_val, target, top_features=None, eps = 0.01):
    """
    Args:
        model 
        criterion : loss function of the model
        x_val : input we want to perturb
        y_val : truth wanted
        features : features we want to perturb
        eps : range we are allowed to perturbate

    Returns:
        x_adv : 
    """
    x_adv = x_val.clone().detach().requires_grad_(True)

    pred = model(x_adv)

    diff = torch.norm(x_adv - x_val)

    # target needs to be class indices, not one-hot. Pred is raw logits
    loss = criterion(pred, target) + diff

    loss.backward()

    with torch.no_grad():
        x_adv = x_adv - eps * x_adv.grad.sign()

    return x_adv.detach(), loss.detach().cpu().numpy(), pred.detach().cpu().numpy()
    

def attack(model, x_adv, target, X_train, dataset, logger, device='cpu'):
    """
    Perform FGSM attack on the given model and dataset.

    Args:
        x_adv : input data to be perturbed
        compute_features : function to compute derived features
        logger : logger for logging information
        device : device to run the computations on ('cpu' or 'cuda')
    Returns:
        x_adv : adversarial examples generated from the input
    """
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    x_adv = x_adv.to(device)
    target = target.to(device)

    X_train = X_train.cpu()
    
    model.eval()
    epsilon = 1

    # Initial prediction
    initial_x_adv, initial_loss, initial_pred = attack_fgsm(
        model=model,
        criterion=criterion,
        x_val=x_adv,
        target=target,
        eps=0.0
    )

    logger.info(f"Initial prediction on adversarial input: {initial_pred.argmax(axis=1)}")

    for iter in range(200):
        new_x_adv, loss, pred = attack_fgsm(
            model=model,
            criterion=criterion,
            x_val=x_adv,
            target=target,
            eps=epsilon
        )

        # Recompute derived features
        x_adv = recompute_features(new_x_adv)

        # Apply bounds constraints
        x_adv = apply_bounds_constraints(
            x_adv=new_x_adv,
            x_original=x_adv,
            modifiable_indices=dataset.MODIFIABLE_FEATURES,
            min_vals=torch.FloatTensor(X_train).min(axis=0).values.to(device),
            max_vals=torch.FloatTensor(X_train).max(axis=0).values.to(device)
        )

        # Check if attack is successful
        if pred.argmax(axis=1).sum() == 0:
            x_adv = apply_integer_constraints(
                x_adv=new_x_adv,
                integer_indices=dataset.INTEGER_INDICES
            )

            # Final evaluation without perturbation
            x_adv, loss, pred = attack_fgsm(
                model=model,
                criterion=criterion,
                x_val=x_adv,
                target=target,
                eps=0.0
            )

            # Check if attack is successful after applying constraints
            if pred.argmax(axis=1).sum() == 0:
                logger.info(f"Successful adversarial example found at iteration {iter+1}")
                break

    logger.info(f"Adversarial input after {iter+1} iterations: {pred.argmax(axis=1)}")
    diff_adv = x_adv - initial_x_adv
    logger.info("Magnitude of perturbation:")
    logger.info(f"Mean: {torch.norm(diff_adv, dim=1).cpu().numpy().mean()}")
    logger.info(f"Min: {torch.norm(diff_adv, dim=1).cpu().numpy().min()}")
    logger.info(f"Max: {torch.norm(diff_adv, dim=1).cpu().numpy().max()}")
    logger.info(f"Std: {torch.norm(diff_adv, dim=1).cpu().numpy().std()}")

    return x_adv.detach().cpu()
