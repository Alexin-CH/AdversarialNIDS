import torch

def attack_fgsm(model,criterion,x_val,target,top_features,eps = 0.01):
    """
    Args:
        model 
        criterion : loss function of the model
        x_val
        y_val : truth wanted
        features : features we want to perturb
        eps : range we are allowed to perturbate

    Returns:
        x_adv : 
    """
    x_adv = x_val.clone().detach().requires_grad_(True)
    pred = model(x_adv)
    loss = criterion(pred,target)
    loss.backward()
    with torch.no_grad():
        x_adv[:, top_features] = x_adv[:, top_features] + eps * x_adv.grad[:, top_features].sign()
    return x_adv
    