import torch

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
    loss = criterion(pred, target) # target needs to be class indices, not one-hot. Pred is raw logits

    loss.backward()

    with torch.no_grad():
        x_adv = x_adv - eps * x_adv.grad.sign()

    return x_adv.detach(), loss.detach().cpu().numpy(), pred.detach().cpu().numpy()
    