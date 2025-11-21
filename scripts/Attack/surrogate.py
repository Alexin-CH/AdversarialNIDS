import torch

def attack_fgsm(model,criterion,x_val,y_val,eps = 0.01):
    x_adv = x_val.clone().detach().requires_grad_(True)
    pred = model(x_adv)
    loss = criterion(pred,y_val)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv
    