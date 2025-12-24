import torch

def apply_integer_constraints(x_adv, integer_indices):
    """
    Round integer features.
    
    Args:
        x_adv: Adversarial tensor (batch_size, n_features)
        integer_indices: List of indices that should be integers
    
    Returns:
        Tensor with integer features rounded
    """
    with torch.no_grad():
        int_mask = torch.zeros(x_adv.shape[1], device=x_adv.device)
        int_mask[integer_indices] = 1.0
        
        rounded = x_adv.round()
        x_constrained = torch.where(int_mask == 1, rounded, x_adv)
    
    return x_constrained