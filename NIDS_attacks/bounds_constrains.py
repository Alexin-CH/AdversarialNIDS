import torch

def apply_bounds_constraints(x_adv, x_original, modifiable_indices, min_vals, max_vals):
    """
    Apply min/max bounds and modifiability constraints.
    
    Args:
        x_adv: Adversarial tensor (batch_size, n_features)
        x_original: Original tensor (for non-modifiable features)
        modifiable_indices: List of indices that can be modified
        min_vals: Tensor of min values per feature
        max_vals: Tensor of max values per feature
    Returns:
        Constrained adversarial tensor
    """
    with torch.no_grad():
        # Keep only modifiable features changed
        mask = torch.zeros(x_adv.shape[1], device=x_adv.device)
        mask[modifiable_indices] = 1.0
        x_constrained = x_original + (x_adv - x_original) * mask
        
        # Clip to bounds
        x_constrained = torch.clamp(x_constrained, min=min_vals, max=max_vals)
    
    return x_constrained


