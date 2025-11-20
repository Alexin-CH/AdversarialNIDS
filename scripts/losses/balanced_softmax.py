import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Cross-Entropy Loss.
    Args:
        sample_per_class : repartition of every classes in the training set
    """
    def __init__(self, samples_per_class):
        super().__init__()
        self.register_buffer("samples_per_class", torch.tensor(samples_per_class, dtype=torch.float))

    def forward(self, logits, targets):
        adjusted_logits = logits + torch.log(self.samples_per_class + 1e-12)

        loss = F.cross_entropy(adjusted_logits, targets)
        return loss