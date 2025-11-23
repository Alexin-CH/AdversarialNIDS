import torch

def compute_features_batch(X, dico,eps=1e-6):
    
    dur    = X[:, dico['Flow Duration']]
    sbytes = X[:, dico['Total Length of Fwd Packets']]
    dbytes = X[:, dico['Total Length of Bwd Packets']]
    spkts  = X[:, dico['Total Fwd Packets']]
    dpkts  = X[:, dico['Total Backward Packets']]

    dur_safe   = torch.clamp(dur, min=eps)
    spkts_safe = torch.clamp(spkts, min=1)
    dpkts_safe = torch.clamp(dpkts, min=1)

    X[:, dico['Fwd Packets/s']]       = spkts / dur_safe
    X[:, dico['Bwd Packets/s']]       = dpkts / dur_safe
    X[:, dico['Avg Fwd Segment Size']] = sbytes / spkts_safe
    X[:, dico['Avg Bwd Segment Size']] = dbytes / dpkts_safe
    X[:, dico['Sintpkt']]             = dur / (spkts - 1 + eps)
    X[:, dico['Dintpkt']]             = dur / (dpkts - 1 + eps)

    return X