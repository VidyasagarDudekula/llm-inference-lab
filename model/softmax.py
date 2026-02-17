import torch
import torch.nn as nn
import torch.nn.functional as F


def stabel_log_softmax(logits: torch.Tensor, targets: torch.Tensor | None = None, dim=-1):
    target_logits = logits
    keep_dim = True
    if targets is not None:
        keep_dim = False
        target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    out = target_logits - torch.logsumexp(logits, dim=dim, keepdim=keep_dim)
    return out

def stable_softmax(logits: torch.Tensor, dim=-1):
    max_values = torch.max(logits, dim=dim).values
    shifted_logits = logits - max_values
    exp_values = torch.exp(shifted_logits)
    out = exp_values/torch.sum(exp_values, dim=dim) 
    return out