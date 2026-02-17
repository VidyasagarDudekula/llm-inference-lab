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


def cross_entropy_custome(logits: torch.Tensor, target_idx: torch.Tensor):
    if logits.shape != target_idx.shape:
        logits = torch.gather(logits, dim=-1, index=target_idx.unsqueeze(-1)).squeeze(-1)
    total_sum = -(torch.sum(logits)/target_idx.shape[0])
    return total_sum

class CrossEntropyLossCustome(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor):
        if logits.shape != target_ids.shape:
            logits = torch.gather(logits, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        total_sum = -(torch.sum(logits)/target_ids.shape[0])
        return total_sum


if __name__ == '__main__':
    gen = torch.manual_seed(42)
    logits = torch.randn((10, 2, 50), generator=gen)
    target_ids = torch.randint(0, 50, (10, 2), generator=gen)
    probs1 = stabel_log_softmax(logits)
    loss1 = cross_entropy_custome(probs1.view(-1, 50), target_ids.view(-1))
    print(probs1, loss1)
    print("*****")
    probs2 = stabel_log_softmax(logits, target_ids)
    loss2 = cross_entropy_custome(probs2.view(-1), target_ids.view(-1))
    print(probs2, loss2)
    

