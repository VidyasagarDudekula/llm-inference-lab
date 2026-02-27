from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid(x):
    return 1 / (1+ torch.exp(-x))


def clip_grad_norm_(model_parameters, max_norm=1.0, eps=0.0001):
    with torch.no_grad():
        mp = list(model_parameters)
        if len(mp)==0:
            return
        val = torch.tensor(0.0, device=mp[0].device)
        for param in mp:
            if param.grad is None:
                continue
            val.add_(torch.linalg.vector_norm(param.grad, ord=2, dtype=torch.float32).square())
        val.sqrt_()
        if val<max_norm:
            return
        scale = max_norm/(val + eps)
        for param in mp:
            if param.grad is None:
                continue
            param.grad.mul_(scale)


class SiLuCustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sigmoid_value = sigmoid(x)
        out = x * sigmoid_value
        ctx.save_for_backward(x, sigmoid_value)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid_values = ctx.saved_tensors
        df_dx = sigmoid_values + x * sigmoid_values * (1 - sigmoid_values)
        return df_dx * grad_output
    

def SiLU(logits: torch.Tensor):
    m = SiLuCustom.apply
    return m(logits)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones((1, self.dim)))
    
    def forward(self, x):
        var_values = torch.mean(x**2, dim=-1, keepdim=True)
        norm_values = x / torch.sqrt(var_values + self.eps)
        out = norm_values * self.alpha
        return out
    
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps: float = 1e-5):
        H = x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        var = (x_centered * x_centered).mean(dim=-1, keepdim=True)

        inv_std = torch.rsqrt(var + eps)
        x_hat = x_centered * inv_std

        y = x_hat * gamma + beta
        ctx.save_for_backward(x_hat, inv_std, gamma)
        ctx.H = H
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x_hat, inv_std, gamma = ctx.saved_tensors
        H = ctx.H

        u = grad_out * gamma 
        sum_u = u.sum(dim=-1, keepdim=True)                  
        sum_u_xhat = (u * x_hat).sum(dim=-1, keepdim=True)

        grad_x = inv_std * (u - sum_u / H - x_hat * (sum_u_xhat / H))

        reduce_dims = tuple(range(grad_out.dim() - 1))
        grad_gamma = (grad_out * x_hat).sum(dim=reduce_dims)


        grad_beta = grad_out.sum(dim=reduce_dims)   

        return grad_x, grad_gamma, grad_beta, None


class MyLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.gamma, self.beta, self.eps)


class SwiGLUCustom(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4):
        super().__init__()
        self.dim = dim
        self.multiplier = multiplier
        self.w_gate = nn.Linear(self.dim, self.multiplier*self.dim, bias=False)
        self.w_value = nn.Linear(self.dim, self.multiplier*self.dim, bias=False)
        self.w_proj = nn.Linear(self.multiplier * self.dim, self.dim, bias=False)
        self.activation = SiLuCustom.apply
        self.activation_default = nn.SiLU()
    
    def forward(self, x, default: bool=False):
        out = None
        if default:
            out = self.w_proj(self.activation_default(self.w_gate(x)) * self.w_value(x))
        else:
            out = self.w_proj(self.activation(self.w_gate(x)) * self.w_value(x))
        return out


def stable_softmax(logits: torch.Tensor, dim=-1):
    max_values = torch.max(logits, dim=dim, keepdim=True).values
    shifted_logits = logits - max_values
    exp_values = torch.exp(shifted_logits)
    out = exp_values/torch.sum(exp_values, dim=dim, keepdim=True) 
    return out

def stable_log_softmax(logits: torch.Tensor, targets: torch.Tensor | None = None, dim=-1):
    target_logits = logits
    keep_dim = True
    if targets is not None:
        keep_dim = False
        target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    out = target_logits - torch.logsumexp(logits, dim=dim, keepdim=keep_dim)
    return out

class CrossEntropyLossCustom(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor):
        if logits.shape != target_ids.shape:
            logits = torch.gather(logits, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        total_sum = -(torch.sum(logits)/target_ids.shape[0])
        return total_sum
    

if __name__ == '__main__':
    logits = torch.randn((2, 50))
    out = stable_softmax(logits)


