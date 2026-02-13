import torch
import torch.nn as nn
from config import device
device = torch.device('cpu')

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
    

if __name__ == '__main__':
    dim = 10
    seq = 2
    gen = torch.manual_seed(42)
    x = torch.randn((2, seq, dim), requires_grad=True, device=device, generator=gen)
    m = MyLayerNorm(dim).to(device)
    out = m(x)
    default_module = nn.LayerNorm(dim)
    out1 = default_module(x)
    print(list(m.parameters()))
    print((out-out1).abs().max())