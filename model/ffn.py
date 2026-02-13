import torch.nn as nn
import torch
import torch.nn.functional as F
from model.activations import SiLuCustome


class SwiGLUCustome(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4):
        super().__init__()
        self.dim = dim
        self.multiplier = multiplier
        self.w_gate = nn.Linear(self.dim, self.multiplier*self.dim, bias=False)
        self.w_value = nn.Linear(self.dim, self.multiplier*self.dim, bias=False)
        self.w_proj = nn.Linear(self.multiplier * self.dim, self.dim, bias=False)
        self.activation = SiLuCustome.apply
        self.activation_default = nn.SiLU()
    
    def forward(self, x, default: bool=False):
        out = None
        if default:
            out = self.w_proj(self.activation_default(self.w_gate(x)) * self.w_value(x))
        else:
            out = self.w_proj(self.activation(self.w_gate(x)) * self.w_value(x))
        return out


if __name__ == '__main__':
    dim=15
    m = SwiGLUCustome(dim)
    gen = torch.manual_seed(42)
    x = torch.randn((2, 10, dim), requires_grad=True, generator=gen)
    out1 = m(x)
    out2 = m(x, default=True)
    print(torch.allclose(out1, out2))