import torch.nn as nn
import torch
import torch.nn.functional as F
from model.ops import SiLuCustom, stable_softmax


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


class SelfAttention(nn.Module):
    def __init__(self, dim: int, seq_len: int, q_heads: int, kv_heads :int, head_dim: int, is_causal: bool):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.is_causal = is_causal
        self.w_q = nn.Linear(dim, self.q_heads * head_dim)
        self.w_k = nn.Linear(dim, self.head_dim * self.kv_heads)
        self.w_v = nn.Linear(dim, self.head_dim * self.kv_heads)
        self.w_o = nn.Linear(self.dim, self.dim)
        self.multiplier = self.q_heads // self.kv_heads

        self.register_buffer('tril', torch.tril(torch.ones(self.seq_len, self.seq_len).bool()))

    def forward(self, x, mask: torch.Tensor | None):
        B, T, C = x.shape
        
        q = self.w_q(x).reshape(B, T, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).reshape(B, T, self.kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.multiplier, dim=1)
        v = self.w_v(x).reshape(B, T, self.kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.multiplier, dim=1)

        qk = q @ k.transpose(-2, -1)
        qk = qk * (self.head_dim ** -0.5)

        if self.is_causal:
            qk.masked_fill(~self.tril[:T, :T], value=float('-inf'))
        
        if mask is not None:
            qk.masked_fill(~mask[:, None, None, :], value=float('-inf'))
        
        qk = stable_softmax(qk, dim=-1)
        
        attn = qk @ v
        attn = attn.transpose(1, 2).reshape(B, T, C)
        attn = self.w_o(attn)
        return attn


if __name__ == '__main__':
    dim=512
    seq_len = 10
    
    q_heads = 4
    head_dim = dim//q_heads
    kv_heads = q_heads//2
    # m = SwiGLUCustom(dim)
    gen = torch.manual_seed(42)
    x = torch.randn((2, 10, dim), requires_grad=True, generator=gen)
    # out1 = m(x)
    # out2 = m(x, default=True)
    # print(torch.allclose(out1, out2))
    m = SelfAttention(dim, seq_len, q_heads, kv_heads, head_dim, True)
    out = m(x, None)
    print(out.shape)