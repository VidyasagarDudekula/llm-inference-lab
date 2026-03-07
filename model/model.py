import torch
import torch.nn as nn
from model.ops import RMSNorm
from model.layers import SwiGLUCustom, SelfAttention



class DecoderBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, q_heads: int, kv_heads: int, head_dim: int, is_causal: bool):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.is_causal = is_causal

        self.self_attns = SelfAttention(self.dim, self.seq_len, self.q_heads, self.kv_heads, self.head_dim, self.is_causal)
        self.mlp_layer = SwiGLUCustom(self.dim)
        self.norm1 = RMSNorm(self.dim)
        self.norm2 = RMSNorm(self.dim)
    
    def forward(self, x, mask = None):
        out = x + self.self_attns(self.norm1(x), mask)
        return out + self.mlp_layer(self.norm2(out))


if __name__ == '__main__':
    dim=512
    seq_len = 10
    
    q_heads = 4
    head_dim = dim//q_heads
    kv_heads = q_heads//2
    with torch.inference_mode():
        gen = torch.manual_seed(42)
        x = torch.randn((2, 10, dim), generator=gen)

        m = DecoderBlock(dim, seq_len, q_heads, kv_heads, head_dim, True)
        out = m(x)
        print(out.shape)

        