import torch
import config
from bench.timer import Timer
from model.activations import SiLuCustome
import torch.nn.functional as F

m_custome = SiLuCustome()

def test_func_prod(inputs, targets):
    out = F.silu(inputs)
    loss = (out-targets).pow(2).sum()
    loss.backward()


def test_func_custome(inputs, targets):
    out = m_custome(inputs)
    loss = (out-targets).pow(2).sum()
    loss.backward()


if __name__ == '__main__':
    timer_obj = Timer()
    
    inputs = torch.randn((1024, 1024, 16), dtype=torch.float32, requires_grad=True, device=config.device)
    targets = torch.randn((1024, 1024, 16), dtype=torch.float32, requires_grad=True, device=config.device)
    
    print("Default PyTorch C++ SiLu:- ", timer_obj(test_func_prod, inputs, targets))
    print("Custome Python SiLu:- ", timer_obj(test_func_custome, inputs, targets))