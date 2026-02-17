import torch
import model.ops as ops
import torch.nn as nn
import torch.nn.functional as F
from bench.timer import Timer
import config

emb_dim = 50

rms_custom = ops.RMSNorm(dim=emb_dim).to(config.device)
rms_default = nn.RMSNorm(emb_dim).to(config.device)


def test_custom_overall(logits: torch.Tensor):
    logits = ops.SiLU(logits)
    logits = rms_custom(logits)
    probs = ops.stable_softmax(logits, dim=-1)
    
    #backward
    loss = torch.sum(probs)
    loss.backward()

def test_default_overall(logits: torch.Tensor):
    logits = F.silu(logits)
    logits = rms_default(logits)
    probs = F.softmax(logits, dim=-1)
    
    #backward
    loss = torch.sum(probs)
    loss.backward()
    

if __name__ == '__main__':
    time = Timer(iter_steps=1000)
    gen = torch.Generator(config.device)
    gen.manual_seed(42)
    
    logits = torch.randn((10, 2, emb_dim), device=config.device, generator=gen, requires_grad=True)
    print("Custom:- ")
    torch.cuda.reset_max_memory_allocated()
    print(f"Total time:- {time(test_custom_overall, logits)}")
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"Memory:- {max_allocated_memory}\n\n")
    
    print("Default:- ")
    torch.cuda.reset_max_memory_allocated()
    print(f"Total time:- {time(test_default_overall, logits)}")
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"Memory:- {max_allocated_memory}\n\n")
    
    
    
    


