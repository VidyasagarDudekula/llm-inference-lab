import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random



class SGDVanilla():
    def __init__(self, model_parameters, lr=0.0001):
        self.lr = lr
        self.model_parameters = list(model_parameters)
    
    def step(self):
        with torch.no_grad():
            for p in self.model_parameters:
                if p.grad is None:
                    continue
                p.sub_(p.grad, alpha=self.lr)
    
    def zero_grad(self):
        with torch.no_grad():
            for param in self.model_parameters:
                if param.grad is None:
                    continue
                param.grad.zero_()



class SGDOptimizer():
    def __init__(self, model_parameters, lr=0.0001, M=0.9):
        self.M = M
        self.lr = lr
        self.model_parameters = list(model_parameters)
        self.states = defaultdict(dict)
    
    def step(self):
        print("New")
        with torch.no_grad():
            for param in self.model_parameters:
                print("Updating Prameters")
                if param.grad is None:
                    continue
                state = self.states[param]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['moving_average'] = torch.zeros_like(param.data, dtype=param.dtype)
                
                state['step'] += 1
                state['moving_average'].mul_(self.M).add_(param.grad)
                
                param.sub_(state['moving_average'], alpha=self.lr)
                
    def zero_grad(self):
        with torch.no_grad():
            for param in self.model_parameters:
                if param.grad is None:
                    continue
                param.grad.zero_()
                
        



if __name__ == '__main__':
    from config import device
    from model.ops import CrossEntropyLossCustom, stable_softmax
    gen = torch.Generator(device)
    gen.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 30),
        nn.Linear(30, 10),
    ).to(device)
    optimizer = SGDOptimizer(model.parameters())
    logits = torch.randn((5, 10), device=device, generator=gen, requires_grad=True)
    target_ids = torch.randint(0, 10, (5, 1), device=device, generator=gen)
    
    out = stable_softmax(model(logits), dim=-1)
    criterion = CrossEntropyLossCustom()
    loss = criterion(out, target_ids.view(-1,))
    print(loss)
    loss.backward()
    optimizer.step()
    optimizer.step()
    optimizer.zero_grad()

