from typing import DefaultDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        with torch.no_grad():
            for param in self.model_parameters:
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
                

class AdamOptimizer:
    def __init__(self, model_parameters, lr=0.0001, M1 = 0.9, M2 = 0.9, bias = 0.9, eps = 1e-7):
        self.M1 = M1
        self.M2 = M2
        self.lr = lr
        self.bias = bias
        self.model_parameters = list(model_parameters)
        self.states = defaultdict(dict)
        self.eps = eps
    
    def step(self):
        with torch.no_grad():
            for param in self.model_parameters:
                state = self.states[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['moving_average_m1'] = torch.zeros_like(param.data)
                    state['moving_average_m2'] = torch.zeros_like(param.data)
                state['step'] += 1
                state['moving_average_m1'] .mul_(self.M1).add_(param.grad, alpha=(1 - self.M1))
                moment1 = state['moving_average_m1'] / (1-(self.bias**state['step']))

                state['moving_average_m2'].mul_(self.M2).addcmul_(param.grad, param.grad, value=(1 - self.M2))
                moment2 = state['moving_average_m2'] / (1-(self.bias**state['step']))

                moment = moment1/(moment2**0.5 + self.eps)

                param.sub_(moment, alpha=self.lr)
    
    def zero_grad(self):
        with torch.no_grad():
            for param in self.model_parameters:
                if param.grad is None:
                    continue
                param.grad.zero_()


class AdamWOptimizer:
    def __init__(self, model_parameters, lr=0.0001, M1 = 0.9, M2 = 0.9, bias1 = 0.9, bias2 = 0.999, eps = 1e-7, weight_decay = 0.2):
        self.M1 = M1
        self.M2 = M2
        self.lr = lr
        self.bias1 = bias1
        self.bias2 = bias2
        self.model_parameters = list(model_parameters)
        self.states = defaultdict(dict)
        self.eps = eps
        self.weight_decay = weight_decay
    
    def step(self):
        with torch.no_grad():
            for param in self.model_parameters:
                if param.grad is None:
                    continue
                state = self.states[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['moving_average_m1'] = torch.zeros_like(param.data)
                    state['moving_average_m2'] = torch.zeros_like(param.data)
                state['step'] += 1
                state['moving_average_m1'] .mul_(self.M1).add_(param.grad, alpha=(1 - self.M1))
                moment1 = state['moving_average_m1'] / (1-(self.bias1**state['step']))

                state['moving_average_m2'].mul_(self.M2).addcmul_(param.grad, param.grad, value=(1 - self.M2))
                moment2 = state['moving_average_m2'] / (1-(self.bias2**state['step']))

                moment = moment1/(moment2**0.5 + self.eps)
                param.mul_(1 - self.lr * self.weight_decay)
                param.sub_(moment, alpha=self.lr)
    
    def zero_grad(self):
        with torch.no_grad():
            for param in self.model_parameters:
                if param.grad is None:
                    continue
                param.grad.zero_()



if __name__ == '__main__':
    from config import device
    from model.ops import CrossEntropyLossCustom, stable_softmax, clip_grad_norm_
    gen = torch.Generator(device)
    gen.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 30),
        nn.Linear(30, 10),
    ).to(device)
    clip_grad_norm_(model.parameters())
    optimizer = AdamWOptimizer(model.parameters())
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

