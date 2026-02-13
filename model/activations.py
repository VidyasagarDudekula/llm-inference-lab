import torch
import torch.nn as nn


def sigmoid(x):
    return 1 / (1+ torch.exp(-x))


class SiLuCustome(torch.autograd.Function):
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


if __name__ == '__main__':
    m = SiLuCustome.apply
    x = torch.randn((2, 10), requires_grad=True)
    out = m(x)
    y = torch.randn((2, 10), requires_grad=True)
    loss = (y-out).pow(2).sum()
    loss.backward()
    print(x)
    print(x.grad)