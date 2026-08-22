import torch
from torch import nn
from einops import einsum, rearrange

def softmax(x, dim):
    """对dim维做softmax"""
    max_x = torch.amax(x, dim=dim, keepdim=True)
    x = x - max_x   ## 加常数不会影响softmax 的结果，这里减去最大值后，所有的x 都小于0，e^x 就不会溢出了
    exp_x = torch.exp(x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def silu(x):
    return x * torch.sigmoid(x)