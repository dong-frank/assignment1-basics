import torch
from torch import nn
from einops import einsum, rearrange
import math

def softmax(x, dim):
    """对dim维做softmax"""
    max_x = torch.amax(x, dim=dim, keepdim=True)
    x = x - max_x   ## 加常数不会影响softmax 的结果，这里减去最大值后，所有的x 都小于0，e^x 就不会溢出了
    exp_x = torch.exp(x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def silu(x):
    return x * torch.sigmoid(x)

def scaled_dot_product_attention(Q, K, V, mask):
    attention_score = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    attention_score = attention_score / math.sqrt(Q.shape[-1])

    attention_score = torch.masked_fill(attention_score, ~mask, float("-inf"))
    
    softmax_attention_score = softmax(attention_score, -1)
    return einsum(softmax_attention_score, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
