import torch
from torch import nn
from einops import einsum, rearrange
import math
from cs336_basics.nn_utils import silu

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        ## 参数初始化
        std = math.sqrt(2 / (in_features + out_features))

        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))  ## gain


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        ## 防止平方溢出，先转为 fp32
        in_dtype = x.dtype
        x = x.to(torch.float32)

        squared = x ** 2

        mean_sq = squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        normalized = x / rms
        out = normalized * self.weight
        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            times = round((8 / 3 * self.d_model) / 64)
            self.d_ff = times * 64
            assert self.d_ff % 64 == 0

        else:
            self.d_ff = d_ff

        if self.d_ff < 1:
            raise ValueError("d_ff must be at least 1, got 0")
        
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k

        j_list = torch.arange(start=0, end=d_k // 2, dtype=torch.float32, device=device)
        f_k = theta ** -(2 * j_list / d_k)
        place_list = torch.arange(start=0, end=max_seq_len, dtype=torch.float32, device=device)
        degree = einsum(place_list, f_k, "i, j -> i j")

        self.register_buffer("cos", torch.cos(degree), persistent=False)
        self.register_buffer("sin", torch.sin(degree), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_table = self.cos[token_positions]
        sin_table = self.sin[token_positions]
        x_pairs = rearrange(x, "... seq (pairs two) -> ... seq pairs two", pairs=self.d_k // 2, two=2)
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        a_rotate = a * cos_table - b * sin_table
        b_rotate = a * sin_table + b * cos_table

        x_pairs_rotate = torch.stack([a_rotate, b_rotate], dim=-1)

        x = rearrange(x_pairs_rotate, "... seq pairs two -> ... seq (pairs two)", pairs= self.d_k // 2, two=2)

        return x


if __name__ == "__main__":
    swiglu = SwiGLU(d_model=8)