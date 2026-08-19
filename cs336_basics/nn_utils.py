import torch
from torch import nn
from einops import einsum, rearrange

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