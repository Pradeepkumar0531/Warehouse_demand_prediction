# First-principles scaled dot-product self-attention.

import math

import torch
from torch import nn


class SelfAttention(nn.Module):

    def __init__(self, d_model, d_k=None, d_v=None, scale=True):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        self.scale = scale

        self.W_Q = nn.Parameter(torch.randn(d_model, self.d_k))
        self.W_K = nn.Parameter(torch.randn(d_model, self.d_k))
        self.W_V = nn.Parameter(torch.randn(d_model, self.d_v))

    def forward(self, X):
        # X: (batch, sequence_length, d_model)
        Q = X @ self.W_Q  
        K = X @ self.W_K
        V = X @ self.W_V

        scores = Q @ K.transpose(-2, -1) # Q*K^T

        if self.scale:
            scores = scores / math.sqrt(self.d_k)  # scaling

        # Stable softmax: subtract the largest score before exp().
        scores = scores - scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(scores)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        output = weights @ V

        return output, weights
