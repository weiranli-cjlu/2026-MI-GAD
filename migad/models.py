from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    """Simple GCN layer, kept for compatibility with the original implementation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        self.bias = Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(inputs, self.weight)
        output = torch.sparse.mm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        return output + self.bias if self.bias is not None else output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class GAD(nn.Module):
    """MI-GAD encoder used by the official implementation."""

    def __init__(self, feat_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(feat_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return torch.tanh(self.lin(x))
