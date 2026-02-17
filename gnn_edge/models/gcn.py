import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Basic Graph Convolution Layer.
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Node features (N x F)
        adj: Adjacency matrix (N x N)
        """

        # Add self-loops
        identity = torch.eye(adj.size(0), device=adj.device)
        adj_hat = adj + identity

        # Degree matrix (clamp to avoid division by zero / inf)
        degree = torch.sum(adj_hat, dim=1)
        degree = torch.clamp(degree, min=self.eps)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt_mat = torch.diag(degree_inv_sqrt)

        # Symmetric normalization
        adj_normalized = degree_inv_sqrt_mat @ adj_hat @ degree_inv_sqrt_mat

        x = self.linear(x)
        return F.relu(adj_normalized @ x)
