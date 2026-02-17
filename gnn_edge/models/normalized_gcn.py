import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedGCNLayer(nn.Module):
    """
    Stability-aware Graph Convolution Layer with sparse adjacency support.
    Designed for CPU-efficient inference.
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (N x F)
        adj: (N x N) dense or sparse
        """

        if adj.is_sparse:
            return self._forward_sparse(x, adj)
        else:
            return self._forward_dense(x, adj)

    def _forward_dense(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

        degree = torch.sum(adj_hat, dim=1)
        degree = torch.clamp(degree, min=self.eps)
        degree_inv_sqrt = torch.pow(degree, -0.5)

        adj_normalized = adj_hat * degree_inv_sqrt.view(-1, 1)
        adj_normalized = adj_normalized * degree_inv_sqrt.view(1, -1)

        x = self.linear(x)
        return F.relu(adj_normalized @ x)

    def _forward_sparse(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        # Add self loops
        identity = torch.eye(adj.size(0), device=adj.device).to_sparse()
        adj_hat = adj + identity

        # Degree calculation
        degree = torch.sparse.sum(adj_hat, dim=1).to_dense()
        degree = torch.clamp(degree, min=self.eps)
        degree_inv_sqrt = torch.pow(degree, -0.5)

        # Normalize sparse matrix
        row, col = adj_hat.indices()
        values = adj_hat.values()

        norm_values = (
            degree_inv_sqrt[row] *
            values *
            degree_inv_sqrt[col]
        )

        adj_normalized = torch.sparse_coo_tensor(
            adj_hat.indices(),
            norm_values,
            adj_hat.size()
        )

        x = self.linear(x)
        return F.relu(torch.sparse.mm(adj_normalized, x))
