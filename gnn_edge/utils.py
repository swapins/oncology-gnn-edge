import torch


def ensure_symmetric(adj: torch.Tensor) -> torch.Tensor:
    """
    Ensures adjacency matrix is symmetric.
    """
    return (adj + adj.T) / 2


def add_self_loops(adj: torch.Tensor) -> torch.Tensor:
    """
    Adds identity matrix to adjacency.
    """
    identity = torch.eye(adj.size(0), device=adj.device)
    return adj + identity
