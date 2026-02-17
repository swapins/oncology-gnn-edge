import torch


def generate_synthetic_ppi(num_nodes: int = 100,
                           feature_dim: int = 64,
                           sparse: bool = True,
                           device: torch.device | None = None):

    device = device or torch.device("cpu")

    adj = torch.randint(0, 2, (num_nodes, num_nodes), device=device).float()
    adj = (adj + adj.T) / 2
    adj = (adj > 0).float()

    if sparse:
        adj = adj.to_sparse()

    features = torch.randn(num_nodes, feature_dim, device=device)

    return {
        "adjacency": adj,
        "features": features
    }
